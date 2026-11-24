"""LLM client for chat completion interactions."""

import json
import logging
import time
from typing import Any

import requests

from .prompts import PromptConfig, load_prompt_config

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM service is unavailable, unreachable, or returns empty response."""


class LLMClient:
    """Client for OpenAI-compatible LLM endpoints."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "qwen2.5:3b",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        base_url: str = "http://localhost:11434/v1",
        timeout: tuple[float, float] | float = (5.0, 120.0),
        max_tokens: int = 1024,
        prompt_config: PromptConfig | None = None,
        max_total_seconds: float = 180.0,
    ):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        if isinstance(timeout, tuple):
            self.timeout: tuple[float, float] = (float(timeout[0]), float(timeout[1]))
        else:
            self.timeout = (5.0, float(timeout))
        self.max_tokens = max_tokens
        # Upper limit for ONE question, counting every attempt and every pause between them.
        self.max_total_seconds = float(max_total_seconds)
        self.prompt_config = prompt_config or load_prompt_config()

        clean_url = base_url.rstrip("/")
        if not clean_url.endswith("/chat/completions"):
            self.base_url = f"{clean_url}/chat/completions"
        else:
            self.base_url = clean_url

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def check_availability(self) -> tuple[bool, int | None, str]:
        """Check availability of the configured model with a minimal prompt."""
        headers = self._get_headers()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        }
        try:
            response = requests.post(
                self.base_url, headers=headers, json=payload, timeout=self.timeout
            )
            if response.status_code == 200:
                return True, 200, "OK"
            short_msg = response.text[:200].strip()
            return False, response.status_code, short_msg
        except requests.RequestException as e:
            logger.warning("LLM availability check failed: %s", e)
            return False, None, "Service unavailable"

    def check_server(self, timeout: tuple[float, float] = (3.0, 5.0)) -> tuple[bool, str]:
        """Quickly ask the server which models it has. Does not generate any text.

        Returns:
            (ok, short description). ok is False if the server is unreachable, returns an
            error, or does not list the configured model.
        """
        models_url = self.base_url.removesuffix("/chat/completions") + "/models"
        try:
            response = requests.get(models_url, headers=self._get_headers(), timeout=timeout)
        except requests.RequestException:
            return False, "server unreachable"

        if response.status_code in (404, 405):
            return True, "server reachable (it has no model list)"
        if response.status_code != 200:
            return False, f"server returned HTTP {response.status_code}"

        try:
            model_ids = [item.get("id") for item in response.json().get("data", [])]
        except (ValueError, AttributeError):
            return True, "server reachable"
        if self.model in model_ids:
            return True, f"model '{self.model}' is available"
        return False, f"server reachable, but model '{self.model}' is not listed"

    def _timeout_within(self, seconds_left: float) -> tuple[float, float]:
        """Return the (connect, read) timeouts, shortened so they never pass the deadline."""
        return (min(self.timeout[0], seconds_left), min(self.timeout[1], seconds_left))

    def _wait_before_retry(self, delay: float, deadline: float) -> bool:
        """Sleep before the next attempt. Return False if that would pass the deadline."""
        if time.monotonic() + delay >= deadline:
            logger.warning(
                "Not retrying the LLM request: too little of the %.0f s time limit is left",
                self.max_total_seconds,
            )
            return False
        time.sleep(delay)
        return True

    def send_request(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str | None:
        """Send chat completion request with exponential backoff retry.

        All attempts together never take longer than max_total_seconds.
        """
        headers = self._get_headers()
        tokens_limit = max_tokens if max_tokens is not None else self.max_tokens
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": tokens_limit,
        }
        deadline = time.monotonic() + self.max_total_seconds

        for attempt in range(self.max_retries):
            seconds_left = deadline - time.monotonic()
            if seconds_left <= 0:
                logger.error(
                    "Giving up on the LLM request: the %.0f s time limit is used up",
                    self.max_total_seconds,
                )
                return None

            delay = self.retry_delay * (2**attempt)
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_within(seconds_left),
                )

                if response.status_code == 200:
                    data = response.json()
                    return self._extract_response(data)

                if response.status_code == 429:
                    short_msg = response.text[:200].strip()
                    logger.warning(
                        "Rate limit hit (HTTP 429: %s), attempt %d/%d",
                        short_msg,
                        attempt + 1,
                        self.max_retries,
                    )
                    if (
                        attempt == 0
                        and self.max_retries > 1
                        and self._wait_before_retry(delay, deadline)
                    ):
                        continue
                    return None

                if response.status_code == 404 or "not found" in response.text.lower():
                    short_msg = response.text[:200].strip()
                    logger.error(
                        "Configured model '%s' was not found on LLM endpoint (HTTP %d: %s). "
                        "Verify model availability on your LLM server and update LLM_MODEL in .env.",
                        self.model,
                        response.status_code,
                        short_msg,
                    )
                    return None

                short_msg = response.text[:200].strip()
                logger.error(
                    "LLM API request failed: HTTP %s - %s",
                    response.status_code,
                    short_msg,
                )
                if attempt < self.max_retries - 1 and self._wait_before_retry(delay, deadline):
                    continue
                return None

            except requests.exceptions.Timeout:
                logger.warning(
                    "Request timeout after %ss, attempt %d/%d",
                    self.timeout,
                    attempt + 1,
                    self.max_retries,
                )
                if attempt < self.max_retries - 1 and self._wait_before_retry(delay, deadline):
                    continue
                return None

            except (requests.RequestException, json.JSONDecodeError) as e:
                logger.error("Request failed: %s", e)
                if attempt < self.max_retries - 1 and self._wait_before_retry(delay, deadline):
                    continue
                return None

        return None

    def _extract_response(self, data: dict[str, Any]) -> str | None:
        """Extract response text from API response data."""
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            if finish_reason == "length" and content is not None:
                content = f"{content.rstrip()}\n\n(Answer was cut off; ask a narrower question for the full detail.)"
            return content
        except (KeyError, IndexError) as e:
            logger.error("Failed to extract response: %s", e)
            return None

    def format_prompt(self, question: str, context: str) -> list[dict[str, str]]:
        """Format question and context into prompt messages."""
        system_message = {"role": "system", "content": self.prompt_config.system_prompt}
        user_message = {
            "role": "user",
            "content": self.prompt_config.format_user_prompt(context=context, question=question),
        }
        return [system_message, user_message]

    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer for a question using context.

        Raises:
            LLMUnavailableError: If request fails or returns empty response.
        """
        if not context:
            return "I don't have any relevant information to answer this question. Please upload a PDF document first."

        messages = self.format_prompt(question, context)
        raw = self.send_request(messages)
        if not raw or not raw.strip():
            raise LLMUnavailableError(
                f"LLM model '{self.model}' unavailable or returned empty response."
            )
        return raw.strip()

    def validate_response(self, response: str | None) -> str:
        """Clean response text, stripping whitespace."""
        return response.strip() if response else ""
