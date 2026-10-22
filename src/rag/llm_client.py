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

    def send_request(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str | None:
        """Send chat completion request with exponential backoff retry."""
        headers = self._get_headers()
        tokens_limit = max_tokens if max_tokens is not None else self.max_tokens
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": tokens_limit,
        }

        for attempt in range(self.max_retries):
            delay = self.retry_delay * (2**attempt)
            try:
                response = requests.post(
                    self.base_url, headers=headers, json=payload, timeout=self.timeout
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
                    if attempt == 0 and self.max_retries > 1:
                        time.sleep(delay)
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
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    continue
                return None

            except requests.exceptions.Timeout:
                logger.warning(
                    "Request timeout after %ss, attempt %d/%d",
                    self.timeout,
                    attempt + 1,
                    self.max_retries,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    continue
                return None

            except (requests.RequestException, json.JSONDecodeError) as e:
                logger.error("Request failed: %s", e)
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
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
