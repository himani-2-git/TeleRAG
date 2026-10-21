"""LLM client for chat completion interactions."""

import logging
import time
from typing import Any

import requests

from .prompts import PromptConfig, load_prompt_config

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for OpenAI-compatible LLM endpoints."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "qwen2.5:3b",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        base_url: str = "http://localhost:11434/v1",
        timeout: float = 300.0,
        max_tokens: int = 300,
        prompt_config: PromptConfig | None = None,
    ):
        """Initialize LLM client.

        Args:
            api_key: API key (optional for local endpoints)
            model: Model identifier to use
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            base_url: Base URL for LLM API (chat/completions endpoint)
            timeout: Request timeout in seconds (defaults to 300s for local models)
            max_tokens: Maximum tokens in generated completion (default 300)
            prompt_config: Optional PromptConfig instance (defaults to loaded config)
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.prompt_config = prompt_config or load_prompt_config()

        # Ensure base_url ends with /chat/completions
        clean_url = base_url.rstrip("/")
        if not clean_url.endswith("/chat/completions"):
            self.base_url = f"{clean_url}/chat/completions"
        else:
            self.base_url = clean_url

    def check_availability(self) -> tuple[bool, int | None, str]:
        """Check availability of the configured model with a minimal prompt.

        Returns:
            Tuple of (is_ok, status_code, reason)
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

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
            else:
                short_msg = response.text[:200].strip()
                return False, response.status_code, short_msg
        except Exception as e:
            return False, None, str(e)

    def send_request(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str | None:
        """Send chat completion request to LLM server.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Optional token limit override (defaults to self.max_tokens)

        Returns:
            Generated response text or None if failed
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        tokens_limit = max_tokens if max_tokens is not None else self.max_tokens
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": tokens_limit,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url, headers=headers, json=payload, timeout=self.timeout
                )

                if response.status_code == 200:
                    data = response.json()
                    return self._extract_response(data)

                elif response.status_code == 429:
                    # Rate limit hit: log response message (never keys) and do not retry more than once
                    short_msg = response.text[:200].strip()
                    logger.warning(
                        f"Rate limit hit (HTTP 429: {short_msg}), attempt {attempt + 1}/{self.max_retries}"
                    )
                    # Retry at most once (only when attempt == 0)
                    if attempt == 0 and self.max_retries > 1:
                        time.sleep(self.retry_delay)
                        continue
                    return None

                elif response.status_code == 404 or "not found" in response.text.lower():
                    # Extract short message if available in response
                    short_msg = response.text[:200].strip()
                    logger.error(
                        "Configured model '%s' was not found on LLM endpoint (HTTP %d: %s). "
                        "Verify model availability on your LLM server and update LLM_MODEL in your .env file.",
                        self.model,
                        response.status_code,
                        short_msg,
                    )
                    return None

                else:
                    short_msg = response.text[:200].strip()
                    logger.error(
                        f"LLM API request failed: HTTP {response.status_code} - {short_msg}"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return None

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Request timeout after {self.timeout}s, attempt {attempt + 1}/{self.max_retries}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return None

            except Exception as e:
                logger.error(f"Request failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return None

        return None

    def _extract_response(self, data: dict[str, Any]) -> str | None:
        """Extract response text from API response.

        Args:
            data: API response data

        Returns:
            Response text or None
        """
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract response: {e}")
            return None

    def format_prompt(self, question: str, context: str) -> list[dict[str, str]]:
        """Format question and context into prompt messages.

        Args:
            question: User's question
            context: Retrieved context from documents

        Returns:
            List of message dictionaries
        """
        system_message = {"role": "system", "content": self.prompt_config.system_prompt}

        user_message = {
            "role": "user",
            "content": self.prompt_config.format_user_prompt(context=context, question=question),
        }

        return [system_message, user_message]

    def generate_answer(self, question: str, context: str) -> str | None:
        """Generate answer for a question using context.

        Args:
            question: User's question
            context: Retrieved document context

        Returns:
            Generated answer or None
        """
        if not context:
            return "I don't have any relevant information to answer this question. Please upload a PDF document first."

        messages = self.format_prompt(question, context)
        return self.send_request(messages)

    def validate_response(self, response: str | None) -> str:
        """Validate and clean response.

        Args:
            response: Raw response from API

        Returns:
            Cleaned response
        """
        if not response:
            return "I'm sorry, I couldn't generate a response at this time. Please try again."

        # Basic cleaning
        response = response.strip()

        return response
