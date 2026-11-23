"""Checks that run once at startup, so the bot fails fast with a clear message."""

import logging

logger = logging.getLogger(__name__)


class StartupError(Exception):
    """Raised when the bot cannot start. The message is written for the person running it."""


def verify_llm_endpoint(llm_client) -> None:
    """Make sure the LLM server answers and knows the configured model.

    Raises:
        StartupError: If the server is unreachable, rejects the key, lacks the model or fails.
    """
    is_ok, status_code, reason = llm_client.check_availability()
    if is_ok:
        return

    url = llm_client.base_url
    model = llm_client.model

    if status_code is None:
        raise StartupError(
            f"Cannot reach the LLM server at {url}. If you use Ollama, make sure it is "
            "running (start it with 'ollama serve'). Otherwise check LLM_BASE_URL in your .env."
        )
    if status_code == 404:
        raise StartupError(
            f"The LLM server is running but does not know the model '{model}'. "
            f"With Ollama, download it with: ollama pull {model}"
        )
    if status_code in (401, 403):
        raise StartupError(
            f"The LLM server rejected the API key (HTTP {status_code}). Check LLM_API_KEY."
        )
    if status_code == 429:
        # The server is up, it is just limiting requests right now. Do not block startup.
        logger.warning("LLM server is rate limiting requests (HTTP 429); continuing anyway.")
        return
    raise StartupError(f"The LLM server returned HTTP {status_code}: {reason}")
