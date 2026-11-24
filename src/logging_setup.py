"""Logging setup: JSON log file with rotation, a request id on every line, secrets hidden."""

import contextvars
import functools
import json
import logging
import os
import re
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from urllib.parse import urlsplit

# The id of the question or upload being handled. It is stored in a context variable, which
# asyncio copies into every task and into every thread started with asyncio.to_thread. So log
# lines written deep inside the RAG pipeline carry the same id as the handler that started it.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# Libraries that log full URLs (the Telegram bot token is part of the URL) or are very chatty.
NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "telegram", "telegram.ext")

REDACTED = "[REDACTED]"
# No word boundary at the start: in API URLs the token follows the text "bot" directly.
TELEGRAM_TOKEN_PATTERN = re.compile(r"\d{6,12}:[A-Za-z0-9_-]{30,}")
BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")


def new_request_id() -> str:
    """Return a short random id such as 'a3f9c2d1'."""
    return uuid.uuid4().hex[:8]


@contextmanager
def request_context(request_id: str | None = None):
    """Give everything logged inside this block the same request id."""
    token = request_id_var.set(request_id or new_request_id())
    try:
        yield request_id_var.get()
    finally:
        request_id_var.reset(token)


def with_request_id(handler):
    """Decorator for async Telegram handlers: every call gets a new request id."""

    @functools.wraps(handler)
    async def wrapper(*args, **kwargs):
        with request_context():
            return await handler(*args, **kwargs)

    return wrapper


def redact_url(url: str) -> str:
    """Return the URL without credentials, query string and fragment (they can hold secrets)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "[invalid url]"
    if not parts.hostname:
        return url[:100] if url else ""
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{parts.hostname}{port}{parts.path}"


class Redactor:
    """Replaces secrets in text. Used on every finished log line, so tracebacks are covered."""

    def __init__(self, secrets: Iterable[str] = ()):
        # Very short values would also hide ordinary words, so only longer secrets are listed.
        self._secrets = sorted({s for s in secrets if s and len(s) >= 8}, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        text = TELEGRAM_TOKEN_PATTERN.sub(REDACTED, text)
        return BEARER_PATTERN.sub(f"Bearer {REDACTED}", text)


class RequestIdFilter(logging.Filter):
    """Adds the current request id to every record, including records from other libraries."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


# Names that every LogRecord has. Anything else was passed with extra={...}.
_STANDARD_RECORD_FIELDS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "request_id",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, easy to search with tools such as jq."""

    def __init__(self, redactor: Redactor):
        super().__init__()
        self._redact = redactor

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS:
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return self._redact(json.dumps(entry, ensure_ascii=False, default=str))


class RedactingTextFormatter(logging.Formatter):
    """Readable one-line format for the console."""

    def __init__(self, redactor: Redactor):
        super().__init__("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
        self._redact = redactor

    def format(self, record: logging.LogRecord) -> str:
        return self._redact(super().format(record))


def configure_logging(config, logger: logging.Logger | None = None) -> list[logging.Handler]:
    """Set up a rotating JSON log file and a readable console log.

    Safe to call more than once: handlers added by an earlier call are replaced.
    """
    logger = logger if logger is not None else logging.getLogger()
    redactor = Redactor([config.telegram_bot_token, config.llm_api_key or ""])

    log_dir = os.path.dirname(config.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = RotatingFileHandler(
        config.log_file,
        maxBytes=config.log_max_bytes,
        backupCount=config.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonFormatter(redactor))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(RedactingTextFormatter(redactor))

    for old_handler in [h for h in logger.handlers if getattr(h, "_telerag_handler", False)]:
        logger.removeHandler(old_handler)
        old_handler.close()

    handlers = [file_handler, console_handler]
    for handler in handlers:
        handler.addFilter(RequestIdFilter())
        handler._telerag_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    logger.setLevel(config.log_level)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return handlers
