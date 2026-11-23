"""Configuration management for PDF RAG Chatbot."""

import ipaddress
import os
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

RETRIEVAL_MODES = ("dense", "hybrid", "hybrid_rerank")
VECTOR_BACKENDS = ("numpy", "chroma")
ORCHESTRATORS = ("plain", "langgraph")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigError(ValueError):
    """Raised when settings are missing or invalid. The message lists every problem found."""


@dataclass
class Config:
    """Application configuration."""

    # API Keys & Endpoints
    telegram_bot_token: str
    llm_api_key: str | None = None
    llm_base_url: str = "http://localhost:11434/v1"

    # File Processing
    max_file_size_mb: int = 20
    max_chunks_per_document: int = 5000
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # RAG Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k_results: int = 2
    similarity_threshold: float = 0.3
    retrieval_mode: str = "hybrid"
    orchestrator: str = "plain"

    # LLM Settings (Ollama default, configurable to hosted providers)
    llm_model: str = "qwen2.5:3b"
    llm_max_tokens: int = 1024
    llm_timeout_seconds: int = 120
    llm_max_total_seconds: int = 180
    max_retries: int = 3
    retry_delay: float = 1.0
    prompt_version: str = "v1"

    # Grounding Check Settings
    grounding_threshold: float = 0.45
    grounding_enforce: bool = False
    grounding_min_coverage: float = 0.5

    # Database & Vector Store
    database_path: str = "data/documents.db"
    embeddings_path: str = "data/embeddings"
    vector_backend: str = "numpy"
    chroma_path: str = "data/chroma"

    # Abuse protection. An empty allowlist means everyone may use the bot. For the limits,
    # 0 means "no limit".
    allowed_user_ids: frozenset[int] = frozenset()
    question_rate_limit: int = 5
    upload_rate_limit: int = 3
    rate_limit_window_seconds: int = 60
    max_documents_per_user: int = 20

    # Operations. Admins may use /status. If admin_user_ids is empty, the allowlist is used.
    admin_user_ids: frozenset[int] = frozenset()
    concurrent_updates: int = 16

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/bot.log"
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 3


def parse_user_ids(raw: str, setting_name: str = "ALLOWED_USER_IDS") -> frozenset[int]:
    """Parse a comma-separated list of Telegram user ids such as '123, 456'."""
    user_ids = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            user_ids.add(int(part))
        except ValueError:
            raise ValueError(
                f"{setting_name} must be a comma-separated list of numeric Telegram user "
                f"ids, but '{part}' is not a number"
            ) from None
    return frozenset(user_ids)


def is_local_endpoint(url: str) -> bool:
    """Return True if the URL points at this machine or an internal network.

    Such servers (a local Ollama, or the "ollama" service in Docker Compose) normally need no
    API key. Remote hosted services do.
    """
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    if host in ("localhost", "host.docker.internal"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # A name without any dot, such as "ollama", can only be an internal service name.
        return "." not in host
    return address.is_loopback or address.is_private


class _EnvReader:
    """Reads settings from the environment and remembers every problem instead of stopping."""

    def __init__(self):
        self.errors: list[str] = []

    @staticmethod
    def _raw(name: str) -> str | None:
        value = os.getenv(name)
        if value is None or not value.strip():
            return None
        return value.strip()

    def text(self, name: str, default: str) -> str:
        return self._raw(name) or default

    def integer(self, name: str, default: int) -> int:
        raw = self._raw(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            self.errors.append(f"{name} must be a whole number, but it is '{raw}'")
            return default

    def number(self, name: str, default: float) -> float:
        raw = self._raw(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            self.errors.append(f"{name} must be a number, but it is '{raw}'")
            return default

    def boolean(self, name: str, default: bool) -> bool:
        raw = self._raw(name)
        if raw is None:
            return default
        if raw.lower() in ("true", "1", "yes", "on"):
            return True
        if raw.lower() in ("false", "0", "no", "off"):
            return False
        self.errors.append(f"{name} must be true or false, but it is '{raw}'")
        return default

    def user_ids(self, name: str) -> frozenset[int]:
        try:
            return parse_user_ids(os.getenv(name, ""), name)
        except ValueError as e:
            self.errors.append(str(e))
            return frozenset()


def _check_range(
    errors: list[str], name: str, value: float, minimum: float, maximum: float | None = None
) -> None:
    if value < minimum or (maximum is not None and value > maximum):
        wanted = f"at least {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        errors.append(f"{name} must be {wanted}, but it is {value}")


def _check_choice(errors: list[str], name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        errors.append(f"{name} must be one of {', '.join(choices)}, but it is '{value}'")


def validate_config(config: Config) -> list[str]:
    """Check value ranges and choices. Returns a list of problems (empty if all is fine)."""
    errors: list[str] = []
    _check_range(errors, "MAX_FILE_SIZE_MB", config.max_file_size_mb, 1)
    _check_range(errors, "MAX_CHUNKS_PER_DOCUMENT", config.max_chunks_per_document, 1)
    _check_range(errors, "CHUNK_SIZE", config.chunk_size, 1)
    _check_range(errors, "CHUNK_OVERLAP", config.chunk_overlap, 0)
    if 0 <= config.chunk_overlap and config.chunk_size <= config.chunk_overlap:
        errors.append(
            f"CHUNK_OVERLAP ({config.chunk_overlap}) must be smaller than "
            f"CHUNK_SIZE ({config.chunk_size})"
        )
    _check_range(errors, "TOP_K_RESULTS", config.top_k_results, 1)
    _check_range(errors, "SIMILARITY_THRESHOLD", config.similarity_threshold, 0, 1)
    _check_range(errors, "GROUNDING_THRESHOLD", config.grounding_threshold, 0, 1)
    _check_range(errors, "GROUNDING_MIN_COVERAGE", config.grounding_min_coverage, 0, 1)
    _check_range(errors, "LLM_MAX_TOKENS", config.llm_max_tokens, 1)
    _check_range(errors, "LLM_TIMEOUT_SECONDS", config.llm_timeout_seconds, 1)
    _check_range(errors, "LLM_MAX_TOTAL_SECONDS", config.llm_max_total_seconds, 1)
    _check_range(errors, "MAX_RETRIES", config.max_retries, 1)
    _check_range(errors, "RETRY_DELAY", config.retry_delay, 0)
    _check_range(errors, "QUESTION_RATE_LIMIT", config.question_rate_limit, 0)
    _check_range(errors, "UPLOAD_RATE_LIMIT", config.upload_rate_limit, 0)
    _check_range(errors, "RATE_LIMIT_WINDOW_SECONDS", config.rate_limit_window_seconds, 1)
    _check_range(errors, "MAX_DOCUMENTS_PER_USER", config.max_documents_per_user, 0)
    _check_range(errors, "CONCURRENT_UPDATES", config.concurrent_updates, 1)
    _check_range(errors, "LOG_MAX_BYTES", config.log_max_bytes, 1024)
    _check_range(errors, "LOG_BACKUP_COUNT", config.log_backup_count, 0)
    _check_choice(errors, "RETRIEVAL_MODE", config.retrieval_mode, RETRIEVAL_MODES)
    _check_choice(errors, "VECTOR_BACKEND", config.vector_backend, VECTOR_BACKENDS)
    _check_choice(errors, "ORCHESTRATOR", config.orchestrator, ORCHESTRATORS)
    _check_choice(errors, "LOG_LEVEL", config.log_level, LOG_LEVELS)
    return errors


def get_config(require_telegram: bool = True) -> Config:
    """Read the configuration from environment variables and check it.

    Raises:
        ConfigError: If anything is missing or invalid. The message lists all problems.
    """
    reader = _EnvReader()
    telegram_token = reader.text("TELEGRAM_BOT_TOKEN", "")
    llm_key = reader.text("LLM_API_KEY", "")
    llm_base_url = reader.text("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")

    config = Config(
        telegram_bot_token=telegram_token,
        llm_api_key=llm_key,
        llm_base_url=llm_base_url,
        max_file_size_mb=reader.integer("MAX_FILE_SIZE_MB", 20),
        max_chunks_per_document=reader.integer("MAX_CHUNKS_PER_DOCUMENT", 5000),
        chunk_size=reader.integer("CHUNK_SIZE", 1000),
        chunk_overlap=reader.integer("CHUNK_OVERLAP", 200),
        embedding_model=reader.text("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        top_k_results=reader.integer("TOP_K_RESULTS", 2),
        similarity_threshold=reader.number("SIMILARITY_THRESHOLD", 0.3),
        retrieval_mode=reader.text("RETRIEVAL_MODE", "hybrid").lower(),
        orchestrator=reader.text("ORCHESTRATOR", "plain").lower(),
        llm_model=reader.text("LLM_MODEL", "qwen2.5:3b"),
        llm_max_tokens=reader.integer("LLM_MAX_TOKENS", 1024),
        llm_timeout_seconds=reader.integer("LLM_TIMEOUT_SECONDS", 120),
        llm_max_total_seconds=reader.integer("LLM_MAX_TOTAL_SECONDS", 180),
        max_retries=reader.integer("MAX_RETRIES", 3),
        retry_delay=reader.number("RETRY_DELAY", 1.0),
        prompt_version=reader.text("PROMPT_VERSION", "v1"),
        grounding_threshold=reader.number("GROUNDING_THRESHOLD", 0.45),
        grounding_enforce=reader.boolean("GROUNDING_ENFORCE", False),
        grounding_min_coverage=reader.number("GROUNDING_MIN_COVERAGE", 0.5),
        database_path=reader.text("DATABASE_PATH", "data/documents.db"),
        embeddings_path=reader.text("EMBEDDINGS_PATH", "data/embeddings"),
        vector_backend=reader.text("VECTOR_BACKEND", "numpy").lower(),
        chroma_path=reader.text("CHROMA_PATH", "data/chroma"),
        allowed_user_ids=reader.user_ids("ALLOWED_USER_IDS"),
        question_rate_limit=reader.integer("QUESTION_RATE_LIMIT", 5),
        upload_rate_limit=reader.integer("UPLOAD_RATE_LIMIT", 3),
        rate_limit_window_seconds=reader.integer("RATE_LIMIT_WINDOW_SECONDS", 60),
        max_documents_per_user=reader.integer("MAX_DOCUMENTS_PER_USER", 20),
        admin_user_ids=reader.user_ids("ADMIN_USER_IDS"),
        concurrent_updates=reader.integer("CONCURRENT_UPDATES", 16),
        log_level=reader.text("LOG_LEVEL", "INFO").upper(),
        log_file=reader.text("LOG_FILE", "logs/bot.log"),
        log_max_bytes=reader.integer("LOG_MAX_BYTES", 5_000_000),
        log_backup_count=reader.integer("LOG_BACKUP_COUNT", 3),
    )

    errors = list(reader.errors)
    errors.extend(validate_config(config))

    if require_telegram and not telegram_token:
        errors.insert(
            0,
            "TELEGRAM_BOT_TOKEN is not set. Create a bot with @BotFather on Telegram and put "
            "its token in your .env file.",
        )
    if require_telegram and not llm_key and not is_local_endpoint(llm_base_url):
        errors.append(
            "LLM_API_KEY is required because LLM_BASE_URL points to a remote service "
            f"({llm_base_url}). Local servers such as Ollama need no key."
        )

    if errors:
        bullet_list = "\n".join(f"  - {problem}" for problem in errors)
        raise ConfigError(f"Invalid configuration:\n{bullet_list}")
    return config
