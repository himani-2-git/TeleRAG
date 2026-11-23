"""Configuration management for PDF RAG Chatbot."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/bot.log"


def parse_user_ids(raw: str) -> frozenset[int]:
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
                f"ALLOWED_USER_IDS must be a comma-separated list of numeric Telegram user "
                f"ids, but '{part}' is not a number"
            ) from None
    return frozenset(user_ids)


def get_config(require_telegram: bool = True) -> Config:
    """Get application configuration from environment variables."""

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    llm_key = os.getenv("LLM_API_KEY", "")
    llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    llm_model = os.getenv("LLM_MODEL", "qwen2.5:3b")

    if require_telegram and not telegram_token:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

    # Ollama or local servers do not require an API key when pointing to localhost/127.0.0.1
    is_local = "localhost" in llm_base_url or "127.0.0.1" in llm_base_url
    if require_telegram and not llm_key and not is_local:
        raise ValueError(
            "LLM_API_KEY environment variable is required when using a remote endpoint"
        )

    return Config(
        telegram_bot_token=telegram_token or "",
        llm_api_key=llm_key or "",
        llm_base_url=llm_base_url,
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "20")),
        max_chunks_per_document=int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "5000")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        top_k_results=int(os.getenv("TOP_K_RESULTS", "2")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.3")),
        retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid"),
        orchestrator=os.getenv("ORCHESTRATOR", "plain").lower(),
        llm_model=llm_model,
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        retry_delay=float(os.getenv("RETRY_DELAY", "1.0")),
        prompt_version=os.getenv("PROMPT_VERSION", "v1"),
        grounding_threshold=float(os.getenv("GROUNDING_THRESHOLD", "0.45")),
        grounding_enforce=os.getenv("GROUNDING_ENFORCE", "false").lower() in ("true", "1", "yes"),
        grounding_min_coverage=float(os.getenv("GROUNDING_MIN_COVERAGE", "0.5")),
        database_path=os.getenv("DATABASE_PATH", "data/documents.db"),
        embeddings_path=os.getenv("EMBEDDINGS_PATH", "data/embeddings"),
        vector_backend=os.getenv("VECTOR_BACKEND", "numpy").lower(),
        chroma_path=os.getenv("CHROMA_PATH", "data/chroma"),
        allowed_user_ids=parse_user_ids(os.getenv("ALLOWED_USER_IDS", "")),
        question_rate_limit=int(os.getenv("QUESTION_RATE_LIMIT", "5")),
        upload_rate_limit=int(os.getenv("UPLOAD_RATE_LIMIT", "3")),
        rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
        max_documents_per_user=int(os.getenv("MAX_DOCUMENTS_PER_USER", "20")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "logs/bot.log"),
    )
