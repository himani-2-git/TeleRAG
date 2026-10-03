"""Configuration management for PDF RAG Chatbot."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Application configuration."""
    
    # API Keys
    telegram_bot_token: str
    openrouter_api_key: str
    
    # File Processing
    max_file_size_mb: int = 20
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # RAG Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k_results: int = 5
    similarity_threshold: float = 0.3
    
    # OpenRouter Settings
    openrouter_model: str = "qwen/qwen3.8-27b:free"
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Database
    database_path: str = "data/documents.db"
    embeddings_path: str = "data/embeddings"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/bot.log"

def get_config() -> Config:
    """Get application configuration from environment variables."""
    
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    if not telegram_token:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
    
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    return Config(
        telegram_bot_token=telegram_token,
        openrouter_api_key=openrouter_key,
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "20")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        top_k_results=int(os.getenv("TOP_K_RESULTS", "5")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.3")),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "qwen/qwen3.8-27b:free"),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        retry_delay=float(os.getenv("RETRY_DELAY", "1.0")),
        database_path=os.getenv("DATABASE_PATH", "data/documents.db"),
        embeddings_path=os.getenv("EMBEDDINGS_PATH", "data/embeddings"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "logs/bot.log")
    )