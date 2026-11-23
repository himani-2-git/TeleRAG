"""Main entry point for the TeleRAG Telegram bot."""

import logging
import sys

from telegram.error import InvalidToken, NetworkError

from src.bot.application import build_application
from src.bot.handlers import BotHandlers
from src.config import ConfigError, get_config
from src.logging_setup import configure_logging
from src.processors.pdf_processor import PDFProcessor
from src.processors.text_chunker import TextChunker
from src.rag.embeddings import EmbeddingGenerator
from src.rag.llm_client import LLMClient
from src.rag.prompts import load_prompt_config
from src.rag.retriever import DocumentRetriever
from src.startup import StartupError, verify_llm_endpoint
from src.storage.document_store import DocumentStore

logger = logging.getLogger(__name__)


def create_bot_handlers(config) -> BotHandlers:
    """Create all components. Cheap checks come first, so common mistakes fail fast.

    Raises:
        StartupError: With a message that says what to fix.
    """
    try:
        prompt_config = load_prompt_config(config.prompt_version)
    except Exception as e:
        raise StartupError(
            f"Cannot load prompt version '{config.prompt_version}' ({type(e).__name__}). "
            "Check PROMPT_VERSION: the files are in the prompts/ folder."
        ) from e
    logger.info("Loaded prompt version: %s", config.prompt_version)

    llm_client = LLMClient(
        api_key=config.llm_api_key or "",
        model=config.llm_model,
        max_retries=config.max_retries,
        retry_delay=config.retry_delay,
        base_url=config.llm_base_url,
        timeout=(5.0, float(config.llm_timeout_seconds)),
        max_tokens=config.llm_max_tokens,
        prompt_config=prompt_config,
        max_total_seconds=float(config.llm_max_total_seconds),
    )
    logger.info("Checking the LLM server (model %s)...", config.llm_model)
    verify_llm_endpoint(llm_client)
    logger.info("LLM server is working.")

    try:
        document_store = DocumentStore(
            db_path=config.database_path,
            embeddings_path=config.embeddings_path,
            vector_backend=config.vector_backend,
            chroma_path=config.chroma_path,
        )
    except Exception as e:
        raise StartupError(
            f"Cannot open the database at '{config.database_path}' ({type(e).__name__}). "
            "Check DATABASE_PATH and that the folder can be written to."
        ) from e

    try:
        embedding_generator = EmbeddingGenerator(model_name=config.embedding_model)
    except Exception as e:
        raise StartupError(
            f"Cannot load the embedding model '{config.embedding_model}' ({type(e).__name__}). "
            "The first start downloads it, so an internet connection is needed. "
            "Also check EMBEDDING_MODEL."
        ) from e

    retriever = DocumentRetriever(
        top_k=config.top_k_results,
        similarity_threshold=config.similarity_threshold,
        retrieval_mode=config.retrieval_mode,
    )
    # When a user adds or deletes a document, their cached search index is dropped.
    document_store.add_change_listener(retriever.invalidate_cache)

    return BotHandlers(
        pdf_processor=PDFProcessor(max_file_size_mb=config.max_file_size_mb),
        text_chunker=TextChunker(chunk_size=config.chunk_size, overlap=config.chunk_overlap),
        document_store=document_store,
        embedding_generator=embedding_generator,
        retriever=retriever,
        llm_client=llm_client,
        config=config,
        prompt_config=prompt_config,
    )


def main():
    """Start the bot. Any problem with settings or services stops it with a clear message."""
    try:
        config = get_config()
    except ConfigError as e:
        # Logging is not set up yet, so write to the console directly.
        print(f"Cannot start the bot. {e}", file=sys.stderr)
        sys.exit(1)

    configure_logging(config)
    logger.info("Starting the TeleRAG bot...")

    try:
        bot_handlers = create_bot_handlers(config)
        application = build_application(config, bot_handlers)
    except StartupError as e:
        logger.error("Cannot start the bot: %s", e)
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error while starting the bot")
        sys.exit(1)

    logger.info("Bot is ready! Starting polling...")
    try:
        # Ctrl+C or SIGTERM stops polling, finishes running requests and then returns.
        application.run_polling(allowed_updates=["message"])
    except InvalidToken:
        logger.error("Telegram rejected TELEGRAM_BOT_TOKEN. Check the token from @BotFather.")
        sys.exit(1)
    except NetworkError as e:
        logger.error("Cannot reach Telegram (%s). Check your internet connection.", e)
        sys.exit(1)
    except Exception:
        logger.exception("The bot stopped because of an unexpected error")
        sys.exit(1)
    logger.info("Goodbye.")


if __name__ == "__main__":
    main()
