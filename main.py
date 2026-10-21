"""Main entry point for PDF RAG Chatbot."""

import logging
import os
import sys

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from src.bot.handlers import BotHandlers
from src.config import get_config
from src.processors.pdf_processor import PDFProcessor
from src.processors.text_chunker import TextChunker
from src.rag.embeddings import EmbeddingGenerator
from src.rag.llm_client import LLMClient
from src.rag.pipeline import RAGPipeline, get_orchestrator
from src.rag.prompts import load_prompt_config
from src.rag.retriever import DocumentRetriever
from src.storage.document_store import DocumentStore


def setup_logging(config):
    """Set up logging configuration."""
    log_dir = os.path.dirname(config.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=getattr(logging, config.log_level),
        handlers=[logging.FileHandler(config.log_file), logging.StreamHandler()],
    )

    # Silence verbose library loggers that could print bot tokens in request URLs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext").setLevel(logging.WARNING)


def main():
    """Main application entry point."""
    logger = logging.getLogger(__name__)

    try:
        config = get_config()
    except ValueError as e:
        logging.basicConfig(level=logging.ERROR)
        logger.error("Configuration error: %s", e)
        logger.error("Please set TELEGRAM_BOT_TOKEN and LLM_API_KEY (if remote) in .env file")
        sys.exit(1)

    setup_logging(config)
    logger.info("Starting PDF RAG Chatbot...")
    logger.info("Loaded prompt version: %s", config.prompt_version)

    try:
        logger.info("Initializing components...")

        prompt_config = load_prompt_config(config.prompt_version)
        pdf_processor = PDFProcessor(max_file_size_mb=config.max_file_size_mb)
        text_chunker = TextChunker(chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        document_store = DocumentStore(
            db_path=config.database_path,
            embeddings_path=config.embeddings_path,
            vector_backend=config.vector_backend,
            chroma_path=config.chroma_path,
        )
        embedding_generator = EmbeddingGenerator(model_name=config.embedding_model)
        retriever = DocumentRetriever(
            top_k=config.top_k_results,
            similarity_threshold=config.similarity_threshold,
            retrieval_mode=config.retrieval_mode,
        )
        llm_client = LLMClient(
            api_key=config.llm_api_key,
            model=config.llm_model,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            base_url=config.llm_base_url,
            max_tokens=config.llm_max_tokens,
            prompt_config=prompt_config,
        )
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=llm_client,
            embedding_generator=embedding_generator,
            prompt_config=prompt_config,
            grounding_threshold=config.grounding_threshold,
            grounding_enforce=config.grounding_enforce,
            grounding_min_coverage=config.grounding_min_coverage,
        )
        orchestrator = get_orchestrator(pipeline, config.orchestrator)

        logger.info(
            "Checking LLM model availability (%s at %s)...",
            config.llm_model,
            config.llm_base_url,
        )
        is_ok, status_code, reason = llm_client.check_availability()
        if is_ok:
            logger.info("LLM OK: Model is available and responding.")
            try:
                logger.info("Sending LLM model warm-up request...")
                llm_client.send_request([{"role": "user", "content": "hi"}], max_tokens=1)
                logger.info("LLM model warm-up complete.")
            except Exception as e:
                logger.warning("LLM warm-up request failed [%s]: %s", type(e).__name__, e)
        else:
            logger.warning("LLM Check Failed: HTTP %s - %s", status_code, reason)
            if status_code in (404, 429):
                hint = (
                    "Hint: Verify that your local LLM server (e.g. Ollama) is running, "
                    "or check model availability on your provider and update LLM_MODEL in .env."
                )
                logger.warning(hint)

        bot_handlers = BotHandlers(
            pdf_processor=pdf_processor,
            text_chunker=text_chunker,
            document_store=document_store,
            embedding_generator=embedding_generator,
            orchestrator=orchestrator,
            config=config,
            prompt_config=prompt_config,
        )

        application = (
            Application.builder()
            .token(config.telegram_bot_token)
            .connect_timeout(30.0)
            .read_timeout(30.0)
            .write_timeout(30.0)
            .pool_timeout(30.0)
            .get_updates_read_timeout(30.0)
            .build()
        )

        application.add_handler(CommandHandler("start", bot_handlers.start_command))
        application.add_handler(CommandHandler("help", bot_handlers.help_command))
        application.add_handler(CommandHandler("list", bot_handlers.list_command))
        application.add_handler(CommandHandler("delete", bot_handlers.delete_command))
        application.add_handler(CommandHandler("clear", bot_handlers.clear_command))
        application.add_handler(CommandHandler("addurl", bot_handlers.addurl_command))
        application.add_handler(MessageHandler(filters.Document.ALL, bot_handlers.handle_document))
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, bot_handlers.handle_message)
        )
        application.add_error_handler(bot_handlers.error_handler)

        logger.info("Bot is ready! Starting polling...")
        application.run_polling(allowed_updates=["message"])

    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
