"""Main entry point for PDF RAG Chatbot."""

import logging
import os
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from src.config import get_config
from src.processors.pdf_processor import PDFProcessor
from src.processors.text_chunker import TextChunker
from src.storage.document_store import DocumentStore
from src.rag.embeddings import EmbeddingGenerator
from src.rag.retriever import DocumentRetriever
from src.rag.openrouter_client import OpenRouterClient
from src.bot.handlers import BotHandlers

def setup_logging(config):
    """Set up logging configuration."""
    log_dir = os.path.dirname(config.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=getattr(logging, config.log_level),
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler()
        ]
    )
    
    # Silence verbose library loggers that could print bot tokens in request URLs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext").setLevel(logging.WARNING)

def main():
    """Main application entry point."""
    # Load configuration
    try:
        config = get_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set TELEGRAM_BOT_TOKEN and OPENROUTER_API_KEY in .env file")
        return
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Starting PDF RAG Chatbot...")
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        
        pdf_processor = PDFProcessor(max_file_size_mb=config.max_file_size_mb)
        text_chunker = TextChunker(chunk_size=config.chunk_size, 
                                   overlap=config.chunk_overlap)
        document_store = DocumentStore(db_path=config.database_path,
                                      embeddings_path=config.embeddings_path)
        embedding_generator = EmbeddingGenerator(model_name=config.embedding_model)
        retriever = DocumentRetriever(top_k=config.top_k_results,
                                     similarity_threshold=config.similarity_threshold,
                                     retrieval_mode=config.retrieval_mode)
        openrouter_client = OpenRouterClient(api_key=config.openrouter_api_key,
                                            model=config.openrouter_model,
                                            max_retries=config.max_retries,
                                            retry_delay=config.retry_delay,
                                            base_url=config.llm_base_url,
                                            timeout=300.0)
        
        # Check LLM availability on startup
        logger.info(f"Checking LLM model availability ({config.openrouter_model} at {config.llm_base_url})...")
        is_ok, status_code, reason = openrouter_client.check_availability()
        if is_ok:
            print("LLM OK")
            logger.info("LLM OK: Model is available and responding.")
        else:
            print(f"LLM Check Failed: HTTP {status_code} - {reason}")
            logger.warning(f"LLM Check Failed: HTTP {status_code} - {reason}")
            if status_code in (404, 429):
                hint = (
                    "Hint: Free model names change often on OpenRouter. "
                    "Pick another free model from https://openrouter.ai/models "
                    "or verify your local Ollama model name, and update OPENROUTER_MODEL in your .env file."
                )
                print(hint)
                logger.warning(hint)
        
        # Initialize bot handlers
        bot_handlers = BotHandlers(
            pdf_processor=pdf_processor,
            text_chunker=text_chunker,
            document_store=document_store,
            embedding_generator=embedding_generator,
            retriever=retriever,
            openrouter_client=openrouter_client,
            config=config
        )
        
        # Create application with generous timeouts for slow local LLMs / CPU inference
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
        
        # Register handlers
        application.add_handler(CommandHandler("start", bot_handlers.start_command))
        application.add_handler(CommandHandler("help", bot_handlers.help_command))
        application.add_handler(CommandHandler("list", bot_handlers.list_command))
        application.add_handler(CommandHandler("delete", bot_handlers.delete_command))
        application.add_handler(CommandHandler("clear", bot_handlers.clear_command))
        application.add_handler(MessageHandler(filters.Document.PDF, bot_handlers.handle_document))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, 
                                              bot_handlers.handle_message))
        
        # Register error handler
        application.add_error_handler(bot_handlers.error_handler)
        
        logger.info("Bot is ready! Starting polling...")
        
        # Start the bot
        application.run_polling(allowed_updates=["message"])
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
