"""Telegram bot handlers package."""

import asyncio
import logging

from telegram import Update
from telegram.ext import ContextTypes

from ...rag.pipeline import RAGPipeline, get_orchestrator
from ...rag.prompts import load_prompt_config
from .commands import CommandHandlersMixin
from .documents import DocumentHandlersMixin
from .questions import QuestionHandlersMixin

logger = logging.getLogger(__name__)


class BotHandlers(CommandHandlersMixin, DocumentHandlersMixin, QuestionHandlersMixin):
    """Coordinates Telegram bot commands, document ingestion, and QA interactions."""

    def __init__(
        self,
        pdf_processor,
        text_chunker,
        document_store,
        embedding_generator,
        config,
        orchestrator=None,
        retriever=None,
        llm_client=None,
        prompt_config=None,
    ):
        self.pdf_processor = pdf_processor
        self.text_chunker = text_chunker
        self.document_store = document_store
        self.embedding_generator = embedding_generator
        self.config = config
        self.prompt_config = prompt_config or load_prompt_config(config.prompt_version)
        if orchestrator is not None:
            self.orchestrator = orchestrator
        elif retriever is not None and llm_client is not None:
            pipeline = RAGPipeline(
                retriever=retriever,
                llm_client=llm_client,
                embedding_generator=embedding_generator,
                prompt_config=self.prompt_config,
                grounding_threshold=config.grounding_threshold,
                grounding_enforce=config.grounding_enforce,
                grounding_min_coverage=config.grounding_min_coverage,
            )
            self.orchestrator = get_orchestrator(pipeline, config.orchestrator)
        else:
            self.orchestrator = None
        self.llm_semaphore = asyncio.Semaphore(1)
        self.active_requests = 0

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Log unhandled errors from Telegram update processing."""
        err = context.error
        err_type = type(err).__name__ if err else "UnknownError"
        logger.error(f"Update caused error [{err_type}]: {err}")


__all__ = ["BotHandlers"]
