"""Telegram bot handlers package."""

import asyncio
import logging
import math

from telegram import Update
from telegram.error import TelegramError
from telegram.ext import (
    ApplicationHandlerStop,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)

from ...rag.pipeline import RAGPipeline, get_orchestrator
from ...rag.prompts import load_prompt_config
from ..rate_limiter import SlidingWindowRateLimiter
from .commands import CommandHandlersMixin
from .documents import DocumentHandlersMixin
from .questions import QuestionHandlersMixin

logger = logging.getLogger(__name__)

PRIVATE_BOT_MESSAGE = (
    "🔒 Sorry, this bot is private. "
    "If you think you should have access, please contact the person who runs it."
)

# Handlers in a lower group number run first. Group 0 holds the normal handlers.
ACCESS_GATE_GROUP = -1


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


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

        window = config.rate_limit_window_seconds
        self.question_limiter = SlidingWindowRateLimiter(config.question_rate_limit, window)
        self.upload_limiter = SlidingWindowRateLimiter(config.upload_rate_limit, window)

    def register_handlers(self, application) -> None:
        """Attach every handler to the Telegram application."""
        # The access gate is in its own early group, so it runs before every other handler
        # and no handler can forget to check who is calling.
        application.add_handler(TypeHandler(Update, self.access_gate), group=ACCESS_GATE_GROUP)

        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("list", self.list_command))
        application.add_handler(CommandHandler("delete", self.delete_command))
        application.add_handler(CommandHandler("clear", self.clear_command))
        application.add_handler(CommandHandler("addurl", self.addurl_command))
        application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
        application.add_error_handler(self.error_handler)

    async def access_gate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Turn away everyone who is not in ALLOWED_USER_IDS. Does nothing if the list is empty."""
        allowed_ids = self.config.allowed_user_ids
        if not allowed_ids:
            return

        user = update.effective_user
        if user is not None and user.id in allowed_ids:
            return

        logger.info("Refused update from user %s: the bot is private", user.id if user else None)
        message = update.effective_message
        if message is not None:
            try:
                await message.reply_text(PRIVATE_BOT_MESSAGE)
            except TelegramError as e:
                logger.warning(
                    "Could not send the private-bot notice [%s]: %s", type(e).__name__, e
                )
        raise ApplicationHandlerStop

    async def _is_rate_limited(
        self, update: Update, limiter: SlidingWindowRateLimiter, what: str
    ) -> bool:
        """Tell the user to slow down and return True if they are over the limit."""
        wait_seconds = limiter.try_acquire(update.effective_user.id)
        if wait_seconds <= 0:
            return False

        seconds = max(1, math.ceil(wait_seconds))
        await update.message.reply_text(
            f"⏳ You are sending {what} too quickly. "
            f"Please wait {_plural(seconds, 'second')} and try again."
        )
        return True

    async def _document_limit_reached(self, update: Update, user_id: int) -> bool:
        """Tell the user and return True if they already store the maximum number of documents."""
        limit = self.config.max_documents_per_user
        if limit <= 0:
            return False

        documents = await asyncio.to_thread(self.document_store.list_documents, user_id)
        if len(documents) < limit:
            return False

        await update.message.reply_text(
            f"❌ You have reached the limit of {_plural(limit, 'document')}. "
            "Delete one with /delete (or everything with /clear) and try again."
        )
        return True

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Log unhandled errors from Telegram update processing."""
        err = context.error
        err_type = type(err).__name__ if err else "UnknownError"
        logger.error(f"Update caused error [{err_type}]: {err}")


__all__ = ["BotHandlers"]
