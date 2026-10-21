"""Question-answering message handlers and streaming typing actions."""

import asyncio
import logging
import time
from typing import Any

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import NetworkError, TimedOut
from telegram.ext import ContextTypes

from ...rag.pipeline import format_source_footer

logger = logging.getLogger(__name__)


class QuestionHandlersMixin:
    """Mixin providing question processing and typing management for BotHandlers."""

    _format_source_footer = staticmethod(format_source_footer)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages as questions or URL uploads."""
        user_id = update.effective_user.id
        message_text = update.message.text

        if self._is_url(message_text):
            await self._handle_url_upload(update, context, message_text)
            return

        question = message_text
        documents = self.document_store.list_documents(user_id)
        if not documents:
            await update.message.reply_text(
                "📭 You haven't uploaded any documents yet.\n"
                "Please send me a PDF file or a link to a PDF!"
            )
            return

        start_time = time.time()
        self.active_requests += 1

        if self.llm_semaphore.locked() or self.active_requests > 1:
            k = self.active_requests - 1
            init_text = f"Queued: {k} question(s) ahead of you. I'll reply when it's your turn."
        else:
            init_text = "Thinking..."

        thinking_msg = await self._reply_with_retry(update, init_text)
        await self._send_typing_action(context, update.effective_chat.id)

        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(
            self._keep_typing(context, update.effective_chat.id, stop_typing)
        )

        try:
            async with self.llm_semaphore:
                state = await asyncio.to_thread(self._run_pipeline, user_id, question)

            stop_typing.set()
            await typing_task

            elapsed_s = round(time.time() - start_time)
            final_response = state.get("final_response", "")

            if not state.get("refused", False):
                model_name = self.config.llm_model
                timing_line = f"Answered in {elapsed_s} s with {model_name}"
                final_response = f"{final_response}\n\n{timing_line}"

            await self._deliver_answer(update, thinking_msg, final_response)

        except Exception as e:
            stop_typing.set()
            try:
                await typing_task
            except Exception:
                pass
            logger.error("Failed to answer question [%s]: %s", type(e).__name__, e)
            await self._deliver_answer(
                update,
                thinking_msg,
                "❌ Sorry, I encountered an error while processing your question. Please try again.",
            )
        finally:
            self.active_requests = max(0, self.active_requests - 1)

    def _run_pipeline(self, user_id: int, question: str) -> dict[str, Any]:
        """Fetch user chunks and execute RAG pipeline via orchestrator."""
        user_chunks = self.document_store.get_user_chunks_with_embeddings(user_id)
        return self.orchestrator.run(
            question=question,
            user_id=user_id,
            user_chunks=user_chunks,
        )

    async def _deliver_answer(self, update: Update, thinking_msg: Any, text: str):
        """Deliver final response as a new reply to user message, then delete thinking message."""
        await self._reply_with_retry(update, text)
        if thinking_msg:
            try:
                await thinking_msg.delete()
            except Exception as e:
                logger.warning(f"Could not delete thinking message [{type(e).__name__}]: {e}")

    async def _send_typing_action(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int):
        """Send chat typing action with retry on transient network errors."""
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except (TimedOut, NetworkError) as e:
            logger.warning(
                f"Typing action timed out/failed [{type(e).__name__}]: {e}. Retrying once..."
            )
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception as retry_err:
                logger.warning(
                    f"Typing action retry failed [{type(retry_err).__name__}]: {retry_err}"
                )
        except Exception as e:
            logger.warning(f"Typing action failed [{type(e).__name__}]: {e}")

    async def _keep_typing(
        self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, stop_event: asyncio.Event
    ):
        """Periodically refresh typing action every 4 seconds until stop_event is set."""
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                if not stop_event.is_set():
                    await self._send_typing_action(context, chat_id)

    async def _reply_with_retry(self, update: Update, text: str):
        """Send message reply with automatic single retry on TimedOut or NetworkError."""
        try:
            return await update.message.reply_text(text)
        except (TimedOut, NetworkError) as e:
            logger.warning(f"Sending reply failed [{type(e).__name__}]: {e}. Retrying once...")
            return await update.message.reply_text(text)
        except Exception as e:
            logger.warning(f"Sending reply failed [{type(e).__name__}]: {e}")
            return None
