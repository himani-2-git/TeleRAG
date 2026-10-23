"""Question-answering message handlers and streaming typing actions."""

import asyncio
import html
import logging
import re
import time
from typing import Any

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, NetworkError, TelegramError, TimedOut
from telegram.ext import ContextTypes

from ...rag.pipeline import format_source_footer

logger = logging.getLogger(__name__)


def format_telegram_html(text: str) -> str:
    """Format markdown text for Telegram HTML parse mode.

    Converts fenced code blocks to <pre><code>...</code></pre> with escaped content,
    and applies html.escape to all other text (including Sources footers).
    """
    pattern = re.compile(r"```[a-zA-Z0-9_\-\+\.]*\n?(.*?)```", re.DOTALL)
    parts = []
    last_idx = 0
    for match in pattern.finditer(text):
        pre_text = text[last_idx : match.start()]
        if pre_text:
            parts.append(html.escape(pre_text, quote=False))
        code_body = match.group(1)
        if code_body.endswith("\n"):
            code_body = code_body[:-1]
        parts.append(f"<pre><code>{html.escape(code_body, quote=False)}</code></pre>")
        last_idx = match.end()

    tail = text[last_idx:]
    if tail:
        parts.append(html.escape(tail, quote=False))

    return "".join(parts)


def utf16_len(s: str) -> int:
    """Calculate length in UTF-16 code units (Telegram's message length metric)."""
    return len(s.encode("utf-16-le")) // 2


def split_telegram_message(text: str, limit: int = 4000) -> list[str]:
    """Split a Telegram message (plain text or HTML) into chunks within the UTF-16 code unit limit.

    Splits on paragraph, then line, then sentence boundaries, then hard-splits as a last resort.
    Never splits inside an HTML tag or entity. If a split falls inside <pre><code>...</code></pre>,
    closes the tags at the end of the chunk and reopens them at the start of the next.
    """
    if utf16_len(text) <= limit:
        return [text]

    chunks = []
    remaining = text
    tag_pattern = re.compile(r"<pre><code>|</code></pre>|<[^>]+>|&[a-zA-Z0-9#]+;")

    while remaining:
        if utf16_len(remaining) <= limit:
            chunks.append(remaining)
            break

        tags = [(m.start(), m.end(), m.group(0)) for m in tag_pattern.finditer(remaining)]

        inside_token = [False] * (len(remaining) + 1)
        in_code_at = [False] * (len(remaining) + 1)

        depth = 0
        tag_idx = 0
        n_tags = len(tags)

        for start, end, _tag in tags:
            for p in range(start + 1, end):
                inside_token[p] = True

        for p in range(len(remaining) + 1):
            while tag_idx < n_tags and tags[tag_idx][1] <= p:
                tag = tags[tag_idx][2]
                if tag in ("<pre><code>", "<pre>", "<code>"):
                    depth += 1
                elif tag in ("</code></pre>", "</pre>", "</code>"):
                    depth = max(0, depth - 1)
                tag_idx += 1
            in_code_at[p] = depth > 0

        best_para = -1
        best_line = -1
        best_sent = -1
        best_word = -1
        best_hard = -1

        current_utf16 = 0
        for pos in range(1, len(remaining) + 1):
            char = remaining[pos - 1]
            current_utf16 += 2 if ord(char) > 0xFFFF else 1

            in_code = in_code_at[pos]
            extra = 13 if in_code else 0  # utf16_len("</code></pre>") == 13
            if current_utf16 + extra > limit:
                break

            if inside_token[pos]:
                continue

            best_hard = pos
            if char == "\n":
                if pos >= 2 and remaining[pos - 2] == "\n":
                    best_para = pos
                else:
                    best_line = pos
            elif char in (".", "!", "?") and pos < len(remaining) and remaining[pos] in (" ", "\n"):
                best_sent = pos
            elif char == " ":
                best_word = pos

        chosen = (
            best_para
            if best_para != -1
            else (
                best_line
                if best_line != -1
                else (
                    best_sent if best_sent != -1 else (best_word if best_word != -1 else best_hard)
                )
            )
        )

        if chosen == -1 or chosen == 0:
            chosen = min(limit, len(remaining))

        in_code = in_code_at[chosen]
        chunk = remaining[:chosen]
        if in_code:
            chunk = chunk + "</code></pre>"
            remaining = "<pre><code>" + remaining[chosen:]
        else:
            chunk = chunk.rstrip("\n")
            remaining = remaining[chosen:].lstrip("\n")

        chunks.append(chunk)

    return chunks


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
            except (asyncio.CancelledError, TelegramError) as typing_err:
                logger.debug("Typing task cancellation error: %s", typing_err)
            logger.exception("Failed to answer question: %s", e)
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
        """Deliver final response formatted as HTML with fallback to plain text, chunked if necessary."""
        formatted_html = format_telegram_html(text)
        chunks = split_telegram_message(formatted_html, limit=4000)

        def _to_plain_text(chunk_html: str) -> str:
            cleaned = chunk_html.replace("<pre><code>", "```\n").replace("</code></pre>", "\n```")
            cleaned = re.sub(r"<[^>]+>", "", cleaned)
            return html.unescape(cleaned)

        for chunk in chunks:
            fb_text = text if len(chunks) == 1 else _to_plain_text(chunk)
            sent = await self._reply_with_retry(
                update, chunk, parse_mode="HTML", fallback_text=fb_text
            )
            if sent is None:
                logger.error("Failed to send chunk of response to user.")
                try:
                    await update.message.reply_text("❌ Failed to send response. Please try again.")
                except Exception as send_err:
                    logger.warning("Could not send failure notice: %s", send_err)
                break

        if thinking_msg:
            try:
                await thinking_msg.delete()
            except TelegramError as e:
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
            except TelegramError as retry_err:
                logger.warning(
                    f"Typing action retry failed [{type(retry_err).__name__}]: {retry_err}"
                )
        except TelegramError as e:
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

    async def _reply_with_retry(
        self,
        update: Update,
        text: str,
        parse_mode: str | None = None,
        fallback_text: str | None = None,
    ):
        """Send message reply with automatic single retry on TimedOut or NetworkError,
        and optional plain text fallback if HTML parsing is rejected.
        """
        kwargs = {}
        if parse_mode is not None:
            kwargs["parse_mode"] = parse_mode

        try:
            return await update.message.reply_text(text, **kwargs)
        except BadRequest as e:
            if parse_mode and fallback_text is not None:
                logger.warning(
                    f"Sending reply rejected with {parse_mode} [{type(e).__name__}]: {e}. Falling back to plain text."
                )
                return await self._reply_with_retry(update, fallback_text, parse_mode=None)
            logger.warning(f"Sending reply failed [{type(e).__name__}]: {e}")
            return None
        except (TimedOut, NetworkError) as e:
            logger.warning(f"Sending reply failed [{type(e).__name__}]: {e}. Retrying once...")
            try:
                return await update.message.reply_text(text, **kwargs)
            except TelegramError as retry_err:
                if parse_mode and fallback_text is not None:
                    logger.warning(
                        f"Reply retry failed with {parse_mode} [{type(retry_err).__name__}]: {retry_err}. Falling back to plain text."
                    )
                    return await self._reply_with_retry(update, fallback_text, parse_mode=None)
                logger.warning(f"Sending reply failed [{type(retry_err).__name__}]: {retry_err}")
                return None
        except TelegramError as e:
            if parse_mode and fallback_text is not None:
                logger.warning(
                    f"Sending reply failed with {parse_mode} [{type(e).__name__}]: {e}. Falling back to plain text."
                )
                return await self._reply_with_retry(update, fallback_text, parse_mode=None)
            logger.warning(f"Sending reply failed [{type(e).__name__}]: {e}")
            return None
