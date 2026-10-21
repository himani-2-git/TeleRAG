"""Telegram bot handlers for PDF RAG chatbot."""

import asyncio
import hashlib
import logging
import os
import re
import socket  # noqa: F401
import tempfile
import time
from typing import Any
from urllib.parse import urlparse

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import NetworkError, TimedOut
from telegram.ext import ContextTypes

from ..processors.pdf_fetcher import fetch_pdf_bytes
from ..rag.pipeline import RAGPipeline, format_source_footer, get_orchestrator
from ..rag.prompts import load_prompt_config

logger = logging.getLogger(__name__)


class BotHandlers:
    """Handles Telegram bot interactions."""

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

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        welcome_message = (
            "👋 Welcome to PDF RAG Chatbot!\n\n"
            "I can help you analyze PDF, Markdown, and text documents, as well as web pages, and answer questions about them.\n\n"
            "📤 Send me a PDF, .md, or .txt file\n"
            "🌐 Send /addurl <url> to ingest a static web page\n"
            "🔗 Or send a direct link to a PDF\n"
            "❓ Ask questions about your uploaded documents\n\n"
            "Commands:\n"
            "/list - Show your uploaded documents\n"
            "/delete <filename> - Delete a specific document\n"
            "/clear - Delete all your documents\n"
            "/addurl <url> - Ingest a web page\n"
            "/help - Show this message"
        )
        await update.message.reply_text(welcome_message)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        await self.start_command(update, context)

    async def list_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /list command."""
        user_id = update.effective_user.id

        try:
            documents = self.document_store.list_documents(user_id)

            if not documents:
                await update.message.reply_text("📭 You haven't uploaded any documents yet.")
                return

            message = "📚 Your uploaded documents:\n\n"
            for doc in documents:
                size_mb = doc["file_size"] / (1024 * 1024)
                message += f"📄 {doc['filename']}\n"
                message += f"   Size: {size_mb:.2f} MB | Chunks: {doc['chunk_count']}\n"
                message += f"   Uploaded: {doc['upload_date']}\n\n"

            await update.message.reply_text(message)

        except Exception as e:
            logger.error(f"Failed to list documents [{type(e).__name__}]: {e}")
            await update.message.reply_text("❌ Failed to retrieve document list.")

    async def delete_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /delete command."""
        user_id = update.effective_user.id

        if not context.args:
            await update.message.reply_text("❌ Please specify a filename: /delete <filename>")
            return

        filename = " ".join(context.args)

        try:
            deleted = self.document_store.delete_document(user_id, filename)

            if deleted:
                await update.message.reply_text(f"✅ Deleted document: {filename}")
            else:
                await update.message.reply_text(f"❌ Document not found: {filename}")

        except Exception as e:
            logger.error(f"Failed to delete document [{type(e).__name__}]: {e}")
            await update.message.reply_text("❌ Failed to delete document.")

    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command."""
        user_id = update.effective_user.id

        try:
            count = self.document_store.clear_user_documents(user_id)

            if count > 0:
                await update.message.reply_text(f"✅ Deleted {count} document(s).")
            else:
                await update.message.reply_text("📭 You don't have any documents to delete.")

        except Exception as e:
            logger.error(f"Failed to clear documents [{type(e).__name__}]: {e}")
            await update.message.reply_text("❌ Failed to clear documents.")

    async def addurl_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /addurl <url> command."""

        if not context.args:
            await update.message.reply_text("❌ Please specify a URL: /addurl <url>")
            return

        url = context.args[0].strip()
        await self._handle_web_url_ingest(update, context, url)

    async def _handle_web_url_ingest(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, url: str
    ):
        """Fetch and ingest a web page for the user."""
        user_id = update.effective_user.id

        from ..processors.web_processor import WebProcessor, validate_url_ssrf

        is_safe, error_msg = validate_url_ssrf(url)
        if not is_safe:
            await update.message.reply_text(f"❌ {error_msg}")
            return

        await update.message.reply_text("🌐 Fetching web page...")
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        try:
            web_processor = WebProcessor(
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
                timeout=15,
                max_bytes=2 * 1024 * 1024,
            )

            citation_label, clean_text, raw_bytes, chunks = await asyncio.to_thread(
                web_processor.process_url, url
            )

            if not chunks:
                await update.message.reply_text(
                    "❌ No text content could be extracted from web page."
                )
                return

            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(chunk_texts)

            chunks_data = [
                {
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "char_count": chunk.char_count,
                    "page": None,
                    "page_start": None,
                    "page_end": None,
                    "section_label": chunk.section_label,
                    "source_type": "web",
                }
                for chunk in chunks
            ]

            self.document_store.store_document(
                user_id=user_id,
                filename=citation_label,
                file_content=raw_bytes,
                chunks=chunks_data,
                embeddings=embeddings,
                source_type="web",
            )

            await update.message.reply_text(
                f"✅ Successfully ingested web page: {citation_label}\n"
                f"📊 Created {len(chunks)} chunks\n\n"
                f"You can now ask questions about this page!"
            )
        except ValueError as e:
            await update.message.reply_text(f"❌ {e}")
        except Exception:
            logger.exception("Failed to ingest web page: %s", url)
            await update.message.reply_text(
                "❌ An unexpected error occurred while processing the web page. Please try again."
            )

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document uploads (PDF, Markdown, Text)."""
        user_id = update.effective_user.id
        document = update.message.document
        fn = document.file_name or "document"
        fn_lower = fn.lower()

        is_pdf = fn_lower.endswith(".pdf")
        is_md = fn_lower.endswith(".md")
        is_txt = fn_lower.endswith(".txt")

        if not (is_pdf or is_md or is_txt):
            await update.message.reply_text(
                "❌ Please send a PDF (.pdf), Markdown (.md), or text (.txt) file."
            )
            return

        # Check file size
        if document.file_size > self.config.max_file_size_mb * 1024 * 1024:
            await update.message.reply_text(
                f"❌ File too large. Maximum size is {self.config.max_file_size_mb}MB."
            )
            return

        doc_type_label = "PDF" if is_pdf else ("Markdown" if is_md else "text")
        await update.message.reply_text(f"📥 Processing your {doc_type_label} document...")
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        temp_file = None
        try:
            # Download file
            file = await document.get_file()
            suffix = ".pdf" if is_pdf else (".md" if is_md else ".txt")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            await file.download_to_drive(temp_file.name)

            with open(temp_file.name, "rb") as f:
                file_content = f.read()

            if is_pdf:
                is_valid, error_msg = self.pdf_processor.validate_pdf(
                    temp_file.name, document.file_size
                )
                if not is_valid:
                    await update.message.reply_text(f"❌ {error_msg}")
                    return
                pages_data = self.pdf_processor.extract_text_with_pages(temp_file.name)
                chunks = self.text_chunker.create_chunks_with_pages(pages_data)
                source_type = "pdf"
            elif is_md:
                from ..processors.markdown_processor import MarkdownProcessor

                md_processor = MarkdownProcessor(
                    chunk_size=self.config.chunk_size, overlap=self.config.chunk_overlap
                )
                md_text = file_content.decode("utf-8", errors="replace")
                chunks = md_processor.process_markdown(md_text, source_name=fn)
                source_type = "markdown"
            else:  # is_txt
                txt_text = file_content.decode("utf-8", errors="replace")
                if not txt_text.strip():
                    await update.message.reply_text("❌ No text content found in file.")
                    return
                raw_chunks = self.text_chunker.create_chunks(txt_text)
                chunks = []
                for i, rc in enumerate(raw_chunks):
                    from ..processors.text_chunker import TextChunk

                    c = TextChunk(
                        content=rc.content,
                        chunk_index=i,
                        start_pos=rc.start_pos,
                        end_pos=rc.end_pos,
                        char_count=rc.char_count,
                        source_type="text",
                        source_name=fn,
                        section_label=fn,
                    )
                    chunks.append(c)
                source_type = "text"

            if not chunks:
                await update.message.reply_text(
                    f"❌ No text content found in {doc_type_label} file."
                )
                return

            # Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(chunk_texts)

            # Prepare chunks for storage
            chunks_data = [
                {
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "char_count": chunk.char_count,
                    "page": chunk.page,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_label": chunk.section_label,
                    "source_type": chunk.source_type or source_type,
                }
                for chunk in chunks
            ]

            # Store document
            self.document_store.store_document(
                user_id=user_id,
                filename=fn,
                file_content=file_content,
                chunks=chunks_data,
                embeddings=embeddings,
                source_type=source_type,
            )

            await update.message.reply_text(
                f"✅ Successfully processed {fn}\n"
                f"📊 Created {len(chunks)} chunks\n\n"
                f"You can now ask questions about this document!"
            )

        except ValueError as e:
            await update.message.reply_text(f"❌ {e}")
        except Exception:
            logger.exception("Failed to process document: %s", fn)
            await update.message.reply_text(
                "❌ An unexpected error occurred while processing your document. Please try again."
            )

        finally:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages as questions or URL uploads."""
        user_id = update.effective_user.id
        message_text = update.message.text

        # Check if message contains a URL
        if self._is_url(message_text):
            await self._handle_url_upload(update, context, message_text)
            return

        # Otherwise treat as a question
        question = message_text

        # Check if user has documents
        documents = self.document_store.list_documents(user_id)
        if not documents:
            await update.message.reply_text(
                "📭 You haven't uploaded any documents yet.\n"
                "Please send me a PDF file or a link to a PDF!"
            )
            return

        start_time = time.time()

        self.active_requests += 1
        # Check if another LLM request is currently active/busy
        if self.llm_semaphore.locked() or self.active_requests > 1:
            k = self.active_requests - 1
            init_text = f"Queued: {k} question(s) ahead of you. I'll reply when it's your turn."
        else:
            init_text = "Thinking..."

        thinking_msg = await self._reply_with_retry(update, init_text)

        await self._send_typing_action(context, update.effective_chat.id)

        # Periodic typing loop while waiting for blocking retrieval and model inference
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

    _format_source_footer = staticmethod(format_source_footer)

    def _run_pipeline(self, user_id: int, question: str) -> dict[str, Any]:
        """Fetch user chunks and execute RAG pipeline via orchestrator."""
        user_chunks = self.document_store.get_user_chunks_with_embeddings(user_id)
        return self.orchestrator.run(
            question=question,
            user_id=user_id,
            user_chunks=user_chunks,
        )

    async def _deliver_answer(self, update: Update, thinking_msg: Any, text: str):
        """Deliver final response as a new reply to user's message, then delete thinking message."""
        # 1. Send final answer as a NEW reply to the user's original message
        await self._reply_with_retry(update, text)

        # 2. Delete the temporary "Thinking..." / "Queued..." message
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

    def _is_url(self, text: str) -> bool:
        """Check if text contains a URL.

        Args:
            text: Text to check

        Returns:
            True if text contains a URL
        """
        # Simple URL pattern
        url_pattern = r"https?://[^\s]+"
        return bool(re.search(url_pattern, text))

    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from URL.

        Args:
            url: URL string

        Returns:
            Filename or generated name
        """
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)

        if not filename or not filename.endswith(".pdf"):
            suffix = hashlib.sha256(url.encode()).hexdigest()[:8]
            filename = f"document_{suffix}.pdf"

        return filename

    def _is_safe_url(self, url: str) -> tuple[bool, str]:
        """Validate URL to protect against SSRF and unsupported schemes."""
        from ..processors.web_processor import validate_url_ssrf

        return validate_url_ssrf(url)

    async def _handle_url_upload(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, url: str
    ):
        """Handle PDF upload from URL safely with async thread download."""
        user_id = update.effective_user.id

        url_match = re.search(r"https?://[^\s]+", url)
        if url_match:
            url = url_match.group(0)

        if not url.lower().endswith(".pdf") and ".pdf" not in url.lower():
            await update.message.reply_text(
                "❌ The URL doesn't appear to be a PDF file.\n"
                "Please send a direct link to a PDF file."
            )
            return

        is_safe, error_msg = self._is_safe_url(url)
        if not is_safe:
            await update.message.reply_text(f"❌ {error_msg}")
            return

        await update.message.reply_text("🔗 Downloading PDF from URL...")
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        max_bytes = self.config.max_file_size_mb * 1024 * 1024
        try:
            pdf_bytes = await asyncio.to_thread(fetch_pdf_bytes, url, max_bytes, 30)
        except ValueError as e:
            await update.message.reply_text(f"❌ {e}")
            return
        except Exception:
            logger.exception("Failed to download PDF from URL: %s", url)
            await update.message.reply_text(
                "❌ An unexpected error occurred while downloading the PDF. Please try again."
            )
            return

        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.write(pdf_bytes)
            temp_file.close()

            file_size = len(pdf_bytes)
            await update.message.reply_text("📥 Processing PDF...")

            is_valid, error_msg = self.pdf_processor.validate_pdf(temp_file.name, file_size)
            if not is_valid:
                await update.message.reply_text(f"❌ {error_msg}")
                return

            pages_data = self.pdf_processor.extract_text_with_pages(temp_file.name)
            chunks = self.text_chunker.create_chunks_with_pages(pages_data)
            if not chunks:
                await update.message.reply_text("❌ No text content found in PDF.")
                return

            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(chunk_texts)

            chunks_data = [
                {
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "char_count": chunk.char_count,
                    "page": chunk.page,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                }
                for chunk in chunks
            ]

            filename = self._extract_filename_from_url(url)
            self.document_store.store_document(
                user_id=user_id,
                filename=filename,
                file_content=pdf_bytes,
                chunks=chunks_data,
                embeddings=embeddings,
            )

            await update.message.reply_text(
                f"✅ Successfully processed {filename}\n"
                f"📊 Created {len(chunks)} chunks\n"
                f"🔗 Source: {url[:50]}...\n\n"
                f"You can now ask questions about this document!"
            )
        except ValueError as e:
            await update.message.reply_text(f"❌ {e}")
        except Exception:
            logger.exception("Failed to process downloaded PDF from URL: %s", url)
            await update.message.reply_text(
                "❌ An unexpected error occurred while processing the PDF. Please try again."
            )
        finally:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors."""
        err = context.error
        err_type = type(err).__name__ if err else "UnknownError"
        logger.error(f"Update caused error [{err_type}]: {err}")
