"""Document and URL upload handlers for Telegram bot."""

import asyncio
import hashlib
import logging
import os
import re
import tempfile
from urllib.parse import urlparse

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from ...processors.pdf_fetcher import fetch_pdf_bytes

logger = logging.getLogger(__name__)


class DocumentHandlersMixin:
    """Mixin providing document and URL processing handlers for BotHandlers."""

    async def _embed_chunks_with_progress(
        self,
        update: Update,
        chunk_texts: list[str],
    ) -> list[list[float]] | None:
        """Validate chunk limit, notify user, and generate embeddings with progress every 10% for docs >500 chunks."""
        total = len(chunk_texts)
        if total > self.config.max_chunks_per_document:
            await update.message.reply_text(
                f"❌ Document exceeds the maximum limit of {self.config.max_chunks_per_document} chunks "
                f"({total} chunks created). Please upload a smaller document."
            )
            return None

        status_msg = await update.message.reply_text(
            f"📊 Created {total} chunks. Ingestion and embedding of large files can take several minutes..."
        )

        if total > 500:
            step = max(1, total // 10)
            embeddings: list[list[float]] = []
            for i in range(0, total, step):
                batch = chunk_texts[i : i + step]
                batch_embs = await asyncio.to_thread(
                    self.embedding_generator.generate_embeddings, batch
                )
                embeddings.extend(batch_embs)
                pct = min(100, int((len(embeddings) / total) * 100))
                try:
                    await status_msg.edit_text(
                        f"⏳ Embedding chunks: {pct}% ({len(embeddings)}/{total})...\n"
                        "Large files can take several minutes."
                    )
                except Exception as e:
                    logger.debug("Failed to edit progress message: %s", e)
            return embeddings

        return await asyncio.to_thread(self.embedding_generator.generate_embeddings, chunk_texts)

    async def _handle_web_url_ingest(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, url: str
    ):
        """Fetch and ingest a web page for the user."""
        user_id = update.effective_user.id
        from ...processors.web_processor import WebProcessor, validate_url_ssrf

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

            citation_label, _clean_text, raw_bytes, chunks = await asyncio.to_thread(
                web_processor.process_url, url
            )

            if not chunks:
                await update.message.reply_text(
                    "❌ No text content could be extracted from web page."
                )
                return

            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self._embed_chunks_with_progress(update, chunk_texts)
            if embeddings is None:
                return

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
        """Handle document uploads for PDF, Markdown, and text formats."""
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
                from ...processors.markdown_processor import MarkdownProcessor

                md_processor = MarkdownProcessor(
                    chunk_size=self.config.chunk_size, overlap=self.config.chunk_overlap
                )
                md_text = file_content.decode("utf-8", errors="replace")
                chunks = md_processor.process_markdown(md_text, source_name=fn)
                source_type = "markdown"
            else:
                txt_text = file_content.decode("utf-8", errors="replace")
                if not txt_text.strip():
                    await update.message.reply_text("❌ No text content found in file.")
                    return
                raw_chunks = self.text_chunker.create_chunks(txt_text)
                chunks = []
                for i, rc in enumerate(raw_chunks):
                    from ...processors.text_chunker import TextChunk

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

            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self._embed_chunks_with_progress(update, chunk_texts)
            if embeddings is None:
                return

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
                except OSError:
                    pass

    def _is_url(self, text: str) -> bool:
        """Check if text contains an HTTP or HTTPS URL."""
        return bool(re.search(r"https?://[^\s]+", text))

    def _extract_filename_from_url(self, url: str) -> str:
        """Derive document filename from URL path or hash fallback."""
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename or not filename.endswith(".pdf"):
            suffix = hashlib.sha256(url.encode()).hexdigest()[:8]
            filename = f"document_{suffix}.pdf"
        return filename

    def _is_safe_url(self, url: str) -> tuple[bool, str]:
        """Validate URL against SSRF and unsupported schemes."""
        from ...processors.web_processor import validate_url_ssrf

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
            embeddings = await self._embed_chunks_with_progress(update, chunk_texts)
            if embeddings is None:
                return

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
