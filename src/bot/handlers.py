"""Telegram bot handlers for PDF RAG chatbot."""

import os
import tempfile
import requests
import re
import socket
import ipaddress
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction
from telegram.error import TimedOut, NetworkError
import asyncio
import logging
import time
from ..rag.prompts import PromptConfig, load_prompt_config
from ..rag.grounding import grounding_report

logger = logging.getLogger(__name__)

class BotHandlers:
    """Handles Telegram bot interactions."""
    
    def __init__(self, pdf_processor, text_chunker, document_store, 
                 embedding_generator, retriever, llm_client, config):
        """Initialize bot handlers.
        
        Args:
            pdf_processor: PDF processing instance
            text_chunker: Text chunking instance
            document_store: Document storage instance
            embedding_generator: Embedding generation instance
            retriever: Document retrieval instance
            llm_client: LLM client instance
            config: Application configuration
        """
        self.pdf_processor = pdf_processor
        self.text_chunker = text_chunker
        self.document_store = document_store
        self.embedding_generator = embedding_generator
        self.retriever = retriever
        self.llm_client = llm_client
        self.config = config
        pv = getattr(self.config, 'prompt_version', 'v1')
        if not isinstance(pv, str):
            pv = "v1"
        self.prompt_config = load_prompt_config(pv)
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
                size_mb = doc['file_size'] / (1024 * 1024)
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
        user_id = update.effective_user.id
        
        if not context.args:
            await update.message.reply_text("❌ Please specify a URL: /addurl <url>")
            return
            
        url = context.args[0].strip()
        await self._handle_web_url_ingest(update, context, url)

    async def _handle_web_url_ingest(self, update: Update, context: ContextTypes.DEFAULT_TYPE, url: str):
        """Fetch and ingest a web page for the user."""
        user_id = update.effective_user.id
        
        from ..processors.web_processor import WebProcessor, validate_url_ssrf
        
        is_safe, error_msg = validate_url_ssrf(url)
        if not is_safe:
            await update.message.reply_text(f"❌ {error_msg}")
            return
            
        await update.message.reply_text("🌐 Fetching web page...")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        try:
            web_processor = WebProcessor(
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
                timeout=15,
                max_bytes=2 * 1024 * 1024
            )
            
            citation_label, clean_text, raw_bytes, chunks = await asyncio.to_thread(
                web_processor.process_url, url
            )
            
            if not chunks:
                await update.message.reply_text("❌ No text content could be extracted from web page.")
                return
                
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(chunk_texts)
            
            chunks_data = [
                {
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content,
                    'char_count': chunk.char_count,
                    'page': None,
                    'page_start': None,
                    'page_end': None,
                    'section_label': chunk.section_label,
                    'source_type': "web"
                }
                for chunk in chunks
            ]
            
            self.document_store.store_document(
                user_id=user_id,
                filename=citation_label,
                file_content=raw_bytes,
                chunks=chunks_data,
                embeddings=embeddings,
                source_type="web"
            )
            
            await update.message.reply_text(
                f"✅ Successfully ingested web page: {citation_label}\n"
                f"📊 Created {len(chunks)} chunks\n\n"
                f"You can now ask questions about this page!"
            )
        except Exception as e:
            logger.error(f"Failed to ingest web page [{type(e).__name__}]: {e}")
            await update.message.reply_text(f"❌ Failed to ingest web page: {str(e)}")

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document uploads (PDF, Markdown, Text)."""
        user_id = update.effective_user.id
        document = update.message.document
        fn = document.file_name or "document"
        fn_lower = fn.lower()
        
        is_pdf = fn_lower.endswith('.pdf')
        is_md = fn_lower.endswith('.md')
        is_txt = fn_lower.endswith('.txt')
        
        if not (is_pdf or is_md or is_txt):
            await update.message.reply_text("❌ Please send a PDF (.pdf), Markdown (.md), or text (.txt) file.")
            return
        
        # Check file size
        if document.file_size > self.config.max_file_size_mb * 1024 * 1024:
            await update.message.reply_text(
                f"❌ File too large. Maximum size is {self.config.max_file_size_mb}MB."
            )
            return
        
        doc_type_label = "PDF" if is_pdf else ("Markdown" if is_md else "text")
        await update.message.reply_text(f"📥 Processing your {doc_type_label} document...")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, 
                                          action=ChatAction.TYPING)
        
        temp_file = None
        try:
            # Download file
            file = await document.get_file()
            suffix = '.pdf' if is_pdf else ('.md' if is_md else '.txt')
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            await file.download_to_drive(temp_file.name)
            
            with open(temp_file.name, 'rb') as f:
                file_content = f.read()
            
            if is_pdf:
                is_valid, error_msg = self.pdf_processor.validate_pdf(temp_file.name, document.file_size)
                if not is_valid:
                    await update.message.reply_text(f"❌ {error_msg}")
                    return
                pages_data = self.pdf_processor.extract_text_with_pages(temp_file.name)
                chunks = self.text_chunker.create_chunks_with_pages(pages_data)
                source_type = "pdf"
            elif is_md:
                from ..processors.markdown_processor import MarkdownProcessor
                md_processor = MarkdownProcessor(chunk_size=self.config.chunk_size, overlap=self.config.chunk_overlap)
                md_text = file_content.decode('utf-8', errors='replace')
                chunks = md_processor.process_markdown(md_text, source_name=fn)
                source_type = "markdown"
            else:  # is_txt
                txt_text = file_content.decode('utf-8', errors='replace')
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
                        section_label=fn
                    )
                    chunks.append(c)
                source_type = "text"
            
            if not chunks:
                await update.message.reply_text(f"❌ No text content found in {doc_type_label} file.")
                return
            
            # Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(chunk_texts)
            
            # Prepare chunks for storage
            chunks_data = [
                {
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content,
                    'char_count': chunk.char_count,
                    'page': chunk.page,
                    'page_start': chunk.page_start,
                    'page_end': chunk.page_end,
                    'section_label': chunk.section_label,
                    'source_type': chunk.source_type or source_type
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
                source_type=source_type
            )
            
            await update.message.reply_text(
                f"✅ Successfully processed {fn}\n"
                f"📊 Created {len(chunks)} chunks\n\n"
                f"You can now ask questions about this document!"
            )
            
        except Exception as e:
            logger.error(f"Failed to process document [{type(e).__name__}]: {e}")
            await update.message.reply_text(f"❌ Failed to process document: {str(e)}")
            
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
        
        # Check if another LLM request is currently active/busy
        if self.llm_semaphore.locked() or self.active_requests > 0:
            k = self.active_requests
            init_text = f"Queued: {k} question(s) ahead of you. I'll reply when it's your turn."
        else:
            init_text = "Thinking..."
        
        thinking_msg = await self._reply_with_retry(update, init_text)
        self.active_requests += 1
        
        await self._send_typing_action(context, update.effective_chat.id)
        
        # Periodic typing loop while waiting for blocking retrieval and model inference
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(self._keep_typing(context, update.effective_chat.id, stop_typing))
        
        try:
            async with self.llm_semaphore:
                # Run blocking document loading, embedding, retrieval, and LLM generation in worker thread
                def _blocking_pipeline():
                    # Get all user chunks with embeddings
                    user_chunks = self.document_store.get_user_chunks_with_embeddings(user_id)
                    if not user_chunks:
                        return "NO_CHUNKS", None, None, 0.0, 0.0
                    
                    t_retrieval_start = time.time()
                    # Check if this is a summary/overview question
                    if self._is_summary_query(question):
                        logger.info(f"Summary/overview query detected for user {user_id}: '{question[:50]}'")
                        sorted_chunks = sorted(
                            user_chunks,
                            key=lambda c: (
                                c.get('page_start') if c.get('page_start') is not None else (c.get('page') or 0),
                                c.get('chunk_index', 0)
                            )
                        )
                        retrieved_chunks = sorted_chunks[:8]
                    else:
                        query_embedding = self.embedding_generator.generate_query_embedding(question)
                        retrieved_chunks = self.retriever.retrieve(
                            query=question, query_embedding=query_embedding, chunks=user_chunks
                        )
                    retrieval_seconds = time.time() - t_retrieval_start
                    
                    if not retrieved_chunks:
                        return "NO_RELEVANT", None, None, retrieval_seconds, 0.0
                    
                    context_text = self.retriever.format_context(retrieved_chunks)
                    t_llm_start = time.time()
                    raw_ans = self.llm_client.generate_answer(question, context_text)
                    llm_seconds = time.time() - t_llm_start
                    return "SUCCESS", raw_ans, retrieved_chunks, retrieval_seconds, llm_seconds

                status, raw_answer, relevant_chunks, retrieval_s, llm_s = await asyncio.to_thread(_blocking_pipeline)
            
            # Stop typing indicator
            stop_typing.set()
            await typing_task
            
            total_seconds = time.time() - start_time
            logger.info(
                f"Question timings: retrieval={retrieval_s:.2f}s, llm={llm_s:.2f}s, total={total_seconds:.2f}s"
            )
            
            if status == "NO_CHUNKS":
                await self._deliver_answer(update, thinking_msg, "❌ No document content available.")
                return
            
            if status == "NO_RELEVANT":
                await self._deliver_answer(
                    update,
                    thinking_msg,
                    self.prompt_config.refusal_message
                )
                return
            
            if not raw_answer:
                logger.warning(f"LLM generation failed for question: {question[:50]}...")
                answer = self.llm_client.validate_response(None)
                await self._deliver_answer(update, thinking_msg, answer)
                return
            
            answer = self.llm_client.validate_response(raw_answer)
            if self._is_insufficient_context_response(answer):
                await self._deliver_answer(
                    update,
                    thinking_msg,
                    self.prompt_config.refusal_message
                )
                return

            # Post-generation grounding check
            try:
                grounding_thresh = getattr(self.config, "grounding_threshold", 0.45)
                if not isinstance(grounding_thresh, (int, float)):
                    grounding_thresh = 0.45
                report = grounding_report(
                    answer=answer,
                    chunks=relevant_chunks,
                    embedder=self.embedding_generator,
                    threshold=grounding_thresh,
                )
                logger.info(
                    f"Grounding check: applicable={report.applicable}, "
                    f"coverage={report.citation_coverage:.2f} ({report.supported_sentences}/{report.total_sentences} supported), "
                    f"threshold={grounding_thresh}"
                )

                grounding_enforce = getattr(self.config, "grounding_enforce", False)
                if not isinstance(grounding_enforce, bool):
                    grounding_enforce = False
                min_coverage = getattr(self.config, "grounding_min_coverage", 0.5)
                if not isinstance(min_coverage, (int, float)):
                    min_coverage = 0.5

                if grounding_enforce and report.applicable and report.citation_coverage < min_coverage:
                    logger.warning(
                        f"Grounding check failed enforcement: coverage={report.citation_coverage:.2f} < min_coverage={min_coverage:.2f}. "
                        f"Refusing answer."
                    )
                    await self._deliver_answer(
                        update,
                        thinking_msg,
                        self.prompt_config.refusal_message
                    )
                    return
            except Exception as ge:
                logger.warning(f"Grounding check evaluation error [{type(ge).__name__}]: {ge}")

            footer = self._format_source_footer(relevant_chunks)
            elapsed_s = round(time.time() - start_time)
            model_name = getattr(self.config, 'llm_model', getattr(self.llm_client, 'model', 'LLM'))
            timing_line = f"Answered in {elapsed_s} s with {model_name}"
            if footer:
                final_response = f"{answer}{footer}\n\n{timing_line}"
            else:
                final_response = f"{answer}\n\n{timing_line}"
            
            await self._deliver_answer(update, thinking_msg, final_response)
            
        except Exception as e:
            stop_typing.set()
            try:
                await typing_task
            except Exception:
                pass
            logger.error(f"Failed to answer question [{type(e).__name__}]: {e}")
            await self._deliver_answer(
                update,
                thinking_msg,
                "❌ Sorry, I encountered an error while processing your question. Please try again."
            )
        finally:
            self.active_requests = max(0, self.active_requests - 1)

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
            logger.warning(f"Typing action timed out/failed [{type(e).__name__}]: {e}. Retrying once...")
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception as retry_err:
                logger.warning(f"Typing action retry failed [{type(retry_err).__name__}]: {retry_err}")
        except Exception as e:
            logger.warning(f"Typing action failed [{type(e).__name__}]: {e}")

    async def _keep_typing(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, stop_event: asyncio.Event):
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
            
    def _is_summary_query(self, query: str) -> bool:
        """Check if query is asking for a summary, overview, or document-level explanation."""
        q = query.lower().strip()
        patterns = [
            "summarize",
            "summary",
            "overview",
            "what is this document about",
            "what is this book about",
            "what is the document about",
            "what is the book about",
            "what is this pdf about",
            "what is the pdf about",
            "tell me about this document",
            "tell me about this pdf",
            "explain this document",
            "brief overview",
            "main points",
            "key takeaways"
        ]
        return any(pattern in q for pattern in patterns)
    
    def _is_insufficient_context_response(self, text: str) -> bool:
        """Check if the LLM response indicates the context does not contain the answer."""
        if not text:
            return False
        clean = text.strip()
        lower = clean.lower()
        
        # Starts with insufficient context marker (trimmed, case-insensitive, punctuation-stripped)
        stripped = clean.strip("`'\" \t\r\n").lower()
        marker = getattr(self.prompt_config, "insufficient_context_marker", "INSUFFICIENT_CONTEXT").strip().lower()
        if stripped.startswith(marker) or stripped.startswith("insufficient_context"):
            return True
            
        # Common phrasings where the model clearly says the context does not contain the answer
        patterns = [
            "insufficient_context",
            "context does not contain",
            "context doesn't contain",
            "context does not provide",
            "context doesn't provide",
            "not mentioned in the context",
            "not mentioned in the provided context",
            "not provided in the context",
            "not found in the context",
            "not contain relevant information",
            "no information in the context",
            "no information in the provided context",
            "provided context does not contain",
            "provided context doesn't contain",
            "provided context does not provide",
            "provided context doesn't provide",
            "does not contain the answer",
            "doesn't contain the answer",
            "cannot find the answer in the context",
            "can't find the answer in the context",
            "no relevant information in the context",
            "no relevant information in the provided context"
        ]
        return any(pattern in lower for pattern in patterns)

    def _format_source_footer(self, chunks: List[Dict[str, Any]]) -> str:
        """Format verified sources footer from retrieved chunks."""
        if not chunks:
            return ""
        
        sources = []
        seen = set()
        for chunk in chunks:
            filename = chunk.get("filename", "document.pdf")
            source_type = chunk.get("source_type", "pdf")
            section_label = chunk.get("section_label")
            p_start = chunk.get("page_start")
            p_end = chunk.get("page_end")
            page = chunk.get("page")
            
            if source_type == "web":
                entry = section_label if section_label else filename
            elif source_type == "markdown" and section_label:
                entry = f"{filename} ({section_label})"
            elif p_start is not None and p_end is not None:
                page_str = f"p. {p_start}" if p_start == p_end else f"p. {p_start}-{p_end}"
                entry = f"{filename} ({page_str})"
            elif page is not None:
                entry = f"{filename} (p. {page})"
            elif section_label:
                entry = f"{filename} ({section_label})"
            elif source_type == "text":
                entry = f"{filename}"
            else:
                entry = f"{filename} (p. 1)"
                
            if entry not in seen:
                seen.add(entry)
                sources.append(entry)
                
        if not sources:
            return ""
            
        footer_lines = ["\n\nSources:"] + [f"• {s}" for s in sources]
        return "\n".join(footer_lines)

    def _is_url(self, text: str) -> bool:
        """Check if text contains a URL.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains a URL
        """
        # Simple URL pattern
        url_pattern = r'https?://[^\s]+'
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
        
        if not filename or not filename.endswith('.pdf'):
            filename = f"document_{hash(url) % 10000}.pdf"
        
        return filename
    
    def _is_safe_url(self, url: str) -> Tuple[bool, str]:
        """Validate URL to protect against SSRF and unsupported schemes."""
        from ..processors.web_processor import validate_url_ssrf
        return validate_url_ssrf(url)
    
    async def _handle_url_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE, url: str):
        """Handle PDF upload from URL.
        
        Args:
            update: Telegram update
            context: Bot context
            url: URL to PDF file
        """
        user_id = update.effective_user.id
        
        # Extract URL if there's extra text
        url_match = re.search(r'https?://[^\s]+', url)
        if url_match:
            url = url_match.group(0)
        
        # Check if URL ends with .pdf
        if not url.lower().endswith('.pdf') and '.pdf' not in url.lower():
            await update.message.reply_text(
                "❌ The URL doesn't appear to be a PDF file.\n"
                "Please send a direct link to a PDF file."
            )
            return
        
        # Validate URL for SSRF protection
        is_safe, error_msg = self._is_safe_url(url)
        if not is_safe:
            await update.message.reply_text(f"❌ {error_msg}")
            return
        
        await update.message.reply_text("🔗 Downloading PDF from URL...")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, 
                                          action=ChatAction.TYPING)
        
        temp_file = None
        try:
            # Download file from URL with manual redirect following to prevent SSRF via redirects
            download_headers = {
                "User-Agent": "TeleRAG/1.0 (PDF Document Assistant)",
                "Accept": "application/pdf, application/octet-stream;q=0.9, */*;q=0.8",
            }
            
            current_url = url
            max_redirects = 3
            redirect_count = 0
            response = None
            
            while True:
                response = requests.get(
                    current_url, 
                    headers=download_headers, 
                    timeout=30, 
                    stream=True, 
                    allow_redirects=False
                )
                
                # Check for redirects (301, 302, 303, 307, 308)
                if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
                    redirect_count += 1
                    if redirect_count > max_redirects:
                        response.close()
                        await update.message.reply_text("❌ Too many redirects (maximum 3 allowed).")
                        return
                    
                    location = response.headers.get('Location')
                    if not location:
                        response.close()
                        await update.message.reply_text("❌ Redirect without location header.")
                        return
                    
                    # Resolve relative redirect URLs against current URL
                    next_url = urljoin(current_url, location)
                    
                    # Validate next hop URL for SSRF protection
                    is_safe, error_msg = self._is_safe_url(next_url)
                    if not is_safe:
                        response.close()
                        await update.message.reply_text(f"❌ Redirect blocked: {error_msg}")
                        return
                    
                    response.close()
                    current_url = next_url
                    continue
                
                # If not a redirect, proceed to process the response
                response.raise_for_status()
                break
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower() and not current_url.lower().endswith('.pdf'):
                response.close()
                await update.message.reply_text(
                    "❌ The URL doesn't point to a PDF file.\n"
                    f"Content type: {content_type}"
                )
                return
            
            # Check declared file size
            max_bytes = self.config.max_file_size_mb * 1024 * 1024
            content_length = response.headers.get('content-length')
            if content_length:
                try:
                    file_size = int(content_length)
                    if file_size > max_bytes:
                        response.close()
                        await update.message.reply_text(
                            f"❌ File too large. Maximum size is {self.config.max_file_size_mb}MB."
                        )
                        return
                except ValueError:
                    pass
            
            # Save to temp file with streaming cap to enforce MAX_FILE_SIZE_MB during streaming
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            downloaded_bytes = 0
            try:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > max_bytes:
                            temp_file.close()
                            response.close()
                            await update.message.reply_text(
                                f"❌ File too large. Maximum size is {self.config.max_file_size_mb}MB."
                            )
                            return
                        temp_file.write(chunk)
            finally:
                response.close()
            temp_file.close()
            
            # Get actual downloaded file size
            file_size = os.path.getsize(temp_file.name)
            
            await update.message.reply_text("📥 Processing PDF...")
            
            # Read file content
            with open(temp_file.name, 'rb') as f:
                file_content = f.read()
            
            # Validate PDF
            is_valid, error_msg = self.pdf_processor.validate_pdf(temp_file.name, file_size)
            if not is_valid:
                await update.message.reply_text(f"❌ {error_msg}")
                return
            
            # Extract text with pages
            pages_data = self.pdf_processor.extract_text_with_pages(temp_file.name)
            
            # Create chunks with page bounds
            chunks = self.text_chunker.create_chunks_with_pages(pages_data)
            
            if not chunks:
                await update.message.reply_text("❌ No text content found in PDF.")
                return
            
            # Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(chunk_texts)
            
            # Prepare chunks for storage
            chunks_data = [
                {
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content,
                    'char_count': chunk.char_count,
                    'page': chunk.page,
                    'page_start': chunk.page_start,
                    'page_end': chunk.page_end
                }
                for chunk in chunks
            ]
            
            # Get filename
            filename = self._extract_filename_from_url(current_url)
            
            # Store document
            self.document_store.store_document(
                user_id=user_id,
                filename=filename,
                file_content=file_content,
                chunks=chunks_data,
                embeddings=embeddings
            )
            
            await update.message.reply_text(
                f"✅ Successfully processed {filename}\n"
                f"📊 Created {len(chunks)} chunks\n"
                f"🔗 Source: {url[:50]}...\n\n"
                f"You can now ask questions about this document!"
            )
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout downloading from URL [{type(e).__name__}]: {e}")
            await update.message.reply_text(
                "❌ Download timeout. The file took too long to download.\n"
                "Please try again or use a different link."
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download from URL [{type(e).__name__}]: {e}")
            await update.message.reply_text(
                f"❌ Failed to download PDF from URL.\n"
                f"Error: {str(e)}\n\n"
                f"Please check the URL and try again."
            )
        except Exception as e:
            logger.error(f"Failed to process URL upload [{type(e).__name__}]: {e}")
            await update.message.reply_text(
                f"❌ Failed to process PDF: {str(e)}"
            )
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors."""
        err = context.error
        err_type = type(err).__name__ if err else "UnknownError"
        logger.error(f"Update caused error [{err_type}]: {err}")
