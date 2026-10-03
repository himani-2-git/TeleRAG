"""Telegram bot handlers for PDF RAG chatbot."""

import os
import tempfile
import requests
import re
import socket
import ipaddress
from typing import Tuple
from urllib.parse import urlparse
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction
import logging

logger = logging.getLogger(__name__)

class BotHandlers:
    """Handles Telegram bot interactions."""
    
    def __init__(self, pdf_processor, text_chunker, document_store, 
                 embedding_generator, retriever, openrouter_client, config):
        """Initialize bot handlers.
        
        Args:
            pdf_processor: PDF processing instance
            text_chunker: Text chunking instance
            document_store: Document storage instance
            embedding_generator: Embedding generation instance
            retriever: Document retrieval instance
            openrouter_client: OpenRouter API client instance
            config: Application configuration
        """
        self.pdf_processor = pdf_processor
        self.text_chunker = text_chunker
        self.document_store = document_store
        self.embedding_generator = embedding_generator
        self.retriever = retriever
        self.openrouter_client = openrouter_client
        self.config = config
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        welcome_message = (
            "👋 Welcome to PDF RAG Chatbot!\n\n"
            "I can help you analyze PDF documents and answer questions about them.\n\n"
            "📤 Send me a PDF file to get started\n"
            "🔗 Or send a direct link to a PDF\n"
            "❓ Ask questions about your uploaded documents\n\n"
            "Commands:\n"
            "/list - Show your uploaded documents\n"
            "/delete <filename> - Delete a specific document\n"
            "/clear - Delete all your documents\n"
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
            logger.error(f"Failed to list documents: {e}")
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
            logger.error(f"Failed to delete document: {e}")
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
            logger.error(f"Failed to clear documents: {e}")
            await update.message.reply_text("❌ Failed to clear documents.")
    
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle PDF document uploads."""
        user_id = update.effective_user.id
        document = update.message.document
        
        # Check if it's a PDF
        if not document.file_name.lower().endswith('.pdf'):
            await update.message.reply_text("❌ Please send a PDF file.")
            return
        
        # Check file size
        if document.file_size > self.config.max_file_size_mb * 1024 * 1024:
            await update.message.reply_text(
                f"❌ File too large. Maximum size is {self.config.max_file_size_mb}MB."
            )
            return
        
        await update.message.reply_text("📥 Processing your PDF...")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, 
                                          action=ChatAction.TYPING)
        
        temp_file = None
        try:
            # Download file
            file = await document.get_file()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            await file.download_to_drive(temp_file.name)
            
            # Read file content for storage
            with open(temp_file.name, 'rb') as f:
                file_content = f.read()
            
            # Validate PDF
            is_valid, error_msg = self.pdf_processor.validate_pdf(temp_file.name, 
                                                                  document.file_size)
            if not is_valid:
                await update.message.reply_text(f"❌ {error_msg}")
                return
            
            # Extract text
            text = self.pdf_processor.extract_text(temp_file.name)
            
            # Create chunks
            chunks = self.text_chunker.create_chunks(text)
            
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
                    'char_count': chunk.char_count
                }
                for chunk in chunks
            ]
            
            # Store document
            self.document_store.store_document(
                user_id=user_id,
                filename=document.file_name,
                file_content=file_content,
                chunks=chunks_data,
                embeddings=embeddings
            )
            
            await update.message.reply_text(
                f"✅ Successfully processed {document.file_name}\n"
                f"📊 Created {len(chunks)} chunks\n\n"
                f"You can now ask questions about this document!"
            )
            
        except Exception as e:
            logger.error(f"Failed to process document: {e}")
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
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, 
                                          action=ChatAction.TYPING)
        
        try:
            # Get all user chunks with embeddings
            chunks = self.document_store.get_user_chunks_with_embeddings(user_id)
            
            if not chunks:
                await update.message.reply_text("❌ No document content available.")
                return
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_query_embedding(question)
            
            # Search for relevant chunks
            relevant_chunks = self.retriever.search_similar(query_embedding, chunks)
            
            if not relevant_chunks:
                await update.message.reply_text(
                    "🤔 I couldn't find relevant information in your documents to answer this question."
                )
                return
            
            # Format context
            context_text = self.retriever.format_context(relevant_chunks)
            
            # Generate answer
            answer = self.openrouter_client.generate_answer(question, context_text)
            answer = self.openrouter_client.validate_response(answer)
            
            await update.message.reply_text(answer)
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            await update.message.reply_text(
                "❌ Sorry, I encountered an error while processing your question. Please try again."
            )
    
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
        """Validate URL to protect against SSRF and unsupported schemes.
        
        Args:
            url: URL string to validate
            
        Returns:
            Tuple of (is_safe, error_message)
        """
        try:
            parsed = urlparse(url)
            if parsed.scheme.lower() not in ('http', 'https'):
                return False, "Only HTTP and HTTPS URLs are supported."
            
            hostname = parsed.hostname
            if not hostname:
                return False, "Invalid URL: missing hostname."
            
            # Resolve DNS and verify IP addresses are not private/loopback/link-local
            addr_info = socket.getaddrinfo(hostname, None)
            for addr in addr_info:
                ip_str = addr[4][0]
                ip = ipaddress.ip_address(ip_str)
                if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                    return False, "Access to private or local network resources is forbidden."
            
            return True, ""
        except Exception as e:
            return False, f"Invalid or unresolvable URL: {str(e)}"
    
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
            # Download file from URL
            download_headers = {
                "User-Agent": "TeleRAG/1.0 (PDF Document Assistant)",
                "Accept": "application/pdf, application/octet-stream;q=0.9, */*;q=0.8",
            }
            response = requests.get(url, headers=download_headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower() and not url.lower().endswith('.pdf'):
                await update.message.reply_text(
                    "❌ The URL doesn't point to a PDF file.\n"
                    f"Content type: {content_type}"
                )
                return
            
            # Check declared file size
            max_bytes = self.config.max_file_size_mb * 1024 * 1024
            content_length = response.headers.get('content-length')
            if content_length:
                file_size = int(content_length)
                if file_size > max_bytes:
                    await update.message.reply_text(
                        f"❌ File too large. Maximum size is {self.config.max_file_size_mb}MB."
                    )
                    return
            
            # Save to temp file with streaming cap to prevent unbounded download
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            downloaded_bytes = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > max_bytes:
                        temp_file.close()
                        await update.message.reply_text(
                            f"❌ File too large. Maximum size is {self.config.max_file_size_mb}MB."
                        )
                        return
                    temp_file.write(chunk)
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
            
            # Extract text
            text = self.pdf_processor.extract_text(temp_file.name)
            
            # Create chunks
            chunks = self.text_chunker.create_chunks(text)
            
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
                    'char_count': chunk.char_count
                }
                for chunk in chunks
            ]
            
            # Get filename
            filename = self._extract_filename_from_url(url)
            
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
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout downloading from URL: {url}")
            await update.message.reply_text(
                "❌ Download timeout. The file took too long to download.\n"
                "Please try again or use a different link."
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download from URL: {e}")
            await update.message.reply_text(
                f"❌ Failed to download PDF from URL.\n"
                f"Error: {str(e)}\n\n"
                f"Please check the URL and try again."
            )
        except Exception as e:
            logger.error(f"Failed to process URL upload: {e}")
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
        logger.error(f"Update {update} caused error {context.error}")
