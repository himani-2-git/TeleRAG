"""Tests for bot handlers and security validations."""

import unittest
from unittest.mock import Mock, AsyncMock, patch
from src.bot.handlers import BotHandlers

class DummyConfig:
    """Mock configuration for testing."""
    max_file_size_mb = 20

class TestBotHandlersSSRF(unittest.IsolatedAsyncioTestCase):
    """Test SSRF validation and redirect handling in BotHandlers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DummyConfig()
        self.handlers = BotHandlers(
            pdf_processor=Mock(),
            text_chunker=Mock(),
            document_store=Mock(),
            embedding_generator=Mock(),
            retriever=Mock(),
            openrouter_client=Mock(),
            config=self.config
        )
    
    def test_non_http_scheme_rejected(self):
        """Test that non-http/https schemes (ftp, file, gopher, etc.) are rejected."""
        schemes = [
            "ftp://example.com/doc.pdf",
            "file:///etc/passwd",
            "gopher://example.com/",
            "data:text/plain;base64,SGVsbG8="
        ]
        for url in schemes:
            is_safe, error = self.handlers._is_safe_url(url)
            self.assertFalse(is_safe, f"Scheme in {url} should be rejected")
            self.assertIn("Only HTTP and HTTPS URLs are supported", error)
    
    def test_private_ip_literal_rejected(self):
        """Test that private, loopback, and link-local IP literals are rejected."""
        private_urls = [
            "http://127.0.0.1/test.pdf",
            "http://10.0.0.1/test.pdf",
            "http://192.168.1.1/test.pdf",
            "http://172.16.0.1/test.pdf",
            "http://169.254.169.254/latest/meta-data/"
        ]
        for url in private_urls:
            is_safe, error = self.handlers._is_safe_url(url)
            self.assertFalse(is_safe, f"URL {url} should be rejected")
            self.assertIn("forbidden", error.lower())
    
    @patch('src.bot.handlers.socket.getaddrinfo')
    def test_domain_resolving_to_private_ip_rejected(self, mock_getaddrinfo):
        """Test that domain names resolving to internal IPs are rejected."""
        mock_getaddrinfo.return_value = [
            (2, 1, 0, '', ('127.0.0.1', 80))
        ]
        is_safe, error = self.handlers._is_safe_url("http://internal.service.local/test.pdf")
        self.assertFalse(is_safe)
        self.assertIn("forbidden", error.lower())
    
    @patch('src.bot.handlers.requests.get')
    async def test_redirect_to_loopback_rejected(self, mock_requests_get):
        """Test that redirects to 127.0.0.1 are detected and blocked."""
        # Initial request returns a 302 redirect to 127.0.0.1
        mock_redirect = Mock()
        mock_redirect.status_code = 302
        mock_redirect.is_redirect = True
        mock_redirect.headers = {"Location": "http://127.0.0.1/secret.pdf"}
        mock_redirect.close = Mock()
        
        mock_requests_get.return_value = mock_redirect
        
        update = Mock()
        update.effective_user.id = 123
        update.effective_chat.id = 456
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.bot.send_chat_action = AsyncMock()
        
        with patch.object(self.handlers, '_is_safe_url') as mock_safe:
            # First hop is safe, second hop (127.0.0.1) is unsafe
            mock_safe.side_effect = [
                (True, ""),
                (False, "Access to private or local network resources is forbidden.")
            ]
            await self.handlers._handle_url_upload(update, context, "https://example.com/download.pdf")
        
        update.message.reply_text.assert_called()
        replies = [call.args[0] for call in update.message.reply_text.call_args_list]
        self.assertTrue(any("Redirect blocked" in r and "forbidden" in r for r in replies))
        self.assertEqual(mock_requests_get.call_count, 1)
        mock_redirect.close.assert_called()
    
    @patch('src.bot.handlers.requests.get')
    async def test_redirect_to_cloud_metadata_rejected(self, mock_requests_get):
        """Test that redirects to 169.254.169.254 are detected and blocked."""
        mock_redirect = Mock()
        mock_redirect.status_code = 301
        mock_redirect.is_redirect = True
        mock_redirect.headers = {"Location": "http://169.254.169.254/latest/meta-data/credentials.pdf"}
        mock_redirect.close = Mock()
        
        mock_requests_get.return_value = mock_redirect
        
        update = Mock()
        update.effective_user.id = 123
        update.effective_chat.id = 456
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.bot.send_chat_action = AsyncMock()
        
        with patch.object(self.handlers, '_is_safe_url') as mock_safe:
            mock_safe.side_effect = [
                (True, ""),
                (False, "Access to private or local network resources is forbidden.")
            ]
            await self.handlers._handle_url_upload(update, context, "https://example.com/doc.pdf")
        
        update.message.reply_text.assert_called()
        replies = [call.args[0] for call in update.message.reply_text.call_args_list]
        self.assertTrue(any("Redirect blocked" in r and "forbidden" in r for r in replies))
        mock_redirect.close.assert_called()
    
    @patch('src.bot.handlers.requests.get')
    async def test_streaming_max_file_size_enforced(self, mock_requests_get):
        """Test that MAX_FILE_SIZE_MB is strictly enforced while streaming even without Content-Length."""
        self.config.max_file_size_mb = 1  # 1 MB limit
        limit_bytes = 1 * 1024 * 1024
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.is_redirect = False
        mock_response.headers = {'content-type': 'application/pdf'}  # No content-length header
        mock_response.close = Mock()
        # Stream chunks totaling over 1 MB (e.g., 3 chunks of 500KB)
        chunk = b"A" * (500 * 1024)
        mock_response.iter_content.return_value = [chunk, chunk, chunk]
        mock_requests_get.return_value = mock_response
        
        update = Mock()
        update.effective_user.id = 123
        update.effective_chat.id = 456
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.bot.send_chat_action = AsyncMock()
        
        with patch.object(self.handlers, '_is_safe_url', return_value=(True, "")):
            await self.handlers._handle_url_upload(update, context, "https://example.com/large.pdf")
        
        update.message.reply_text.assert_called()
        replies = [call.args[0] for call in update.message.reply_text.call_args_list]
        self.assertTrue(any("File too large" in r for r in replies))
        mock_response.close.assert_called()

class TestBotHandlersQuestionAnswering(unittest.IsolatedAsyncioTestCase):
    """Test question answering workflows, footer suppression, and summary handling."""
    
    def setUp(self):
        self.config = DummyConfig()
        self.mock_doc_store = Mock()
        self.mock_embedder = Mock()
        self.mock_retriever = Mock()
        self.mock_llm = Mock()
        
        self.handlers = BotHandlers(
            pdf_processor=Mock(),
            text_chunker=Mock(),
            document_store=self.mock_doc_store,
            embedding_generator=self.mock_embedder,
            retriever=self.mock_retriever,
            openrouter_client=self.mock_llm,
            config=self.config
        )
    
    async def test_no_source_footer_on_generation_failure(self):
        """When LLM generation fails (returns None), no source footer should be appended."""
        user_id = 123
        self.mock_doc_store.list_documents.return_value = [{'id': 1, 'filename': 'doc.pdf'}]
        chunks = [
            {'content': 'Some content', 'filename': 'doc.pdf', 'page_start': 1, 'page_end': 1, 'chunk_index': 0}
        ]
        self.mock_doc_store.get_user_chunks_with_embeddings.return_value = chunks
        self.mock_embedder.generate_query_embedding.return_value = [0.1, 0.2]
        self.mock_retriever.retrieve.return_value = chunks
        self.mock_retriever.format_context.return_value = "Formatted context"
        
        # LLM generation fails (returns None)
        self.mock_llm.generate_answer.return_value = None
        self.mock_llm.validate_response.return_value = (
            "I'm sorry, I couldn't generate a response at this time. Please try again."
        )
        
        update = Mock()
        update.effective_user.id = user_id
        update.effective_chat.id = 456
        update.message.text = "What is the algorithm?"
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.bot.send_chat_action = AsyncMock()
        
        await self.handlers.handle_message(update, context)
        
        update.message.reply_text.assert_called_once()
        reply = update.message.reply_text.call_args[0][0]
        self.assertIn("couldn't generate a response", reply)
        self.assertNotIn("Sources:", reply)
    
    async def test_summary_query_skips_cutoff_and_uses_first_chunks(self):
        """Summary/overview queries should bypass retriever cutoff and take first ~8 chunks in page order."""
        user_id = 123
        self.mock_doc_store.list_documents.return_value = [{'id': 1, 'filename': 'book.pdf'}]
        
        # 10 chunks out of order
        unordered_chunks = [
            {'content': f'Chunk {i}', 'filename': 'book.pdf', 'page_start': 10 - i, 'page_end': 10 - i, 'chunk_index': 10 - i}
            for i in range(10)
        ]
        self.mock_doc_store.get_user_chunks_with_embeddings.return_value = unordered_chunks
        self.mock_retriever.format_context.return_value = "Summary context"
        self.mock_llm.generate_answer.return_value = "This document is about Python programming."
        self.mock_llm.validate_response.return_value = "This document is about Python programming."
        
        update = Mock()
        update.effective_user.id = user_id
        update.effective_chat.id = 456
        update.message.text = "Can you summarize this document for me?"
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.bot.send_chat_action = AsyncMock()
        
        await self.handlers.handle_message(update, context)
        
        # Retriever.retrieve should NOT be called because cutoff was skipped
        self.mock_retriever.retrieve.assert_not_called()
        self.mock_embedder.generate_query_embedding.assert_not_called()
        
        # format_context should receive 8 chunks sorted by page_start
        self.mock_retriever.format_context.assert_called_once()
        passed_chunks = self.mock_retriever.format_context.call_args[0][0]
        self.assertEqual(len(passed_chunks), 8)
        self.assertEqual(passed_chunks[0]['page_start'], 1)
        self.assertEqual(passed_chunks[-1]['page_start'], 8)
        
        update.message.reply_text.assert_called_once()
        reply = update.message.reply_text.call_args[0][0]
        self.assertIn("This document is about Python programming.", reply)
        self.assertIn("Sources:", reply)
    
    def test_is_summary_query_patterns(self):
        """Test _is_summary_query detects common summary/overview phrasings."""
        summary_queries = [
            "summarize this document",
            "Give me a summary",
            "Can I get an overview of the book?",
            "What is this document about?",
            "What is the book about?",
            "what is this pdf about",
            "Brief overview of topics",
            "What are the main points?"
        ]
        for q in summary_queries:
            self.assertTrue(self.handlers._is_summary_query(q), f"Failed to detect summary query: {q}")
            
        non_summary_queries = [
            "What is a variable?",
            "How do for loops work in Python?",
            "Explain recursion on page 42"
        ]
        for q in non_summary_queries:
            self.assertFalse(self.handlers._is_summary_query(q), f"Incorrectly flagged non-summary query: {q}")

    async def test_send_typing_action_retries_on_timeout(self):
        """Test _send_typing_action retries once when telegram.error.TimedOut occurs."""
        from telegram.error import TimedOut
        context = Mock()
        # First call fails with TimedOut, second succeeds
        context.bot.send_chat_action = AsyncMock(side_effect=[TimedOut("Connection timed out"), None])
        
        await self.handlers._send_typing_action(context, 12345)
        
        self.assertEqual(context.bot.send_chat_action.call_count, 2)

    async def test_reply_with_retry_on_network_error(self):
        """Test _reply_with_retry retries reply_text once if NetworkError is raised."""
        from telegram.error import NetworkError
        update = Mock()
        # First attempt fails with NetworkError, retry succeeds
        update.message.reply_text = AsyncMock(side_effect=[NetworkError("Connection reset"), None])
        
        await self.handlers._reply_with_retry(update, "Hello world")
        
        self.assertEqual(update.message.reply_text.call_count, 2)
        update.message.reply_text.assert_called_with("Hello world")

if __name__ == '__main__':
    unittest.main()
