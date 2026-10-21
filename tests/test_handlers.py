"""Tests for bot handlers and security validations."""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from src.bot.handlers import BotHandlers


class DummyConfig:
    """Mock configuration for testing."""

    max_file_size_mb = 20
    llm_model = "qwen2.5:3b"


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
            llm_client=Mock(),
            config=self.config,
        )

    def test_non_http_scheme_rejected(self):
        """Test that non-http/https schemes (ftp, file, gopher, etc.) are rejected."""
        schemes = [
            "ftp://example.com/doc.pdf",
            "file:///etc/passwd",
            "gopher://example.com/",
            "data:text/plain;base64,SGVsbG8=",
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
            "http://169.254.169.254/latest/meta-data/",
        ]
        for url in private_urls:
            is_safe, error = self.handlers._is_safe_url(url)
            self.assertFalse(is_safe, f"URL {url} should be rejected")
            self.assertIn("forbidden", error.lower())

    @patch("src.bot.handlers.socket.getaddrinfo")
    def test_domain_resolving_to_private_ip_rejected(self, mock_getaddrinfo):
        """Test that domain names resolving to internal IPs are rejected."""
        mock_getaddrinfo.return_value = [(2, 1, 0, "", ("127.0.0.1", 80))]
        is_safe, error = self.handlers._is_safe_url("http://internal.service.local/test.pdf")
        self.assertFalse(is_safe)
        self.assertIn("forbidden", error.lower())

    @patch("src.bot.handlers.requests.get")
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

        with patch.object(self.handlers, "_is_safe_url") as mock_safe:
            # First hop is safe, second hop (127.0.0.1) is unsafe
            mock_safe.side_effect = [
                (True, ""),
                (False, "Access to private or local network resources is forbidden."),
            ]
            await self.handlers._handle_url_upload(
                update, context, "https://example.com/download.pdf"
            )

        update.message.reply_text.assert_called()
        replies = [call.args[0] for call in update.message.reply_text.call_args_list]
        self.assertTrue(any("Redirect blocked" in r and "forbidden" in r for r in replies))
        self.assertEqual(mock_requests_get.call_count, 1)
        mock_redirect.close.assert_called()

    @patch("src.bot.handlers.requests.get")
    async def test_redirect_to_cloud_metadata_rejected(self, mock_requests_get):
        """Test that redirects to 169.254.169.254 are detected and blocked."""
        mock_redirect = Mock()
        mock_redirect.status_code = 301
        mock_redirect.is_redirect = True
        mock_redirect.headers = {
            "Location": "http://169.254.169.254/latest/meta-data/credentials.pdf"
        }
        mock_redirect.close = Mock()

        mock_requests_get.return_value = mock_redirect

        update = Mock()
        update.effective_user.id = 123
        update.effective_chat.id = 456
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.bot.send_chat_action = AsyncMock()

        with patch.object(self.handlers, "_is_safe_url") as mock_safe:
            mock_safe.side_effect = [
                (True, ""),
                (False, "Access to private or local network resources is forbidden."),
            ]
            await self.handlers._handle_url_upload(update, context, "https://example.com/doc.pdf")

        update.message.reply_text.assert_called()
        replies = [call.args[0] for call in update.message.reply_text.call_args_list]
        self.assertTrue(any("Redirect blocked" in r and "forbidden" in r for r in replies))
        mock_redirect.close.assert_called()

    @patch("src.bot.handlers.requests.get")
    async def test_streaming_max_file_size_enforced(self, mock_requests_get):
        """Test that MAX_FILE_SIZE_MB is strictly enforced while streaming even without Content-Length."""
        self.config.max_file_size_mb = 1  # 1 MB limit
        1 * 1024 * 1024

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.is_redirect = False
        mock_response.headers = {"content-type": "application/pdf"}  # No content-length header
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

        with patch.object(self.handlers, "_is_safe_url", return_value=(True, "")):
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
            llm_client=self.mock_llm,
            config=self.config,
        )

    async def test_no_source_footer_on_generation_failure(self):
        """When LLM generation fails (returns None), no source footer should be appended."""
        user_id = 123
        self.mock_doc_store.list_documents.return_value = [{"id": 1, "filename": "doc.pdf"}]
        chunks = [
            {
                "content": "Some content",
                "filename": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "chunk_index": 0,
            }
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

        mock_thinking_msg = Mock()
        mock_thinking_msg.delete = AsyncMock()
        update = Mock()
        update.effective_user.id = user_id
        update.effective_chat.id = 456
        update.message.text = "What is the algorithm?"
        update.message.reply_text = AsyncMock(return_value=mock_thinking_msg)
        context = Mock()
        context.bot.send_chat_action = AsyncMock()

        await self.handlers.handle_message(update, context)

        self.assertEqual(update.message.reply_text.call_count, 2)
        self.assertEqual("Thinking...", update.message.reply_text.call_args_list[0][0][0])
        reply = update.message.reply_text.call_args_list[1][0][0]
        self.assertIn("couldn't generate a response", reply)
        self.assertNotIn("Sources:", reply)
        self.assertNotIn("Answered in", reply)
        mock_thinking_msg.delete.assert_called_once()

    async def test_summary_query_skips_cutoff_and_uses_first_chunks(self):
        """Summary/overview queries should bypass retriever cutoff and take first ~8 chunks in page order."""
        user_id = 123
        self.mock_doc_store.list_documents.return_value = [{"id": 1, "filename": "book.pdf"}]

        # 10 chunks out of order
        unordered_chunks = [
            {
                "content": f"Chunk {i}",
                "filename": "book.pdf",
                "page_start": 10 - i,
                "page_end": 10 - i,
                "chunk_index": 10 - i,
            }
            for i in range(10)
        ]
        self.mock_doc_store.get_user_chunks_with_embeddings.return_value = unordered_chunks
        self.mock_retriever.format_context.return_value = "Summary context"
        self.mock_llm.generate_answer.return_value = "This document is about Python programming."
        self.mock_llm.validate_response.return_value = "This document is about Python programming."

        mock_thinking_msg = Mock()
        mock_thinking_msg.delete = AsyncMock()
        update = Mock()
        update.effective_user.id = user_id
        update.effective_chat.id = 456
        update.message.text = "Can you summarize this document for me?"
        update.message.reply_text = AsyncMock(return_value=mock_thinking_msg)
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
        self.assertEqual(passed_chunks[0]["page_start"], 1)
        self.assertEqual(passed_chunks[-1]["page_start"], 8)

        self.assertEqual(update.message.reply_text.call_count, 2)
        self.assertEqual("Thinking...", update.message.reply_text.call_args_list[0][0][0])
        reply = update.message.reply_text.call_args_list[1][0][0]
        self.assertIn("This document is about Python programming.", reply)
        self.assertIn("Sources:", reply)
        self.assertIn("Answered in", reply)
        mock_thinking_msg.delete.assert_called_once()

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
            "What are the main points?",
        ]
        for q in summary_queries:
            self.assertTrue(
                self.handlers._is_summary_query(q), f"Failed to detect summary query: {q}"
            )

        non_summary_queries = [
            "What is a variable?",
            "How do for loops work in Python?",
            "Explain recursion on page 42",
        ]
        for q in non_summary_queries:
            self.assertFalse(
                self.handlers._is_summary_query(q), f"Incorrectly flagged non-summary query: {q}"
            )

    async def test_send_typing_action_retries_on_timeout(self):
        """Test _send_typing_action retries once when telegram.error.TimedOut occurs."""
        from telegram.error import TimedOut

        context = Mock()
        # First call fails with TimedOut, second succeeds
        context.bot.send_chat_action = AsyncMock(
            side_effect=[TimedOut("Connection timed out"), None]
        )

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

    async def test_answer_sent_as_new_reply_and_thinking_deleted_with_timing_footer(self):
        """Answer is delivered as a new reply, thinking message is deleted, and timing footer is appended."""
        user_id = 123
        self.mock_doc_store.list_documents.return_value = [{"id": 1, "filename": "doc.pdf"}]
        chunks = [
            {
                "content": "Test content",
                "filename": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "chunk_index": 0,
            }
        ]
        self.mock_doc_store.get_user_chunks_with_embeddings.return_value = chunks
        self.mock_embedder.generate_query_embedding.return_value = [0.1, 0.2]
        self.mock_retriever.retrieve.return_value = chunks
        self.mock_retriever.format_context.return_value = "Test content"
        self.mock_llm.generate_answer.return_value = "Test answer"
        self.mock_llm.validate_response.return_value = "Test answer"

        mock_thinking_msg = Mock()
        mock_thinking_msg.delete = AsyncMock()

        update = Mock()
        update.effective_user.id = user_id
        update.effective_chat.id = 456
        update.message.text = "What is this?"
        update.message.reply_text = AsyncMock(return_value=mock_thinking_msg)
        context = Mock()
        context.bot.send_chat_action = AsyncMock()

        await self.handlers.handle_message(update, context)

        # First reply is Thinking...
        self.assertEqual(update.message.reply_text.call_count, 2)
        first_call = update.message.reply_text.call_args_list[0][0][0]
        self.assertEqual("Thinking...", first_call)

        # Second reply is final answer with Sources and Answered in footer
        final_call = update.message.reply_text.call_args_list[1][0][0]
        self.assertIn("Test answer", final_call)
        self.assertIn("Sources:", final_call)
        self.assertIn("Answered in", final_call)
        self.assertIn("with qwen2.5:3b", final_call)

        # Thinking message deleted
        mock_thinking_msg.delete.assert_called_once()

    async def test_queue_position_message_when_semaphore_busy(self):
        """When the LLM semaphore is busy, immediately reply with queue position."""
        user_id = 999
        self.mock_doc_store.list_documents.return_value = [{"id": 1, "filename": "doc.pdf"}]
        chunks = [
            {
                "content": "Test chunk",
                "filename": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "chunk_index": 0,
            }
        ]
        self.mock_doc_store.get_user_chunks_with_embeddings.return_value = chunks
        self.mock_embedder.generate_query_embedding.return_value = [0.1, 0.2]
        self.mock_retriever.retrieve.return_value = chunks
        self.mock_retriever.format_context.return_value = "Context text"
        self.mock_llm.generate_answer.return_value = "Grounded response"
        self.mock_llm.validate_response.return_value = "Grounded response"

        mock_queued_msg = Mock()
        mock_queued_msg.delete = AsyncMock()

        update = Mock()
        update.effective_user.id = user_id
        update.effective_chat.id = 888
        update.message.text = "What is the result?"
        update.message.reply_text = AsyncMock(return_value=mock_queued_msg)
        context = Mock()
        context.bot.send_chat_action = AsyncMock()

        # Simulate semaphore locked and 1 active request ahead
        await self.handlers.llm_semaphore.acquire()
        self.handlers.active_requests = 1

        # Release semaphore after brief sleep to allow handle_message to finish
        async def release_later():
            await asyncio.sleep(0.01)
            self.handlers.llm_semaphore.release()

        asyncio.create_task(release_later())

        await self.handlers.handle_message(update, context)

        # 1. Verify queue notice was sent first (active_requests ahead was 1)
        reply_calls = [call.args[0] for call in update.message.reply_text.call_args_list]
        self.assertTrue(
            any(
                "Queued: 1 question(s) ahead of you. I'll reply when it's your turn." in r
                for r in reply_calls
            )
        )

        # 2. Verify answer delivered as a new reply
        self.assertEqual(update.message.reply_text.call_count, 2)
        final_reply = update.message.reply_text.call_args_list[1][0][0]
        self.assertIn("Grounded response", final_reply)
        self.assertIn("Sources:", final_reply)
        self.assertIn("Answered in", final_reply)

        # 3. Queued message was deleted
        mock_queued_msg.delete.assert_called_once()

    async def test_insufficient_context_response_suppresses_sources_footer(self):
        """When the LLM replies with INSUFFICIENT_CONTEXT or indicates missing context, suppress Sources footer."""
        user_id = 123
        self.mock_doc_store.list_documents.return_value = [{"id": 1, "filename": "guide.pdf"}]
        chunks = [
            {
                "content": "Some unrelated programming tips.",
                "filename": "guide.pdf",
                "page_start": 5,
                "page_end": 5,
                "chunk_index": 0,
            }
        ]
        self.mock_doc_store.get_user_chunks_with_embeddings.return_value = chunks
        self.mock_embedder.generate_query_embedding.return_value = [0.1, 0.2]
        self.mock_retriever.retrieve.return_value = chunks
        self.mock_retriever.format_context.return_value = "Formatted context"

        # Model returns INSUFFICIENT_CONTEXT
        self.mock_llm.generate_answer.return_value = "INSUFFICIENT_CONTEXT"
        self.mock_llm.validate_response.return_value = "INSUFFICIENT_CONTEXT"

        mock_thinking_msg = Mock()
        mock_thinking_msg.delete = AsyncMock()
        update = Mock()
        update.effective_user.id = user_id
        update.effective_chat.id = 456
        update.message.text = "How do I bake sourdough bread?"
        update.message.reply_text = AsyncMock(return_value=mock_thinking_msg)
        context = Mock()
        context.bot.send_chat_action = AsyncMock()

        await self.handlers.handle_message(update, context)

        self.assertEqual(update.message.reply_text.call_count, 2)
        reply = update.message.reply_text.call_args_list[1][0][0]
        self.assertIn(
            "I couldn't find relevant information in your documents to answer this question.", reply
        )
        self.assertNotIn("Sources:", reply)
        self.assertNotIn("Answered in", reply)
        self.assertNotIn("guide.pdf", reply)
        mock_thinking_msg.delete.assert_called_once()

    async def test_normal_answer_includes_sources_footer(self):
        """A normal grounded LLM answer must still retain its verified Sources footer."""
        user_id = 123
        self.mock_doc_store.list_documents.return_value = [
            {"id": 1, "filename": "python_tutorial.pdf"}
        ]
        chunks = [
            {
                "content": "Functions are defined using the def keyword.",
                "filename": "python_tutorial.pdf",
                "page_start": 12,
                "page_end": 13,
                "chunk_index": 2,
            }
        ]
        self.mock_doc_store.get_user_chunks_with_embeddings.return_value = chunks
        self.mock_embedder.generate_query_embedding.return_value = [0.1, 0.2]
        self.mock_retriever.retrieve.return_value = chunks
        self.mock_retriever.format_context.return_value = "Context text"

        # Model returns a valid answer
        self.mock_llm.generate_answer.return_value = (
            "In Python, functions are defined using the def keyword."
        )
        self.mock_llm.validate_response.return_value = (
            "In Python, functions are defined using the def keyword."
        )

        mock_thinking_msg = Mock()
        mock_thinking_msg.delete = AsyncMock()
        update = Mock()
        update.effective_user.id = user_id
        update.effective_chat.id = 456
        update.message.text = "How do you define a function in Python?"
        update.message.reply_text = AsyncMock(return_value=mock_thinking_msg)
        context = Mock()
        context.bot.send_chat_action = AsyncMock()

        await self.handlers.handle_message(update, context)

        self.assertEqual(update.message.reply_text.call_count, 2)
        reply = update.message.reply_text.call_args_list[1][0][0]
        self.assertIn("In Python, functions are defined using the def keyword.", reply)
        self.assertIn("Sources:", reply)
        self.assertIn("python_tutorial.pdf (p. 12-13)", reply)
        self.assertIn("Answered in", reply)
        mock_thinking_msg.delete.assert_called_once()


if __name__ == "__main__":
    unittest.main()
