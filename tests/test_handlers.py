"""Tests for bot handlers and security validations."""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from telegram.error import BadRequest

from src.bot.handlers import BotHandlers
from src.bot.handlers.questions import format_telegram_html, split_telegram_message, utf16_len
from src.config import Config
from src.rag.pipeline import is_insufficient_context, is_summary_query


class TestBotHandlersSSRF(unittest.IsolatedAsyncioTestCase):
    """Test SSRF validation and redirect handling in BotHandlers."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(telegram_bot_token="test")
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

    @patch("src.processors.web_processor.socket.getaddrinfo")
    def test_domain_resolving_to_private_ip_rejected(self, mock_getaddrinfo):
        """Test that domain names resolving to internal IPs are rejected."""
        mock_getaddrinfo.return_value = [(2, 1, 0, "", ("127.0.0.1", 80))]
        is_safe, error = self.handlers._is_safe_url("http://internal.service.local/test.pdf")
        self.assertFalse(is_safe)
        self.assertIn("forbidden", error.lower())

    @patch("src.processors.pdf_fetcher.requests.get")
    async def test_redirect_to_loopback_rejected(self, mock_requests_get):
        """Test that redirects to 127.0.0.1 are detected and blocked."""
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

        with patch("src.processors.pdf_fetcher.validate_url_ssrf") as mock_safe:
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

    @patch("src.processors.pdf_fetcher.requests.get")
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

        with patch("src.processors.pdf_fetcher.validate_url_ssrf") as mock_safe:
            mock_safe.side_effect = [
                (True, ""),
                (False, "Access to private or local network resources is forbidden."),
            ]
            await self.handlers._handle_url_upload(update, context, "https://example.com/doc.pdf")

        update.message.reply_text.assert_called()
        replies = [call.args[0] for call in update.message.reply_text.call_args_list]
        self.assertTrue(any("Redirect blocked" in r and "forbidden" in r for r in replies))
        mock_redirect.close.assert_called()

    @patch("src.processors.pdf_fetcher.requests.get")
    async def test_streaming_max_file_size_enforced(self, mock_requests_get):
        """Test that MAX_FILE_SIZE_MB is strictly enforced while streaming even without Content-Length."""
        self.config.max_file_size_mb = 1

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.is_redirect = False
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.close = Mock()
        chunk = b"A" * (500 * 1024)
        mock_response.iter_content.return_value = [chunk, chunk, chunk]
        mock_requests_get.return_value = mock_response

        update = Mock()
        update.effective_user.id = 123
        update.effective_chat.id = 456
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.bot.send_chat_action = AsyncMock()

        with patch("src.processors.pdf_fetcher.validate_url_ssrf", return_value=(True, "")):
            await self.handlers._handle_url_upload(update, context, "https://example.com/large.pdf")

        update.message.reply_text.assert_called()
        replies = [call.args[0] for call in update.message.reply_text.call_args_list]
        self.assertTrue(any("File too large" in r for r in replies))
        mock_response.close.assert_called()

    @patch("src.processors.web_processor.socket.getaddrinfo")
    def test_dns_rebinding_style_resolution_rejected(self, mock_getaddrinfo):
        """Test that hostname resolving to a mix of public and private IPs is rejected."""
        mock_getaddrinfo.return_value = [
            (2, 1, 0, "", ("93.184.216.34", 80)),
            (2, 1, 0, "", ("127.0.0.1", 80)),
        ]
        is_safe, error = self.handlers._is_safe_url("http://rebinding.attack.com/doc.pdf")
        self.assertFalse(is_safe)
        self.assertIn("forbidden", error.lower())


class TestBotHandlersQuestionAnswering(unittest.IsolatedAsyncioTestCase):
    """Test question answering workflows, footer suppression, and summary handling."""

    def setUp(self):
        self.config = Config(telegram_bot_token="test")
        self.mock_doc_store = Mock()
        self.mock_embedder = Mock()
        self.mock_embedder.generate_query_embedding.return_value = [0.1, 0.2]
        self.mock_embedder.generate_embeddings.side_effect = lambda texts: (
            [[0.1, 0.2] for _ in texts] if isinstance(texts, list) else [0.1, 0.2]
        )
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

        # LLM generation fails (raises LLMUnavailableError)
        from src.rag.llm_client import LLMUnavailableError

        self.mock_llm.generate_answer.side_effect = LLMUnavailableError("Service down")

        mock_thinking_msg = Mock()
        mock_thinking_msg.delete = AsyncMock()
        update = Mock()
        update.effective_user.id = user_id
        update.effective_chat.id = 456
        update.message.text = "What is the algorithm?"
        update.message.reply_text = AsyncMock(return_value=mock_thinking_msg)
        context = Mock()
        context.bot.send_chat_action = AsyncMock()

        with self.assertNoLogs("src.rag.pipeline", level="WARNING"):
            await self.handlers.handle_message(update, context)

        self.assertEqual(update.message.reply_text.call_count, 2)
        self.assertEqual("Thinking...", update.message.reply_text.call_args_list[0][0][0])
        reply = update.message.reply_text.call_args_list[1][0][0]
        self.assertIn("model is currently unavailable", reply)
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

        with self.assertNoLogs("src.rag.pipeline", level="WARNING"):
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
        """Test is_summary_query detects common summary/overview phrasings."""
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
            self.assertTrue(is_summary_query(q), f"Failed to detect summary query: {q}")

        non_summary_queries = [
            "What is a variable?",
            "How do for loops work in Python?",
            "Explain recursion on page 42",
        ]
        for q in non_summary_queries:
            self.assertFalse(is_summary_query(q), f"Incorrectly flagged non-summary query: {q}")

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

        with self.assertNoLogs("src.rag.pipeline", level="WARNING"):
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

        with self.assertNoLogs("src.rag.pipeline", level="WARNING"):
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

        with self.assertNoLogs("src.rag.pipeline", level="WARNING"):
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

        with self.assertNoLogs("src.rag.pipeline", level="WARNING"):
            await self.handlers.handle_message(update, context)

        self.assertEqual(update.message.reply_text.call_count, 2)
        reply = update.message.reply_text.call_args_list[1][0][0]
        self.assertIn("In Python, functions are defined using the def keyword.", reply)
        self.assertIn("Sources:", reply)
        self.assertIn("python_tutorial.pdf (p. 12-13)", reply)
        self.assertIn("Answered in", reply)
        mock_thinking_msg.delete.assert_called_once()

    async def test_handle_message_calls_orchestrator(self):
        """Verify handle_message routes through the orchestrator."""
        user_id = 123
        chunks = [{"content": "hello", "filename": "doc.pdf", "page": 1, "chunk_index": 0}]
        self.mock_doc_store.get_user_chunks_with_embeddings.return_value = chunks
        mock_orch = Mock()
        mock_orch.run.return_value = {
            "answer": "Test answer",
            "final_response": "Test answer\n\nSources:\n• doc.pdf (p. 1)",
            "refused": False,
        }
        self.handlers.orchestrator = mock_orch

        mock_thinking_msg = Mock()
        mock_thinking_msg.delete = AsyncMock()
        update = Mock()
        update.effective_user.id = user_id
        update.effective_chat.id = 456
        update.message.text = "What is hello?"
        update.message.reply_text = AsyncMock(return_value=mock_thinking_msg)
        context = Mock()
        context.bot.send_chat_action = AsyncMock()

        with self.assertNoLogs("src.rag.pipeline", level="WARNING"):
            await self.handlers.handle_message(update, context)

        mock_orch.run.assert_called_once_with(
            question="What is hello?",
            user_id=user_id,
            user_chunks=chunks,
        )

    def test_marker_only_insufficient_context(self):
        """Verify marker-only check strips whitespace/quotes/backticks and checks prefix."""
        self.assertTrue(is_insufficient_context("INSUFFICIENT_CONTEXT"))
        self.assertTrue(is_insufficient_context("`INSUFFICIENT_CONTEXT`"))
        self.assertTrue(is_insufficient_context(" 'INSUFFICIENT_CONTEXT' "))
        self.assertTrue(is_insufficient_context('"INSUFFICIENT_CONTEXT"'))
        self.assertTrue(is_insufficient_context("insufficient_context: cannot find it"))
        # Non-marker phrases should not be caught
        self.assertFalse(is_insufficient_context("I don't have enough information."))
        self.assertFalse(is_insufficient_context("The context does not mention this."))
        self.assertFalse(is_insufficient_context(""))

    async def test_queue_accounting_increments_before_notice_and_decrements_in_finally(self):
        """Verify active_requests increments before queue notice and decrements in finally."""
        user_id = 999
        self.mock_doc_store.list_documents.return_value = [{"id": 1, "filename": "doc.pdf"}]
        self.mock_doc_store.get_user_chunks_with_embeddings.return_value = [
            {"content": "content", "filename": "doc.pdf", "page": 1, "chunk_index": 0}
        ]

        await self.handlers.llm_semaphore.acquire()
        self.handlers.active_requests = 1

        update = Mock()
        update.effective_user.id = user_id
        update.effective_chat.id = 888
        update.message.text = "Question?"
        mock_queued_msg = Mock()
        mock_queued_msg.delete = AsyncMock()
        update.message.reply_text = AsyncMock(return_value=mock_queued_msg)
        context = Mock()
        context.bot.send_chat_action = AsyncMock()

        async def release_later():
            await asyncio.sleep(0.01)
            self.handlers.llm_semaphore.release()

        asyncio.create_task(release_later())

        with self.assertNoLogs("src.rag.pipeline", level="WARNING"):
            await self.handlers.handle_message(update, context)

        self.assertEqual(self.handlers.active_requests, 1)


class TestCommandHandlers(unittest.IsolatedAsyncioTestCase):
    """Tests for bot slash commands (/start, /help, /delete, etc.)."""

    def setUp(self):
        self.doc_store = Mock()
        self.handlers = BotHandlers(
            pdf_processor=Mock(),
            text_chunker=Mock(),
            document_store=self.doc_store,
            embedding_generator=Mock(),
            retriever=Mock(),
            llm_client=Mock(),
            config=Config(telegram_bot_token="test"),
        )

    async def test_start_and_help_command_text(self):
        update = Mock()
        update.message.reply_text = AsyncMock()
        context = Mock()

        await self.handlers.start_command(update, context)
        start_msg = update.message.reply_text.call_args[0][0]
        self.assertIn("/delete 2 or /delete part-of-name", start_msg)
        self.assertNotIn("<filename>", start_msg)

        update.message.reply_text.reset_mock()
        await self.handlers.help_command(update, context)
        help_msg = update.message.reply_text.call_args[0][0]
        self.assertIn("/delete 2 or /delete part-of-name", help_msg)
        self.assertNotIn("<filename>", help_msg)

    async def test_delete_missing_args(self):
        update = Mock()
        update.effective_user.id = 123
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.args = []

        await self.handlers.delete_command(update, context)
        msg = update.message.reply_text.call_args[0][0]
        self.assertIn("Please specify a document to delete", msg)

    async def test_delete_by_1_based_index(self):
        update = Mock()
        update.effective_user.id = 123
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.args = ["2"]

        self.doc_store.list_documents.return_value = [
            {"filename": "first.pdf"},
            {"filename": "second.pdf"},
        ]
        self.doc_store.delete_document.return_value = True

        await self.handlers.delete_command(update, context)
        self.doc_store.delete_document.assert_called_once_with(123, "second.pdf")
        update.message.reply_text.assert_called_once_with("✅ Deleted document: second.pdf")

    async def test_delete_strip_brackets_and_quotes(self):
        update = Mock()
        update.effective_user.id = 123
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.args = ['<"sample', 'doc.pdf">']

        self.doc_store.list_documents.return_value = [
            {"filename": "sample doc.pdf"},
        ]
        self.doc_store.delete_document.return_value = True

        await self.handlers.delete_command(update, context)
        self.doc_store.delete_document.assert_called_once_with(123, "sample doc.pdf")
        update.message.reply_text.assert_called_once_with("✅ Deleted document: sample doc.pdf")

    async def test_delete_case_insensitive_exact_match(self):
        update = Mock()
        update.effective_user.id = 123
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.args = ["SAMPLE.PDF"]

        self.doc_store.list_documents.return_value = [
            {"filename": "sample.pdf"},
            {"filename": "sample.pdf.bak"},
        ]
        self.doc_store.delete_document.return_value = True

        await self.handlers.delete_command(update, context)
        self.doc_store.delete_document.assert_called_once_with(123, "sample.pdf")
        update.message.reply_text.assert_called_once_with("✅ Deleted document: sample.pdf")

    async def test_delete_unique_substring_match(self):
        update = Mock()
        update.effective_user.id = 123
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.args = ["report"]

        self.doc_store.list_documents.return_value = [
            {"filename": "annual_report_2024.pdf"},
            {"filename": "invoice.pdf"},
        ]
        self.doc_store.delete_document.return_value = True

        await self.handlers.delete_command(update, context)
        self.doc_store.delete_document.assert_called_once_with(123, "annual_report_2024.pdf")
        update.message.reply_text.assert_called_once_with(
            "✅ Deleted document: annual_report_2024.pdf"
        )

    async def test_delete_multiple_matches_disambiguation(self):
        update = Mock()
        update.effective_user.id = 123
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.args = ["report"]

        self.doc_store.list_documents.return_value = [
            {"filename": "annual_report_2023.pdf"},
            {"filename": "annual_report_2024.pdf"},
        ]

        await self.handlers.delete_command(update, context)
        msg = update.message.reply_text.call_args[0][0]
        self.assertIn("Multiple documents match 'report':", msg)
        self.assertIn("• annual_report_2023.pdf", msg)
        self.assertIn("• annual_report_2024.pdf", msg)
        self.assertIn("Please specify the exact name or document number from /list.", msg)
        self.doc_store.delete_document.assert_not_called()

    async def test_list_command_empty(self):
        update = Mock()
        update.effective_user.id = 123
        update.message.reply_text = AsyncMock()
        context = Mock()

        self.doc_store.list_documents.return_value = []

        await self.handlers.list_command(update, context)
        update.message.reply_text.assert_called_once_with(
            "📭 You haven't uploaded any documents yet."
        )

    async def test_list_command_formatting(self):
        update = Mock()
        update.effective_user.id = 123
        update.message.reply_text = AsyncMock()
        context = Mock()

        self.doc_store.list_documents.return_value = [
            {
                "filename": "alpha.pdf",
                "file_size": 1024 * 1024,
                "chunk_count": 5,
                "upload_date": "2026-10-21T22:51:00",
            },
            {
                "filename": "beta.md",
                "file_size": 2 * 1024 * 1024,
                "chunk_count": 12,
                "upload_date": "2026-10-22T08:15:00",
            },
        ]

        await self.handlers.list_command(update, context)
        msg = update.message.reply_text.call_args[0][0]
        self.assertIn("1. 📄 alpha.pdf", msg)
        self.assertIn("   Size: 1.00 MB | Chunks: 5", msg)
        self.assertIn("   Uploaded: 21 Oct 2026, 22:51", msg)

        self.assertIn("2. 📄 beta.md", msg)
        self.assertIn("   Size: 2.00 MB | Chunks: 12", msg)
        self.assertIn("   Uploaded: 22 Oct 2026, 08:15", msg)

    async def test_clear_and_list_lifecycle(self):
        """Test uploading two documents, calling /clear, verifying /list is empty, and /clear on empty list returns no-documents message."""
        import tempfile

        from src.storage.document_store import DocumentStore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            emb_path = os.path.join(temp_dir, "embeddings")
            real_doc_store = DocumentStore(db_path=db_path, embeddings_path=emb_path)

            mock_embedder = Mock()
            mock_embedder.generate_embeddings.side_effect = lambda texts: (
                [[0.1] * 10] * len(texts) if isinstance(texts, list) else [0.1] * 10
            )

            handlers = BotHandlers(
                pdf_processor=Mock(),
                text_chunker=Mock(),
                document_store=real_doc_store,
                embedding_generator=mock_embedder,
                retriever=Mock(),
                llm_client=Mock(),
                config=Config(telegram_bot_token="test"),
            )

            user_id = 777

            # Upload two documents
            real_doc_store.store_document(
                user_id=user_id,
                filename="first.pdf",
                file_content=b"pdf content 1",
                chunks=[{"chunk_index": 0, "content": "text 1", "char_count": 6}],
                embeddings=[[0.1] * 10],
                source_type="pdf",
            )
            real_doc_store.store_document(
                user_id=user_id,
                filename="second.pdf",
                file_content=b"pdf content 2",
                chunks=[{"chunk_index": 0, "content": "text 2", "char_count": 6}],
                embeddings=[[0.2] * 10],
                source_type="pdf",
            )

            # Call /clear
            update = Mock()
            update.effective_user.id = user_id
            update.message.reply_text = AsyncMock()
            context = Mock()

            await handlers.clear_command(update, context)
            update.message.reply_text.assert_called_once_with("✅ Deleted 2 document(s).")

            # Assert /list is empty
            update.message.reply_text.reset_mock()
            await handlers.list_command(update, context)
            update.message.reply_text.assert_called_once_with(
                "📭 You haven't uploaded any documents yet."
            )

            # Call /clear on an empty list
            update.message.reply_text.reset_mock()
            await handlers.clear_command(update, context)
            update.message.reply_text.assert_called_once_with(
                "📭 You don't have any documents to delete."
            )

    async def test_delete_not_found(self):
        update = Mock()
        update.effective_user.id = 123
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.args = ["nonexistent"]

        self.doc_store.list_documents.return_value = [
            {"filename": "annual_report_2024.pdf"},
        ]

        await self.handlers.delete_command(update, context)
        update.message.reply_text.assert_called_once_with(
            "Document not found. Send /list to see names and numbers."
        )
        self.doc_store.delete_document.assert_not_called()


class TestTelegramFormatting(unittest.IsolatedAsyncioTestCase):
    """Tests for Telegram HTML code formatting and plain-text fallback."""

    def setUp(self):
        self.handlers = BotHandlers(
            pdf_processor=Mock(),
            text_chunker=Mock(),
            document_store=Mock(),
            embedding_generator=Mock(),
            retriever=Mock(),
            llm_client=Mock(),
            config=Config(telegram_bot_token="test"),
        )

    def test_format_telegram_html_plain_and_sources(self):
        raw = "Condition: x < 10 & y > 5\n\nSources:\n• <data.pdf> (p. 1)"
        formatted = format_telegram_html(raw)
        self.assertIn("x &lt; 10 &amp; y &gt; 5", formatted)
        self.assertIn("&lt;data.pdf&gt;", formatted)

    def test_format_telegram_html_fenced_code_blocks(self):
        raw = (
            "Here is a snippet:\n"
            "```python\n"
            "def check(a, b):\n"
            "    return a < b & True\n"
            "```\n"
            "And another:\n"
            "```\n"
            "echo <hello>\n"
            "```\n\n"
            "Sources:\n• doc.txt"
        )
        formatted = format_telegram_html(raw)
        self.assertIn(
            "<pre><code>def check(a, b):\n    return a &lt; b &amp; True</code></pre>", formatted
        )
        self.assertIn("<pre><code>echo &lt;hello&gt;</code></pre>", formatted)
        self.assertIn("Sources:\n• doc.txt", formatted)

    async def test_deliver_answer_sends_html(self):
        update = Mock()
        update.message.reply_text = AsyncMock()
        thinking_msg = Mock()
        thinking_msg.delete = AsyncMock()

        raw_answer = "Hello ```python\nprint(1 < 2)\n```\n\nSources:\n• <doc.pdf>"
        await self.handlers._deliver_answer(update, thinking_msg, raw_answer)

        update.message.reply_text.assert_called_once()
        call_args, call_kwargs = update.message.reply_text.call_args
        self.assertEqual(call_kwargs.get("parse_mode"), "HTML")
        self.assertIn("<pre><code>print(1 &lt; 2)</code></pre>", call_args[0])
        self.assertIn("&lt;doc.pdf&gt;", call_args[0])
        thinking_msg.delete.assert_called_once()

    async def test_deliver_answer_fallback_on_telegram_reject(self):
        update = Mock()
        # First call with HTML fails with BadRequest, second plain text call succeeds
        update.message.reply_text = AsyncMock(
            side_effect=[
                BadRequest("Can't parse entities: can't find end tag"),
                Mock(),
            ]
        )
        thinking_msg = Mock()
        thinking_msg.delete = AsyncMock()

        raw_answer = "Hello ```python\nprint(1 < 2)\n```\n\nSources:\n• <doc.pdf>"
        await self.handlers._deliver_answer(update, thinking_msg, raw_answer)

        self.assertEqual(update.message.reply_text.call_count, 2)
        # First attempt: HTML
        first_call = update.message.reply_text.call_args_list[0]
        self.assertEqual(first_call[1].get("parse_mode"), "HTML")
        # Fallback attempt: plain text (no parse_mode) with raw text
        second_call = update.message.reply_text.call_args_list[1]
        self.assertNotIn("parse_mode", second_call[1])
        self.assertEqual(second_call[0][0], raw_answer)
        thinking_msg.delete.assert_called_once()

    def test_split_telegram_message_lengths(self):
        # 4096-character answer
        ans_4096 = ("This is a paragraph of text explaining details.\n\n") * 100
        self.assertGreaterEqual(len(ans_4096), 4096)
        chunks_4096 = split_telegram_message(ans_4096, limit=4000)
        self.assertGreaterEqual(len(chunks_4096), 2)
        for c in chunks_4096:
            self.assertLessEqual(utf16_len(c), 4000)

        # 5000-character answer
        ans_5000 = ("Key section in the document describes distributed consensus.\n\n") * 85
        self.assertGreaterEqual(len(ans_5000), 5000)
        chunks_5000 = split_telegram_message(ans_5000, limit=4000)
        self.assertGreaterEqual(len(chunks_5000), 2)
        for c in chunks_5000:
            self.assertLessEqual(utf16_len(c), 4000)

        # 9000-character answer
        ans_9000 = ("Sentence explaining system architecture in complete detail. ") * 160
        self.assertGreaterEqual(len(ans_9000), 9000)
        chunks_9000 = split_telegram_message(ans_9000, limit=4000)
        self.assertGreaterEqual(len(chunks_9000), 3)
        for c in chunks_9000:
            self.assertLessEqual(utf16_len(c), 4000)

    def test_split_telegram_message_code_blocks(self):
        # Answer with code block spanning split point
        code_lines = "\n".join(f"    item_{i} = calculate_value({i})" for i in range(250))
        code_body = f"<pre><code>def process_items():\n{code_lines}\n    return True</code></pre>"
        self.assertGreater(len(code_body), 6000)

        # When splitting code directly, chunk 0 must close with </code></pre> and chunk 1 re-open with <pre><code>
        code_chunks = split_telegram_message(code_body, limit=4000)
        self.assertGreaterEqual(len(code_chunks), 2)
        self.assertTrue(code_chunks[0].endswith("</code></pre>"))
        self.assertTrue(code_chunks[1].startswith("<pre><code>"))

        # When surrounded with headers and footers
        raw = (
            f"Header explanation before code block.\n\n{code_body}\n\nTrailing remarks after code."
        )
        chunks = split_telegram_message(raw, limit=4000)
        self.assertGreaterEqual(len(chunks), 3)
        for i, chunk in enumerate(chunks):
            self.assertLessEqual(utf16_len(chunk), 4000)
            self.assertEqual(
                chunk.count("<pre><code>"),
                chunk.count("</code></pre>"),
                f"Chunk {i} tags unbalanced: {chunk[:30]}...{chunk[-30:]}",
            )

    def test_split_telegram_message_emoji_near_limit(self):
        # 3999 ASCII characters + 2-unit emoji + trailing text
        text = ("a" * 3999) + "😀" + ("b" * 100)
        chunks = split_telegram_message(text, limit=4000)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(utf16_len(chunks[0]), 3999)
        self.assertEqual(chunks[0], "a" * 3999)
        self.assertTrue(chunks[1].startswith("😀"))
        self.assertLessEqual(utf16_len(chunks[1]), 4000)

    async def test_deliver_answer_multichunk_ordering_and_footer(self):
        update = Mock()
        update.message.reply_text = AsyncMock()
        thinking_msg = Mock()
        thinking_msg.delete = AsyncMock()

        answer_part = ("Detailed explanation of distributed systems.\n\n") * 100
        sources_part = "Sources:\n• consensus.pdf (p. 1-5)"
        timing_part = "Answered in 3 s with test-model"
        full_text = f"{answer_part}\n{sources_part}\n\n{timing_part}"

        await self.handlers._deliver_answer(update, thinking_msg, full_text)

        # Delivered in order across multiple messages
        self.assertGreaterEqual(update.message.reply_text.call_count, 2)
        calls = update.message.reply_text.call_args_list

        # First chunk does NOT have Sources footer or timing line
        first_chunk_text = calls[0][0][0]
        self.assertNotIn("Sources:", first_chunk_text)
        self.assertNotIn("Answered in", first_chunk_text)

        # Last chunk has Sources footer and timing line
        last_chunk_text = calls[-1][0][0]
        self.assertIn("Sources:", last_chunk_text)
        self.assertIn("Answered in 3 s with test-model", last_chunk_text)

        thinking_msg.delete.assert_called_once()

    async def test_deliver_answer_multichunk_mocked_bad_request_on_one_chunk(self):
        update = Mock()
        # 2 chunks expected:
        # Chunk 1: HTML fails with BadRequest, plain text fallback succeeds
        # Chunk 2: HTML succeeds
        update.message.reply_text = AsyncMock(
            side_effect=[
                BadRequest("Can't parse entities: can't find end tag"),
                Mock(),  # Plain text fallback for chunk 1 succeeds
                Mock(),  # Chunk 2 HTML succeeds
            ]
        )
        thinking_msg = Mock()
        thinking_msg.delete = AsyncMock()

        answer_text = ("Part of long response.\n\n") * 200
        await self.handlers._deliver_answer(update, thinking_msg, answer_text)

        self.assertEqual(update.message.reply_text.call_count, 3)
        # Call 1: HTML for chunk 1
        self.assertEqual(update.message.reply_text.call_args_list[0][1].get("parse_mode"), "HTML")
        # Call 2: plain text for chunk 1
        self.assertNotIn("parse_mode", update.message.reply_text.call_args_list[1][1])
        # Call 3: HTML for chunk 2
        self.assertEqual(update.message.reply_text.call_args_list[2][1].get("parse_mode"), "HTML")
        thinking_msg.delete.assert_called_once()

    async def test_deliver_answer_failure_replies_with_plain_text_error(self):
        update = Mock()
        # When _reply_with_retry returns None (both HTML and fallback fail)
        update.message.reply_text = AsyncMock(
            side_effect=[
                BadRequest("Failed HTML"),
                BadRequest("Failed fallback"),
                Mock(),  # Error notice succeeds
            ]
        )
        thinking_msg = Mock()
        thinking_msg.delete = AsyncMock()

        await self.handlers._deliver_answer(update, thinking_msg, "Short test answer")

        # 3 calls: HTML attempt -> plain text fallback attempt -> error notice
        self.assertEqual(update.message.reply_text.call_count, 3)
        error_notice_call = update.message.reply_text.call_args_list[2]
        self.assertIn("❌ Failed to send response", error_notice_call[0][0])
        thinking_msg.delete.assert_called_once()


class TestIngestionProgressAndLimits(unittest.IsolatedAsyncioTestCase):
    """Tests for chunk limits and progress notifications during ingestion."""

    def setUp(self):
        self.doc_store = Mock()
        self.embedding_gen = Mock()
        self.embedding_gen.generate_embeddings.side_effect = lambda texts: (
            [[0.1] * 10] * len(texts) if isinstance(texts, list) else [0.1] * 10
        )
        self.config = Config(
            telegram_bot_token="test",
            max_chunks_per_document=100,
        )
        self.handlers = BotHandlers(
            pdf_processor=Mock(),
            text_chunker=Mock(),
            document_store=self.doc_store,
            embedding_generator=self.embedding_gen,
            retriever=Mock(),
            llm_client=Mock(),
            config=self.config,
        )

    def test_config_max_chunks_default(self):
        default_cfg = Config(telegram_bot_token="test")
        self.assertEqual(default_cfg.max_chunks_per_document, 5000)

    async def test_embed_chunks_rejects_exceeding_max(self):
        update = Mock()
        update.message.reply_text = AsyncMock()

        chunks = [f"chunk {i}" for i in range(120)]
        result = await self.handlers._embed_chunks_with_progress(update, chunks)

        self.assertIsNone(result)
        self.embedding_gen.generate_embeddings.assert_not_called()
        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.call_args[0][0]
        self.assertIn("exceeds the maximum limit of 100 chunks", msg)
        self.assertIn("120 chunks created", msg)

    async def test_embed_chunks_small_file_notifies_user(self):
        update = Mock()
        update.message.reply_text = AsyncMock()

        chunks = ["chunk 1", "chunk 2", "chunk 3"]
        result = await self.handlers._embed_chunks_with_progress(update, chunks)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.call_args[0][0]
        self.assertIn("Created 3 chunks", msg)
        self.assertIn("take several minutes", msg)

    async def test_embed_chunks_large_file_progress_updates(self):
        # Configure higher limit so 600 chunks is accepted (> 500 chunks triggers progress)
        self.handlers.config.max_chunks_per_document = 1000
        status_msg = Mock()
        status_msg.edit_text = AsyncMock()
        update = Mock()
        update.message.reply_text = AsyncMock(return_value=status_msg)

        chunks = [f"chunk {i}" for i in range(600)]
        result = await self.handlers._embed_chunks_with_progress(update, chunks)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 600)
        # Verify status message was edited multiple times with progress percentages
        self.assertGreater(status_msg.edit_text.call_count, 5)
        edit_texts = [call.args[0] for call in status_msg.edit_text.call_args_list]
        self.assertTrue(any("10%" in t for t in edit_texts))
        self.assertTrue(any("50%" in t for t in edit_texts))
        self.assertTrue(any("100%" in t for t in edit_texts))


if __name__ == "__main__":
    unittest.main()
