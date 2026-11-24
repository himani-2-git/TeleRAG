"""Tests for how the running bot behaves: concurrency, errors, status, logging and shutdown."""

import asyncio
import logging
import os
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, Mock, patch

from telegram.error import TelegramError

from src.bot.application import build_application
from src.bot.handlers import GENERIC_ERROR_MESSAGE, BotHandlers
from src.config import Config
from src.logging_setup import JsonFormatter, Redactor, RequestIdFilter, request_id_var
from src.processors.text_chunker import TextChunker
from src.rag.retriever import DocumentRetriever
from src.storage.document_store import DocumentStore
from tests.bot_helpers import make_context, make_handlers, make_update, reply_texts


class ConstantEmbedder:
    """Every text gets the same vector, so retrieval always finds the user's own chunks."""

    def generate_embeddings(self, texts):
        if isinstance(texts, str):
            return [1.0, 0.0]
        return [[1.0, 0.0] for _ in texts]

    def generate_query_embedding(self, query):
        return [1.0, 0.0]


class ContextEchoLLM:
    """Fake LLM. It answers with the last line of the context it was given.

    Optionally it blocks inside generate_answer until release is set, like a slow model.
    """

    def __init__(self, started=None, release=None):
        self.started = started
        self.release = release
        self.timed_out = False

    def generate_answer(self, question, context):
        if self.release is not None:
            self.started.set()
            if not self.release.wait(timeout=5):
                self.timed_out = True
        return context.strip().splitlines()[-1]

    def validate_response(self, text):
        return text.strip() if text else ""

    def check_server(self):
        return True, "fake llm"


class RealComponentsCase(unittest.IsolatedAsyncioTestCase):
    """BotHandlers wired to a real DocumentStore, retriever and pipeline in a temp folder."""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._temp.cleanup)
        self.store = DocumentStore(
            os.path.join(self._temp.name, "docs.db"), os.path.join(self._temp.name, "emb")
        )
        self.retriever = DocumentRetriever(top_k=2, similarity_threshold=0.3)
        self.store.add_change_listener(self.retriever.invalidate_cache)
        self.llm = ContextEchoLLM()
        self.handlers = self.make_handlers()

    def make_handlers(self, llm=None, **config_values):
        config_values.setdefault("question_rate_limit", 0)
        config_values.setdefault("upload_rate_limit", 0)
        return BotHandlers(
            pdf_processor=Mock(),
            text_chunker=TextChunker(chunk_size=500, overlap=50),
            document_store=self.store,
            embedding_generator=ConstantEmbedder(),
            retriever=self.retriever,
            llm_client=llm or self.llm,
            config=Config(telegram_bot_token="test", **config_values),
        )

    def add_document(self, user_id, filename, text):
        chunks = [{"chunk_index": 0, "content": text, "char_count": len(text)}]
        self.store.store_document(
            user_id, filename, text.encode(), chunks, [[1.0, 0.0]], source_type="text"
        )


class TestTwoUsersAtTheSameTime(RealComponentsCase):
    async def test_each_user_gets_an_answer_from_their_own_document_only(self):
        self.add_document(1, "one.txt", "ALPHA note that belongs to the first user.")
        self.add_document(2, "two.txt", "BRAVO note that belongs to the second user.")
        context = make_context()

        for _ in range(4):
            first = make_update(user_id=1, text="What does my note say?")
            second = make_update(user_id=2, text="What does my note say?")
            await asyncio.gather(
                self.handlers.handle_message(first, context),
                self.handlers.handle_message(second, context),
            )

            first_reply = " ".join(reply_texts(first))
            second_reply = " ".join(reply_texts(second))
            self.assertIn("ALPHA", first_reply)
            self.assertNotIn("BRAVO", first_reply)
            self.assertIn("BRAVO", second_reply)
            self.assertNotIn("ALPHA", second_reply)

    async def test_other_users_are_served_while_a_slow_answer_is_running(self):
        started, release = threading.Event(), threading.Event()
        slow_llm = ContextEchoLLM(started, release)
        handlers = self.make_handlers(llm=slow_llm)
        self.add_document(1, "one.txt", "ALPHA note that belongs to the first user.")
        self.add_document(2, "two.txt", "BRAVO note that belongs to the second user.")
        context = make_context()

        slow_question = asyncio.create_task(
            handlers.handle_message(make_update(user_id=1, text="What does it say?"), context)
        )
        for _ in range(500):  # wait without blocking the event loop
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        self.assertTrue(started.is_set(), "the slow answer never started")

        listing = make_update(user_id=2)
        await asyncio.wait_for(handlers.list_command(listing, context), timeout=2)
        self.assertTrue(any("two.txt" in text for text in reply_texts(listing)))

        release.set()
        await asyncio.wait_for(slow_question, timeout=5)
        self.assertFalse(slow_llm.timed_out, "the event loop was blocked by the slow answer")

    def test_the_application_is_built_to_process_updates_at_the_same_time(self):
        config = Config(telegram_bot_token="123456:TEST", concurrent_updates=12)
        application = build_application(config, self.handlers)

        self.assertEqual(application.update_processor.max_concurrent_updates, 12)
        self.assertIsNotNone(application.post_shutdown)


class TestBlockingWorkStaysOffTheEventLoop(unittest.IsolatedAsyncioTestCase):
    async def test_url_validation_runs_in_a_worker_thread(self):
        """DNS lookups can take seconds and must not freeze the bot for everyone else."""
        handlers = make_handlers()
        loop_thread = threading.get_ident()
        threads_used = []

        def fake_validate(url):
            threads_used.append(threading.get_ident())
            return False, "blocked for test"

        context = make_context()
        context.args = ["https://example.com/page"]
        with patch("src.processors.web_processor.validate_url_ssrf", fake_validate):
            await handlers.addurl_command(make_update(), context)
            await handlers.handle_message(make_update(text="https://example.com/a.pdf"), context)

        self.assertEqual(len(threads_used), 2)
        self.assertNotIn(loop_thread, threads_used)


class TestFriendlyErrors(unittest.IsolatedAsyncioTestCase):
    async def test_unexpected_errors_get_a_friendly_reply_without_details(self):
        handlers = make_handlers()
        update = make_update()
        context = make_context()
        context.error = RuntimeError("secret detail: /home/bot/data/documents.db is corrupt")

        with self.assertLogs("src.bot.handlers", level="ERROR") as logs:
            await handlers.error_handler(update, context)

        reply = update.effective_message.reply_text.call_args.args[0]
        self.assertEqual(reply, GENERIC_ERROR_MESSAGE)
        self.assertNotIn("secret detail", reply)
        self.assertTrue(any("RuntimeError" in line for line in logs.output))

    async def test_errors_without_a_message_are_only_logged(self):
        handlers = make_handlers()
        context = make_context()
        context.error = TelegramError("network problem while polling")

        with self.assertLogs("src.bot.handlers", level="ERROR"):
            await handlers.error_handler(None, context)

    async def test_error_handler_survives_a_failing_reply(self):
        handlers = make_handlers()
        update = make_update()
        update.effective_message.reply_text.side_effect = TelegramError("bot was blocked")
        context = make_context()
        context.error = RuntimeError("boom")

        with self.assertLogs("src.bot.handlers", level="WARNING"):
            await handlers.error_handler(update, context)

    async def test_embedding_errors_never_show_model_internals_to_the_user(self):
        handlers = make_handlers()
        handlers.text_chunker = TextChunker(chunk_size=500, overlap=50)
        handlers.embedding_generator.generate_embeddings.side_effect = ValueError(
            "tensor shape mismatch at 0x7f3a (secret internals)"
        )
        update = make_update()
        update.message.document.file_name = "notes.txt"
        update.message.document.file_size = 100

        def write_file(path):
            with open(path, "wb") as f:
                f.write(b"Some ordinary text for the document.")

        file_mock = Mock()
        file_mock.download_to_drive = AsyncMock(side_effect=write_file)
        update.message.document.get_file = AsyncMock(return_value=file_mock)
        handlers.document_store.list_documents.return_value = []

        with self.assertLogs("src.bot.handlers.documents", level="ERROR"):
            await handlers.handle_document(update, make_context())

        replies = " ".join(reply_texts(update))
        self.assertNotIn("tensor", replies)
        self.assertNotIn("secret internals", replies)
        self.assertIn("could not process", replies.lower())


class TestShutdown(unittest.IsolatedAsyncioTestCase):
    async def test_questions_still_waiting_when_the_bot_stops_are_not_started(self):
        handlers = make_handlers()
        handlers._run_pipeline = Mock(return_value={"final_response": "answer", "refused": False})
        context = make_context()
        context.application.running = False
        update = make_update(text="What is the fee?")

        await handlers.handle_message(update, context)

        handlers._run_pipeline.assert_not_called()
        self.assertTrue(any("restart" in text.lower() for text in reply_texts(update)))

    async def test_questions_run_normally_while_the_bot_is_running(self):
        handlers = make_handlers()
        handlers._run_pipeline = Mock(return_value={"final_response": "answer", "refused": False})
        context = make_context()
        context.application.running = True

        await handlers.handle_message(make_update(text="What is the fee?"), context)

        handlers._run_pipeline.assert_called_once()


class TestStatusCommand(unittest.IsolatedAsyncioTestCase):
    def healthy_handlers(self, **config_values):
        handlers = make_handlers(**config_values)
        handlers.document_store.db.ping.return_value = True
        handlers.embedding_generator.generate_query_embedding.return_value = [0.1, 0.2]
        handlers.llm_client.check_server.return_value = (True, "model 'qwen2.5:3b' is available")
        return handlers

    async def test_admin_sees_the_state_of_every_component(self):
        handlers = self.healthy_handlers(admin_user_ids=frozenset({9}))
        update = make_update(user_id=9)

        await handlers.status_command(update, make_context())

        report = reply_texts(update)[-1]
        for name in ("Database", "Embeddings", "LLM"):
            self.assertIn(name, report)
        self.assertEqual(report.count("✅"), 3)
        self.assertNotIn("❌", report)

    async def test_other_users_are_refused_and_nothing_is_checked(self):
        handlers = self.healthy_handlers(admin_user_ids=frozenset({9}))
        update = make_update(user_id=1)

        await handlers.status_command(update, make_context())

        self.assertIn("admin", reply_texts(update)[-1].lower())
        handlers.llm_client.check_server.assert_not_called()
        handlers.document_store.db.ping.assert_not_called()

    async def test_allowed_users_are_admins_when_no_admin_list_is_set(self):
        handlers = self.healthy_handlers(allowed_user_ids=frozenset({5}))
        update = make_update(user_id=5)

        await handlers.status_command(update, make_context())

        self.assertIn("Database", reply_texts(update)[-1])

    async def test_command_is_off_when_nobody_is_configured_as_admin(self):
        handlers = self.healthy_handlers()
        update = make_update(user_id=1)

        await handlers.status_command(update, make_context())

        self.assertIn("ADMIN_USER_IDS", reply_texts(update)[-1])
        handlers.llm_client.check_server.assert_not_called()

    async def test_a_failing_component_is_marked_without_leaking_the_error_text(self):
        handlers = self.healthy_handlers(admin_user_ids=frozenset({9}))
        handlers.document_store.db.ping.side_effect = sqlite3.OperationalError(
            "unable to open /secret/path/documents.db"
        )
        handlers.llm_client.check_server.return_value = (False, "server unreachable")
        update = make_update(user_id=9)

        await handlers.status_command(update, make_context())

        report = reply_texts(update)[-1]
        self.assertEqual(report.count("❌"), 2)
        self.assertEqual(report.count("✅"), 1)
        self.assertIn("OperationalError", report)
        self.assertNotIn("/secret/path", report)

    async def test_a_check_that_raises_unexpectedly_is_reported_as_failed(self):
        handlers = self.healthy_handlers(admin_user_ids=frozenset({9}))
        handlers.embedding_generator.generate_query_embedding.side_effect = RuntimeError("oom")
        update = make_update(user_id=9)

        await handlers.status_command(update, make_context())

        self.assertIn("❌ Embeddings", reply_texts(update)[-1])

    def test_real_database_answers_the_ping(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            store = DocumentStore(os.path.join(folder, "d.db"), os.path.join(folder, "emb"))

            self.assertTrue(store.db.ping())


class ListHandler(logging.Handler):
    """Collects log records after the request id has been added to them."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []
        self.addFilter(RequestIdFilter())

    def emit(self, record):
        self.records.append(record)


class TestLoggingDuringRequests(RealComponentsCase):
    def setUp(self):
        super().setUp()
        self.collector = ListHandler()
        root = logging.getLogger()
        self._old_level = root.level
        root.setLevel(logging.DEBUG)
        root.addHandler(self.collector)
        self.addCleanup(self.restore_logging)

    def restore_logging(self):
        root = logging.getLogger()
        root.removeHandler(self.collector)
        root.setLevel(self._old_level)

    def log_lines(self):
        formatter = JsonFormatter(Redactor())
        return [formatter.format(record) for record in self.collector.records]

    async def test_every_question_gets_its_own_request_id_in_all_its_log_lines(self):
        self.add_document(1, "one.txt", "ALPHA note that belongs to the first user.")
        seen_in_worker = []
        original = self.handlers._run_pipeline

        def spy(user_id, question):
            seen_in_worker.append(request_id_var.get())
            return original(user_id, question)

        self.handlers._run_pipeline = spy
        context = make_context()

        await self.handlers.handle_message(make_update(user_id=1, text="First question?"), context)
        await self.handlers.handle_message(make_update(user_id=1, text="Second question?"), context)

        question_logs = [
            r
            for r in self.collector.records
            if r.name == "src.bot.handlers.questions"
            and r.getMessage() in ("Question received", "Question answered")
        ]
        self.assertEqual(len(question_logs), 4)
        ids = [record.request_id for record in question_logs]
        self.assertNotIn("-", ids)
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[2], ids[3])
        self.assertNotEqual(ids[0], ids[2])
        self.assertEqual(seen_in_worker, [ids[0], ids[2]], "worker thread lost the request id")
        answered = question_logs[1]
        self.assertEqual(answered.user_id, 1)
        self.assertIn("elapsed_ms", answered.__dict__)

    async def test_document_text_questions_and_answers_never_appear_in_logs(self):
        document_secret = "SENTINELDOC_9F3A"
        question_secret = "SENTINELQUESTION_7C1B"
        context = make_context()

        # Upload through the real text path, then ask a question and get an echoed answer.
        update = make_update(user_id=1)
        update.message.document.file_name = "private.txt"
        update.message.document.file_size = 100

        def write_file(path):
            with open(path, "wb") as f:
                f.write(f"Confidential memo {document_secret} for the board.".encode())

        file_mock = Mock()
        file_mock.download_to_drive = AsyncMock(side_effect=write_file)
        update.message.document.get_file = AsyncMock(return_value=file_mock)
        await self.handlers.handle_document(update, context)

        question = make_update(user_id=1, text=f"What does {question_secret} mean in the memo?")
        await self.handlers.handle_message(question, context)

        self.assertIn(document_secret, " ".join(reply_texts(question)), "the answer should work")
        self.assertTrue(self.collector.records, "nothing was logged, so this test proves nothing")
        for line in self.log_lines():
            self.assertNotIn(document_secret, line)
            self.assertNotIn(question_secret, line)

    async def test_url_secrets_are_not_logged_when_an_import_fails(self):
        url = "https://example.com/page?token=URLSECRET_55"
        context = make_context()
        context.args = [url]

        with (
            patch("src.processors.web_processor.validate_url_ssrf", return_value=(True, "")),
            patch(
                "src.processors.web_processor.WebProcessor.process_url",
                side_effect=OSError("connection reset"),
            ),
        ):
            await self.handlers.addurl_command(make_update(), context)

        with (
            patch.object(self.handlers, "_is_safe_url", return_value=(True, "")),
            patch("src.bot.handlers.documents.fetch_pdf_bytes", side_effect=RuntimeError("x")),
        ):
            await self.handlers.handle_message(
                make_update(text="https://example.com/a.pdf?sig=URLSECRET_55"), context
            )

        failures = [line for line in self.log_lines() if "Failed to" in line]
        self.assertEqual(len(failures), 2)
        for line in self.log_lines():
            self.assertNotIn("URLSECRET_55", line)


if __name__ == "__main__":
    unittest.main()
