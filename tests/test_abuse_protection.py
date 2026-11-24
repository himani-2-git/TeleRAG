"""Tests for abuse protection: private-bot allowlist, rate limits and document limit."""

import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from telegram import Bot, Update
from telegram.error import TelegramError
from telegram.ext import Application, ApplicationHandlerStop

from src.bot.rate_limiter import SlidingWindowRateLimiter
from src.config import get_config
from tests.bot_helpers import make_context, make_handlers, make_update, reply_texts


class FakeClock:
    """Controllable clock so rate limit tests never have to sleep."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TestSlidingWindowRateLimiter(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.limiter = SlidingWindowRateLimiter(max_events=3, window_seconds=60, clock=self.clock)

    def test_allows_up_to_the_limit_then_blocks(self):
        for _ in range(3):
            self.assertEqual(self.limiter.try_acquire("user"), 0.0)

        self.assertGreater(self.limiter.try_acquire("user"), 0.0)

    def test_reports_how_long_until_the_oldest_event_expires(self):
        self.limiter.try_acquire("user")  # t = 0
        self.clock.advance(10)
        self.limiter.try_acquire("user")  # t = 10
        self.clock.advance(10)
        self.limiter.try_acquire("user")  # t = 20
        self.clock.advance(10)  # t = 30

        self.assertAlmostEqual(self.limiter.try_acquire("user"), 30.0)

    def test_allows_again_after_the_window_has_passed(self):
        for _ in range(3):
            self.limiter.try_acquire("user")
        self.clock.advance(61)

        self.assertEqual(self.limiter.try_acquire("user"), 0.0)

    def test_blocked_attempts_do_not_extend_the_block(self):
        for _ in range(3):
            self.limiter.try_acquire("user")
        self.clock.advance(30)
        for _ in range(20):
            self.assertGreater(self.limiter.try_acquire("user"), 0.0)
        self.clock.advance(31)

        self.assertEqual(self.limiter.try_acquire("user"), 0.0)

    def test_users_are_limited_independently(self):
        for _ in range(3):
            self.limiter.try_acquire("alice")

        self.assertGreater(self.limiter.try_acquire("alice"), 0.0)
        self.assertEqual(self.limiter.try_acquire("bob"), 0.0)

    def test_zero_disables_the_limit(self):
        limiter = SlidingWindowRateLimiter(max_events=0, window_seconds=60, clock=self.clock)

        for _ in range(1000):
            self.assertEqual(limiter.try_acquire("user"), 0.0)

    def test_idle_users_are_forgotten_so_memory_stays_bounded(self):
        limiter = SlidingWindowRateLimiter(
            max_events=1, window_seconds=10, clock=self.clock, max_tracked_keys=3
        )
        for user in ("a", "b", "c"):
            limiter.try_acquire(user)
        self.clock.advance(11)

        limiter.try_acquire("d")

        self.assertEqual(len(limiter._events), 1)


class TestAbuseProtectionConfig(unittest.TestCase):
    KEYS = (
        "ALLOWED_USER_IDS",
        "QUESTION_RATE_LIMIT",
        "UPLOAD_RATE_LIMIT",
        "RATE_LIMIT_WINDOW_SECONDS",
        "MAX_DOCUMENTS_PER_USER",
    )

    def load_config(self, **env):
        with patch.dict(os.environ, env, clear=False):
            for key in self.KEYS:
                if key not in env:
                    os.environ.pop(key, None)
            return get_config(require_telegram=False)

    def test_defaults(self):
        config = self.load_config()

        self.assertEqual(config.allowed_user_ids, frozenset())
        self.assertEqual(config.question_rate_limit, 5)
        self.assertEqual(config.upload_rate_limit, 3)
        self.assertEqual(config.rate_limit_window_seconds, 60)
        self.assertEqual(config.max_documents_per_user, 20)

    def test_allowed_user_ids_are_parsed_from_a_comma_separated_list(self):
        config = self.load_config(ALLOWED_USER_IDS=" 123, 456 ,")

        self.assertEqual(config.allowed_user_ids, frozenset({123, 456}))

    def test_blank_allowed_user_ids_means_everyone_is_allowed(self):
        config = self.load_config(ALLOWED_USER_IDS="   ")

        self.assertEqual(config.allowed_user_ids, frozenset())

    def test_non_numeric_user_id_fails_with_a_clear_message(self):
        with self.assertRaises(ValueError) as caught:
            self.load_config(ALLOWED_USER_IDS="123,abc")

        self.assertIn("ALLOWED_USER_IDS", str(caught.exception))
        self.assertIn("abc", str(caught.exception))

    def test_limits_can_be_changed_from_the_environment(self):
        config = self.load_config(
            QUESTION_RATE_LIMIT="9",
            UPLOAD_RATE_LIMIT="2",
            RATE_LIMIT_WINDOW_SECONDS="30",
            MAX_DOCUMENTS_PER_USER="7",
        )

        self.assertEqual(config.question_rate_limit, 9)
        self.assertEqual(config.upload_rate_limit, 2)
        self.assertEqual(config.rate_limit_window_seconds, 30)
        self.assertEqual(config.max_documents_per_user, 7)


class TestAccessGate(unittest.IsolatedAsyncioTestCase):
    async def test_everyone_passes_when_no_allowlist_is_set(self):
        handlers = make_handlers()
        update = make_update(user_id=555)

        await handlers.access_gate(update, make_context())

        update.effective_message.reply_text.assert_not_called()

    async def test_allowed_user_passes(self):
        handlers = make_handlers(allowed_user_ids=frozenset({1, 2}))
        update = make_update(user_id=2)

        await handlers.access_gate(update, make_context())

        update.effective_message.reply_text.assert_not_called()

    async def test_other_users_get_a_polite_private_message_and_are_stopped(self):
        handlers = make_handlers(allowed_user_ids=frozenset({1}))
        update = make_update(user_id=555)

        with self.assertRaises(ApplicationHandlerStop):
            await handlers.access_gate(update, make_context())

        update.effective_message.reply_text.assert_awaited_once()
        self.assertIn("private", update.effective_message.reply_text.call_args.args[0])

    async def test_user_is_still_stopped_when_the_private_reply_cannot_be_sent(self):
        handlers = make_handlers(allowed_user_ids=frozenset({1}))
        update = make_update(user_id=555)
        update.effective_message.reply_text.side_effect = TelegramError("network down")

        with self.assertRaises(ApplicationHandlerStop):
            await handlers.access_gate(update, make_context())

    async def test_update_without_a_user_is_stopped_when_bot_is_private(self):
        handlers = make_handlers(allowed_user_ids=frozenset({1}))
        update = make_update()
        update.effective_user = None

        with self.assertRaises(ApplicationHandlerStop):
            await handlers.access_gate(update, make_context())

    def test_gate_is_registered_in_a_group_that_runs_before_all_other_handlers(self):
        handlers = make_handlers(allowed_user_ids=frozenset({1}))
        application = Application.builder().token("123456:TEST").build()

        handlers.register_handlers(application)

        self.assertEqual(min(application.handlers), -1)
        gate_group = application.handlers[-1]
        self.assertEqual(len(gate_group), 1)
        self.assertEqual(gate_group[0].callback, handlers.access_gate)
        other_callbacks = [h.callback for h in application.handlers[0]]
        self.assertNotIn(handlers.access_gate, other_callbacks)
        self.assertIn(handlers.handle_message, other_callbacks)
        self.assertIn(handlers.handle_document, other_callbacks)


class TestAccessGateEndToEnd(unittest.IsolatedAsyncioTestCase):
    """Send real Update objects through a real Application (only network calls are replaced)."""

    @staticmethod
    def message_update(bot, user_id, text, is_command=False):
        message = {
            "message_id": 1,
            "date": 1700000000,
            "chat": {"id": user_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "first_name": "Test"},
            "text": text,
        }
        if is_command:
            message["entities"] = [{"type": "bot_command", "offset": 0, "length": len(text)}]
        return Update.de_json({"update_id": 1, "message": message}, bot)

    async def test_stranger_never_reaches_the_command_handlers(self):
        handlers = make_handlers(allowed_user_ids=frozenset({1}))
        handlers.document_store.list_documents.return_value = []
        application = Application.builder().token("123456:TEST").build()
        handlers.register_handlers(application)
        send_message = AsyncMock()

        with (
            patch.object(Bot, "send_message", send_message),
            patch.object(Bot, "initialize", AsyncMock()),
            patch.object(Bot, "shutdown", AsyncMock()),
        ):
            await application.initialize()
            stranger_command = self.message_update(application.bot, 999, "/list", is_command=True)
            await application.process_update(stranger_command)

            handlers.document_store.list_documents.assert_not_called()
            sent_text = send_message.call_args.kwargs["text"]
            self.assertIn("private", sent_text)

            # A plain message keeps this test independent of the bot username that command
            # matching needs. It reaches the question handler, which looks up the user documents.
            allowed_question = self.message_update(application.bot, 1, "What is the fee?")
            await application.process_update(allowed_question)

            handlers.document_store.list_documents.assert_called_once_with(1)


class TestRateLimitsInHandlers(unittest.IsolatedAsyncioTestCase):
    async def test_questions_over_the_limit_are_refused_without_running_the_pipeline(self):
        handlers = make_handlers(question_rate_limit=2)
        handlers._run_pipeline = Mock(return_value={"final_response": "answer", "refused": False})
        context = make_context()

        for _ in range(2):
            await handlers.handle_message(make_update(user_id=1, text="What is a fee?"), context)
        blocked = make_update(user_id=1, text="What is a fee?")
        await handlers.handle_message(blocked, context)

        self.assertEqual(handlers._run_pipeline.call_count, 2)
        self.assertTrue(any("too quickly" in text for text in reply_texts(blocked)))

    async def test_one_users_questions_do_not_use_up_another_users_limit(self):
        handlers = make_handlers(question_rate_limit=1)
        handlers._run_pipeline = Mock(return_value={"final_response": "answer", "refused": False})
        context = make_context()

        await handlers.handle_message(make_update(user_id=1, text="What is a fee?"), context)
        await handlers.handle_message(make_update(user_id=2, text="What is a fee?"), context)

        self.assertEqual(handlers._run_pipeline.call_count, 2)

    async def test_question_limit_zero_disables_the_limit(self):
        handlers = make_handlers(question_rate_limit=0)
        handlers._run_pipeline = Mock(return_value={"final_response": "answer", "refused": False})
        context = make_context()

        for _ in range(10):
            await handlers.handle_message(make_update(user_id=1, text="What is a fee?"), context)

        self.assertEqual(handlers._run_pipeline.call_count, 10)

    async def test_file_uploads_over_the_limit_are_refused(self):
        handlers = make_handlers(upload_rate_limit=2)
        context = make_context()

        def unsupported_file_update():
            update = make_update(user_id=1)
            update.message.document.file_name = "notes.docx"
            update.message.document.file_size = 10
            return update

        for _ in range(2):
            allowed = unsupported_file_update()
            await handlers.handle_document(allowed, context)
            self.assertTrue(any("Please send a PDF" in t for t in reply_texts(allowed)))
        blocked = unsupported_file_update()
        await handlers.handle_document(blocked, context)

        self.assertTrue(any("too quickly" in text for text in reply_texts(blocked)))

    async def test_pdf_links_count_as_uploads(self):
        handlers = make_handlers(upload_rate_limit=1)
        handlers._is_safe_url = Mock(return_value=(False, "blocked for test"))
        context = make_context()

        await handlers.handle_message(make_update(text="https://example.com/a.pdf"), context)
        blocked = make_update(text="https://example.com/b.pdf")
        await handlers.handle_message(blocked, context)

        self.assertTrue(any("too quickly" in text for text in reply_texts(blocked)))

    @patch(
        "src.processors.web_processor.validate_url_ssrf", return_value=(False, "blocked for test")
    )
    async def test_addurl_counts_as_an_upload(self, _mock_ssrf):
        handlers = make_handlers(upload_rate_limit=1)
        context = make_context()
        context.args = ["https://example.com/page"]

        await handlers.addurl_command(make_update(), context)
        blocked = make_update()
        await handlers.addurl_command(blocked, context)

        self.assertTrue(any("too quickly" in text for text in reply_texts(blocked)))

    async def test_uploads_and_questions_have_separate_limits(self):
        handlers = make_handlers(question_rate_limit=1, upload_rate_limit=1)
        handlers._run_pipeline = Mock(return_value={"final_response": "answer", "refused": False})
        context = make_context()
        upload = make_update()
        upload.message.document.file_name = "notes.docx"
        upload.message.document.file_size = 10

        await handlers.handle_document(upload, context)
        await handlers.handle_message(make_update(text="What is a fee?"), context)

        handlers._run_pipeline.assert_called_once()

    async def test_message_tells_the_user_how_long_to_wait(self):
        handlers = make_handlers(question_rate_limit=1, rate_limit_window_seconds=60)
        handlers._run_pipeline = Mock(return_value={"final_response": "answer", "refused": False})
        context = make_context()

        await handlers.handle_message(make_update(text="What is a fee?"), context)
        blocked = make_update(text="What is a fee?")
        await handlers.handle_message(blocked, context)

        message = next(t for t in reply_texts(blocked) if "too quickly" in t)
        self.assertRegex(message, r"\d+ seconds?")


class TestDocumentLimit(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def text_file_update(user_id=1):
        update = make_update(user_id=user_id)
        update.message.document.file_name = "notes.txt"
        update.message.document.file_size = 100
        update.message.document.get_file = AsyncMock(side_effect=RuntimeError("stop here"))
        return update

    async def test_upload_is_refused_when_the_user_is_at_the_limit(self):
        handlers = make_handlers(max_documents_per_user=2)
        handlers.document_store.list_documents.return_value = [{"id": 1}, {"id": 2}]
        update = self.text_file_update()

        await handlers.handle_document(update, make_context())

        update.message.document.get_file.assert_not_awaited()
        self.assertTrue(any("limit of 2 documents" in t for t in reply_texts(update)))
        handlers.document_store.store_document.assert_not_called()

    async def test_upload_is_allowed_below_the_limit(self):
        handlers = make_handlers(max_documents_per_user=2)
        handlers.document_store.list_documents.return_value = [{"id": 1}]
        update = self.text_file_update()

        await handlers.handle_document(update, make_context())

        update.message.document.get_file.assert_awaited_once()

    async def test_limit_zero_means_unlimited(self):
        handlers = make_handlers(max_documents_per_user=0)
        handlers.document_store.list_documents.return_value = [{"id": i} for i in range(100)]
        update = self.text_file_update()

        await handlers.handle_document(update, make_context())

        update.message.document.get_file.assert_awaited_once()

    async def test_the_limit_is_checked_for_the_requesting_user_only(self):
        handlers = make_handlers(max_documents_per_user=1)
        handlers.document_store.list_documents.return_value = []

        await handlers.handle_document(self.text_file_update(user_id=42), make_context())

        handlers.document_store.list_documents.assert_called_with(42)

    @patch("src.bot.handlers.documents.fetch_pdf_bytes")
    async def test_pdf_link_is_refused_at_the_limit_before_downloading(self, mock_fetch):
        handlers = make_handlers(max_documents_per_user=1)
        handlers.document_store.list_documents.return_value = [{"id": 1}]
        handlers._is_safe_url = Mock(return_value=(True, ""))
        update = make_update(text="https://example.com/a.pdf")

        await handlers.handle_message(update, make_context())

        mock_fetch.assert_not_called()
        self.assertTrue(any("limit of 1 document" in t for t in reply_texts(update)))

    @patch("src.processors.web_processor.WebProcessor.process_url")
    @patch("src.processors.web_processor.validate_url_ssrf", return_value=(True, ""))
    async def test_web_page_is_refused_at_the_limit_before_fetching(self, _ssrf, mock_process):
        handlers = make_handlers(max_documents_per_user=1)
        handlers.document_store.list_documents.return_value = [{"id": 1}]
        context = make_context()
        context.args = ["https://example.com/page"]
        update = make_update()

        await handlers.addurl_command(update, context)

        mock_process.assert_not_called()
        self.assertTrue(any("limit of 1 document" in t for t in reply_texts(update)))


if __name__ == "__main__":
    unittest.main()
