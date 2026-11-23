"""Tests for structured logging, request ids, rotation and secret redaction."""

import asyncio
import json
import logging
import os
import tempfile
import unittest

from src.config import Config
from src.logging_setup import (
    JsonFormatter,
    Redactor,
    configure_logging,
    redact_url,
    request_context,
    request_id_var,
    with_request_id,
)

FAKE_TELEGRAM_TOKEN = "123456789:AAFakeTokenForTestsOnly_abcdefghijk123"
FAKE_API_KEY = "sk-test-secret-key-0123456789"


def make_record(message="hello", level=logging.INFO, exc_info=None, **extra):
    record = logging.LogRecord("tests.logger", level, __file__, 1, message, (), exc_info)
    record.request_id = request_id_var.get()
    for key, value in extra.items():
        setattr(record, key, value)
    return record


class TestRequestIds(unittest.IsolatedAsyncioTestCase):
    def test_default_request_id_is_a_dash(self):
        self.assertEqual(request_id_var.get(), "-")

    def test_request_context_sets_and_resets_the_id(self):
        with request_context() as request_id:
            self.assertEqual(request_id_var.get(), request_id)
            self.assertEqual(len(request_id), 8)

        self.assertEqual(request_id_var.get(), "-")

    async def test_each_call_of_a_decorated_handler_gets_its_own_id(self):
        seen = []

        @with_request_id
        async def handler():
            seen.append(request_id_var.get())

        await handler()
        await handler()

        self.assertEqual(len(set(seen)), 2)
        self.assertNotIn("-", seen)
        self.assertEqual(request_id_var.get(), "-")

    async def test_concurrent_handlers_keep_separate_ids(self):
        seen = {}

        @with_request_id
        async def handler(name):
            before = request_id_var.get()
            await asyncio.sleep(0.01)
            seen[name] = (before, request_id_var.get())

        await asyncio.gather(handler("a"), handler("b"))

        self.assertEqual(seen["a"][0], seen["a"][1])
        self.assertEqual(seen["b"][0], seen["b"][1])
        self.assertNotEqual(seen["a"][0], seen["b"][0])

    async def test_the_id_follows_work_moved_to_a_thread(self):
        """Pipeline and LLM code runs in a thread, and its log lines must carry the same id."""

        @with_request_id
        async def handler():
            return request_id_var.get(), await asyncio.to_thread(request_id_var.get)

        outer, inner = await handler()

        self.assertEqual(outer, inner)
        self.assertNotEqual(outer, "-")


class TestJsonFormatter(unittest.TestCase):
    def setUp(self):
        self.formatter = JsonFormatter(Redactor())

    def test_output_is_one_json_object_with_the_main_fields(self):
        with request_context("abc12345"):
            line = self.formatter.format(make_record("Question answered"))

        entry = json.loads(line)
        self.assertNotIn("\n", line)
        self.assertEqual(entry["message"], "Question answered")
        self.assertEqual(entry["level"], "INFO")
        self.assertEqual(entry["logger"], "tests.logger")
        self.assertEqual(entry["request_id"], "abc12345")
        self.assertIn("time", entry)

    def test_extra_fields_are_included(self):
        entry = json.loads(self.formatter.format(make_record(user_id=42, elapsed_ms=1500)))

        self.assertEqual(entry["user_id"], 42)
        self.assertEqual(entry["elapsed_ms"], 1500)

    def test_exceptions_are_included_as_text(self):
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            import sys

            record = make_record("failed", level=logging.ERROR, exc_info=sys.exc_info())

        entry = json.loads(self.formatter.format(record))

        self.assertIn("RuntimeError: boom", entry["exception"])

    def test_non_ascii_text_is_kept_readable(self):
        line = self.formatter.format(make_record("документ"))

        self.assertIn("документ", line)


class TestRedaction(unittest.TestCase):
    def test_telegram_bot_tokens_are_hidden_even_when_unknown(self):
        redactor = Redactor()

        text = redactor(f"GET https://api.telegram.org/bot{FAKE_TELEGRAM_TOKEN}/getUpdates")

        self.assertNotIn(FAKE_TELEGRAM_TOKEN, text)
        self.assertIn("[REDACTED]", text)

    def test_known_secrets_are_hidden(self):
        redactor = Redactor([FAKE_API_KEY, "short"])

        text = redactor(f"key={FAKE_API_KEY} and the word short stays")

        self.assertNotIn(FAKE_API_KEY, text)
        self.assertIn("word short stays", text)

    def test_bearer_tokens_are_hidden(self):
        text = Redactor()("Authorization: Bearer abcdef1234567890xyz")

        self.assertNotIn("abcdef1234567890xyz", text)

    def test_secrets_are_hidden_inside_tracebacks_in_json(self):
        formatter = JsonFormatter(Redactor([FAKE_API_KEY]))
        try:
            raise RuntimeError(f"request with {FAKE_API_KEY} failed")
        except RuntimeError:
            import sys

            record = make_record("failed", level=logging.ERROR, exc_info=sys.exc_info())

        self.assertNotIn(FAKE_API_KEY, formatter.format(record))

    def test_urls_are_logged_without_query_or_credentials(self):
        url = "https://user:pw@example.com/files/a.pdf?token=SECRET&x=1#frag"

        self.assertEqual(redact_url(url), "https://example.com/files/a.pdf")

    def test_redact_url_survives_garbage(self):
        self.assertIsInstance(redact_url("not a url"), str)
        self.assertEqual(redact_url(""), "")


class TestConfigureLogging(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.log_file = os.path.join(self._temp.name, "logs", "bot.log")
        self.logger = logging.getLogger("telerag.test.logging")
        self.logger.propagate = False
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()
        self._temp.cleanup()

    def config(self, **overrides):
        values = {
            "telegram_bot_token": FAKE_TELEGRAM_TOKEN,
            "llm_api_key": FAKE_API_KEY,
            "log_file": self.log_file,
        }
        values.update(overrides)
        return Config(**values)

    def read_log_lines(self):
        for handler in self.logger.handlers:
            handler.flush()
        with open(self.log_file, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_file_receives_json_lines_with_the_request_id(self):
        configure_logging(self.config(), logger=self.logger)

        with request_context("req00001"):
            self.logger.info("Question received", extra={"user_id": 7})

        (entry,) = self.read_log_lines()
        self.assertEqual(entry["message"], "Question received")
        self.assertEqual(entry["request_id"], "req00001")
        self.assertEqual(entry["user_id"], 7)

    def test_the_log_directory_is_created(self):
        configure_logging(self.config(), logger=self.logger)

        self.assertTrue(os.path.isdir(os.path.dirname(self.log_file)))

    def test_tokens_and_keys_never_reach_the_file(self):
        configure_logging(self.config(), logger=self.logger)

        self.logger.info("calling https://api.telegram.org/bot%s/getMe", FAKE_TELEGRAM_TOKEN)
        self.logger.error("auth failed for key %s", FAKE_API_KEY)

        for handler in self.logger.handlers:
            handler.flush()
        with open(self.log_file, encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn(FAKE_TELEGRAM_TOKEN, content)
        self.assertNotIn(FAKE_API_KEY, content)

    def test_the_log_file_rotates_and_keeps_a_limited_number_of_backups(self):
        configure_logging(self.config(log_max_bytes=1024, log_backup_count=2), logger=self.logger)

        for i in range(200):
            self.logger.info("line number %d with some padding text to fill the file", i)

        log_dir = os.path.dirname(self.log_file)
        files = sorted(os.listdir(log_dir))
        self.assertEqual(files, ["bot.log", "bot.log.1", "bot.log.2"])
        for name in files:
            self.assertLess(os.path.getsize(os.path.join(log_dir, name)), 1024 + 400)

    def test_log_level_is_applied(self):
        configure_logging(self.config(log_level="WARNING"), logger=self.logger)

        self.assertEqual(self.logger.level, logging.WARNING)

    def test_chatty_libraries_stay_quiet_even_in_debug_mode(self):
        configure_logging(self.config(log_level="DEBUG"), logger=self.logger)

        for name in ("httpx", "httpcore", "urllib3", "telegram"):
            self.assertGreaterEqual(logging.getLogger(name).level, logging.WARNING, name)

    def test_calling_it_twice_does_not_duplicate_handlers(self):
        configure_logging(self.config(), logger=self.logger)
        configure_logging(self.config(), logger=self.logger)

        self.assertEqual(len(self.logger.handlers), 2)


if __name__ == "__main__":
    unittest.main()
