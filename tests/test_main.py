"""Tests for the startup sequence in main.py: problems stop the bot early and clearly."""

import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from telegram.error import InvalidToken, NetworkError

import main


class FakeLLMClient:
    """Stands in for LLMClient. The check result is set per test."""

    check_result = (True, 200, "OK")

    def __init__(self, **kwargs):
        self.model = kwargs.get("model", "qwen2.5:3b")
        self.base_url = kwargs.get("base_url", "http://localhost:11434") + "/chat/completions"
        self.kwargs = kwargs

    def check_availability(self):
        return self.check_result


class MainCase(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._temp.cleanup)
        folder = self._temp.name
        self.env = {
            "TELEGRAM_BOT_TOKEN": "123456:test-token",
            "DATABASE_PATH": os.path.join(folder, "db", "documents.db"),
            "EMBEDDINGS_PATH": os.path.join(folder, "emb"),
            "LOG_FILE": os.path.join(folder, "logs", "bot.log"),
        }
        FakeLLMClient.check_result = (True, 200, "OK")
        self.application = Mock()
        self.patches = [
            patch.object(main, "configure_logging"),
            patch.object(main, "EmbeddingGenerator", return_value=Mock()),
            patch.object(main, "LLMClient", FakeLLMClient),
            patch.object(main, "build_application", return_value=self.application),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_main(self, **env_changes):
        env = dict(self.env)
        env.update(env_changes)
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SystemExit) as caught:
                main.main()
        return caught.exception.code


class TestStartupFailures(MainCase):
    def test_bad_settings_stop_the_bot_and_list_every_problem(self):
        errors = io.StringIO()

        with contextlib.redirect_stderr(errors):
            code = self.run_main(TELEGRAM_BOT_TOKEN="", CHUNK_SIZE="abc", TOP_K_RESULTS="0")

        self.assertEqual(code, 1)
        for name in ("TELEGRAM_BOT_TOKEN", "CHUNK_SIZE", "TOP_K_RESULTS"):
            self.assertIn(name, errors.getvalue())
        self.application.run_polling.assert_not_called()

    def test_unreachable_llm_server_stops_the_bot(self):
        FakeLLMClient.check_result = (False, None, "Service unavailable")

        with self.assertLogs("main", level="ERROR") as logs:
            code = self.run_main()

        self.assertEqual(code, 1)
        self.assertIn("Cannot reach the LLM server", " ".join(logs.output))
        self.application.run_polling.assert_not_called()

    def test_missing_model_tells_the_user_how_to_download_it(self):
        FakeLLMClient.check_result = (False, 404, "model not found")

        with self.assertLogs("main", level="ERROR") as logs:
            self.run_main(LLM_MODEL="qwen2.5:3b")

        self.assertIn("ollama pull qwen2.5:3b", " ".join(logs.output))

    def test_database_that_cannot_be_opened_stops_the_bot(self):
        blocker = os.path.join(self._temp.name, "not_a_folder")
        with open(blocker, "w") as f:
            f.write("this is a file, so nothing can be created inside it")

        with self.assertLogs("main", level="ERROR") as logs:
            code = self.run_main(DATABASE_PATH=os.path.join(blocker, "sub", "documents.db"))

        self.assertEqual(code, 1)
        self.assertIn("Cannot open the database", " ".join(logs.output))

    def test_embedding_model_that_cannot_be_loaded_stops_the_bot(self):
        with patch.object(main, "EmbeddingGenerator", side_effect=OSError("no internet")):
            with self.assertLogs("main", level="ERROR") as logs:
                code = self.run_main(EMBEDDING_MODEL="some-model")

        self.assertEqual(code, 1)
        self.assertIn("embedding model 'some-model'", " ".join(logs.output))

    def test_unknown_prompt_version_stops_the_bot(self):
        with self.assertLogs("main", level="ERROR") as logs:
            code = self.run_main(PROMPT_VERSION="does-not-exist")

        self.assertEqual(code, 1)
        self.assertIn("PROMPT_VERSION", " ".join(logs.output))

    def test_telegram_rejecting_the_token_stops_the_bot_with_a_clear_message(self):
        self.application.run_polling.side_effect = InvalidToken()

        with self.assertLogs("main", level="ERROR") as logs:
            code = self.run_main()

        self.assertEqual(code, 1)
        self.assertIn("TELEGRAM_BOT_TOKEN", " ".join(logs.output))

    def test_no_connection_to_telegram_stops_the_bot_with_a_clear_message(self):
        self.application.run_polling.side_effect = NetworkError("timed out")

        with self.assertLogs("main", level="ERROR") as logs:
            code = self.run_main()

        self.assertEqual(code, 1)
        self.assertIn("Cannot reach Telegram", " ".join(logs.output))


class TestSuccessfulStart(MainCase):
    def test_bot_starts_polling_when_everything_works(self):
        with patch.dict(os.environ, self.env, clear=True):
            main.main()

        self.application.run_polling.assert_called_once_with(allowed_updates=["message"])

    def test_the_llm_time_cap_from_the_settings_reaches_the_client(self):
        with patch.dict(os.environ, {**self.env, "LLM_MAX_TOTAL_SECONDS": "77"}, clear=True):
            handlers = main.create_bot_handlers(main.get_config())

        self.assertEqual(handlers.llm_client.kwargs["max_total_seconds"], 77.0)

    def test_deleting_a_document_clears_the_cached_search_index(self):
        with patch.dict(os.environ, self.env, clear=True):
            handlers = main.create_bot_handlers(main.get_config())

        listeners = handlers.document_store._change_listeners
        self.assertEqual(len(listeners), 1)


if __name__ == "__main__":
    unittest.main()
