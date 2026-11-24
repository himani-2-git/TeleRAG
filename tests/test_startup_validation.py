"""Tests for startup validation: settings are checked together and bad values fail fast."""

import logging
import os
import unittest
from unittest.mock import patch

from src.config import ConfigError, get_config, is_local_endpoint
from src.startup import StartupError, verify_llm_endpoint


def load(**env):
    """Load config from exactly these environment variables (the real .env is ignored)."""
    values = {"TELEGRAM_BOT_TOKEN": "123456:test-token"}
    values.update(env)
    with patch.dict(os.environ, values, clear=True):
        return get_config()


class TestConfigValidation(unittest.TestCase):
    def test_config_error_is_a_value_error(self):
        self.assertTrue(issubclass(ConfigError, ValueError))

    def test_missing_token_message_explains_how_to_fix_it(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigError) as caught:
                get_config()

        self.assertIn("TELEGRAM_BOT_TOKEN", str(caught.exception))
        self.assertIn("BotFather", str(caught.exception))

    def test_every_problem_is_reported_at_once(self):
        with self.assertRaises(ConfigError) as caught:
            load(
                CHUNK_SIZE="abc",
                TOP_K_RESULTS="0",
                SIMILARITY_THRESHOLD="1.5",
                RETRIEVAL_MODE="fancy",
            )

        message = str(caught.exception)
        for name in ("CHUNK_SIZE", "TOP_K_RESULTS", "SIMILARITY_THRESHOLD", "RETRIEVAL_MODE"):
            self.assertIn(name, message)

    def test_non_numeric_value_names_the_setting_and_the_bad_text(self):
        with self.assertRaises(ConfigError) as caught:
            load(MAX_FILE_SIZE_MB="twenty")

        self.assertIn("MAX_FILE_SIZE_MB", str(caught.exception))
        self.assertIn("twenty", str(caught.exception))

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with self.assertRaises(ConfigError) as caught:
            load(CHUNK_SIZE="500", CHUNK_OVERLAP="500")

        self.assertIn("CHUNK_OVERLAP", str(caught.exception))

    def test_blank_values_use_the_default(self):
        config = load(CHUNK_SIZE="", TOP_K_RESULTS="  ")

        self.assertEqual(config.chunk_size, 1000)
        self.assertEqual(config.top_k_results, 2)

    def test_yes_and_no_style_booleans_are_accepted(self):
        self.assertTrue(load(GROUNDING_ENFORCE="yes").grounding_enforce)
        self.assertFalse(load(GROUNDING_ENFORCE="off").grounding_enforce)

    def test_unclear_boolean_is_rejected(self):
        with self.assertRaises(ConfigError) as caught:
            load(GROUNDING_ENFORCE="maybe")

        self.assertIn("GROUNDING_ENFORCE", str(caught.exception))

    def test_log_level_is_case_insensitive_and_checked(self):
        self.assertEqual(load(LOG_LEVEL="debug").log_level, "DEBUG")
        with self.assertRaises(ConfigError) as caught:
            load(LOG_LEVEL="loud")
        self.assertIn("LOG_LEVEL", str(caught.exception))

    def test_unknown_choices_are_rejected(self):
        for name, value in (
            ("VECTOR_BACKEND", "mongo"),
            ("ORCHESTRATOR", "magic"),
            ("RETRIEVAL_MODE", "keyword"),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ConfigError) as caught:
                    load(**{name: value})
                self.assertIn(name, str(caught.exception))

    def test_new_runtime_settings_have_sensible_defaults(self):
        config = load()

        self.assertEqual(config.concurrent_updates, 16)
        self.assertEqual(config.llm_max_total_seconds, 180)
        self.assertEqual(config.log_max_bytes, 5_000_000)
        self.assertEqual(config.log_backup_count, 3)
        self.assertEqual(config.admin_user_ids, frozenset())

    def test_admin_ids_are_parsed_and_checked(self):
        self.assertEqual(load(ADMIN_USER_IDS="7, 8").admin_user_ids, frozenset({7, 8}))
        with self.assertRaises(ConfigError) as caught:
            load(ADMIN_USER_IDS="me")
        self.assertIn("ADMIN_USER_IDS", str(caught.exception))

    def test_token_is_not_required_for_offline_tools(self):
        with patch.dict(os.environ, {}, clear=True):
            config = get_config(require_telegram=False)

        self.assertEqual(config.telegram_bot_token, "")


class TestLocalEndpointDetection(unittest.TestCase):
    def test_local_and_internal_addresses_need_no_api_key(self):
        for url in (
            "http://localhost:11434/v1",
            "http://127.0.0.1:11434/v1",
            "http://[::1]:11434/v1",
            "http://ollama:11434/v1",  # Docker Compose service name
            "http://host.docker.internal:11434/v1",
            "http://192.168.1.20:11434/v1",
        ):
            with self.subTest(url=url):
                self.assertTrue(is_local_endpoint(url))
                self.assertEqual(load(LLM_BASE_URL=url).llm_base_url, url)

    def test_remote_service_requires_an_api_key(self):
        url = "https://openrouter.ai/api/v1"
        self.assertFalse(is_local_endpoint(url))

        with self.assertRaises(ConfigError) as caught:
            load(LLM_BASE_URL=url)
        self.assertIn("LLM_API_KEY", str(caught.exception))

        self.assertEqual(load(LLM_BASE_URL=url, LLM_API_KEY="k").llm_api_key, "k")


class FakeLLM:
    base_url = "http://localhost:11434/v1/chat/completions"
    model = "qwen2.5:3b"

    def __init__(self, result):
        self.result = result

    def check_availability(self):
        return self.result


class TestVerifyLlmEndpoint(unittest.TestCase):
    def test_working_endpoint_passes(self):
        verify_llm_endpoint(FakeLLM((True, 200, "OK")))

    def test_unreachable_server_gives_a_helpful_message(self):
        with self.assertRaises(StartupError) as caught:
            verify_llm_endpoint(FakeLLM((False, None, "Service unavailable")))

        message = str(caught.exception)
        self.assertIn("Cannot reach", message)
        self.assertIn("Ollama", message)
        self.assertIn("LLM_BASE_URL", message)

    def test_missing_model_tells_you_the_pull_command(self):
        with self.assertRaises(StartupError) as caught:
            verify_llm_endpoint(FakeLLM((False, 404, "model not found")))

        self.assertIn("ollama pull qwen2.5:3b", str(caught.exception))

    def test_rejected_key_points_to_llm_api_key(self):
        for status in (401, 403):
            with self.subTest(status=status):
                with self.assertRaises(StartupError) as caught:
                    verify_llm_endpoint(FakeLLM((False, status, "denied")))
                self.assertIn("LLM_API_KEY", str(caught.exception))

    def test_server_error_stops_startup(self):
        with self.assertRaises(StartupError) as caught:
            verify_llm_endpoint(FakeLLM((False, 500, "boom")))

        self.assertIn("500", str(caught.exception))

    def test_rate_limit_only_warns_because_the_server_is_up(self):
        with self.assertLogs("src.startup", level=logging.WARNING):
            verify_llm_endpoint(FakeLLM((False, 429, "slow down")))


if __name__ == "__main__":
    unittest.main()
