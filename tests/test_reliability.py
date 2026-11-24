"""Tests for timeouts, retry limits and thread safety."""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import requests

from src.processors.pdf_fetcher import fetch_pdf_bytes
from src.processors.web_processor import WebProcessor
from src.rag.embeddings import EmbeddingGenerator
from src.rag.llm_client import LLMClient, LLMUnavailableError
from src.rag.reranker import CrossEncoderReranker


class FakeClock:
    """A clock that only moves when the code under test sleeps or "waits" for the network."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def as_time_module(self):
        return SimpleNamespace(monotonic=self.monotonic, sleep=self.sleep)


class TestLlmRetryTimeCap(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.read_timeouts = []

        def hanging_post(url, headers=None, json=None, timeout=None):
            # The server never answers: the call lasts for the whole read timeout.
            self.read_timeouts.append(timeout[1])
            self.clock.now += timeout[1]
            raise requests.exceptions.Timeout()

        self.hanging_post = hanging_post

    def send(self, **client_options):
        client = LLMClient(
            api_key="k",
            model="m",
            max_retries=3,
            retry_delay=1.0,
            timeout=(5.0, 120.0),
            **client_options,
        )
        with (
            patch("src.rag.llm_client.time", self.clock.as_time_module()),
            patch("src.rag.llm_client.requests.post", side_effect=self.hanging_post),
            self.assertLogs("src.rag.llm_client", level="WARNING"),
        ):
            return client.send_request([{"role": "user", "content": "hi"}])

    def test_total_time_never_exceeds_the_cap(self):
        result = self.send(max_total_seconds=180.0)

        self.assertIsNone(result)
        self.assertLessEqual(self.clock.now, 180.0)

    def test_last_attempt_only_gets_the_time_that_is_left(self):
        self.send(max_total_seconds=180.0)

        # 120 s for the first try, 1 s pause, so only 59 s remain for the second try.
        self.assertEqual(self.read_timeouts, [120.0, 59.0])

    def test_no_new_attempt_starts_when_the_cap_is_already_used_up(self):
        self.send(max_total_seconds=10.0)

        self.assertEqual(self.read_timeouts, [10.0])

    def test_a_pause_that_would_pass_the_cap_is_skipped(self):
        self.send(max_total_seconds=120.5)

        # After the first 120 s attempt only 0.5 s remain, less than the 1 s retry pause.
        self.assertEqual(len(self.read_timeouts), 1)

    def test_default_cap_is_three_minutes(self):
        self.assertEqual(LLMClient().max_total_seconds, 180.0)

    def test_generate_answer_reports_the_model_as_unavailable(self):
        client = LLMClient(max_retries=3, retry_delay=1.0, max_total_seconds=30.0)

        with (
            patch("src.rag.llm_client.time", self.clock.as_time_module()),
            patch("src.rag.llm_client.requests.post", side_effect=self.hanging_post),
            self.assertLogs("src.rag.llm_client", level="WARNING"),
            self.assertRaises(LLMUnavailableError),
        ):
            client.generate_answer("question", "context")


class TestLlmServerCheck(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient(model="qwen2.5:3b", base_url="http://localhost:11434/v1")

    @staticmethod
    def models_response(model_ids, status=200):
        response = Mock()
        response.status_code = status
        response.json.return_value = {"data": [{"id": model_id} for model_id in model_ids]}
        return response

    @patch("src.rag.llm_client.requests.get")
    def test_reports_ok_when_the_model_is_listed(self, mock_get):
        mock_get.return_value = self.models_response(["qwen2.5:3b", "llama3.2:3b"])

        is_ok, detail = self.client.check_server()

        self.assertTrue(is_ok)
        self.assertIn("qwen2.5:3b", detail)
        self.assertEqual(mock_get.call_args.args[0], "http://localhost:11434/v1/models")
        self.assertLessEqual(max(mock_get.call_args.kwargs["timeout"]), 10)

    @patch("src.rag.llm_client.requests.get")
    def test_reports_a_missing_model(self, mock_get):
        mock_get.return_value = self.models_response(["llama3.2:3b"])

        is_ok, detail = self.client.check_server()

        self.assertFalse(is_ok)
        self.assertIn("not listed", detail)

    @patch("src.rag.llm_client.requests.get", side_effect=requests.ConnectionError("refused"))
    def test_reports_an_unreachable_server(self, _mock_get):
        is_ok, detail = self.client.check_server()

        self.assertFalse(is_ok)
        self.assertIn("unreachable", detail)

    @patch("src.rag.llm_client.requests.get")
    def test_reports_an_http_error(self, mock_get):
        mock_get.return_value = self.models_response([], status=500)

        is_ok, detail = self.client.check_server()

        self.assertFalse(is_ok)
        self.assertIn("500", detail)


def slow_download_response(clock, content_type, seconds_per_chunk=4.0):
    """A response whose body arrives slowly: each chunk takes seconds_per_chunk."""

    def chunks(chunk_size=8192):
        for _ in range(100):
            clock.now += seconds_per_chunk
            yield b"x" * 10

    response = Mock()
    response.status_code = 200
    response.is_redirect = False
    response.headers = {"content-type": content_type}
    response.iter_content.side_effect = chunks
    return response


class TestDownloadDeadlines(unittest.TestCase):
    """A server that sends data very slowly must not hold a download open for ever."""

    def test_web_page_download_stops_after_the_total_time(self):
        clock = FakeClock()
        processor = WebProcessor(max_total_seconds=10.0)

        with (
            patch("src.processors.web_processor.time", clock.as_time_module()),
            patch("src.processors.web_processor.validate_url_ssrf", return_value=(True, "")),
            patch(
                "src.processors.web_processor.requests.get",
                return_value=slow_download_response(clock, "text/html"),
            ),
            self.assertRaises(ValueError) as caught,
        ):
            processor.fetch_url("https://example.com/slow")

        self.assertIn("too long", str(caught.exception))
        self.assertLess(clock.now, 20)

    def test_pdf_download_stops_after_the_total_time(self):
        clock = FakeClock()

        with (
            patch("src.processors.pdf_fetcher.time", clock.as_time_module()),
            patch("src.processors.pdf_fetcher.validate_url_ssrf", return_value=(True, "")),
            patch(
                "src.processors.pdf_fetcher.requests.get",
                return_value=slow_download_response(clock, "application/pdf"),
            ),
            self.assertRaises(ValueError) as caught,
        ):
            fetch_pdf_bytes("https://example.com/a.pdf", 10**6, timeout=30, total_timeout=10.0)

        self.assertIn("too long", str(caught.exception))
        self.assertLess(clock.now, 20)


class ConcurrencyProbe:
    """Counts how many threads are inside a call at the same time."""

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def enter(self):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def leave(self):
        with self._guard:
            self.active -= 1


def run_in_threads(function, count=6):
    threads = [threading.Thread(target=function) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)


class TestThreadSafetyOfModels(unittest.TestCase):
    """Updates now run at the same time, so shared models must not be used by two threads."""

    def test_embedding_model_is_used_by_one_thread_at_a_time(self):
        probe = ConcurrencyProbe()

        class FakeSentenceTransformer:
            def __init__(self, name):
                pass

            def encode(self, texts, **kwargs):
                probe.enter()
                time.sleep(0.02)
                probe.leave()
                if isinstance(texts, str):
                    return np.zeros(3)
                return np.zeros((len(texts), 3))

        with patch("src.rag.embeddings.SentenceTransformer", FakeSentenceTransformer):
            generator = EmbeddingGenerator("fake-model")
            run_in_threads(lambda: generator.generate_embeddings("some text"))

        self.assertEqual(probe.max_active, 1)

    def test_reranker_model_is_used_by_one_thread_at_a_time(self):
        probe = ConcurrencyProbe()

        class FakeCrossEncoder:
            def predict(self, pairs):
                probe.enter()
                time.sleep(0.02)
                probe.leave()
                return [0.5] * len(pairs)

        with patch.dict(CrossEncoderReranker._models, {"fake-ce": FakeCrossEncoder()}):
            reranker = CrossEncoderReranker(model_name="fake-ce")
            run_in_threads(lambda: reranker.rerank("query", [{"content": "text"}], top_k=1))

        self.assertEqual(probe.max_active, 1)


if __name__ == "__main__":
    unittest.main()
