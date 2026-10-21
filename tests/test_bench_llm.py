"""Tests for the LLM benchmark harness (eval/bench_llm.py)."""

import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from eval.bench_llm import is_refusal_response, run_llm_benchmark


class TestBenchLLM(unittest.TestCase):
    """Test LLM benchmarking loop, abstention handling, and metric aggregation."""

    def test_is_refusal_response_detection(self):
        """Verify is_refusal_response recognizes refusal strings and INSUFFICIENT_CONTEXT."""
        self.assertTrue(is_refusal_response("INSUFFICIENT_CONTEXT"))
        self.assertTrue(is_refusal_response("insufficient_context\nSome explanation"))
        self.assertTrue(is_refusal_response("The provided context does not contain the answer."))
        self.assertTrue(is_refusal_response("This information is not found in the context."))
        self.assertTrue(is_refusal_response(None))
        self.assertTrue(is_refusal_response(""))
        self.assertFalse(is_refusal_response("In Python, functions are defined using def."))

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "mock_token_for_ci"}, clear=False)
    @patch("eval.bench_llm.LLMClient")
    def test_bench_llm_completes_with_mocked_client_and_exercises_abstention(self, mock_client_cls):
        """Benchmark runs across 12 CI questions, exercises abstention on unanswerables, and outputs table fields."""
        mock_client = Mock()
        mock_client.send_request.return_value = {"choices": [{"message": {"content": "ok"}}]}
        # For answerable questions, return a valid grounded answer
        mock_client.generate_answer.return_value = "This is a valid answer grounded in context."
        mock_client_cls.return_value = mock_client

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_json = os.path.join(tmp_dir, "llm_bench.json")

            results = run_llm_benchmark(
                models=["test-mock-3b"],
                pdf_path="eval/fixtures/ci_fixture.pdf",
                golden_path="eval/golden_ci.json",
                output_path=out_json,
            )

            # 1. Assert model key present
            self.assertIn("test-mock-3b", results)
            m = results["test-mock-3b"]

            # 2. Assert table fields are present
            required_fields = [
                "median_latency_s",
                "p95_latency_s",
                "answerable_answered_pct",
                "unanswerable_refused_pct",
                "sample_size",
            ]
            for f in required_fields:
                self.assertIn(f, m, f"Missing required table field: {f}")

            # Also verify human-readable table column aliases
            self.assertIn("median latency", m)
            self.assertIn("p95 latency", m)
            self.assertIn("answerable answered %", m)
            self.assertIn("unanswerable refused %", m)

            # 3. Assert all 12 questions were evaluated (9 answerable + 3 unanswerable)
            self.assertEqual(m["sample_size"], 12)
            self.assertEqual(m["failed_count"], 0)

            # 4. Assert abstention branch was exercised:
            # The 3 unanswerable questions have similarity < 0.3 on ci_fixture.pdf,
            # so they must abstain and NEVER call client.generate_answer.
            # Thus generate_answer should only be called 9 times (for answerable questions).
            self.assertEqual(mock_client.generate_answer.call_count, 9)

            # Unanswerable questions were counted as refused without LLM calls
            self.assertEqual(m["unanswerable_refused_pct"], 100.0)
            self.assertEqual(m["answerable_answered_pct"], 100.0)

            # 5. Assert output file was written to disk
            self.assertTrue(os.path.exists(out_json))
            with open(out_json, encoding="utf-8") as f:
                saved_data = json.load(f)
            self.assertIn("test-mock-3b", saved_data)
            self.assertEqual(saved_data["test-mock-3b"]["sample_size"], 12)

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "mock_token_for_ci"}, clear=False)
    @patch("eval.bench_llm.LLMClient")
    def test_bench_llm_robust_against_question_exceptions(self, mock_client_cls):
        """Loop catches per-question exceptions, logs them, increments failure count, and continues."""
        mock_client = Mock()
        mock_client.send_request.return_value = {"choices": [{"message": {"content": "ok"}}]}

        # Raise an exception on the 1st answerable question, succeed on the rest
        call_counter = {"count": 0}

        def generate_with_one_error(q, ctx):
            call_counter["count"] += 1
            if call_counter["count"] == 1:
                raise RuntimeError("Simulated transient socket error")
            return "Valid answer."

        mock_client.generate_answer.side_effect = generate_with_one_error
        mock_client_cls.return_value = mock_client

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_json = os.path.join(tmp_dir, "llm_bench_robust.json")
            results = run_llm_benchmark(
                models=["robust-test-model"],
                pdf_path="eval/fixtures/ci_fixture.pdf",
                golden_path="eval/golden_ci.json",
                output_path=out_json,
            )

            m = results["robust-test-model"]
            # 1 failure was recorded, but the run completed
            self.assertEqual(m["failed_count"], 1)
            self.assertEqual(m["sample_size"], 12)
            # Remaining answerable questions succeeded (8 of 9 answered)
            self.assertAlmostEqual(m["answerable_answered_pct"], 100.0 * 8 / 9, places=1)
            self.assertEqual(m["unanswerable_refused_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
