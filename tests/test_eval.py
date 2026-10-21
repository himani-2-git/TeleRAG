"""Unit tests for evaluation metrics, rate calculation, and summary reporting."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from eval.report import (
    print_faithfulness_summary_table,
    print_validation_set_summary,
    rate,
)
from eval.run_eval import rate as run_eval_rate
from eval.run_eval import run_evaluation
from eval.run_faithfulness import rate as run_faithfulness_rate


class TestEvalRateAndMetrics(unittest.TestCase):
    """Test rate function, zero-denominator safety, and reporting formatting."""

    def test_rate_calculation(self):
        """Test rate computes accurate division and returns None when denominator is 0."""
        self.assertEqual(rate(3, 4), 0.75)
        self.assertEqual(rate(0, 5), 0.0)
        self.assertIsNone(rate(0, 0))
        self.assertIsNone(rate(5, 0))

        # Check re-exports from run_eval and run_faithfulness
        self.assertEqual(run_eval_rate(1, 2), 0.5)
        self.assertIsNone(run_eval_rate(0, 0))
        self.assertEqual(run_faithfulness_rate(2, 4), 0.5)
        self.assertIsNone(run_faithfulness_rate(0, 0))

    def test_mean_faithfulness_with_no_applicable_answers_is_na(self):
        """When results contain only refusals or errors, mean faithfulness is n/a."""
        results = {
            "q_0": {
                "id": "q_0",
                "question": "What is Python?",
                "applicable": False,
                "abstained": True,
                "unscorable": False,
                "faithfulness": None,
            },
            "q_1": {
                "id": "q_1",
                "question": "How to bake bread?",
                "applicable": False,
                "abstained": False,
                "unscorable": False,
                "faithfulness": None,
            },
        }

        buf = io.StringIO()
        with redirect_stdout(buf):
            mean_f, unscorable_count = print_faithfulness_summary_table(
                results=results,
                active_judge_name="test-judge",
                judge_val_verdict="VALID",
                heldout_accept_rate=0.85,
                heldout_false_accept_rate=0.0,
            )

        self.assertIsNone(mean_f)
        self.assertEqual(unscorable_count, 0)
        output = buf.getvalue()
        self.assertIn("**Mean Answer Faithfulness**: `n/a`", output)
        self.assertNotIn("`1.000`", output)

    def test_print_validation_set_summary_handles_none_rates(self):
        """When validation metrics are None, print n/a rather than numbers or crash."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_validation_set_summary(
                name="EMPTY SET",
                chosen_threshold=0.50,
                refs=[],
                negs=[],
                m_ref=None,
                m_neg=None,
                max_neg=None,
                ref_accept_rate=None,
                ref_accept_cnt=0,
                scorable_ref_count=0,
                neg_false_accept_rate=None,
                neg_false_accept_cnt=0,
                scorable_neg_count=0,
                neg_fully_accepted_rate=None,
                neg_fully_accepted_cnt=0,
                unscorable_ref_cnt=0,
                unscorable_neg_cnt=0,
                passed=False,
            )
        output = buf.getvalue()
        self.assertIn("Mean Reference Faithfulness:         n/a", output)
        self.assertIn("Reference Accept Rate:               n/a (0/0)", output)
        self.assertIn("Negative Control False-Accept Rate:  n/a (0/0)", output)
        self.assertNotIn("1.000 (0/0)", output)

    @patch("eval.run_eval.extract_chunks_with_pages")
    @patch("eval.run_eval.EmbeddingGenerator")
    def test_run_eval_zero_unanswerable_hard_reports_na(self, mock_emb_cls, mock_extract):
        """When golden data has zero unanswerable hard questions, abstention_accuracy_hard is n/a."""
        mock_extract.return_value = [
            {"chunk_index": 0, "page": 1, "page_start": 1, "page_end": 1, "content": "Paxos"}
        ]
        mock_emb_inst = Mock()
        mock_emb_inst.generate_embeddings.return_value = [[0.1] * 10]
        mock_emb_inst.generate_query_embedding.return_value = [0.1] * 10
        mock_emb_cls.return_value = mock_emb_inst

        golden_data = [
            {
                "id": "q1",
                "question": "What is Paxos?",
                "type": "answerable",
                "ground_truth_pages": [1],
            }
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(golden_data, f)
            golden_path = f.name

        try:
            summary = run_evaluation(
                pdf_path="eval/fixtures/ci_fixture.pdf",
                golden_path=golden_path,
                mode="dense",
            )
            self.assertEqual(summary["abstention_accuracy_hard"], "n/a")
            self.assertEqual(summary["abstention_accuracy_easy"], "n/a")
            self.assertEqual(summary["abstention_accuracy"], "n/a")
            self.assertNotEqual(summary["abstention_accuracy_hard"], 1.0)
        finally:
            if os.path.exists(golden_path):
                os.unlink(golden_path)
