"""Unit tests for faithfulness evaluation with mocked LLM judge."""

import json
import os
import tempfile
import unittest
from unittest.mock import Mock

from eval.run_faithfulness import (
    extract_json_block,
    judge_answer_faithfulness,
    run_faithfulness_evaluation,
)


class TestFaithfulness(unittest.TestCase):
    """Test suite for faithfulness evaluation module."""

    def test_extract_json_block(self):
        """Verify extraction from markdown code blocks and raw JSON."""
        md_json = "Here is the response:\n```json\n{\"claims\": [{\"claim\": \"test\", \"entailed\": true}]}\n```\nDone."
        extracted = extract_json_block(md_json)
        self.assertIsNotNone(extracted)
        self.assertIn("claims", extracted)
        self.assertEqual(len(extracted["claims"]), 1)

        raw_json = "{\"claims\": []}"
        extracted_raw = extract_json_block(raw_json)
        self.assertIsNotNone(extracted_raw)
        self.assertEqual(extracted_raw["claims"], [])

    def test_judge_answer_faithfulness_mocked(self):
        """Verify judge parses claims and entailment correctly."""
        mock_judge = Mock()
        mock_judge.send_request.return_value = json.dumps(
            {
                "claims": [
                    {"claim": "Replication provides high availability", "entailed": True, "reason": "Matches doc"},
                    {"claim": "Raft uses 5 phases", "entailed": False, "reason": "Doc states 3 states"},
                ]
            }
        )

        report = judge_answer_faithfulness(
            judge_client=mock_judge,
            question="What is high availability?",
            context="Replication is the primary technique for high availability.",
            answer="Replication provides high availability. Raft uses 5 phases.",
        )

        self.assertEqual(len(report.claims), 2)
        self.assertTrue(report.claims[0].entailed)
        self.assertFalse(report.claims[1].entailed)

    def test_run_faithfulness_evaluation_pipeline_mocked(self):
        """Verify end-to-end evaluation pipeline with mocked generator and judge."""
        mock_gen = Mock()
        mock_gen.generate_answer.return_value = "Replication is the primary technique for high availability."

        mock_judge = Mock()
        mock_judge.send_request.return_value = json.dumps(
            {
                "claims": [
                    {
                        "claim": "Replication is primary technique for high availability",
                        "entailed": True,
                        "reason": "Directly stated in Chapter 1",
                    }
                ]
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "faithfulness_test.json")

            summary = run_faithfulness_evaluation(
                golden_path="eval/golden_ci.json",
                pdf_path="eval/fixtures/ci_fixture.pdf",
                output_path=out_json,
                n=2,
                judge_model="mock-judge",
                gen_model="mock-gen",
                resume=False,
                judge_client=mock_judge,
                gen_client=mock_gen,
            )

            self.assertIn("mean_faithfulness", summary)
            self.assertEqual(summary["mean_faithfulness"], 1.0)
            self.assertTrue(os.path.exists(out_json))

            with open(out_json, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            self.assertIn("questions", saved_data)
            self.assertEqual(len(saved_data["questions"]), 2)


if __name__ == "__main__":
    unittest.main()
