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
    validate_judge,
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
            self.assertEqual(summary["pdf_name"], "ci_fixture.pdf")
            self.assertEqual(summary["n"], 2)
            self.assertEqual(summary["generator"], "mock-gen")
            self.assertEqual(summary["judge"], "mock-judge")
            self.assertIn("prompt_version", summary)
            self.assertTrue(os.path.exists(out_json))

            with open(out_json, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            self.assertIn("questions", saved_data)
            self.assertEqual(len(saved_data["questions"]), 2)
            self.assertEqual(saved_data["pdf_name"], "ci_fixture.pdf")

    def test_validate_judge_untrustworthy_when_scoring_all_positive(self):
        """Verify judge scoring 1.0 on both references and negative controls fails validation."""
        mock_judge = Mock()
        # Fake judge marks every claim as entailed (cannot detect negative controls)
        mock_judge.send_request.return_value = json.dumps(
            {
                "claims": [
                    {"claim": "Some claim", "entailed": True, "reason": "Always yes"}
                ]
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "judge_val_test.json")
            # Create a small fixture with 1 reference and 1 negative control
            sample_answers = [
                {
                    "id": "ref_01",
                    "question": "What is the primary technique used for high availability in distributed systems?",
                    "answer": "Replication is the primary technique for high availability.",
                    "is_negative_control": False,
                },
                {
                    "id": "neg_01",
                    "question": "What is the primary technique used for high availability in distributed systems?",
                    "answer": "Quantum entanglement relays teleport packets across interstellar wormholes.",
                    "is_negative_control": True,
                },
            ]
            ans_file = os.path.join(tmpdir, "answers.json")
            with open(ans_file, "w", encoding="utf-8") as f:
                json.dump(sample_answers, f)

            res = validate_judge(
                answers_path=ans_file,
                pdf_path="eval/fixtures/ci_fixture.pdf",
                output_path=out_json,
                judge_model="gullible-judge",
                judge_client=mock_judge,
            )

            self.assertEqual(res["status"], "NOT TRUSTWORTHY")
            self.assertFalse(res["passed"])
            self.assertEqual(res["mean_reference_faithfulness"], 1.0)
            self.assertEqual(res["mean_negative_control_faithfulness"], 1.0)
            self.assertTrue(os.path.exists(out_json))

            with open(out_json, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertIn("judge_validation", saved)
            self.assertEqual(saved["judge_validation"]["status"], "NOT TRUSTWORTHY")

    def test_validate_judge_valid_when_separating_correctly(self):
        """Verify judge that entails references and rejects negative controls passes validation."""
        mock_judge = Mock()

        def side_effect(messages, max_tokens=None):
            user_content = messages[1]["content"]
            # If evaluating the unsupported negative answer
            if "Quantum" in user_content or "interstellar" in user_content:
                return json.dumps(
                    {
                        "claims": [
                            {"claim": "Quantum entanglement is used", "entailed": False, "reason": "Not in text"}
                        ]
                    }
                )
            return json.dumps(
                {
                    "claims": [
                        {"claim": "Replication is primary technique", "entailed": True, "reason": "In text"}
                    ]
                }
            )

        mock_judge.send_request.side_effect = side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "judge_val_test.json")
            sample_answers = [
                {
                    "id": "ref_01",
                    "question": "What is the primary technique used for high availability in distributed systems?",
                    "answer": "Replication is the primary technique for high availability.",
                    "is_negative_control": False,
                },
                {
                    "id": "neg_01",
                    "question": "What is the primary technique used for high availability in distributed systems?",
                    "answer": "Quantum entanglement relays teleport packets across interstellar wormholes.",
                    "is_negative_control": True,
                },
            ]
            ans_file = os.path.join(tmpdir, "answers.json")
            with open(ans_file, "w", encoding="utf-8") as f:
                json.dump(sample_answers, f)

            res = validate_judge(
                answers_path=ans_file,
                pdf_path="eval/fixtures/ci_fixture.pdf",
                output_path=out_json,
                judge_model="accurate-judge",
                judge_client=mock_judge,
            )

            self.assertEqual(res["status"], "VALID")
            self.assertTrue(res["passed"])
            self.assertGreaterEqual(res["mean_reference_faithfulness"], 0.80)
            self.assertTrue(all(s <= 0.50 for s in res["negative_control_scores"]))

            with open(out_json, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertIn("judge_validation", saved)
            self.assertEqual(saved["judge_validation"]["status"], "VALID")

    def test_faithfulness_refusal_and_abstention(self):
        """Verify refusal answers are marked applicable=False and faithfulness=None."""
        mock_gen = Mock()
        mock_gen.generate_answer.return_value = "INSUFFICIENT_CONTEXT"

        mock_judge = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "faithfulness_refusal.json")

            summary = run_faithfulness_evaluation(
                golden_path="eval/golden_ci.json",
                pdf_path="eval/fixtures/ci_fixture.pdf",
                output_path=out_json,
                n=1,
                judge_model="mock-judge",
                gen_model="mock-gen",
                resume=False,
                judge_client=mock_judge,
                gen_client=mock_gen,
            )

            # Refusal should have 0 applicable items, mean_faithfulness default 1.0
            self.assertEqual(summary["mean_faithfulness"], 1.0)
            q_id = list(summary["questions"].keys())[0]
            rec = summary["questions"][q_id]
            self.assertFalse(rec["applicable"])
            self.assertIsNone(rec["faithfulness"])


if __name__ == "__main__":
    unittest.main()
