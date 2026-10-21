"""Unit tests for faithfulness evaluation with NLI CrossEncoder and LLM judges."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock

import numpy as np

from eval.run_faithfulness import (
    build_premise_windows,
    extract_json_block,
    get_nli_label_indices,
    judge_answer_faithfulness_llm,
    judge_answer_faithfulness_nli,
    make_cache_key,
    parse_premise_windows,
    run_faithfulness_evaluation,
    validate_judge,
)


class TestFaithfulness(unittest.TestCase):
    """Test suite for faithfulness evaluation module."""

    def setUp(self):
        self._stdout_trap = io.StringIO()
        self._stdout_redirector = redirect_stdout(self._stdout_trap)
        self._stdout_redirector.__enter__()

    def tearDown(self):
        self._stdout_redirector.__exit__(None, None, None)

    def test_extract_json_block(self):
        """Verify extraction from markdown code blocks and raw JSON."""
        md_json = 'Here is the response:\n```json\n{"claims": [{"claim": "test", "entailed": true}]}\n```\nDone.'
        extracted = extract_json_block(md_json)
        self.assertIsNotNone(extracted)
        self.assertIn("claims", extracted)
        self.assertEqual(len(extracted["claims"]), 1)

        raw_json = '{"claims": []}'
        extracted_raw = extract_json_block(raw_json)
        self.assertIsNotNone(extracted_raw)
        self.assertEqual(extracted_raw["claims"], [])

    def test_judge_answer_faithfulness_llm_mocked(self):
        """Verify LLM judge parses claims and entailment correctly."""
        mock_judge = Mock()
        mock_judge.send_request.return_value = json.dumps(
            {
                "claims": [
                    {
                        "claim": "Replication provides high availability",
                        "entailed": True,
                        "reason": "Matches doc",
                    },
                    {
                        "claim": "Raft uses 5 phases",
                        "entailed": False,
                        "reason": "Doc states 3 states",
                    },
                ]
            }
        )

        report = judge_answer_faithfulness_llm(
            judge_client=mock_judge,
            question="What is high availability?",
            context="Replication is the primary technique for high availability.",
            answer="Replication provides high availability. Raft uses 5 phases.",
        )

        self.assertEqual(len(report.claims), 2)
        self.assertFalse(report.unscorable)
        self.assertTrue(report.claims[0].entailed)
        self.assertFalse(report.claims[1].entailed)

    def test_zero_claim_unscorable_handling(self):
        """Verify that an answer with 0 claims is marked UNSCORABLE and excluded from mean."""
        mock_judge = Mock()
        # Returns empty claims
        mock_judge.send_request.return_value = json.dumps({"claims": []})

        report = judge_answer_faithfulness_llm(
            judge_client=mock_judge,
            question="What is consensus?",
            context="Some text",
            answer="Ambiguous text without factual claims",
        )
        self.assertTrue(report.unscorable)
        self.assertEqual(len(report.claims), 0)

        # In evaluation pipeline, unscorable answers must have faithfulness=None and be excluded from mean
        mock_gen = Mock()
        mock_gen.generate_answer.return_value = "Ambiguous text without factual claims."

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "unscorable_eval.json")
            summary = run_faithfulness_evaluation(
                golden_path="eval/golden_ci.json",
                pdf_path="eval/fixtures/ci_fixture.pdf",
                output_path=out_json,
                n=1,
                judge_type="llm",
                judge_model="mock-judge",
                gen_model="mock-gen",
                resume=False,
                judge_client=mock_judge,
                gen_client=mock_gen,
            )

            q_id = list(summary["questions"].keys())[0]
            rec = summary["questions"][q_id]
            self.assertTrue(rec["unscorable"])
            self.assertIsNone(rec["faithfulness"])
            self.assertEqual(summary["unscorable_count"], 1)

    def test_nli_scorer_supported_contradicted_unrelated(self):
        """Verify NLI scorer correctly classifies supported, contradicted, and unrelated sentences."""
        mock_ce = Mock()
        # Mock config with standard label indices: 0: contradiction, 1: entailment, 2: neutral
        mock_ce.config = Mock()
        mock_ce.config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        mock_ce.config.label2id = {"contradiction": 0, "entailment": 1, "neutral": 2}
        mock_ce.tokenizer = None

        chunks = [{"content": "Replication is the primary technique for high availability."}]

        # 1. Supported sentence: high entailment probability
        mock_ce.predict.return_value = np.array([[-2.0, 3.5, -1.0]])
        rep_supp = judge_answer_faithfulness_nli(
            cross_encoder=mock_ce,
            question="What is high availability?",
            chunks=chunks,
            answer="Replication provides high availability in distributed systems.",
            entail_threshold=0.5,
        )
        self.assertFalse(rep_supp["unscorable"])
        self.assertEqual(rep_supp["faithfulness"], 1.0)
        self.assertTrue(rep_supp["sentences"][0]["supported"])

        # 2. Contradicted sentence: high contradiction probability
        mock_ce.predict.return_value = np.array([[3.5, -2.0, -1.0]])
        rep_contra = judge_answer_faithfulness_nli(
            cross_encoder=mock_ce,
            question="What is high availability?",
            chunks=chunks,
            answer="Replication actively destroys system availability.",
            entail_threshold=0.5,
        )
        self.assertFalse(rep_contra["unscorable"])
        self.assertEqual(rep_contra["faithfulness"], 0.0)
        self.assertFalse(rep_contra["sentences"][0]["supported"])

        # 3. Unrelated sentence: high neutral probability
        mock_ce.predict.return_value = np.array([[-1.0, -0.5, 3.5]])
        rep_neutral = judge_answer_faithfulness_nli(
            cross_encoder=mock_ce,
            question="What is high availability?",
            chunks=chunks,
            answer="French hollandaise sauce is an emulsion of egg yolks and butter.",
            entail_threshold=0.5,
        )
        self.assertFalse(rep_neutral["unscorable"])
        self.assertEqual(rep_neutral["faithfulness"], 0.0)
        self.assertFalse(rep_neutral["sentences"][0]["supported"])

    def test_nli_label_order_handling(self):
        """Verify dynamic label order resolution from model config (never hard-coded)."""
        # Config A: 0=contradiction, 1=entailment, 2=neutral
        cfg_a = Mock()
        cfg_a.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        map_a = get_nli_label_indices(cfg_a)
        self.assertEqual(map_a["entailment"], 1)
        self.assertEqual(map_a["contradiction"], 0)
        self.assertEqual(map_a["neutral"], 2)

        # Config B: 0=entailment, 1=neutral, 2=contradiction
        cfg_b = Mock()
        cfg_b.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        map_b = get_nli_label_indices(cfg_b)
        self.assertEqual(map_b["entailment"], 0)
        self.assertEqual(map_b["contradiction"], 2)
        self.assertEqual(map_b["neutral"], 1)

        # Verify NLI scorer respects Config B
        mock_ce = Mock()
        mock_ce.config = cfg_b
        mock_ce.tokenizer = None
        # High score at index 0 (entailment in Config B)
        mock_ce.predict.return_value = np.array([[3.0, -1.0, -2.0]])
        rep = judge_answer_faithfulness_nli(
            cross_encoder=mock_ce,
            question="Test Q",
            chunks=[{"content": "Source text"}],
            answer="Sentence text.",
            entail_threshold=0.5,
        )
        self.assertEqual(rep["faithfulness"], 1.0)
        self.assertTrue(rep["sentences"][0]["supported"])

    def test_cache_key_separation(self):
        """Verify resume/cache key separates runs by judge type, model, generator, prompt, pdf, n, seed."""
        key_nli = make_cache_key(
            judge_type="nli",
            judge_model="cross-encoder/nli-deberta-v3-small",
            gen_model="qwen2.5:3b",
            prompt_version="v1",
            pdf_name="book.pdf",
            n=20,
            seed=42,
        )
        key_llm = make_cache_key(
            judge_type="llm",
            judge_model="llama3.2:3b",
            gen_model="qwen2.5:3b",
            prompt_version="v1",
            pdf_name="book.pdf",
            n=20,
            seed=42,
        )
        self.assertNotEqual(key_nli, key_llm)

        # Test resume cache rejection on mismatch
        mock_gen = Mock()
        mock_gen.generate_answer.return_value = "Answer 1"
        mock_ce = Mock()
        mock_ce.config = Mock(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
        mock_ce.tokenizer = None
        mock_ce.predict.return_value = np.array([[-1.0, 2.0, -1.0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "cache_test.json")
            # Write a fake cache file created by LLM judge
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "cache_key": key_llm,
                        "judge_type": "llm",
                        "judge_model": "llama3.2:3b",
                        "gen_model": "qwen2.5:3b",
                        "prompt_version": "v1",
                        "pdf_name": "book.pdf",
                        "n": 20,
                        "seed": 42,
                        "questions": {"q_old": {"faithfulness": 1.0, "applicable": True}},
                    },
                    f,
                )

            # Running with NLI judge must not resume mismatched cache
            summary = run_faithfulness_evaluation(
                golden_path="eval/golden_ci.json",
                pdf_path="eval/fixtures/ci_fixture.pdf",
                output_path=out_json,
                n=1,
                judge_type="nli",
                nli_model="cross-encoder/nli-deberta-v3-small",
                resume=True,
                cross_encoder=mock_ce,
                gen_client=mock_gen,
            )
            # The old LLM question should not be present in new results
            self.assertNotIn("q_old", summary["questions"])

    def test_build_premise_windows_does_not_cross_chunks(self):
        """Verify premise windows (1, 2, 3 sentences) never cross chunk boundaries."""
        chunks = [
            {"content": "Chunk one sentence A. Chunk one sentence B. Chunk one sentence C."},
            {"content": "Chunk two sentence X. Chunk two sentence Y."},
        ]
        windows = build_premise_windows(chunks, window_sizes=[1, 2, 3])

        # Every window must belong strictly to chunk 1 or chunk 2, never mixing both
        for w in windows:
            in_c1 = "Chunk one" in w
            in_c2 = "Chunk two" in w
            self.assertTrue(in_c1 != in_c2, f"Window crossed chunk boundary: {w}")

        # Chunk 1: 3 sentences -> w=1: 3, w=2: 2, w=3: 1 -> 6 windows
        # Chunk 2: 2 sentences -> w=1: 2, w=2: 1 -> 3 windows
        # Total = 9 windows
        self.assertEqual(len(windows), 9)

        # Check parsing of premise-window parameter
        self.assertEqual(parse_premise_windows("1, 2, 3"), [1, 2, 3])
        self.assertEqual(parse_premise_windows("2,1"), [1, 2])
        self.assertEqual(parse_premise_windows("invalid"), [1, 2, 3])

    def test_threshold_sweep_selection(self):
        """Verify calibration sweep selects the threshold satisfying pass rule with largest margin."""
        mock_ce = Mock()
        mock_ce.config = Mock(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
        mock_ce.tokenizer = None

        def side_effect(pairs, batch_size=32):
            results = []
            for premise, hyp in pairs:
                if "Quantum" in hyp:
                    results.append([2.0, -1.0, 0.0])
                else:
                    results.append([0.0, 1.5, 0.0])
            return np.array(results)

        mock_ce.predict.side_effect = side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            dev_json = os.path.join(tmpdir, "dev_answers.json")
            heldout_json = os.path.join(tmpdir, "heldout_answers.json")
            out_json = os.path.join(tmpdir, "val_out.json")

            dev_data = [
                {
                    "id": "ref_1",
                    "question": "What is high availability?",
                    "answer": "Replication is the primary technique for high availability.",
                    "is_negative_control": False,
                },
                {
                    "id": "neg_1",
                    "question": "What is high availability?",
                    "answer": "Quantum failure sentence.",
                    "is_negative_control": True,
                },
            ]
            heldout_data = [
                {
                    "id": "href_1",
                    "question": "What is high availability?",
                    "answer": "Replication is the primary technique for high availability.",
                    "is_negative_control": False,
                },
                {
                    "id": "hneg_1",
                    "question": "What is high availability?",
                    "answer": "Quantum heldout failure sentence.",
                    "is_negative_control": True,
                },
            ]
            with open(dev_json, "w", encoding="utf-8") as f:
                json.dump(dev_data, f)
            with open(heldout_json, "w", encoding="utf-8") as f:
                json.dump(heldout_data, f)

            res = validate_judge(
                answers_path=dev_json,
                heldout_answers_path=heldout_json,
                pdf_path="eval/fixtures/ci_fixture.pdf",
                output_path=out_json,
                judge_type="nli",
                cross_encoder=mock_ce,
            )

            # Sweep should be populated with 13 thresholds (0.30 to 0.90 in 0.05 steps)
            self.assertEqual(len(res["calibration_sweep"]), 13)
            valid_sw = [s for s in res["calibration_sweep"] if s["passed"]]
            self.assertTrue(len(valid_sw) > 0)
            best_sw = max(valid_sw, key=lambda s: (s["margin"], s["threshold"]))
            self.assertEqual(res["entail_threshold"], best_sw["threshold"])
            self.assertEqual(res["status"], "VALID")

    def test_heldout_never_used_in_calibration(self):
        """Verify held-out items are strictly evaluated once and never included in threshold calibration sweep."""
        mock_ce = Mock()
        mock_ce.config = Mock(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
        mock_ce.tokenizer = None

        def side_effect(pairs, batch_size=32):
            results = []
            for premise, hyp in pairs:
                if "Bogus" in hyp:
                    results.append([3.0, -2.0, -1.0])
                else:
                    results.append([-1.0, 3.0, -1.0])
            return np.array(results)

        mock_ce.predict.side_effect = side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            dev_json = os.path.join(tmpdir, "dev_answers.json")
            heldout_json = os.path.join(tmpdir, "heldout_answers.json")
            out_json = os.path.join(tmpdir, "val_out.json")

            dev_data = [
                {
                    "id": "dev_ref",
                    "question": "What is high availability?",
                    "answer": "Replication is the primary technique for high availability.",
                    "is_negative_control": False,
                },
                {
                    "id": "dev_neg",
                    "question": "What is high availability?",
                    "answer": "Bogus fail.",
                    "is_negative_control": True,
                },
            ]
            heldout_data = [
                {
                    "id": "heldout_ref",
                    "question": "What is high availability?",
                    "answer": "Replication is the primary technique for high availability.",
                    "is_negative_control": False,
                },
                {
                    "id": "heldout_neg",
                    "question": "What is high availability?",
                    "answer": "Bogus fail.",
                    "is_negative_control": True,
                },
            ]
            with open(dev_json, "w", encoding="utf-8") as f:
                json.dump(dev_data, f)
            with open(heldout_json, "w", encoding="utf-8") as f:
                json.dump(heldout_data, f)

            res = validate_judge(
                answers_path=dev_json,
                heldout_answers_path=heldout_json,
                pdf_path="eval/fixtures/ci_fixture.pdf",
                output_path=out_json,
                judge_type="nli",
                cross_encoder=mock_ce,
            )

            # Both dev and heldout sections must exist in results
            self.assertIn("dev", res)
            self.assertIn("heldout", res)

            # Dev section contains only dev IDs
            dev_ids = [
                r["id"]
                for r in res["dev"]["reference_results"] + res["dev"]["negative_control_results"]
            ]
            self.assertIn("dev_ref", dev_ids)
            self.assertIn("dev_neg", dev_ids)
            self.assertNotIn("heldout_ref", dev_ids)
            self.assertNotIn("heldout_neg", dev_ids)

            # Heldout section contains only heldout IDs
            heldout_ids = [
                r["id"]
                for r in res["heldout"]["reference_results"]
                + res["heldout"]["negative_control_results"]
            ]
            self.assertIn("heldout_ref", heldout_ids)
            self.assertIn("heldout_neg", heldout_ids)
            self.assertNotIn("dev_ref", heldout_ids)
            self.assertNotIn("dev_neg", heldout_ids)

    def test_validate_judge_verdicts(self):
        """Verify validate_judge pass rules: reference mean >= 0.80, neg controls <= 0.50, 0 unscorable references."""
        # 1. Successful validation (VALID)
        mock_ce_valid = Mock()
        mock_ce_valid.config = Mock(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
        mock_ce_valid.tokenizer = None

        def side_effect_valid(pairs, batch_size=32):
            results = []
            for premise, hyp in pairs:
                if "Quantum" in hyp or "interstellar" in hyp:
                    # Negative control -> contradiction
                    results.append([3.0, -2.0, -1.0])
                else:
                    # Reference -> entailment
                    results.append([-2.0, 3.0, -1.0])
            return np.array(results)

        mock_ce_valid.predict.side_effect = side_effect_valid

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "judge_val_valid.json")
            sample_answers = [
                {
                    "id": "ref_01",
                    "question": "What is high availability?",
                    "answer": "Replication is the primary technique for high availability.",
                    "is_negative_control": False,
                },
                {
                    "id": "neg_01",
                    "question": "What is high availability?",
                    "answer": "Quantum entanglement relays teleport packets across interstellar wormholes.",
                    "is_negative_control": True,
                },
            ]
            ans_file = os.path.join(tmpdir, "answers.json")
            with open(ans_file, "w", encoding="utf-8") as f:
                json.dump(sample_answers, f)

            res = validate_judge(
                answers_path=ans_file,
                heldout_answers_path=ans_file,
                pdf_path="eval/fixtures/ci_fixture.pdf",
                output_path=out_json,
                judge_type="nli",
                cross_encoder=mock_ce_valid,
            )
            self.assertEqual(res["status"], "VALID")
            self.assertTrue(res["passed"])
            self.assertEqual(res["unscorable_references_count"], 0)
            # Verify explicit error rates
            self.assertIn("reference_accept_rate", res["dev"])
            self.assertIn("negative_control_false_accept_rate", res["dev"])
            self.assertIn("negative_control_fully_accepted_rate", res["dev"])
            self.assertEqual(res["dev"]["reference_accept_rate"], 1.0)
            self.assertEqual(res["dev"]["negative_control_false_accept_rate"], 0.0)
            self.assertEqual(res["dev"]["negative_control_fully_accepted_rate"], 0.0)
            self.assertEqual(res["heldout"]["reference_accept_rate"], 1.0)
            self.assertEqual(res["heldout"]["negative_control_false_accept_rate"], 0.0)
            self.assertEqual(res["heldout"]["negative_control_fully_accepted_rate"], 0.0)

        # 2. Failed validation: reference coverage failure (unscorable reference)
        mock_judge_unscorable = Mock()
        mock_judge_unscorable.send_request.return_value = json.dumps({"claims": []})

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "judge_val_fail.json")
            ans_file = os.path.join(tmpdir, "answers.json")
            with open(ans_file, "w", encoding="utf-8") as f:
                json.dump(sample_answers, f)

            res_fail = validate_judge(
                answers_path=ans_file,
                heldout_answers_path=ans_file,
                pdf_path="eval/fixtures/ci_fixture.pdf",
                output_path=out_json,
                judge_type="llm",
                judge_client=mock_judge_unscorable,
            )
            self.assertEqual(res_fail["status"], "JUDGE NOT TRUSTWORTHY")
            self.assertFalse(res_fail["passed"])
            self.assertGreater(res_fail["unscorable_references_count"], 0)

    def test_faithfulness_refusal_and_abstention(self):
        """Verify refusal answers are marked applicable=False and faithfulness=None."""
        mock_gen = Mock()
        mock_gen.generate_answer.return_value = "INSUFFICIENT_CONTEXT"
        mock_ce = Mock()
        mock_ce.config = Mock(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
        mock_ce.tokenizer = None

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "faithfulness_refusal.json")

            summary = run_faithfulness_evaluation(
                golden_path="eval/golden_ci.json",
                pdf_path="eval/fixtures/ci_fixture.pdf",
                output_path=out_json,
                n=1,
                judge_type="nli",
                resume=False,
                cross_encoder=mock_ce,
                gen_client=mock_gen,
            )

            self.assertEqual(summary["mean_faithfulness"], "n/a")
            q_id = list(summary["questions"].keys())[0]
            rec = summary["questions"][q_id]
            self.assertFalse(rec["applicable"])
            self.assertIsNone(rec["faithfulness"])

    def test_report_header_and_unvalidated_judge_label(self):
        """Verify report outputs and saves judge validation status, error rates, and unvalidated label."""
        mock_gen = Mock()
        mock_gen.generate_answer.return_value = "Replication provides high availability."
        mock_ce = Mock()
        mock_ce.config = Mock(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
        mock_ce.tokenizer = None
        mock_ce.predict.return_value = np.array([[-1.0, 3.0, -1.0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "faithfulness_unvalidated.json")

            summary = run_faithfulness_evaluation(
                golden_path="eval/golden_ci.json",
                pdf_path="eval/fixtures/ci_fixture.pdf",
                output_path=out_json,
                n=1,
                judge_type="nli",
                nli_model="mock-unvalidated-model",
                resume=False,
                cross_encoder=mock_ce,
                gen_client=mock_gen,
            )

            # Since no judge_validation_mock-unvalidated-model.json exists, verdict must be UNVALIDATED
            self.assertEqual(summary["judge_validation_verdict"], "UNVALIDATED")
            self.assertIn("heldout_reference_accept_rate", summary)
            self.assertIn("heldout_negative_control_false_accept_rate", summary)

            with open(out_json, encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["judge_validation_verdict"], "UNVALIDATED")


if __name__ == "__main__":
    unittest.main()
