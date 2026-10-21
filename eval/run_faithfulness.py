"""Local faithfulness evaluation using NLI CrossEncoder or LLM-as-a-judge.

Evaluates whether model-generated answers are strictly faithful to retrieved context.
- NLI Judge (default): splits answers into sentences, checks entailment against each
  retrieved chunk using a cross-encoder NLI model, and assesses factual support.
- LLM Judge: extracts factual claims and checks entailment via structured JSON prompt.

NOTE:
    A 3B local LLM judge exhibits noticeable variance and prompt-sensitivity.
    NLI CrossEncoders (e.g. cross-encoder/nli-deberta-v3-small) provide deterministic,
    sentence-level entailment probabilities.
"""

import argparse
from dataclasses import dataclass, field
import json
import logging
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.run_eval import extract_chunks_with_pages
from src.config import get_config
from src.rag.embeddings import EmbeddingGenerator
from src.rag.grounding import split_sentences, strip_footers
from src.rag.llm_client import LLMClient
from src.rag.prompts import load_prompt_config
from src.rag.retriever import DocumentRetriever

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

try:
    from pydantic import BaseModel, Field

    class ClaimAssessment(BaseModel):
        claim: str
        entailed: bool
        reason: str = ""

    class FaithfulnessReport(BaseModel):
        claims: List[ClaimAssessment] = Field(default_factory=list)
        unscorable: bool = False

except ImportError:
    class ClaimAssessment:  # type: ignore
        def __init__(self, claim: str, entailed: bool, reason: str = ""):
            self.claim = claim
            self.entailed = entailed
            self.reason = reason

        def dict(self):
            return {"claim": self.claim, "entailed": self.entailed, "reason": self.reason}

    class FaithfulnessReport:  # type: ignore
        def __init__(self, claims: Optional[List[Any]] = None, unscorable: bool = False):
            self.claims = claims or []
            self.unscorable = unscorable

        def dict(self):
            return {
                "claims": [c.dict() if hasattr(c, "dict") else c for c in self.claims],
                "unscorable": self.unscorable,
            }


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BOOK_PDF = "eval/fixtures/book.pdf"
DEFAULT_BOOK_GOLDEN = "eval/golden.json"
DEFAULT_CI_PDF = "eval/fixtures/ci_fixture.pdf"
DEFAULT_CI_ANSWERS = "eval/fixtures/ci_answers.json"
DEFAULT_HELDOUT_ANSWERS = "eval/fixtures/ci_answers_heldout.json"
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
DEFAULT_PREMISE_WINDOWS = "1,2,3"
DEFAULT_ENTAIL_THRESHOLD = 0.30

JUDGE_SYSTEM_PROMPT = """You are an objective evaluation judge assessing the factual faithfulness of an answer against provided source context.
Task:
1. Extract every distinct factual statement/claim made in the Answer.
2. For each claim, evaluate whether it is directly entailed (supported) by the Context. If the claim introduces outside facts, contradictions, or ungrounded claims, mark entailed as false.
3. Respond ONLY with a valid JSON object matching this schema:
{
  "claims": [
    {
      "claim": "text of claim",
      "entailed": true,
      "reason": "short explanation"
    }
  ]
}
Do not include any prose before or after the JSON block."""


def to_json_serializable(val: Any) -> Any:
    """Helper to ensure numpy types (e.g. np.bool_, np.floating) serialize cleanly to JSON."""
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return str(val)


def sanitize_filename(name: str) -> str:
    """Sanitize model or file string to be safe for filenames on all platforms."""
    return re.sub(r'[/\\:*"<>|]', "_", name)


def make_cache_key(
    judge_type: str,
    judge_model: str,
    gen_model: str,
    prompt_version: str,
    pdf_name: str,
    n: int,
    seed: int,
) -> str:
    """Generate deterministic resume/cache key so results from different configurations are never mixed."""
    return f"{judge_type}:{judge_model}:{gen_model}:{prompt_version}:{pdf_name}:{n}:{seed}"


def get_nli_label_indices(config: Any) -> Dict[str, int]:
    """Extract mapping from standard label names ('entailment', 'contradiction', 'neutral') to model output indices.

    Reads label order dynamically from model config (never hard-coded).
    """
    label_map: Dict[str, int] = {}
    id2label = getattr(config, "id2label", None) or {}
    for idx, lbl in id2label.items():
        lbl_lower = str(lbl).lower()
        if "entail" in lbl_lower:
            label_map["entailment"] = int(idx)
        elif "contra" in lbl_lower:
            label_map["contradiction"] = int(idx)
        elif "neut" in lbl_lower:
            label_map["neutral"] = int(idx)

    if "entailment" not in label_map or "contradiction" not in label_map:
        label2id = getattr(config, "label2id", None) or {}
        for lbl, idx in label2id.items():
            lbl_lower = str(lbl).lower()
            if "entail" in lbl_lower:
                label_map["entailment"] = int(idx)
            elif "contra" in lbl_lower:
                label_map["contradiction"] = int(idx)
            elif "neut" in lbl_lower:
                label_map["neutral"] = int(idx)

    # Defaults if config has no recognizable mappings (e.g. uninitialized mock)
    if "entailment" not in label_map:
        label_map["entailment"] = 1
    if "contradiction" not in label_map:
        label_map["contradiction"] = 0
    if "neutral" not in label_map:
        label_map["neutral"] = 2
    return label_map


def truncate_premise(tokenizer: Any, text: str, max_tokens: int = 400) -> str:
    """Truncate premise text safely so it fits within the model's sequence length."""
    if not text or not tokenizer:
        return text or ""
    try:
        tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens, add_special_tokens=False)
        return tokenizer.decode(tokens, skip_special_tokens=True)
    except Exception:
        return text[: max_tokens * 4]


def parse_premise_windows(window_str: str) -> List[int]:
    """Parse comma-separated window sizes (e.g. '1,2,3') into a sorted list of positive ints."""
    sizes = []
    for part in str(window_str).split(","):
        part = part.strip()
        if part.isdigit():
            val = int(part)
            if val > 0 and val not in sizes:
                sizes.append(val)
    return sorted(sizes) if sizes else [1, 2, 3]


def build_premise_windows(
    chunks: List[Any],
    window_sizes: Union[List[int], Tuple[int, ...]] = (1, 2, 3),
    tokenizer: Any = None,
    max_tokens: int = 400,
) -> List[str]:
    """Build sliding sentence windows within each chunk (never crossing chunk boundaries).

    For each chunk, sentences are split and sliding windows of length w in window_sizes
    are generated. Windows from different chunks are never combined.
    """
    windows: List[str] = []
    seen = set()
    for c in chunks:
        raw_text = c.get("content", "") if isinstance(c, dict) else str(c)
        raw_text = raw_text.strip()
        if not raw_text:
            continue
        norm_text = " ".join(raw_text.split())
        sents = split_sentences(norm_text)
        if not sents:
            if norm_text:
                trunc = truncate_premise(tokenizer, norm_text, max_tokens=max_tokens)
                if trunc not in seen:
                    seen.add(trunc)
                    windows.append(trunc)
            continue

        n_sents = len(sents)
        for w in window_sizes:
            if w <= 0:
                continue
            if n_sents >= w:
                for i in range(n_sents - w + 1):
                    win_text = " ".join(sents[i : i + w]).strip()
                    if win_text and win_text not in seen:
                        seen.add(win_text)
                        windows.append(truncate_premise(tokenizer, win_text, max_tokens=max_tokens))
            else:
                win_text = " ".join(sents).strip()
                if win_text and win_text not in seen:
                    seen.add(win_text)
                    windows.append(truncate_premise(tokenizer, win_text, max_tokens=max_tokens))

    return windows


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse first valid JSON object from LLM response."""
    if not text:
        return None
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    match_brace = re.search(r"(\{.*\})", text, re.DOTALL)
    if match_brace:
        try:
            return json.loads(match_brace.group(1))
        except Exception:
            pass
    return None


def judge_answer_faithfulness_llm(
    judge_client: LLMClient,
    question: str,
    context: str,
    answer: str,
) -> FaithfulnessReport:
    """Use LLM judge to extract claims and check entailment against context."""
    user_prompt = f"""Context:
{context}

Question:
{question}

Answer:
{answer}

Evaluate the claims in the Answer above:"""

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_resp = judge_client.send_request(messages, max_tokens=600)
        data = extract_json_block(raw_resp or "")
        if data and "claims" in data and isinstance(data["claims"], list):
            validated_claims = []
            for item in data["claims"]:
                if isinstance(item, dict) and "claim" in item and "entailed" in item:
                    validated_claims.append(
                        ClaimAssessment(
                            claim=str(item["claim"]),
                            entailed=bool(item["entailed"]),
                            reason=str(item.get("reason", "")),
                        )
                    )
            if not validated_claims:
                # 0 claims extracted -> unscorable
                return FaithfulnessReport(claims=[], unscorable=True)
            return FaithfulnessReport(claims=validated_claims, unscorable=False)
    except Exception as e:
        logger.warning("Judge LLM call failed or produced invalid format [%s]: %s", type(e).__name__, e)

    return FaithfulnessReport(claims=[], unscorable=True)


# Backwards compatibility alias
judge_answer_faithfulness = judge_answer_faithfulness_llm


def judge_answer_faithfulness_nli(
    cross_encoder: Any,
    question: str,
    chunks: List[Any],
    answer: str,
    entail_threshold: float = 0.5,
    premise_window_sizes: Union[List[int], Tuple[int, ...]] = (1, 2, 3),
) -> Dict[str, Any]:
    """Evaluate answer faithfulness using CrossEncoder NLI across sentence-level sliding premise windows."""
    clean_ans = strip_footers(answer)
    sentences = split_sentences(clean_ans)

    if not sentences:
        return {
            "unscorable": True,
            "faithfulness": None,
            "total_sentences": 0,
            "supported_sentences": 0,
            "sentences": [],
            "status": "UNSCORABLE",
        }

    if not chunks:
        return {
            "unscorable": False,
            "faithfulness": 0.0,
            "total_sentences": len(sentences),
            "supported_sentences": 0,
            "sentences": [
                {
                    "sentence": s,
                    "supported": False,
                    "best_window_index": -1,
                    "best_window": "",
                    "entailment_prob": 0.0,
                    "contradiction_prob": 1.0,
                    "reason": "No context chunks retrieved",
                }
                for s in sentences
            ],
            "status": "EVALUATED",
        }

    tokenizer = getattr(cross_encoder, "tokenizer", None)
    windows = build_premise_windows(
        chunks=chunks,
        window_sizes=premise_window_sizes,
        tokenizer=tokenizer,
        max_tokens=400,
    )

    if not windows:
        return {
            "unscorable": False,
            "faithfulness": 0.0,
            "total_sentences": len(sentences),
            "supported_sentences": 0,
            "sentences": [
                {
                    "sentence": s,
                    "supported": False,
                    "best_window_index": -1,
                    "best_window": "",
                    "entailment_prob": 0.0,
                    "contradiction_prob": 1.0,
                    "reason": "No premise windows generated",
                }
                for s in sentences
            ],
            "status": "EVALUATED",
        }

    model_config = getattr(cross_encoder, "config", None) or getattr(getattr(cross_encoder, "model", None), "config", None)
    label_map = get_nli_label_indices(model_config)
    entail_idx = label_map["entailment"]
    contra_idx = label_map["contradiction"]

    # Batch all pairs across all sentences: [premise, hypothesis]
    all_pairs = [[w, s] for s in sentences for w in windows]
    preds = cross_encoder.predict(all_pairs, batch_size=32)
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)

    exp_p = np.exp(preds - np.max(preds, axis=-1, keepdims=True))
    probs = exp_p / np.sum(exp_p, axis=-1, keepdims=True)

    assessments: List[Dict[str, Any]] = []
    supported_count = 0
    num_windows = len(windows)

    for i, s in enumerate(sentences):
        s_probs = probs[i * num_windows : (i + 1) * num_windows]
        best_win_idx = int(np.argmax(s_probs[:, entail_idx]))
        best_entail = float(s_probs[best_win_idx, entail_idx])
        best_contra = float(s_probs[best_win_idx, contra_idx])

        is_supported = (best_entail >= entail_threshold) and (best_entail > best_contra)
        if is_supported:
            supported_count += 1

        assessments.append(
            {
                "sentence": s,
                "supported": is_supported,
                "best_window_index": best_win_idx,
                "best_window": windows[best_win_idx],
                "entailment_prob": round(best_entail, 4),
                "contradiction_prob": round(best_contra, 4),
                "reason": f"Win {best_win_idx}: entail={best_entail:.3f}, contra={best_contra:.3f}",
            }
        )

    score = supported_count / len(sentences)
    return {
        "unscorable": False,
        "faithfulness": round(score, 3),
        "total_sentences": len(sentences),
        "supported_sentences": supported_count,
        "sentences": assessments,
        "status": "EVALUATED",
    }


def validate_judge(
    answers_path: str = DEFAULT_CI_ANSWERS,
    heldout_answers_path: str = DEFAULT_HELDOUT_ANSWERS,
    pdf_path: str = DEFAULT_CI_PDF,
    output_path: Optional[str] = None,
    judge_type: str = "nli",
    judge_model: str = "llama3.2:3b",
    nli_model: str = DEFAULT_NLI_MODEL,
    entail_threshold: Optional[float] = None,
    premise_window: str = DEFAULT_PREMISE_WINDOWS,
    judge_client: Optional[LLMClient] = None,
    cross_encoder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Validate faithfulness judge using supported references and negative controls.

    Pass Rule:
      1. References average faithfulness >= 0.80 on BOTH DEV and HELD-OUT sets
      2. Every negative control scores <= 0.50 on BOTH DEV and HELD-OUT sets
      3. Zero unscorable references (coverage requirement) on BOTH sets
    Otherwise prints 'JUDGE NOT TRUSTWORTHY'.
    """
    if not os.path.exists(pdf_path):
        abs_path = os.path.abspath(pdf_path)
        print(f"Error: Judge validation PDF not found at: {abs_path}")
        sys.exit(0)

    if not os.path.exists(answers_path):
        abs_path = os.path.abspath(answers_path)
        print(f"Error: Judge validation DEV answers not found at: {abs_path}")
        sys.exit(0)

    try:
        config = get_config(require_telegram=False)
    except Exception:
        from src.config import Config
        config = Config(telegram_bot_token="", llm_base_url="http://localhost:11434/v1")

    logger.info("Extracting fixture for judge validation: %s", pdf_path)
    chunks = extract_chunks_with_pages(pdf_path)

    embedder = EmbeddingGenerator(model_name=config.embedding_model)
    contents = [c.get("content", "") for c in chunks]
    embeddings = embedder.generate_embeddings(contents)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    retriever = DocumentRetriever(
        top_k=config.top_k_results,
        similarity_threshold=config.similarity_threshold,
        retrieval_mode=getattr(config, "retrieval_mode", "hybrid"),
    )

    premise_window_sizes = parse_premise_windows(premise_window)
    active_model_name = nli_model if judge_type == "nli" else judge_model

    # Determine per-judge output file path
    if output_path is None or output_path == "eval/results/faithfulness.json":
        sanitized = sanitize_filename(active_model_name)
        val_output_path = f"eval/results/judge_validation_{sanitized}.json"
    else:
        val_output_path = output_path

    # Initialize judge engine
    if judge_type == "nli":
        if cross_encoder is None:
            if CrossEncoder is None:
                raise ImportError("sentence-transformers is required for NLI judge. Install sentence-transformers.")
            logger.info("Loading NLI CrossEncoder: %s", nli_model)
            cross_encoder = CrossEncoder(nli_model)
    else:
        if judge_client is None:
            judge_client = LLMClient(
                api_key=config.llm_api_key,
                model=judge_model,
                base_url=config.llm_base_url,
                timeout=300.0,
                max_tokens=600,
            )

    def _eval_items(items: List[Dict[str, Any]], current_threshold: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        refs_eval = []
        negs_eval = []
        for item in items:
            q_id = item.get("id", "unknown")
            q_text = item["question"]
            ans_text = item["answer"]
            is_neg = item.get("is_negative_control", False)

            if ans_text.strip() == "INSUFFICIENT_CONTEXT":
                continue

            q_emb = embedder.generate_query_embedding(q_text)
            retrieved = retriever.retrieve(query=q_text, query_embedding=q_emb, chunks=chunks)

            if judge_type == "nli":
                report = judge_answer_faithfulness_nli(
                    cross_encoder=cross_encoder,
                    question=q_text,
                    chunks=retrieved,
                    answer=ans_text,
                    entail_threshold=current_threshold,
                    premise_window_sizes=premise_window_sizes,
                )
                rec = {
                    "id": q_id,
                    "question": q_text,
                    "answer": ans_text,
                    "is_negative_control": is_neg,
                    "unscorable": report["unscorable"],
                    "total_items": report["total_sentences"],
                    "supported_items": report["supported_sentences"],
                    "faithfulness": report["faithfulness"],
                    "details": report["sentences"],
                }
            else:
                context_str = retriever.format_context(retrieved)
                report_llm = judge_answer_faithfulness_llm(
                    judge_client=judge_client,
                    question=q_text,
                    context=context_str,
                    answer=ans_text,
                )
                total_c = len(report_llm.claims)
                entailed_c = sum(1 for c in report_llm.claims if c.entailed)
                if report_llm.unscorable or total_c == 0:
                    rec = {
                        "id": q_id,
                        "question": q_text,
                        "answer": ans_text,
                        "is_negative_control": is_neg,
                        "unscorable": True,
                        "total_items": 0,
                        "supported_items": 0,
                        "faithfulness": None,
                        "details": [],
                    }
                else:
                    score = entailed_c / total_c
                    claims_data = [
                        c.model_dump() if hasattr(c, "model_dump")
                        else (c.dict() if hasattr(c, "dict") else {"claim": c.claim, "entailed": c.entailed, "reason": c.reason})
                        for c in report_llm.claims
                    ]
                    rec = {
                        "id": q_id,
                        "question": q_text,
                        "answer": ans_text,
                        "is_negative_control": is_neg,
                        "unscorable": False,
                        "total_items": total_c,
                        "supported_items": entailed_c,
                        "faithfulness": round(score, 3),
                        "details": claims_data,
                    }

            if is_neg:
                negs_eval.append(rec)
            else:
                refs_eval.append(rec)
        return refs_eval, negs_eval

    # Load DEV items
    with open(answers_path, "r", encoding="utf-8") as f:
        dev_items = json.load(f)

    # Initial threshold to evaluate DEV raw stats
    initial_threshold = entail_threshold if entail_threshold is not None else DEFAULT_ENTAIL_THRESHOLD
    print(f"\n--- Running Judge Validation Mode ---")
    print(f"Judge Type: {judge_type.upper()} | Model: {active_model_name} | Premise Windows: {premise_window}")
    print(f"DEV Set ({answers_path}): {len(dev_items)} items")

    dev_refs, dev_negs = _eval_items(dev_items, initial_threshold)

    # Threshold calibration on DEV items only (if NLI)
    calibration_sweep: List[Dict[str, Any]] = []
    chosen_threshold = initial_threshold

    if judge_type == "nli":
        all_dev_results = dev_refs + dev_negs
        threshold_range = [round(t, 2) for t in np.arange(0.30, 0.901, 0.05)]
        for t_cand in threshold_range:
            t_ref_scores = []
            t_neg_scores = []
            t_unscorable_ref = 0
            for r in all_dev_results:
                if r.get("unscorable"):
                    if not r.get("is_negative_control"):
                        t_unscorable_ref += 1
                    continue
                sentences = r.get("details", [])
                if not sentences:
                    continue
                sup_cnt = sum(
                    1 for s in sentences
                    if s.get("entailment_prob", 0.0) >= t_cand and s.get("entailment_prob", 0.0) > s.get("contradiction_prob", 0.0)
                )
                score = sup_cnt / len(sentences)
                if r.get("is_negative_control"):
                    t_neg_scores.append(score)
                else:
                    t_ref_scores.append(score)

            m_ref = sum(t_ref_scores) / len(t_ref_scores) if t_ref_scores else 0.0
            m_neg = max(t_neg_scores) if t_neg_scores else 0.0
            p_ok = (m_ref >= 0.80) and (m_neg <= 0.50) and (t_unscorable_ref == 0)
            margin = min(m_ref - 0.80, 0.50 - m_neg)
            calibration_sweep.append({
                "threshold": t_cand,
                "mean_reference": round(m_ref, 3),
                "max_negative_control": round(m_neg, 3),
                "margin": round(margin, 3),
                "passed": p_ok,
            })

        print("\n### Threshold Calibration Sweep (DEV Only)\n")
        print("| Threshold | Reference Mean | Max Neg Ctrl | Margin | Status |")
        print("| :--- | :--- | :--- | :--- | :--- |")
        for sw in calibration_sweep:
            st = "PASS" if sw["passed"] else "FAIL"
            print(f"| {sw['threshold']:.2f} | {sw['mean_reference']:.3f} | {sw['max_negative_control']:.3f} | {sw['margin']:+.3f} | {st} |")

        valid_sweeps = [sw for sw in calibration_sweep if sw["passed"]]
        if valid_sweeps:
            best_sweep = max(valid_sweeps, key=lambda sw: (sw["margin"], sw["threshold"]))
            best_calibrated = best_sweep["threshold"]
            print(f"\nOptimal Calibrated Threshold on DEV: {best_calibrated:.2f} (margin: {best_sweep['margin']:+.3f})")
        else:
            best_sweep = max(calibration_sweep, key=lambda sw: sw["margin"])
            best_calibrated = best_sweep["threshold"]
            print(f"\nNo threshold met pass rule on DEV. Best candidate threshold: {best_calibrated:.2f} (margin: {best_sweep['margin']:+.3f})")

        chosen_threshold = entail_threshold if entail_threshold is not None else best_calibrated

        # Re-score DEV items at chosen_threshold if different from initial
        if chosen_threshold != initial_threshold:
            for r in all_dev_results:
                if not r.get("unscorable") and r.get("details"):
                    sup_cnt = 0
                    for s in r["details"]:
                        e_p = s.get("entailment_prob", 0.0)
                        c_p = s.get("contradiction_prob", 0.0)
                        s_ok = (e_p >= chosen_threshold) and (e_p > c_p)
                        s["supported"] = s_ok
                        if s_ok:
                            sup_cnt += 1
                    r["supported_items"] = sup_cnt
                    r["faithfulness"] = round(sup_cnt / len(r["details"]), 3)

    # Evaluate HELD-OUT set ONCE with fixed chosen_threshold
    heldout_refs: List[Dict[str, Any]] = []
    heldout_negs: List[Dict[str, Any]] = []
    has_heldout = os.path.exists(heldout_answers_path)
    if has_heldout:
        with open(heldout_answers_path, "r", encoding="utf-8") as f:
            heldout_items = json.load(f)
        print(f"\nEvaluating HELD-OUT Set ({heldout_answers_path}): {len(heldout_items)} items with fixed threshold {chosen_threshold:.2f}...")
        heldout_refs, heldout_negs = _eval_items(heldout_items, chosen_threshold)
    else:
        logger.warning("Held-out answers file not found at %s", heldout_answers_path)

    # Helper to summarize a set
    def _summarize_set(name: str, refs: List[Dict[str, Any]], negs: List[Dict[str, Any]]) -> Dict[str, Any]:
        scorable_ref = [r for r in refs if not r.get("unscorable") and r.get("faithfulness") is not None]
        scorable_neg = [r for r in negs if not r.get("unscorable") and r.get("faithfulness") is not None]
        unscorable_ref_cnt = sum(1 for r in refs if r.get("unscorable"))
        unscorable_neg_cnt = sum(1 for r in negs if r.get("unscorable"))
        ref_sc = [r["faithfulness"] for r in scorable_ref]
        neg_sc = [r["faithfulness"] for r in scorable_neg]
        m_ref = sum(ref_sc) / len(ref_sc) if ref_sc else 0.0
        m_neg = sum(neg_sc) / len(neg_sc) if neg_sc else 0.0
        max_neg = max(neg_sc) if neg_sc else 0.0
        p_ok = (m_ref >= 0.80) and all(s <= 0.50 for s in neg_sc) and (unscorable_ref_cnt == 0)

        # Explicit judge error rates
        ref_accept_cnt = sum(1 for r in scorable_ref if r["faithfulness"] >= 0.80)
        ref_accept_rate = round(ref_accept_cnt / len(scorable_ref), 3) if scorable_ref else 0.0

        neg_false_accept_cnt = sum(1 for r in scorable_neg if r["faithfulness"] > 0.50)
        neg_false_accept_rate = round(neg_false_accept_cnt / len(scorable_neg), 3) if scorable_neg else 0.0

        neg_fully_accepted_cnt = sum(1 for r in scorable_neg if r["faithfulness"] >= 1.0 - 1e-6)
        neg_fully_accepted_rate = round(neg_fully_accepted_cnt / len(scorable_neg), 3) if scorable_neg else 0.0

        print(f"\n### {name} Summary Table (Threshold: {chosen_threshold:.2f})\n")
        print("| Item ID | Type | Supported | Total | Faithfulness | Status |")
        print("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in refs:
            if r.get("unscorable"):
                f_str, st = "n/a", "UNSCORABLE (COVERAGE FAIL)"
            else:
                f_str = f"{r['faithfulness']:.2f}"
                st = "PASS" if r["faithfulness"] >= 0.80 else "FAIL"
            print(f"| {r['id']} | REFERENCE | {r.get('supported_items', 0)} | {r.get('total_items', 0)} | {f_str} | {st} |")

        for r in negs:
            if r.get("unscorable"):
                f_str, st = "n/a", "UNSCORABLE"
            else:
                f_str = f"{r['faithfulness']:.2f}"
                st = "PASS" if r["faithfulness"] <= 0.50 else "FAIL"
            print(f"| {r['id']} | NEG_CTRL | {r.get('supported_items', 0)} | {r.get('total_items', 0)} | {f_str} | {st} |")

        print(f"\n{name} Metrics:")
        print(f"  Mean Reference Faithfulness:         {m_ref:.3f} (Required >= 0.800)")
        print(f"  Mean Negative Control Faithfulness:  {m_neg:.3f}")
        print(f"  Max Negative Control Faithfulness:   {max_neg:.3f} (Required <= 0.500)")
        print(f"  Reference Accept Rate:               {ref_accept_rate:.3f} ({ref_accept_cnt}/{len(scorable_ref)})")
        print(f"  Negative Control False-Accept Rate:  {neg_false_accept_rate:.3f} ({neg_false_accept_cnt}/{len(scorable_neg)})")
        print(f"  Negative Control Fully-Accepted Rate:{neg_fully_accepted_rate:.3f} ({neg_fully_accepted_count}/{len(scorable_neg)})" if 'neg_fully_accepted_count' in dir() else f"  Negative Control Fully-Accepted Rate:{neg_fully_accepted_rate:.3f} ({neg_fully_accepted_cnt}/{len(scorable_neg)})")
        print(f"  Unscorable References:               {unscorable_ref_cnt} (Required == 0)")
        print(f"  Unscorable Negative Controls:        {unscorable_neg_cnt}")
        print(f"  Set Status:                          {'PASS' if p_ok else 'FAIL'}")

        return {
            "passed": p_ok,
            "mean_reference_faithfulness": round(m_ref, 3),
            "mean_negative_control_faithfulness": round(m_neg, 3),
            "max_negative_control_faithfulness": round(max_neg, 3),
            "reference_accept_rate": ref_accept_rate,
            "negative_control_false_accept_rate": neg_false_accept_rate,
            "negative_control_fully_accepted_rate": neg_fully_accepted_rate,
            "unscorable_references_count": unscorable_ref_cnt,
            "unscorable_negative_controls_count": unscorable_neg_cnt,
            "reference_scores": ref_sc,
            "negative_control_scores": neg_sc,
            "reference_results": refs,
            "negative_control_results": negs,
        }

    dev_summary = _summarize_set("DEV SET", dev_refs, dev_negs)
    heldout_summary = _summarize_set("HELD-OUT SET", heldout_refs, heldout_negs) if has_heldout else {"passed": True}

    is_overall_valid = bool(dev_summary["passed"] and heldout_summary["passed"])
    overall_status = "VALID" if is_overall_valid else "JUDGE NOT TRUSTWORTHY"

    # Requirement 5: If it fails, print which items fail on which windows and stop
    if not is_overall_valid:
        print("\n================================================================================")
        print("[FAIL] JUDGE NOT TRUSTWORTHY: Detailed Failure Diagnostic")
        print("================================================================================")
        for set_label, s_data in [("DEV", dev_summary), ("HELD-OUT", heldout_summary)]:
            all_set_items = s_data.get("reference_results", []) + s_data.get("negative_control_results", [])
            for itm in all_set_items:
                is_nc = bool(itm.get("is_negative_control", False))
                f_val = itm.get("faithfulness")
                unsc = bool(itm.get("unscorable", False))
                item_failed = unsc or (not is_nc and (f_val is None or f_val < 0.80)) or (is_nc and f_val is not None and f_val > 0.50)
                if item_failed:
                    print(f"\nItem [{itm['id']}] ({set_label} | {'NEG_CTRL' if is_nc else 'REFERENCE'}): faithfulness={f_val} unscorable={unsc}")
                    print(f"  Question: {itm.get('question')}")
                    print(f"  Answer:   {itm.get('answer')}")
                    for s_i, s_entry in enumerate(itm.get("details", [])):
                        print(f"    Sentence {s_i+1}: \"{s_entry.get('sentence')}\"")
                        print(f"      Supported: {s_entry.get('supported')} (entail={s_entry.get('entailment_prob')}, contra={s_entry.get('contradiction_prob')})")
                        print(f"      Best Window [{s_entry.get('best_window_index')}]: \"{s_entry.get('best_window')}\"")
        print("================================================================================\n")
    else:
        print(f"\n[PASS] VALID: Judge distinguishes supported answers from unsupported negative controls across both DEV and HELD-OUT sets.\n")

    val_payload = {
        "status": overall_status,
        "judge_type": judge_type,
        "judge_model": active_model_name,
        "passed": bool(is_overall_valid),
        "entail_threshold": round(chosen_threshold, 2),
        "premise_window": premise_window,
        "dev": dev_summary,
        "heldout": heldout_summary,
        "calibration_sweep": calibration_sweep,
        # Backward-compatible top-level fields
        "mean_reference_faithfulness": dev_summary["mean_reference_faithfulness"],
        "mean_negative_control_faithfulness": dev_summary["mean_negative_control_faithfulness"],
        "max_negative_control_faithfulness": dev_summary["max_negative_control_faithfulness"],
        "reference_accept_rate": dev_summary["reference_accept_rate"],
        "negative_control_false_accept_rate": dev_summary["negative_control_false_accept_rate"],
        "negative_control_fully_accepted_rate": dev_summary["negative_control_fully_accepted_rate"],
        "unscorable_references_count": dev_summary["unscorable_references_count"],
        "unscorable_negative_controls_count": dev_summary["unscorable_negative_controls_count"],
        "reference_scores": dev_summary["reference_scores"],
        "negative_control_scores": dev_summary["negative_control_scores"],
        "reference_results": dev_summary["reference_results"],
        "negative_control_results": dev_summary["negative_control_results"],
    }

    out_dir = os.path.dirname(val_output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(val_output_path, "w", encoding="utf-8") as f:
        json.dump(val_payload, f, indent=2, default=to_json_serializable)
    logger.info("Saved judge validation results to: %s", val_output_path)

    return val_payload


def run_faithfulness_evaluation(
    golden_path: str = DEFAULT_BOOK_GOLDEN,
    pdf_path: str = DEFAULT_BOOK_PDF,
    output_path: str = "eval/results/faithfulness.json",
    n: int = 20,
    seed: int = 42,
    judge_type: str = "nli",
    judge_model: str = "llama3.2:3b",
    nli_model: str = DEFAULT_NLI_MODEL,
    entail_threshold: float = DEFAULT_ENTAIL_THRESHOLD,
    premise_window: str = DEFAULT_PREMISE_WINDOWS,
    gen_model: str = "qwen2.5:3b",
    resume: bool = True,
    judge_client: Optional[LLMClient] = None,
    cross_encoder: Optional[Any] = None,
    gen_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Execute faithfulness evaluation pipeline across golden questions."""
    premise_window_sizes = parse_premise_windows(premise_window)
    active_judge_name = nli_model if judge_type == "nli" else judge_model

    # Self-judging bias warning
    if judge_type == "llm" and judge_model == gen_model:
        msg = f"WARNING: Judge model matches generator model ('{judge_model}'). Self-judging introduces systematic bias and inflated faithfulness scores."
        logger.warning(msg)
        print(f"\n[WARNING] {msg}\n")

    if not os.path.exists(pdf_path):
        abs_path = os.path.abspath(pdf_path)
        print(f"Error: Evaluation document not found at: {abs_path}")
        print(f"Please place 'Think Python 2e' PDF at '{abs_path}' or specify --pdf and --golden explicitly (e.g. --pdf eval/fixtures/ci_fixture.pdf --golden eval/golden_ci.json).")
        sys.exit(0)

    if not os.path.exists(golden_path):
        abs_path = os.path.abspath(golden_path)
        print(f"Error: Golden questions file not found at: {abs_path}")
        print(f"Please place golden questions at '{abs_path}' or specify --golden explicitly.")
        sys.exit(0)

    try:
        config = get_config(require_telegram=False)
    except Exception:
        from src.config import Config
        config = Config(telegram_bot_token="", llm_base_url="http://localhost:11434/v1")

    try:
        p_ver = getattr(config, "prompt_version", "v1")
        prompt_cfg = load_prompt_config(p_ver)
        prompt_version = prompt_cfg.version
    except Exception:
        prompt_version = getattr(config, "prompt_version", "v1")

    # Ingest document
    logger.info("Extracting document: %s", pdf_path)
    chunks = extract_chunks_with_pages(pdf_path)

    embedder = EmbeddingGenerator(model_name=config.embedding_model)
    contents = [c.get("content", "") for c in chunks]
    embeddings = embedder.generate_embeddings(contents)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    retriever = DocumentRetriever(
        top_k=config.top_k_results,
        similarity_threshold=config.similarity_threshold,
        retrieval_mode=getattr(config, "retrieval_mode", "hybrid"),
    )

    with open(golden_path, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    # Evaluate answerable questions only
    answerable = [q for q in all_questions if q.get("answerable", True)]

    # Deterministic sampling with seed
    if seed is not None and len(answerable) > n:
        rng = random.Random(seed)
        selected = rng.sample(answerable, n)
    else:
        selected = answerable[:n]

    # Initialize generator client
    if gen_client is None:
        gen_client = LLMClient(
            api_key=config.llm_api_key,
            model=gen_model,
            base_url=config.llm_base_url,
            timeout=300.0,
            max_tokens=config.llm_max_tokens,
        )

    # Initialize judge engine
    if judge_type == "nli":
        if cross_encoder is None:
            if CrossEncoder is None:
                raise ImportError("sentence-transformers is required for NLI judge. Install sentence-transformers.")
            logger.info("Loading NLI CrossEncoder: %s", nli_model)
            cross_encoder = CrossEncoder(nli_model)
    else:
        if judge_client is None:
            judge_client = LLMClient(
                api_key=config.llm_api_key,
                model=judge_model,
                base_url=config.llm_base_url,
                timeout=300.0,
                max_tokens=600,
            )

    pdf_base = os.path.basename(pdf_path)
    current_cache_key = make_cache_key(
        judge_type=judge_type,
        judge_model=active_judge_name,
        gen_model=gen_model,
        prompt_version=prompt_version,
        pdf_name=pdf_base,
        n=n,
        seed=seed,
    )

    results: Dict[str, Any] = {}
    if resume and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict) and "questions" in saved:
                    saved_key = saved.get("cache_key")
                    # Check cache key components to ensure results from different judges are never mixed
                    key_matches = (saved_key == current_cache_key) or (
                        saved.get("judge_type") == judge_type
                        and (saved.get("judge") == active_judge_name or saved.get("judge_model") == active_judge_name)
                        and (saved.get("generator") == gen_model or saved.get("gen_model") == gen_model)
                        and saved.get("prompt_version") == prompt_version
                        and saved.get("pdf_name") == pdf_base
                        and saved.get("n") == n
                        and saved.get("seed") == seed
                    )
                    if key_matches:
                        results = saved["questions"]
                        logger.info("Resuming from %s with %d matching cached evaluations", output_path, len(results))
                    else:
                        logger.info(
                            "Cache mismatch in %s (cached key: '%s', current key: '%s'). Starting fresh.",
                            output_path,
                            saved_key,
                            current_cache_key,
                        )
        except Exception as e:
            logger.warning("Could not read resume cache from %s: %s", output_path, e)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sanitized_judge = active_judge_name.replace("/", "_").replace(":", "_")
    val_file_path = f"eval/results/judge_validation_{sanitized_judge}.json"
    judge_val_verdict = "UNVALIDATED"
    heldout_accept_rate = None
    heldout_false_accept_rate = None

    if os.path.exists(val_file_path):
        try:
            with open(val_file_path, "r", encoding="utf-8") as f:
                val_data = json.load(f)
                judge_val_verdict = val_data.get("status", "UNVALIDATED")
                heldout_sec = val_data.get("heldout", {})
                heldout_accept_rate = heldout_sec.get("reference_accept_rate")
                heldout_false_accept_rate = heldout_sec.get("negative_control_false_accept_rate")
        except Exception as e:
            logger.warning("Could not read judge validation file %s: %s", val_file_path, e)

    print("\nStarting Faithfulness Evaluation...")
    print(f"Judge Type: {judge_type.upper()} | Judge Model: {active_judge_name}")
    print(f"Validation Verdict: {judge_val_verdict}")
    if heldout_accept_rate is not None:
        print(f"HELD-OUT Accept Rate: {heldout_accept_rate:.3f} | HELD-OUT False-Accept Rate: {heldout_false_accept_rate:.3f}")
    print(f"Generator Model: {gen_model} | Prompt Version: {prompt_version}")
    print(f"PDF: {pdf_base} | Questions: {len(selected)} (sampled from {len(answerable)} answerable, seed={seed})")

    def _save_current_progress(mean_score: Optional[float] = None):
        existing_data = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, dict):
                        existing_data = {}
            except Exception:
                existing_data = {}

        summary_payload = {
            "cache_key": current_cache_key,
            "judge_type": judge_type,
            "judge_model": active_judge_name,
            "judge": active_judge_name,
            "judge_validation_verdict": judge_val_verdict,
            "heldout_reference_accept_rate": heldout_accept_rate,
            "heldout_negative_control_false_accept_rate": heldout_false_accept_rate,
            "gen_model": gen_model,
            "generator": gen_model,
            "prompt_version": prompt_version,
            "pdf_name": pdf_base,
            "n": n,
            "seed": seed,
            "sample_size": len(results),
            "questions": results,
        }
        if mean_score is not None:
            summary_payload["mean_faithfulness"] = round(mean_score, 3)

        existing_data.update(summary_payload)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)

    for idx, q in enumerate(selected):
        q_id = q.get("id", f"q_{idx}")
        q_text = q["question"]

        if q_id in results:
            logger.info("[%d/%d] Skipping cached question: %s", idx + 1, len(selected), q_id)
            continue

        logger.info("[%d/%d] Evaluating %s: '%s'", idx + 1, len(selected), q_id, q_text[:40])
        try:
            q_emb = embedder.generate_query_embedding(q_text)
            retrieved = retriever.retrieve(query=q_text, query_embedding=q_emb, chunks=chunks)

            if not retrieved:
                record = {
                    "id": q_id,
                    "question": q_text,
                    "applicable": False,
                    "abstained": True,
                    "unscorable": False,
                    "answer": "INSUFFICIENT_CONTEXT",
                    "total_items": 0,
                    "supported_items": 0,
                    "faithfulness": None,
                    "details": [],
                }
                results[q_id] = record
            else:
                context_str = retriever.format_context(retrieved)
                raw_ans = gen_client.generate_answer(q_text, context_str) or ""
                clean_ans = strip_footers(raw_ans)

                if "insufficient_context" in clean_ans.lower() or not clean_ans.strip():
                    record = {
                        "id": q_id,
                        "question": q_text,
                        "applicable": False,
                        "abstained": False,
                        "unscorable": False,
                        "answer": clean_ans,
                        "total_items": 0,
                        "supported_items": 0,
                        "faithfulness": None,
                        "details": [],
                    }
                else:
                    if judge_type == "nli":
                        rep_nli = judge_answer_faithfulness_nli(
                            cross_encoder=cross_encoder,
                            question=q_text,
                            chunks=retrieved,
                            answer=clean_ans,
                            entail_threshold=entail_threshold,
                            premise_window_sizes=premise_window_sizes,
                        )
                        record = {
                            "id": q_id,
                            "question": q_text,
                            "applicable": True,
                            "abstained": False,
                            "unscorable": rep_nli["unscorable"],
                            "answer": clean_ans,
                            "total_items": rep_nli["total_sentences"],
                            "supported_items": rep_nli["supported_sentences"],
                            "faithfulness": rep_nli["faithfulness"],
                            "details": rep_nli["sentences"],
                        }
                    else:
                        rep_llm = judge_answer_faithfulness_llm(
                            judge_client=judge_client,
                            question=q_text,
                            context=context_str,
                            answer=clean_ans,
                        )
                        total_c = len(rep_llm.claims)
                        entailed_c = sum(1 for c in rep_llm.claims if c.entailed)
                        is_unscorable = rep_llm.unscorable or total_c == 0
                        score = (entailed_c / total_c) if (total_c > 0 and not is_unscorable) else None

                        record = {
                            "id": q_id,
                            "question": q_text,
                            "applicable": True,
                            "abstained": False,
                            "unscorable": is_unscorable,
                            "answer": clean_ans,
                            "total_items": total_c,
                            "supported_items": entailed_c,
                            "faithfulness": round(score, 3) if score is not None else None,
                            "details": [
                                (
                                    c.model_dump()
                                    if hasattr(c, "model_dump")
                                    else (c.dict() if hasattr(c, "dict") else {"claim": c.claim, "entailed": c.entailed, "reason": c.reason})
                                )
                                for c in rep_llm.claims
                            ],
                        }

                results[q_id] = record
            _save_current_progress()

        except Exception as e:
            logger.error("Failed evaluating question %s [%s]: %s", q_id, type(e).__name__, e)
            results[q_id] = {
                "id": q_id,
                "question": q_text,
                "error": str(e),
                "faithfulness": None,
                "applicable": False,
                "unscorable": True,
            }
            _save_current_progress()

    # Summary table
    print("\n### Faithfulness Evaluation Summary Table\n")
    print(f"**Judge Model**: `{active_judge_name}`")
    print(f"**Judge Validation Verdict**: `{judge_val_verdict}`")
    h_acc_str = f"{heldout_accept_rate:.3f}" if heldout_accept_rate is not None else "n/a"
    h_fa_str = f"{heldout_false_accept_rate:.3f}" if heldout_false_accept_rate is not None else "n/a"
    print(f"**HELD-OUT Accept Rate**: `{h_acc_str}`")
    print(f"**HELD-OUT False-Accept Rate**: `{h_fa_str}`\n")
    if judge_val_verdict != "VALID":
        print("UNVALIDATED JUDGE: treat scores as directional\n")

    print("| Question ID | Question Excerpt | Supported | Total | Faithfulness | Status |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    applicable_scores: List[float] = []
    unscorable_count = 0

    for r in results.values():
        if "error" in r:
            status_str = "ERROR"
            f_str = "n/a"
        elif not r.get("applicable", True):
            status_str = "REFUSAL / ABSTAINED"
            f_str = "n/a"
        elif r.get("unscorable", False):
            status_str = "UNSCORABLE"
            f_str = "n/a"
            unscorable_count += 1
        else:
            score_val = r.get("faithfulness")
            if score_val is not None:
                applicable_scores.append(score_val)
                f_str = f"{score_val:.2f}"
                status_str = "PASS" if score_val >= 0.7 else "WARN"
            else:
                f_str = "n/a"
                status_str = "UNSCORABLE"
                unscorable_count += 1

        q_excerpt = (r["question"][:40] + "...") if len(r["question"]) > 40 else r["question"]
        print(f"| {r['id']} | {q_excerpt} | {r.get('supported_items', 0)} | {r.get('total_items', 0)} | {f_str} | {status_str} |")

    mean_f = sum(applicable_scores) / len(applicable_scores) if applicable_scores else 1.0
    print(f"\n**Mean Answer Faithfulness**: `{mean_f:.3f}` across {len(applicable_scores)} applicable answers.")
    if unscorable_count > 0:
        print(f"**Unscorable Answers Excluded from Mean**: {unscorable_count}")

    _save_current_progress(mean_score=mean_f)

    return {
        "cache_key": current_cache_key,
        "judge_type": judge_type,
        "judge_model": active_judge_name,
        "judge": active_judge_name,
        "judge_validation_verdict": judge_val_verdict,
        "heldout_reference_accept_rate": heldout_accept_rate,
        "heldout_negative_control_false_accept_rate": heldout_false_accept_rate,
        "gen_model": gen_model,
        "generator": gen_model,
        "prompt_version": prompt_version,
        "pdf_name": pdf_base,
        "n": n,
        "seed": seed,
        "sample_size": len(results),
        "mean_faithfulness": round(mean_f, 3),
        "unscorable_count": unscorable_count,
        "questions": results,
    }


def main():
    default_pdf = DEFAULT_BOOK_PDF if os.path.exists(DEFAULT_BOOK_PDF) else DEFAULT_BOOK_PDF
    default_golden = DEFAULT_BOOK_GOLDEN if os.path.exists(DEFAULT_BOOK_PDF) else DEFAULT_BOOK_GOLDEN

    parser = argparse.ArgumentParser(description="Evaluate answer faithfulness with NLI or LLM judge.")
    parser.add_argument("--judge-type", type=str, choices=["nli", "llm"], default="nli", help="Judge architecture (nli or llm)")
    parser.add_argument("--nli-model", type=str, default=DEFAULT_NLI_MODEL, help="NLI model name (e.g. cross-encoder/nli-deberta-v3-small, cross-encoder/nli-MiniLM2-L6-H768)")
    parser.add_argument("--entail-threshold", type=float, default=None, help="Entailment probability threshold for sentence support (default calibrated on dev: 0.30)")
    parser.add_argument("--premise-window", type=str, default=DEFAULT_PREMISE_WINDOWS, help="Sliding window sizes of consecutive sentences within chunks (default '1,2,3')")
    parser.add_argument("--judge-model", type=str, default="llama3.2:3b", help="Model identifier for LLM judge")
    parser.add_argument("--gen-model", type=str, default="qwen2.5:3b", help="Model identifier for answer generation")
    parser.add_argument("--golden", type=str, default=default_golden, help="Path to golden questions JSON")
    parser.add_argument("--pdf", type=str, default=default_pdf, help="Path to fixture PDF")
    parser.add_argument("--output", type=str, default="eval/results/faithfulness.json", help="Path to output JSON")
    parser.add_argument("--answers", type=str, default=DEFAULT_CI_ANSWERS, help="Path to answers JSON for judge validation")
    parser.add_argument("--heldout-answers", type=str, default=DEFAULT_HELDOUT_ANSWERS, help="Path to held-out answers JSON for judge validation")
    parser.add_argument("--n", type=int, default=20, help="Number of questions to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for question sampling")
    parser.add_argument("--validate-judge", action="store_true", help="Validate judge model on supported and negative controls")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing results without resuming")
    args = parser.parse_args()

    if args.validate_judge:
        validate_judge(
            answers_path=args.answers,
            heldout_answers_path=args.heldout_answers,
            pdf_path=DEFAULT_CI_PDF,
            output_path=args.output,
            judge_type=args.judge_type,
            judge_model=args.judge_model,
            nli_model=args.nli_model,
            entail_threshold=args.entail_threshold,
            premise_window=args.premise_window,
        )
    else:
        run_faithfulness_evaluation(
            golden_path=args.golden,
            pdf_path=args.pdf,
            output_path=args.output,
            n=args.n,
            seed=args.seed,
            judge_type=args.judge_type,
            judge_model=args.judge_model,
            nli_model=args.nli_model,
            entail_threshold=args.entail_threshold or DEFAULT_ENTAIL_THRESHOLD,
            premise_window=args.premise_window,
            gen_model=args.gen_model,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
