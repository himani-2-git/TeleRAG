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
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

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
) -> Dict[str, Any]:
    """Evaluate answer faithfulness using CrossEncoder NLI model across split sentences against retrieved chunks."""
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
                    "best_chunk_index": -1,
                    "entailment_prob": 0.0,
                    "contradiction_prob": 1.0,
                    "reason": "No context chunks retrieved",
                }
                for s in sentences
            ],
            "status": "EVALUATED",
        }

    tokenizer = getattr(cross_encoder, "tokenizer", None)
    chunk_texts = [
        truncate_premise(tokenizer, c.get("content", "") if isinstance(c, dict) else str(c), max_tokens=400)
        for c in chunks
    ]

    model_config = getattr(cross_encoder, "config", None) or getattr(getattr(cross_encoder, "model", None), "config", None)
    label_map = get_nli_label_indices(model_config)
    entail_idx = label_map["entailment"]
    contra_idx = label_map["contradiction"]

    assessments: List[Dict[str, Any]] = []
    supported_count = 0

    for s in sentences:
        pairs = [[c_text, s] for c_text in chunk_texts]
        preds = cross_encoder.predict(pairs)
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        if preds.ndim == 1:
            preds = preds.reshape(1, -1)

        exp_p = np.exp(preds - np.max(preds, axis=-1, keepdims=True))
        probs = exp_p / np.sum(exp_p, axis=-1, keepdims=True)

        best_idx = int(np.argmax(probs[:, entail_idx]))
        best_entail = float(probs[best_idx, entail_idx])
        best_contra = float(probs[best_idx, contra_idx])

        is_supported = (best_entail >= entail_threshold) and (best_entail > best_contra)
        if is_supported:
            supported_count += 1

        assessments.append(
            {
                "sentence": s,
                "supported": is_supported,
                "best_chunk_index": best_idx,
                "entailment_prob": round(best_entail, 4),
                "contradiction_prob": round(best_contra, 4),
                "reason": f"Chunk {best_idx}: entail={best_entail:.3f}, contra={best_contra:.3f}",
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
    pdf_path: str = DEFAULT_CI_PDF,
    output_path: Optional[str] = None,
    judge_type: str = "nli",
    judge_model: str = "llama3.2:3b",
    nli_model: str = DEFAULT_NLI_MODEL,
    entail_threshold: float = 0.5,
    judge_client: Optional[LLMClient] = None,
    cross_encoder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Validate faithfulness judge using supported references and negative controls.

    Pass Rule:
      1. References average faithfulness >= 0.80
      2. Every negative control scores <= 0.50
      3. Zero unscorable references (coverage requirement)
    Otherwise prints 'JUDGE NOT TRUSTWORTHY'.
    """
    if not os.path.exists(pdf_path):
        abs_path = os.path.abspath(pdf_path)
        print(f"Error: Judge validation PDF not found at: {abs_path}")
        sys.exit(0)

    if not os.path.exists(answers_path):
        abs_path = os.path.abspath(answers_path)
        print(f"Error: Judge validation answers not found at: {abs_path}")
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

    with open(answers_path, "r", encoding="utf-8") as f:
        all_items = json.load(f)

    references = [
        item for item in all_items
        if not item.get("is_negative_control", False) and item.get("answer", "").strip() != "INSUFFICIENT_CONTEXT"
    ]
    negative_controls = [
        item for item in all_items
        if item.get("is_negative_control", False)
    ]

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

    print("\n--- Running Judge Validation Mode ---")
    print(f"Judge Type: {judge_type.upper()} | Model: {active_model_name}")
    print(f"References to evaluate: {len(references)} | Negative controls: {len(negative_controls)}\n")

    def _eval_item(item: Dict[str, Any], is_neg: bool) -> Dict[str, Any]:
        q_id = item.get("id", "unknown")
        q_text = item["question"]
        ans_text = item["answer"]
        q_emb = embedder.generate_query_embedding(q_text)
        retrieved = retriever.retrieve(query=q_text, query_embedding=q_emb, chunks=chunks)

        if judge_type == "nli":
            report = judge_answer_faithfulness_nli(
                cross_encoder=cross_encoder,
                question=q_text,
                chunks=retrieved,
                answer=ans_text,
                entail_threshold=entail_threshold,
            )
            return {
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
                return {
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
            score = entailed_c / total_c
            claims_data = [
                c.model_dump() if hasattr(c, "model_dump")
                else (c.dict() if hasattr(c, "dict") else {"claim": c.claim, "entailed": c.entailed, "reason": c.reason})
                for c in report_llm.claims
            ]
            return {
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

    ref_results: List[Dict[str, Any]] = []
    for item in references:
        try:
            res = _eval_item(item, is_neg=False)
            ref_results.append(res)
        except Exception as e:
            logger.error("Error evaluating reference %s: %s", item.get("id"), e)
            ref_results.append({"id": item.get("id"), "faithfulness": None, "unscorable": True, "error": str(e)})

    neg_results: List[Dict[str, Any]] = []
    for item in negative_controls:
        try:
            res = _eval_item(item, is_neg=True)
            neg_results.append(res)
        except Exception as e:
            logger.error("Error evaluating negative control %s: %s", item.get("id"), e)
            neg_results.append({"id": item.get("id"), "faithfulness": None, "unscorable": True, "error": str(e)})

    # Filter out unscorable items from means
    scorable_ref = [r for r in ref_results if not r.get("unscorable") and r.get("faithfulness") is not None]
    scorable_neg = [r for r in neg_results if not r.get("unscorable") and r.get("faithfulness") is not None]
    unscorable_ref_count = sum(1 for r in ref_results if r.get("unscorable"))
    unscorable_neg_count = sum(1 for r in neg_results if r.get("unscorable"))

    ref_scores = [r["faithfulness"] for r in scorable_ref]
    neg_scores = [r["faithfulness"] for r in scorable_neg]

    mean_ref = sum(ref_scores) / len(ref_scores) if ref_scores else 0.0
    mean_neg = sum(neg_scores) / len(neg_scores) if neg_scores else 0.0
    max_neg = max(neg_scores) if neg_scores else 0.0

    # Pass Rule:
    # 1. references mean >= 0.80
    # 2. every negative control <= 0.50
    # 3. zero unscorable references (coverage failure)
    neg_controls_pass = all(s <= 0.50 for s in neg_scores) if neg_scores else True
    is_valid = (mean_ref >= 0.80) and neg_controls_pass and (unscorable_ref_count == 0)
    status = "VALID" if is_valid else "JUDGE NOT TRUSTWORTHY"

    print("\n### Judge Validation Summary Table\n")
    print("| Item ID | Type | Supported | Total | Faithfulness | Status |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in ref_results:
        if r.get("unscorable"):
            f_str = "n/a"
            st = "UNSCORABLE (COVERAGE FAIL)"
        else:
            f_str = f"{r['faithfulness']:.2f}"
            st = "PASS" if r["faithfulness"] >= 0.80 else "WARN"
        print(f"| {r['id']} | REFERENCE | {r.get('supported_items', 0)} | {r.get('total_items', 0)} | {f_str} | {st} |")

    for r in neg_results:
        if r.get("unscorable"):
            f_str = "n/a"
            st = "UNSCORABLE"
        else:
            f_str = f"{r['faithfulness']:.2f}"
            st = "PASS" if r["faithfulness"] <= 0.50 else "FAIL"
        print(f"| {r['id']} | NEG_CTRL | {r.get('supported_items', 0)} | {r.get('total_items', 0)} | {f_str} | {st} |")

    print(f"\nMean Reference Faithfulness: {mean_ref:.3f} (Required >= 0.800)")
    print(f"Mean Negative Control Faithfulness: {mean_neg:.3f}")
    print(f"Max Negative Control Faithfulness: {max_neg:.3f} (Required <= 0.500)")
    print(f"Unscorable References: {unscorable_ref_count} (Required == 0)")
    print(f"Unscorable Negative Controls: {unscorable_neg_count}")

    if unscorable_ref_count > 0:
        print(f"\nCOVERAGE FAILURE: {unscorable_ref_count} reference answer(s) had 0 checkable claims/sentences and were UNSCORABLE.")

    failed_negs = [r for r in scorable_neg if r["faithfulness"] > 0.50]
    if failed_negs:
        print("\nFailed Negative Controls (> 0.50):")
        for fn in failed_negs:
            print(f"  - {fn['id']}: faithfulness={fn['faithfulness']:.3f} | {fn['question']}")

    if is_valid:
        print("\nVALID: Judge distinguishes supported answers from unsupported negative controls.\n")
    else:
        print("\nJUDGE NOT TRUSTWORTHY\n")

    validation_payload = {
        "status": status,
        "judge_type": judge_type,
        "judge_model": active_model_name,
        "passed": is_valid,
        "mean_reference_faithfulness": round(mean_ref, 3),
        "mean_negative_control_faithfulness": round(mean_neg, 3),
        "max_negative_control_faithfulness": round(max_neg, 3),
        "unscorable_references_count": unscorable_ref_count,
        "unscorable_negative_controls_count": unscorable_neg_count,
        "reference_scores": ref_scores,
        "negative_control_scores": neg_scores,
        "failed_negative_controls": failed_negs,
        "reference_results": ref_results,
        "negative_control_results": neg_results,
    }

    out_dir = os.path.dirname(val_output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(val_output_path, "w", encoding="utf-8") as f:
        json.dump(validation_payload, f, indent=2)
    logger.info("Saved judge validation results to: %s", val_output_path)

    return validation_payload


def run_faithfulness_evaluation(
    golden_path: str = DEFAULT_BOOK_GOLDEN,
    pdf_path: str = DEFAULT_BOOK_PDF,
    output_path: str = "eval/results/faithfulness.json",
    n: int = 20,
    seed: int = 42,
    judge_type: str = "nli",
    judge_model: str = "llama3.2:3b",
    nli_model: str = DEFAULT_NLI_MODEL,
    entail_threshold: float = 0.5,
    gen_model: str = "qwen2.5:3b",
    resume: bool = True,
    judge_client: Optional[LLMClient] = None,
    cross_encoder: Optional[Any] = None,
    gen_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Execute faithfulness evaluation pipeline across golden questions."""
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

    print("\nStarting Faithfulness Evaluation...")
    print(f"Judge Type: {judge_type.upper()} | Judge Model: {active_judge_name}")
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
    parser.add_argument("--entail-threshold", type=float, default=0.5, help="Entailment probability threshold for sentence support (default 0.5)")
    parser.add_argument("--judge-model", type=str, default="llama3.2:3b", help="Model identifier for LLM judge")
    parser.add_argument("--gen-model", type=str, default="qwen2.5:3b", help="Model identifier for answer generation")
    parser.add_argument("--golden", type=str, default=default_golden, help="Path to golden questions JSON")
    parser.add_argument("--pdf", type=str, default=default_pdf, help="Path to fixture PDF")
    parser.add_argument("--output", type=str, default="eval/results/faithfulness.json", help="Path to output JSON")
    parser.add_argument("--answers", type=str, default=DEFAULT_CI_ANSWERS, help="Path to answers JSON for judge validation")
    parser.add_argument("--n", type=int, default=20, help="Number of questions to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for question sampling")
    parser.add_argument("--validate-judge", action="store_true", help="Validate judge model on supported and negative controls")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing results without resuming")
    args = parser.parse_args()

    if args.validate_judge:
        validate_judge(
            answers_path=args.answers,
            pdf_path=DEFAULT_CI_PDF,
            output_path=args.output,
            judge_type=args.judge_type,
            judge_model=args.judge_model,
            nli_model=args.nli_model,
            entail_threshold=args.entail_threshold,
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
            entail_threshold=args.entail_threshold,
            gen_model=args.gen_model,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
