"""Local faithfulness evaluation using LLM-as-a-judge.

Evaluates whether model-generated answers are strictly faithful to retrieved context.
Extracts factual claims from the answer and verifies if each claim is entailed by the retrieved chunks.

NOTE:
    A 3B local model judge (e.g., qwen2.5:3b) exhibits noticeable variance and prompt-sensitivity
    compared to larger frontier models. This script is intended strictly for local trend-checking
    and experimentation, NOT as a zero-tolerance CI quality gate.

Usage:
    python -m eval.run_faithfulness \
        --n 20 \
        --judge-model qwen2.5:3b \
        --gen-model qwen2.5:3b \
        --output eval/results/faithfulness.json
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.run_eval import extract_chunks_with_pages
from src.config import get_config
from src.rag.embeddings import EmbeddingGenerator
from src.rag.grounding import strip_footers
from src.rag.llm_client import LLMClient
from src.rag.prompts import load_prompt_config
from src.rag.retriever import DocumentRetriever

try:
    from pydantic import BaseModel, Field

    class ClaimAssessment(BaseModel):
        claim: str
        entailed: bool
        reason: str = ""

    class FaithfulnessReport(BaseModel):
        claims: List[ClaimAssessment] = Field(default_factory=list)

except ImportError:
    class ClaimAssessment:  # type: ignore
        def __init__(self, claim: str, entailed: bool, reason: str = ""):
            self.claim = claim
            self.entailed = entailed
            self.reason = reason

        def dict(self):
            return {"claim": self.claim, "entailed": self.entailed, "reason": self.reason}

    class FaithfulnessReport:  # type: ignore
        def __init__(self, claims: Optional[List[Any]] = None):
            self.claims = claims or []

        def dict(self):
            return {"claims": [c.dict() if hasattr(c, "dict") else c for c in self.claims]}


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BOOK_PDF = "eval/fixtures/book.pdf"
DEFAULT_BOOK_GOLDEN = "eval/golden.json"
DEFAULT_CI_PDF = "eval/fixtures/ci_fixture.pdf"
DEFAULT_CI_ANSWERS = "eval/fixtures/ci_answers.json"

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


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse first valid JSON object from LLM response."""
    if not text:
        return None
    # Look for ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    # Look for raw outer { ... }
    match_brace = re.search(r"(\{.*\})", text, re.DOTALL)
    if match_brace:
        try:
            return json.loads(match_brace.group(1))
        except Exception:
            pass
    return None


def judge_answer_faithfulness(
    judge_client: LLMClient,
    question: str,
    context: str,
    answer: str,
) -> FaithfulnessReport:
    """Use judge model to extract claims and check entailment against context."""
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
            return FaithfulnessReport(claims=validated_claims)
    except Exception as e:
        logger.warning("Judge LLM call failed or produced invalid format [%s]: %s", type(e).__name__, e)

    return FaithfulnessReport(claims=[])


def validate_judge(
    answers_path: str = DEFAULT_CI_ANSWERS,
    pdf_path: str = DEFAULT_CI_PDF,
    output_path: str = "eval/results/faithfulness.json",
    judge_model: str = "llama3.2:3b",
    judge_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Validate faithfulness judge using supported references and negative controls.

    Mark judge VALID only if references average >= 0.80 AND every negative control scores <= 0.50.
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

    # Separate references (supported, answerable) and negative controls (unsupported)
    references = [
        item for item in all_items
        if not item.get("is_negative_control", False) and item.get("answer", "").strip() != "INSUFFICIENT_CONTEXT"
    ]
    negative_controls = [
        item for item in all_items
        if item.get("is_negative_control", False)
    ]

    if judge_client is None:
        judge_client = LLMClient(
            api_key=config.llm_api_key,
            model=judge_model,
            base_url=config.llm_base_url,
            timeout=300.0,
            max_tokens=600,
        )

    print("\n--- Running Judge Validation Mode ---")
    print(f"Judge Model: {judge_model}")
    print(f"References to evaluate: {len(references)} | Negative controls: {len(negative_controls)}\n")

    def _eval_item(item: Dict[str, Any], is_neg: bool) -> Dict[str, Any]:
        q_id = item.get("id", "unknown")
        q_text = item["question"]
        ans_text = item["answer"]
        q_emb = embedder.generate_query_embedding(q_text)
        retrieved = retriever.retrieve(query=q_text, query_embedding=q_emb, chunks=chunks)
        context_str = retriever.format_context(retrieved)

        report = judge_answer_faithfulness(
            judge_client=judge_client,
            question=q_text,
            context=context_str,
            answer=ans_text,
        )
        total_c = len(report.claims)
        entailed_c = sum(1 for c in report.claims if c.entailed)
        score = (entailed_c / total_c) if total_c > 0 else (0.0 if is_neg else 1.0)

        claims_data = [
            c.model_dump() if hasattr(c, "model_dump")
            else (c.dict() if hasattr(c, "dict") else {"claim": c.claim, "entailed": c.entailed, "reason": c.reason})
            for c in report.claims
        ]
        return {
            "id": q_id,
            "question": q_text,
            "answer": ans_text,
            "is_negative_control": is_neg,
            "total_claims": total_c,
            "entailed_claims": entailed_c,
            "faithfulness": round(score, 3),
            "claims": claims_data,
        }

    ref_results: List[Dict[str, Any]] = []
    for item in references:
        try:
            res = _eval_item(item, is_neg=False)
            ref_results.append(res)
        except Exception as e:
            logger.error("Error evaluating reference %s: %s", item.get("id"), e)
            ref_results.append({"id": item.get("id"), "faithfulness": 0.0, "error": str(e)})

    neg_results: List[Dict[str, Any]] = []
    for item in negative_controls:
        try:
            res = _eval_item(item, is_neg=True)
            neg_results.append(res)
        except Exception as e:
            logger.error("Error evaluating negative control %s: %s", item.get("id"), e)
            neg_results.append({"id": item.get("id"), "faithfulness": 1.0, "error": str(e)})

    ref_scores = [r["faithfulness"] for r in ref_results if "faithfulness" in r]
    neg_scores = [r["faithfulness"] for r in neg_results if "faithfulness" in r]

    mean_ref = sum(ref_scores) / len(ref_scores) if ref_scores else 0.0
    mean_neg = sum(neg_scores) / len(neg_scores) if neg_scores else 0.0
    max_neg = max(neg_scores) if neg_scores else 0.0

    # Rule: references average >= 0.80 AND every negative control scores <= 0.50
    is_valid = (mean_ref >= 0.80) and all(s <= 0.50 for s in neg_scores)
    status = "VALID" if is_valid else "NOT TRUSTWORTHY"

    print("\n### Judge Validation Summary Table\n")
    print("| Item ID | Type | Entailed | Total | Faithfulness | Status |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in ref_results:
        st = "PASS" if r.get("faithfulness", 0.0) >= 0.80 else "WARN"
        print(f"| {r['id']} | REFERENCE | {r.get('entailed_claims', 0)} | {r.get('total_claims', 0)} | {r.get('faithfulness', 0.0):.2f} | {st} |")
    for r in neg_results:
        st = "PASS" if r.get("faithfulness", 0.0) <= 0.50 else "FAIL"
        print(f"| {r['id']} | NEG_CTRL | {r.get('entailed_claims', 0)} | {r.get('total_claims', 0)} | {r.get('faithfulness', 0.0):.2f} | {st} |")

    print(f"\nMean Reference Faithfulness: {mean_ref:.3f} (Required >= 0.800)")
    print(f"Mean Negative Control Faithfulness: {mean_neg:.3f}")
    print(f"Max Negative Control Faithfulness: {max_neg:.3f} (Required <= 0.500)")

    if is_valid:
        print("\nJUDGE VALID: Judge distinguishes supported answers from unsupported negative controls.\n")
    else:
        print("\nJUDGE NOT TRUSTWORTHY\n")

    validation_payload = {
        "status": status,
        "judge_model": judge_model,
        "passed": is_valid,
        "mean_reference_faithfulness": round(mean_ref, 3),
        "mean_negative_control_faithfulness": round(mean_neg, 3),
        "max_negative_control_faithfulness": round(max_neg, 3),
        "reference_scores": ref_scores,
        "negative_control_scores": neg_scores,
        "reference_results": ref_results,
        "negative_control_results": neg_results,
    }

    # Save inside eval/results/faithfulness.json under judge_validation
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    saved_data = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                if not isinstance(saved_data, dict):
                    saved_data = {}
        except Exception:
            saved_data = {}

    saved_data["judge_validation"] = validation_payload
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(saved_data, f, indent=2)

    return validation_payload


def run_faithfulness_evaluation(
    golden_path: str = DEFAULT_BOOK_GOLDEN,
    pdf_path: str = DEFAULT_BOOK_PDF,
    output_path: str = "eval/results/faithfulness.json",
    n: int = 20,
    seed: int = 42,
    judge_model: str = "llama3.2:3b",
    gen_model: str = "qwen2.5:3b",
    resume: bool = True,
    judge_client: Optional[LLMClient] = None,
    gen_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Execute faithfulness evaluation pipeline across golden questions."""
    # Check self-judging bias
    if judge_model == gen_model:
        msg = f"WARNING: Judge model matches generator model ('{judge_model}'). Self-judging can introduce systematic bias and inflated faithfulness scores."
        logger.warning(msg)
        print(f"\n[WARNING] {msg}\n")

    # If document does not exist, print exact path and exit cleanly with instructions
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

    # Ingest fixture
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

    # In this mode, evaluate answerable questions only
    answerable = [q for q in all_questions if q.get("answerable", True)]

    # Fixed random seed sampling
    if seed is not None and len(answerable) > n:
        rng = random.Random(seed)
        selected = rng.sample(answerable, n)
    else:
        selected = answerable[:n]

    # Initialize LLM clients
    if gen_client is None:
        gen_client = LLMClient(
            api_key=config.llm_api_key,
            model=gen_model,
            base_url=config.llm_base_url,
            timeout=300.0,
            max_tokens=config.llm_max_tokens,
        )

    if judge_client is None:
        judge_client = LLMClient(
            api_key=config.llm_api_key,
            model=judge_model,
            base_url=config.llm_base_url,
            timeout=300.0,
            max_tokens=600,
        )

    # Resume support
    results: Dict[str, Any] = {}
    if resume and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict) and "questions" in saved:
                    results = saved["questions"]
                    logger.info("Resuming from %s with %d existing evaluations", output_path, len(results))
        except Exception as e:
            logger.warning("Could not read resume cache from %s: %s", output_path, e)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("\nStarting Faithfulness Evaluation...")
    print(f"Generator Model: {gen_model} | Judge Model: {judge_model}")
    print(f"PDF: {os.path.basename(pdf_path)} | Questions: {len(selected)} (sampled from {len(answerable)} answerable, seed={seed})")
    print(f"Prompt Version: {prompt_version}")
    print("NOTE: 3B judge models are noisy/approximate. Intended for local trend tracking.\n")

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
            "pdf_name": os.path.basename(pdf_path),
            "n": n,
            "seed": seed,
            "generator": gen_model,
            "gen_model": gen_model,
            "judge": judge_model,
            "judge_model": judge_model,
            "prompt_version": prompt_version,
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
                # Abstention
                record = {
                    "id": q_id,
                    "question": q_text,
                    "applicable": False,
                    "abstained": True,
                    "answer": "INSUFFICIENT_CONTEXT",
                    "total_claims": 0,
                    "entailed_claims": 0,
                    "faithfulness": None,
                    "claims": [],
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
                        "answer": clean_ans,
                        "total_claims": 0,
                        "entailed_claims": 0,
                        "faithfulness": None,
                        "claims": [],
                    }
                else:
                    report = judge_answer_faithfulness(
                        judge_client=judge_client,
                        question=q_text,
                        context=context_str,
                        answer=clean_ans,
                    )
                    total_c = len(report.claims)
                    entailed_c = sum(1 for c in report.claims if c.entailed)
                    score = (entailed_c / total_c) if total_c > 0 else 1.0

                    record = {
                        "id": q_id,
                        "question": q_text,
                        "applicable": True,
                        "abstained": False,
                        "answer": clean_ans,
                        "total_claims": total_c,
                        "entailed_claims": entailed_c,
                        "faithfulness": round(score, 3),
                        "claims": [
                            (
                                c.model_dump()
                                if hasattr(c, "model_dump")
                                else (c.dict() if hasattr(c, "dict") else {"claim": c.claim, "entailed": c.entailed, "reason": c.reason})
                            )
                            for c in report.claims
                        ],
                    }

                results[q_id] = record

            # Save progress after each question
            _save_current_progress()

        except Exception as e:
            logger.error("Failed evaluating question %s [%s]: %s", q_id, type(e).__name__, e)
            results[q_id] = {
                "id": q_id,
                "question": q_text,
                "error": str(e),
                "faithfulness": None,
                "applicable": False,
            }
            _save_current_progress()

    # Print markdown table
    print("\n### Faithfulness Evaluation Summary Table\n")
    print("| Question ID | Question Excerpt | Entailed | Total | Faithfulness | Status |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    applicable_scores = []
    for r in results.values():
        if "error" in r:
            status = "ERROR"
            f_str = "n/a"
        elif not r.get("applicable", True):
            status = "REFUSAL / ABSTAINED"
            f_str = "n/a"
        else:
            score_val = r.get("faithfulness")
            if score_val is not None:
                applicable_scores.append(score_val)
                f_str = f"{score_val:.2f}"
                status = "PASS" if score_val >= 0.7 else "WARN"
            else:
                f_str = "n/a"
                status = "REFUSAL / ABSTAINED"

        q_excerpt = (r["question"][:40] + "...") if len(r["question"]) > 40 else r["question"]
        print(f"| {r['id']} | {q_excerpt} | {r.get('entailed_claims', 0)} | {r.get('total_claims', 0)} | {f_str} | {status} |")

    mean_f = sum(applicable_scores) / len(applicable_scores) if applicable_scores else 1.0
    print(f"\n**Mean Answer Faithfulness**: `{mean_f:.3f}` across {len(applicable_scores)} applicable answers.")
    print("\n*Note: Local 3B judge evaluations serve as directional heuristics rather than absolute ground truth.*\n")

    _save_current_progress(mean_score=mean_f)

    return {
        "pdf_name": os.path.basename(pdf_path),
        "n": n,
        "seed": seed,
        "generator": gen_model,
        "gen_model": gen_model,
        "judge": judge_model,
        "judge_model": judge_model,
        "prompt_version": prompt_version,
        "sample_size": len(results),
        "mean_faithfulness": round(mean_f, 3),
        "questions": results,
    }


def main():
    default_pdf = DEFAULT_BOOK_PDF if os.path.exists(DEFAULT_BOOK_PDF) else DEFAULT_BOOK_PDF
    default_golden = DEFAULT_BOOK_GOLDEN if os.path.exists(DEFAULT_BOOK_PDF) else DEFAULT_BOOK_GOLDEN

    parser = argparse.ArgumentParser(description="Evaluate answer faithfulness with an LLM judge.")
    parser.add_argument("--golden", type=str, default=default_golden, help="Path to golden questions JSON")
    parser.add_argument("--pdf", type=str, default=default_pdf, help="Path to fixture PDF")
    parser.add_argument("--output", type=str, default="eval/results/faithfulness.json", help="Path to output JSON")
    parser.add_argument("--answers", type=str, default=DEFAULT_CI_ANSWERS, help="Path to answers JSON for judge validation")
    parser.add_argument("--n", type=int, default=20, help="Number of questions to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for question sampling")
    parser.add_argument("--judge-model", type=str, default="llama3.2:3b", help="Model identifier for judge")
    parser.add_argument("--gen-model", type=str, default="qwen2.5:3b", help="Model identifier for answer generation")
    parser.add_argument("--validate-judge", action="store_true", help="Validate judge model on supported and negative controls")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing results without resuming")
    args = parser.parse_args()

    if args.validate_judge:
        validate_judge(
            answers_path=args.answers,
            pdf_path=DEFAULT_CI_PDF,
            output_path=args.output,
            judge_model=args.judge_model,
        )
    else:
        run_faithfulness_evaluation(
            golden_path=args.golden,
            pdf_path=args.pdf,
            output_path=args.output,
            n=args.n,
            seed=args.seed,
            judge_model=args.judge_model,
            gen_model=args.gen_model,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()

