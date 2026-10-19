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


def run_faithfulness_evaluation(
    golden_path: str = "eval/golden_ci.json",
    pdf_path: str = "eval/fixtures/ci_fixture.pdf",
    output_path: str = "eval/results/faithfulness.json",
    n: int = 20,
    judge_model: str = "qwen2.5:3b",
    gen_model: str = "qwen2.5:3b",
    resume: bool = True,
    judge_client: Optional[LLMClient] = None,
    gen_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Execute faithfulness evaluation pipeline across golden questions."""
    try:
        config = get_config(require_telegram=False)
    except Exception:
        from src.config import Config
        config = Config(telegram_bot_token="", llm_base_url="http://localhost:11434/v1")

    # Ingest fixture
    logger.info("Extracting fixture: %s", pdf_path)
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

    selected = all_questions[:n]

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
    print("NOTE: 3B judge models are noisy/approximate. Intended for local trend tracking.\n")

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
                    "faithfulness": 1.0,
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
                        "faithfulness": 1.0,
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
            summary_payload = {
                "gen_model": gen_model,
                "judge_model": judge_model,
                "sample_size": len(results),
                "questions": results,
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2)

        except Exception as e:
            logger.error("Failed evaluating question %s [%s]: %s", q_id, type(e).__name__, e)
            results[q_id] = {
                "id": q_id,
                "question": q_text,
                "error": str(e),
                "faithfulness": 0.0,
                "applicable": False,
            }

    # Print markdown table
    print("\n### Faithfulness Evaluation Summary Table\n")
    print(f"| Question ID | Question Excerpt | Entailed | Total | Faithfulness | Status |")
    print(f"| :--- | :--- | :--- | :--- | :--- | :--- |")
    applicable_scores = []
    for r in results.values():
        if "error" in r:
            status = "ERROR"
        elif not r.get("applicable", True):
            status = "REFUSAL / ABSTAINED"
        else:
            status = "PASS" if r.get("faithfulness", 0.0) >= 0.7 else "WARN"
            applicable_scores.append(r.get("faithfulness", 0.0))

        q_excerpt = (r["question"][:40] + "...") if len(r["question"]) > 40 else r["question"]
        print(f"| {r['id']} | {q_excerpt} | {r.get('entailed_claims', 0)} | {r.get('total_claims', 0)} | {r.get('faithfulness', 0.0):.2f} | {status} |")

    mean_f = sum(applicable_scores) / len(applicable_scores) if applicable_scores else 1.0
    print(f"\n**Mean Answer Faithfulness**: `{mean_f:.3f}` across {len(applicable_scores)} applicable answers.")
    print("\n*Note: Local 3B judge evaluations serve as directional heuristics rather than absolute ground truth.*\n")

    return {
        "gen_model": gen_model,
        "judge_model": judge_model,
        "mean_faithfulness": round(mean_f, 3),
        "questions": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate answer faithfulness with an LLM judge.")
    parser.add_argument("--golden", type=str, default="eval/golden_ci.json", help="Path to golden questions JSON")
    parser.add_argument("--pdf", type=str, default="eval/fixtures/ci_fixture.pdf", help="Path to fixture PDF")
    parser.add_argument("--output", type=str, default="eval/results/faithfulness.json", help="Path to output JSON")
    parser.add_argument("--n", type=int, default=20, help="Number of questions to evaluate")
    parser.add_argument("--judge-model", type=str, default="qwen2.5:3b", help="Model identifier for judge")
    parser.add_argument("--gen-model", type=str, default="qwen2.5:3b", help="Model identifier for answer generation")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing results without resuming")
    args = parser.parse_args()

    run_faithfulness_evaluation(
        golden_path=args.golden,
        pdf_path=args.pdf,
        output_path=args.output,
        n=args.n,
        judge_model=args.judge_model,
        gen_model=args.gen_model,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
