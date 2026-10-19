"""CI Grounding Gate: Evaluate factual citation coverage and hallucination controls.

Usage:
    python eval/run_grounding_gate.py \
        --pdf eval/fixtures/ci_fixture.pdf \
        --answers eval/fixtures/ci_answers.json \
        --thresholds eval/thresholds.json
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.run_eval import extract_chunks_with_pages
from src.config import get_config
from src.rag.embeddings import EmbeddingGenerator
from src.rag.grounding import grounding_report
from src.rag.retriever import DocumentRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_grounding_gate(
    pdf_path: str,
    answers_path: str,
    thresholds_path: str,
    grounding_threshold: float = 0.45,
) -> bool:
    """Run post-retrieval grounding checks across reference answers and negative controls.
    
    Returns:
        True if all threshold assertions pass, False otherwise.
    """
    try:
        config = get_config(require_telegram=False)
    except Exception:
        from src.config import Config
        config = Config(
            telegram_bot_token="",
            embedding_model="all-MiniLM-L6-v2",
            top_k_results=int(os.getenv("TOP_K_RESULTS", "2")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.3")),
            retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid"),
        )

    # 1. Load thresholds
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    min_mean_coverage = thresholds.get("min_grounding_mean_coverage", 0.85)
    max_neg_coverage = thresholds.get("max_grounding_negative_control_coverage", 0.40)

    # 2. Ingest fixture
    logger.info("Extracting and chunking fixture: %s", pdf_path)
    chunks = extract_chunks_with_pages(pdf_path)

    # 3. Embed chunks in batch
    embedder = EmbeddingGenerator(model_name=config.embedding_model)
    logger.info("Generating embeddings for %d chunks...", len(chunks))
    contents = [c.get("content", "") for c in chunks]
    embeddings = embedder.generate_embeddings(contents)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    retriever = DocumentRetriever(
        top_k=config.top_k_results,
        similarity_threshold=config.similarity_threshold,
        retrieval_mode=getattr(config, "retrieval_mode", "hybrid"),
    )

    # 4. Load answers
    with open(answers_path, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)

    logger.info("Loaded %d test items from %s", len(items), answers_path)

    results_table = []
    reference_coverages = []
    negative_coverages = []
    gate_passed = True

    for item in items:
        q_id = item.get("id", "unknown")
        q_text = item.get("question", "")
        answer = item.get("answer", "")
        is_neg = item.get("is_negative_control", False)

        # Retrieve chunks for the question
        q_emb = embedder.generate_query_embedding(q_text)
        retrieved = retriever.retrieve(query=q_text, query_embedding=q_emb, chunks=chunks)

        # Compute grounding report
        report = grounding_report(
            answer=answer,
            chunks=retrieved if retrieved else chunks,
            embedder=embedder,
            threshold=grounding_threshold,
        )

        item_status = "PASS"
        if is_neg:
            negative_coverages.append(report.citation_coverage)
            if report.citation_coverage > max_neg_coverage:
                item_status = "FAIL (NEGATIVE CONTROL EXCEEDED)"
                gate_passed = False
        else:
            if report.applicable:
                reference_coverages.append(report.citation_coverage)
                if report.citation_coverage < 0.70:
                    item_status = "FAIL (LOW COVERAGE)"
                    gate_passed = False
            else:
                # Valid refusal
                item_status = "PASS (REFUSAL)"

        results_table.append(
            {
                "id": q_id,
                "type": "NEG_CTRL" if is_neg else ("REFUSAL" if not report.applicable else "REFERENCE"),
                "coverage": f"{report.citation_coverage:.2f}",
                "supported": f"{report.supported_sentences}/{report.total_sentences}",
                "status": item_status,
                "preview": answer[:50] + ("..." if len(answer) > 50 else ""),
            }
        )

    # Print summary table
    print("\n" + "=" * 95)
    print(f"{'ID':<8} | {'Type':<10} | {'Coverage':<9} | {'Supported':<10} | {'Status':<30} | {'Answer Preview'}")
    print("-" * 95)
    for r in results_table:
        print(f"{r['id']:<8} | {r['type']:<10} | {r['coverage']:<9} | {r['supported']:<10} | {r['status']:<30} | {r['preview']}")
    print("=" * 95)

    mean_ref_coverage = float(np.mean(reference_coverages)) if reference_coverages else 0.0
    max_observed_neg = float(np.max(negative_coverages)) if negative_coverages else 0.0

    print(f"\nGrounding Gate Metrics:")
    print(f"  Mean Reference Coverage:     {mean_ref_coverage:.3f} (Required >= {min_mean_coverage:.3f})")
    print(f"  Max Negative Control Score:  {max_observed_neg:.3f} (Required <= {max_neg_coverage:.3f})")

    if mean_ref_coverage < min_mean_coverage:
        logger.error(f"FAIL: Mean reference coverage {mean_ref_coverage:.3f} < threshold {min_mean_coverage:.3f}")
        gate_passed = False

    if max_observed_neg > max_neg_coverage:
        logger.error(f"FAIL: Negative control score {max_observed_neg:.3f} > threshold {max_neg_coverage:.3f}")
        gate_passed = False

    if gate_passed:
        print("\n[PASS] GROUNDING GATE PASSED: All reference answers grounded and hallucinations rejected.")
    else:
        print("\n[FAIL] GROUNDING GATE FAILED.")

    return gate_passed


def main():
    parser = argparse.ArgumentParser(description="Run CI grounding gate.")
    parser.add_argument("--pdf", type=str, default="eval/fixtures/ci_fixture.pdf", help="Path to PDF fixture")
    parser.add_argument("--answers", type=str, default="eval/fixtures/ci_answers.json", help="Path to answers JSON")
    parser.add_argument("--thresholds", type=str, default="eval/thresholds.json", help="Path to thresholds JSON")
    parser.add_argument("--threshold", type=float, default=0.45, help="Sentence grounding cosine similarity threshold")
    args = parser.parse_args()

    passed = run_grounding_gate(
        pdf_path=args.pdf,
        answers_path=args.answers,
        thresholds_path=args.thresholds,
        grounding_threshold=args.threshold,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
