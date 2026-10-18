"""Benchmark LLM generation latency and refusal accuracy on CI fixture.

Usage:
    python -m eval.bench_llm --models qwen2.5:3b,llama3.2:3b
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.run_eval import extract_chunks_with_pages
from src.config import get_config
from src.rag.embeddings import EmbeddingGenerator
from src.rag.llm_client import LLMClient
from src.rag.retriever import DocumentRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def is_refusal_response(response: str) -> bool:
    """Check if the LLM response indicates refusal / insufficient context."""
    if not response:
        return True
    clean = response.strip().lower()
    if clean.startswith("insufficient_context") or "insufficient_context" in clean:
        return True
    refusal_patterns = [
        "context does not contain",
        "context doesn't contain",
        "context does not provide",
        "context doesn't provide",
        "not mentioned in the context",
        "not provided in the context",
        "not found in the context",
        "no information in the context",
        "does not contain the answer",
        "doesn't contain the answer",
    ]
    return any(p in clean for p in refusal_patterns)


def run_llm_benchmark(
    models: List[str],
    pdf_path: str = "eval/fixtures/ci_fixture.pdf",
    golden_path: str = "eval/golden_ci.json",
    output_path: str = "eval/results/llm_bench.json"
) -> Dict[str, Any]:
    """Run LLM latency and accuracy benchmark across specified models."""
    config = get_config()
    
    # 1. Ingest fixture
    logger.info("Extracting and chunking %s...", pdf_path)
    chunks = extract_chunks_with_pages(pdf_path)
    
    # 2. Embed chunks
    embedder = EmbeddingGenerator(model_name=config.embedding_model)
    logger.info("Generating embeddings for %d chunks...", len(chunks))
    for chunk in chunks:
        chunk["embedding"] = embedder.generate_embeddings(chunk["content"])
        
    retriever = DocumentRetriever(
        top_k=config.top_k_results,
        similarity_threshold=config.similarity_threshold,
        retrieval_mode="hybrid"
    )
    
    # 3. Load 12 questions (answerable + unanswerable)
    with open(golden_path, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
        
    answerable = [q for q in all_questions if q.get("answerable", True)]
    unanswerable = [q for q in all_questions if not q.get("answerable", True)]
    selected_questions = answerable[:9] + unanswerable[:3]
    logger.info(
        "Selected %d questions (%d answerable, %d unanswerable)",
        len(selected_questions),
        len([q for q in selected_questions if q.get("answerable", True)]),
        len([q for q in selected_questions if not q.get("answerable", True)])
    )
    
    results = {}
    
    for model_name in models:
        model_name = model_name.strip()
        if not model_name:
            continue
            
        logger.info("=" * 60)
        logger.info("Benchmarking model: %s", model_name)
        logger.info("=" * 60)
        
        client = LLMClient(
            api_key=config.llm_api_key,
            model=model_name,
            base_url=config.llm_base_url,
            timeout=300.0,
            max_tokens=config.llm_max_tokens
        )
        
        # Warm-up request
        logger.info("Sending warm-up request to %s...", model_name)
        try:
            client.send_request([{"role": "user", "content": "hi"}], max_tokens=1)
            logger.info("Warm-up complete.")
        except Exception as e:
            logger.warning("Warm-up failed: %s", e)
            
        latencies = []
        ans_answered = 0
        ans_total = 0
        unans_refused = 0
        unans_total = 0
        
        for q in selected_questions:
            q_text = q["question"]
            is_ans = q.get("answerable", True)
            if is_ans:
                ans_total += 1
            else:
                unans_total += 1
                
            # Retrieve context
            q_emb = embedder.generate_query_embedding(q_text)
            top_chunks = retriever.retrieve(query=q_text, query_embedding=q_emb, chunks=chunks)
            if not top_chunks:
                # If similarity threshold filtered chunks, provide top chunks anyway to test INSUFFICIENT_CONTEXT
                similarities = retriever._compute_dense_similarity(q_emb, chunks)
                top_chunks = [c for c, _ in sorted(zip(chunks, similarities), key=lambda x: x[1], reverse=True)[:config.top_k_results]]
                
            context_text = retriever.format_context(top_chunks)
            
            t0 = time.time()
            response = client.generate_answer(q_text, context_text)
            elapsed = time.time() - t0
            latencies.append(elapsed)
            
            refused = is_refusal_response(response)
            if is_ans:
                if not refused:
                    ans_answered += 1
            else:
                if refused:
                    unans_refused += 1
                    
            logger.info(
                "[%s] Q: '%s' -> %.2fs (Refused: %s)",
                model_name, q_text[:30], elapsed, refused
            )
            
        median_lat = float(np.median(latencies)) if latencies else 0.0
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
        ans_pct = float(100.0 * ans_answered / ans_total) if ans_total else 0.0
        unans_pct = float(100.0 * unans_refused / unans_total) if unans_total else 0.0
        
        results[model_name] = {
            "median_latency_s": round(median_lat, 2),
            "p95_latency_s": round(p95_lat, 2),
            "answerable_answered_pct": round(ans_pct, 1),
            "unanswerable_refused_pct": round(unans_pct, 1),
            "sample_size": len(selected_questions),
            "latencies_s": [round(x, 2) for x in latencies]
        }
        
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved benchmark results to %s", output_path)
    
    # Print formatted table
    print("\n" + "=" * 75)
    print(f"{'Model':<20} | {'Median (s)':<10} | {'P95 (s)':<10} | {'Answered %':<12} | {'Refused %':<12}")
    print("-" * 75)
    for model, m in results.items():
        print(f"{model:<20} | {m['median_latency_s']:<10.2f} | {m['p95_latency_s']:<10.2f} | {m['answerable_answered_pct']:<12.1f} | {m['unanswerable_refused_pct']:<12.1f}")
    print("=" * 75 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM latency and refusal on fixture questions.")
    parser.add_argument("--models", type=str, default="qwen2.5:3b,llama3.2:3b",
                        help="Comma-separated list of LLM model identifiers")
    parser.add_argument("--pdf", type=str, default="eval/fixtures/ci_fixture.pdf",
                        help="Path to PDF fixture")
    parser.add_argument("--golden", type=str, default="eval/golden_ci.json",
                        help="Path to golden questions JSON")
    parser.add_argument("--output", type=str, default="eval/results/llm_bench.json",
                        help="Path to save JSON benchmark output")
    args = parser.parse_args()
    
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    run_llm_benchmark(
        models=model_list,
        pdf_path=args.pdf,
        golden_path=args.golden,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
