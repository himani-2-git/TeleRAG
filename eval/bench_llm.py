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
from typing import Any

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


def print_benchmark_table(results: dict[str, Any]) -> None:
    """Print ASCII summary table of benchmark results."""
    print("\n" + "=" * 90)
    print(
        f"{'model':<20} | {'median latency':<16} | {'p95 latency':<13} | {'answerable answered %':<23} | {'unanswerable refused %':<24}"
    )
    print("-" * 90)
    for model, m in results.items():
        med = m.get("median_latency_s", m.get("median latency", 0.0))
        p95 = m.get("p95_latency_s", m.get("p95 latency", 0.0))
        ans = m.get("answerable_answered_pct", m.get("answerable answered %", 0.0))
        unans = m.get("unanswerable_refused_pct", m.get("unanswerable refused %", 0.0))
        med_str = f"{med:.2f}s"
        p95_str = f"{p95:.2f}s"
        ans_str = f"{ans:.1f}%"
        unans_str = f"{unans:.1f}%"
        print(f"{model:<20} | {med_str:<16} | {p95_str:<13} | {ans_str:<23} | {unans_str:<24}")
    print("=" * 90 + "\n")


def run_llm_benchmark(
    models: list[str],
    pdf_path: str = "eval/fixtures/ci_fixture.pdf",
    golden_path: str = "eval/golden_ci.json",
    output_path: str = "eval/results/llm_bench.json",
) -> dict[str, Any]:
    """Run LLM latency and accuracy benchmark across specified models."""
    try:
        config = get_config(require_telegram=False)
    except Exception:
        from src.config import Config

        config = Config(
            telegram_bot_token="",
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/"),
            llm_model=os.getenv("LLM_MODEL", "qwen2.5:3b"),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "300")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            top_k_results=int(os.getenv("TOP_K_RESULTS", "2")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.3")),
            retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid"),
        )

    # 1. Ingest fixture
    logger.info("Extracting and chunking %s...", pdf_path)
    chunks = extract_chunks_with_pages(pdf_path)

    # 2. Embed chunks in batch
    embedder = EmbeddingGenerator(model_name=config.embedding_model)
    logger.info("Generating embeddings for %d chunks...", len(chunks))
    contents = [c.get("content", "") for c in chunks]
    embeddings = embedder.generate_embeddings(contents)
    for chunk, emb in zip(chunks, embeddings, strict=False):
        chunk["embedding"] = emb

    retriever = DocumentRetriever(
        top_k=config.top_k_results,
        similarity_threshold=config.similarity_threshold,
        retrieval_mode=getattr(config, "retrieval_mode", "hybrid"),
    )

    # 3. Load 12 questions (9 answerable + 3 unanswerable)
    with open(golden_path, encoding="utf-8") as f:
        all_questions = json.load(f)

    answerable = [q for q in all_questions if q.get("answerable", True)]
    unanswerable = [q for q in all_questions if not q.get("answerable", True)]
    selected_questions = answerable[:9] + unanswerable[:3]
    logger.info(
        "Selected %d questions (%d answerable, %d unanswerable)",
        len(selected_questions),
        len([q for q in selected_questions if q.get("answerable", True)]),
        len([q for q in selected_questions if not q.get("answerable", True)]),
    )

    # Load existing results if output_path exists so partial results from earlier runs aren't lost
    results: dict[str, Any] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, encoding="utf-8") as f:
                results = json.load(f)
        except Exception as e:
            logger.warning("Could not load existing %s: %s", output_path, e)
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
            max_tokens=config.llm_max_tokens,
        )

        # Warm-up request
        logger.info("Sending warm-up request to %s...", model_name)
        try:
            client.send_request([{"role": "user", "content": "hi"}], max_tokens=1)
            logger.info("Warm-up complete.")
        except Exception as e:
            logger.warning("Warm-up failed: %s", e)

        latencies: list[float] = []
        ans_answered = 0
        ans_total = 0
        unans_refused = 0
        unans_total = 0
        failed_count = 0

        for q in selected_questions:
            q_text = q["question"]
            is_ans = q.get("answerable", True)
            if is_ans:
                ans_total += 1
            else:
                unans_total += 1

            try:
                # 1. Retrieve context using the exact same public path as the bot
                q_emb = embedder.generate_query_embedding(q_text)
                top_chunks = retriever.retrieve(query=q_text, query_embedding=q_emb, chunks=chunks)

                # If top dense score is below SIMILARITY_THRESHOLD, retriever returns [] (abstains)
                if not top_chunks:
                    elapsed = 0.0
                    latencies.append(elapsed)
                    refused = True
                    if not is_ans:
                        unans_refused += 1
                    logger.info(
                        "[%s] Q: '%s' -> %.2fs (Abstained: below threshold, refused=%s)",
                        model_name,
                        q_text[:30],
                        elapsed,
                        refused,
                    )
                    continue

                # 2. Chunks passed threshold: call LLM
                context_text = retriever.format_context(top_chunks)
                t0 = time.time()
                response = client.generate_answer(q_text, context_text)
                elapsed = time.time() - t0

                if response is None:
                    logger.warning(
                        "[%s] Q: '%s' -> LLM returned None (generation failure)",
                        model_name,
                        q_text[:30],
                    )
                    failed_count += 1
                    continue

                latencies.append(elapsed)
                refused = is_refusal_response(response)
                if is_ans:
                    if not refused:
                        ans_answered += 1
                else:
                    if refused:
                        unans_refused += 1

                logger.info(
                    "[%s] Q: '%s' -> %.2fs (Refused: %s)", model_name, q_text[:30], elapsed, refused
                )
            except Exception as e:
                logger.error("[%s] Error evaluating question '%s': %s", model_name, q_text[:30], e)
                failed_count += 1
                continue

        median_lat = float(np.median(latencies)) if latencies else 0.0
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
        ans_pct = float(100.0 * ans_answered / ans_total) if ans_total else 0.0
        unans_pct = float(100.0 * unans_refused / unans_total) if unans_total else 0.0

        model_result = {
            "model": model_name,
            "prompt_version": getattr(config, "prompt_version", "v1"),
            "median_latency_s": round(median_lat, 2),
            "p95_latency_s": round(p95_lat, 2),
            "answerable_answered_pct": round(ans_pct, 1),
            "unanswerable_refused_pct": round(unans_pct, 1),
            "median latency": round(median_lat, 2),
            "p95 latency": round(p95_lat, 2),
            "answerable answered %": round(ans_pct, 1),
            "unanswerable refused %": round(unans_pct, 1),
            "sample_size": len(selected_questions),
            "failed_count": failed_count,
            "latencies_s": [round(x, 2) for x in latencies],
        }
        results[model_name] = model_result

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved benchmark results after model '%s' to %s", model_name, output_path)

    # Print formatted table at the end
    print_benchmark_table(results)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM latency and refusal on fixture questions."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="qwen2.5:3b,llama3.2:3b",
        help="Comma-separated list of LLM model identifiers",
    )
    parser.add_argument(
        "--pdf", type=str, default="eval/fixtures/ci_fixture.pdf", help="Path to PDF fixture"
    )
    parser.add_argument(
        "--golden", type=str, default="eval/golden_ci.json", help="Path to golden questions JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/results/llm_bench.json",
        help="Path to save JSON benchmark output",
    )
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    run_llm_benchmark(
        models=model_list, pdf_path=args.pdf, golden_path=args.golden, output_path=args.output
    )


if __name__ == "__main__":
    main()
