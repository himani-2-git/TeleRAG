"""Retrieval evaluation harness for TeleRAG.

Evaluates retrieval quality against golden question datasets using real
production retrieval pipelines without touching database or external APIs.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pypdf import PdfReader

from eval.report import rate
from src.processors.pdf_processor import PDFProcessor
from src.processors.text_chunker import TextChunker
from src.rag.embeddings import EmbeddingGenerator
from src.rag.retriever import DocumentRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_chunks_with_pages(
    pdf_path: str, chunk_size: int = 1000, overlap: int = 200
) -> list[dict[str, Any]]:
    """Extract and chunk text from PDF, associating each chunk with its 1-based start page.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Target size for chunks in characters.
        overlap: Overlap size between chunks in characters.

    Returns:
        List of chunk dictionaries with content, page, chunk_index, and filename.
    """
    processor = PDFProcessor()
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)

    reader = PdfReader(pdf_path)
    pages_text: list[str] = []
    offsets: list[tuple[int, int, int]] = []
    current_pos = 0
    filename = os.path.basename(pdf_path)

    for page_num, page in enumerate(reader.pages, 1):
        raw_text = page.extract_text() or ""
        cleaned = processor.clean_text(raw_text)
        pages_text.append(cleaned)
        start_offset = current_pos
        end_offset = current_pos + len(cleaned)
        offsets.append((page_num, start_offset, end_offset))
        current_pos = end_offset + 2  # account for \n\n separator

    full_text = "\n\n".join(pages_text)
    raw_chunks = chunker.create_chunks(full_text)

    def find_page_for_pos(pos: int) -> int:
        for p_num, s_off, e_off in offsets:
            if s_off <= pos < e_off:
                return p_num
        return offsets[-1][0] if offsets else 1

    chunks_data: list[dict[str, Any]] = []
    for i, c in enumerate(raw_chunks):
        page_start = getattr(c, "page_start", None)
        if page_start is None:
            page_start = getattr(c, "page", None)
        if page_start is None:
            page_start = find_page_for_pos(c.start_pos)

        page_end = getattr(c, "page_end", None)
        if page_end is None:
            end_char_pos = max(c.start_pos, c.end_pos - 1)
            page_end = find_page_for_pos(end_char_pos)

        chunks_data.append(
            {
                "chunk_index": i,
                "content": c.content,
                "page": page_start,
                "page_start": page_start,
                "page_end": page_end,
                "filename": filename,
                "char_count": len(c.content),
            }
        )

    return chunks_data


def run_evaluation(
    pdf_path: str,
    golden_path: str,
    mode: str = "dense",
    out_path: str | None = None,
    min_hit5: float | None = None,
    similarity_threshold: float | None = None,
) -> dict[str, Any]:
    """Execute retrieval evaluation on the specified dataset.

    Args:
        pdf_path: Path to source PDF.
        golden_path: Path to JSON golden question set.
        mode: Retrieval mode ('dense', 'hybrid', 'hybrid_rerank').
        out_path: Optional path to save JSON metrics report.
        min_hit5: Optional minimum hit@5 threshold to assert.
        similarity_threshold: Optional similarity threshold for dense retrieval abstention.

    Returns:
        Evaluation metrics dictionary.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF fixture not found at {pdf_path}")
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden dataset not found at {golden_path}")

    if similarity_threshold is None:
        try:
            from src.config import get_config

            config = get_config(require_telegram=False)
            similarity_threshold = config.similarity_threshold
        except Exception:
            similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

    with open(golden_path, encoding="utf-8") as f:
        golden_data = json.load(f)

    logger.info("Extracting and chunking %s...", pdf_path)
    chunks = extract_chunks_with_pages(pdf_path)
    logger.info("Generated %d chunks from %s", len(chunks), pdf_path)

    logger.info("Generating embeddings with all-MiniLM-L6-v2...")
    embedding_gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    texts = [c["content"] for c in chunks]
    embeddings = embedding_gen.generate_embeddings(texts)
    for c, emb in zip(chunks, embeddings, strict=False):
        c["embedding"] = emb

    # Select retrieval strategy based on mode
    retriever = DocumentRetriever(top_k=5, similarity_threshold=similarity_threshold)
    hybrid_retriever = None
    reranker = None

    if mode == "hybrid":
        try:
            from src.rag.hybrid import HybridRetriever

            hybrid_retriever = HybridRetriever(top_k=5, similarity_threshold=similarity_threshold)
            hybrid_retriever.index_corpus(chunks)
        except ImportError as err:
            raise NotImplementedError(
                "Hybrid retrieval module src.rag.hybrid is not yet implemented."
            ) from err
    elif mode == "hybrid_rerank":
        try:
            from src.rag.hybrid import HybridRetriever
            from src.rag.reranker import CrossEncoderReranker

            hybrid_retriever = HybridRetriever(top_k=20, similarity_threshold=similarity_threshold)
            hybrid_retriever.index_corpus(chunks)
            reranker = CrossEncoderReranker()
        except ImportError as err:
            raise NotImplementedError(
                "Reranker or hybrid modules are not yet implemented."
            ) from err

    query_results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []

    answerable_count = 0
    strict_hit1_count = 0
    strict_hit5_count = 0
    strict_reciprocal_ranks: list[float] = []

    span_hit1_count = 0
    span_hit5_count = 0
    span_reciprocal_ranks: list[float] = []

    unanswerable_count = 0
    abstention_correct = 0
    unanswerable_easy_count = 0
    abstention_easy_correct = 0
    unanswerable_hard_count = 0
    abstention_hard_correct = 0

    logger.info("Evaluating %d queries in mode='%s'...", len(golden_data), mode)

    for item in golden_data:
        q_id = item.get("id", "")
        question = item["question"]
        expected_pages = set(item.get("expected_pages", []))
        is_answerable = item.get("answerable", True)
        category = item.get("category", "")

        start_time = time.perf_counter()
        query_emb = embedding_gen.generate_query_embedding(question)

        if mode == "dense":
            results = retriever.search_similar(query_emb, chunks)
        elif mode == "hybrid":
            assert hybrid_retriever is not None
            results = hybrid_retriever.search(
                query=question, query_embedding=query_emb, chunks=chunks
            )
        elif mode == "hybrid_rerank":
            assert hybrid_retriever is not None and reranker is not None
            candidates = hybrid_retriever.search(
                query=question, query_embedding=query_emb, chunks=chunks
            )
            results = reranker.rerank(query=question, candidates=candidates, top_k=5)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        latencies_ms.append(elapsed_ms)

        retrieved_pages = [r.get("page_start", r.get("page")) for r in results]
        retrieved_scores = [round(float(r.get("similarity_score", 0.0)), 4) for r in results[:5]]
        retrieved_chunks = [
            {
                "page_start": r.get("page_start", r.get("page")),
                "page_end": r.get("page_end", r.get("page")),
                "score": round(float(r.get("similarity_score", 0.0)), 4),
            }
            for r in results[:5]
        ]

        if is_answerable:
            answerable_count += 1

            # 1. Strict start-page metric
            s_h1 = 1 if (len(retrieved_pages) > 0 and retrieved_pages[0] in expected_pages) else 0
            s_h5 = 1 if any(p in expected_pages for p in retrieved_pages[:5]) else 0
            s_rr = 0.0
            for rank_idx, p in enumerate(retrieved_pages[:5], 1):
                if p in expected_pages:
                    s_rr = 1.0 / rank_idx
                    break
            strict_hit1_count += s_h1
            strict_hit5_count += s_h5
            strict_reciprocal_ranks.append(s_rr)

            # 2. Page-span overlap metric
            def chunk_span_overlaps(chunk_dict: dict[str, Any], exp_pages=expected_pages) -> bool:
                p_s = chunk_dict.get("page_start", chunk_dict.get("page", 1))
                p_e = chunk_dict.get("page_end", p_s)
                return bool(set(range(p_s, p_e + 1)).intersection(exp_pages))

            span_h1 = 1 if (len(results) > 0 and chunk_span_overlaps(results[0])) else 0
            span_h5 = 1 if any(chunk_span_overlaps(c) for c in results[:5]) else 0
            span_rr = 0.0
            for rank_idx, c in enumerate(results[:5], 1):
                if chunk_span_overlaps(c):
                    span_rr = 1.0 / rank_idx
                    break
            span_hit1_count += span_h1
            span_hit5_count += span_h5
            span_reciprocal_ranks.append(span_rr)

            query_results.append(
                {
                    "id": q_id,
                    "question": question,
                    "category": category,
                    "answerable": True,
                    "expected_pages": list(expected_pages),
                    "retrieved_pages": retrieved_pages,
                    "retrieved_scores": retrieved_scores,
                    "retrieved_chunks": retrieved_chunks,
                    "strict_hit1": bool(s_h1),
                    "strict_hit5": bool(s_h5),
                    "strict_rr": s_rr,
                    "span_hit1": bool(span_h1),
                    "span_hit5": bool(span_h5),
                    "span_rr": span_rr,
                    "hit1": bool(span_h1),
                    "hit5": bool(span_h5),
                    "rr": span_rr,
                    "latency_ms": elapsed_ms,
                }
            )
        else:
            unanswerable_count += 1
            # Correct abstention means 0 chunks returned
            abstained = len(results) == 0
            if abstained:
                abstention_correct += 1

            is_hard = (category == "unanswerable_hard") or (
                str(q_id) in [f"u{i}" for i in range(11, 21)]
            )
            if is_hard:
                unanswerable_hard_count += 1
                if abstained:
                    abstention_hard_correct += 1
            else:
                unanswerable_easy_count += 1
                if abstained:
                    abstention_easy_correct += 1

            query_results.append(
                {
                    "id": q_id,
                    "question": question,
                    "category": category,
                    "answerable": False,
                    "expected_pages": [],
                    "retrieved_pages": retrieved_pages,
                    "retrieved_scores": retrieved_scores,
                    "retrieved_chunks": retrieved_chunks,
                    "abstained": abstained,
                    "latency_ms": elapsed_ms,
                }
            )

    def _format_rate(val: float | None, fmt: str = "%.4f") -> str:
        return fmt % val if val is not None else "n/a"

    strict_h1 = rate(strict_hit1_count, answerable_count)
    strict_h5 = rate(strict_hit5_count, answerable_count)
    strict_mrr = rate(sum(strict_reciprocal_ranks), answerable_count)

    span_h1 = rate(span_hit1_count, answerable_count)
    span_h5 = rate(span_hit5_count, answerable_count)
    span_mrr = rate(sum(span_reciprocal_ranks), answerable_count)

    abstention_acc = rate(abstention_correct, unanswerable_count)
    abstention_acc_easy = rate(abstention_easy_correct, unanswerable_easy_count)
    abstention_acc_hard = rate(abstention_hard_correct, unanswerable_hard_count)
    mean_latency_ms = rate(sum(latencies_ms), len(latencies_ms))

    summary = {
        "mode": mode,
        "dataset_pdf": os.path.basename(pdf_path),
        "total_chunks": len(chunks),
        "total_questions": len(golden_data),
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "unanswerable_easy_questions": unanswerable_easy_count,
        "unanswerable_hard_questions": unanswerable_hard_count,
        "strict": {
            "hit_at_1": round(strict_h1, 4) if strict_h1 is not None else "n/a",
            "hit_at_5": round(strict_h5, 4) if strict_h5 is not None else "n/a",
            "mrr": round(strict_mrr, 4) if strict_mrr is not None else "n/a",
        },
        "span": {
            "hit_at_1": round(span_h1, 4) if span_h1 is not None else "n/a",
            "hit_at_5": round(span_h5, 4) if span_h5 is not None else "n/a",
            "mrr": round(span_mrr, 4) if span_mrr is not None else "n/a",
        },
        "hit_at_1": round(span_h1, 4) if span_h1 is not None else "n/a",
        "hit_at_5": round(span_h5, 4) if span_h5 is not None else "n/a",
        "mrr": round(span_mrr, 4) if span_mrr is not None else "n/a",
        "abstention_accuracy": round(abstention_acc, 4) if abstention_acc is not None else "n/a",
        "abstention_accuracy_easy": (
            round(abstention_acc_easy, 4) if abstention_acc_easy is not None else "n/a"
        ),
        "abstention_accuracy_hard": (
            round(abstention_acc_hard, 4) if abstention_acc_hard is not None else "n/a"
        ),
        "mean_retrieval_ms": (round(mean_latency_ms, 2) if mean_latency_ms is not None else "n/a"),
        "query_results": query_results,
    }

    logger.info("Evaluation Summary [%s]:", mode)
    logger.info(
        "  [Span Metric]   hit@1: %s (%d/%d), hit@5: %s (%d/%d), MRR: %s",
        _format_rate(span_h1),
        span_hit1_count,
        answerable_count,
        _format_rate(span_h5),
        span_hit5_count,
        answerable_count,
        _format_rate(span_mrr),
    )
    logger.info(
        "  [Strict Metric] hit@1: %s (%d/%d), hit@5: %s (%d/%d), MRR: %s",
        _format_rate(strict_h1),
        strict_hit1_count,
        answerable_count,
        _format_rate(strict_h5),
        strict_hit5_count,
        answerable_count,
        _format_rate(strict_mrr),
    )
    logger.info(
        "  Abstention Accuracy (all):  %s (%d/%d)",
        _format_rate(abstention_acc),
        abstention_correct,
        unanswerable_count,
    )
    logger.info(
        "  Abstention Accuracy (easy): %s (%d/%d)",
        _format_rate(abstention_acc_easy),
        abstention_easy_correct,
        unanswerable_easy_count,
    )
    logger.info(
        "  Abstention Accuracy (hard): %s (%d/%d)",
        _format_rate(abstention_acc_hard),
        abstention_hard_correct,
        unanswerable_hard_count,
    )
    logger.info(
        "  Mean Latency:               %s ms/query",
        _format_rate(mean_latency_ms, fmt="%.2f"),
    )

    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved evaluation results to %s", out_path)

    hit_at_5 = summary["hit_at_5"]
    if min_hit5 is not None:
        if hit_at_5 < min_hit5:
            logger.error(
                "Quality gate failed: hit@5 (%.4f) < required min_hit5 (%.4f)", hit_at_5, min_hit5
            )
            sys.exit(1)
        else:
            logger.info("Quality gate passed: hit@5 (%.4f) >= min_hit5 (%.4f)", hit_at_5, min_hit5)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation harness for TeleRAG.")
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid", "hybrid_rerank"],
        default="dense",
        help="Retrieval mode to evaluate",
    )
    parser.add_argument(
        "--pdf", default="eval/fixtures/book.pdf", help="Path to PDF document fixture"
    )
    parser.add_argument(
        "--golden", default="eval/golden.json", help="Path to golden question set JSON"
    )
    parser.add_argument("--out", default=None, help="Output path to save evaluation results JSON")
    parser.add_argument(
        "--min-hit5",
        type=float,
        default=None,
        help="Quality gate threshold: fail if hit@5 is below this value",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Path to JSON file with quality gate thresholds (e.g. eval/thresholds.json)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Minimum similarity score threshold for dense retrieval abstention",
    )
    args = parser.parse_args()

    summary = run_evaluation(
        pdf_path=args.pdf,
        golden_path=args.golden,
        mode=args.mode,
        out_path=args.out,
        min_hit5=args.min_hit5,
        similarity_threshold=args.similarity_threshold,
    )

    if args.thresholds:
        if not os.path.exists(args.thresholds):
            logger.error("Thresholds file not found: %s", args.thresholds)
            sys.exit(1)
        with open(args.thresholds, encoding="utf-8") as f:
            th = json.load(f)

        failed = False
        if "min_hit_at_5" in th:
            val = summary["hit_at_5"]
            if val == "n/a" or (isinstance(val, (int, float)) and val < th["min_hit_at_5"]):
                logger.error(
                    "Gate FAILED: hit@5 (%s) < min_hit_at_5 (%.4f)",
                    val,
                    th["min_hit_at_5"],
                )
                failed = True
        if "min_mrr" in th:
            val = summary["mrr"]
            if val == "n/a" or (isinstance(val, (int, float)) and val < th["min_mrr"]):
                logger.error("Gate FAILED: mrr (%s) < min_mrr (%.4f)", val, th["min_mrr"])
                failed = True
        if "min_abstention_easy" in th:
            val = summary["abstention_accuracy_easy"]
            if val == "n/a" or (isinstance(val, (int, float)) and val < th["min_abstention_easy"]):
                logger.error(
                    "Gate FAILED: easy abstention (%s) < min_abstention_easy (%.4f)",
                    val,
                    th["min_abstention_easy"],
                )
                failed = True

        if failed:
            logger.error("Retrieval quality gate FAILED.")
            sys.exit(1)
        else:
            logger.info("All quality gate assertions PASSED.")


if __name__ == "__main__":
    main()
