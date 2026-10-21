"""Hybrid retrieval module combining dense semantic search and BM25 via Reciprocal Rank Fusion."""

import hashlib
import logging
import re
from collections import defaultdict
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """Combines dense semantic retrieval and sparse BM25 via Reciprocal Rank Fusion (RRF)."""

    def __init__(self, top_k: int = 5, similarity_threshold: float = 0.3, rrf_k: int = 60):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.rrf_k = rrf_k
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict[str, Any]] = []
        self._corpus_tokenized: list[list[str]] = []
        self._bm25_cache: dict[
            tuple[Any, str], tuple[BM25Okapi, list[dict[str, Any]], list[list[str]]]
        ] = {}

    def _compute_corpus_fingerprint(self, chunks: list[dict[str, Any]]) -> str:
        """Compute deterministic hash of chunks corpus."""
        parts = [
            f"{c.get('document_id', c.get('doc_id', ''))}:{c.get('chunk_index', '')}:{len(c.get('content', ''))}"
            for c in chunks
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def invalidate_cache(self, user_id: int | None = None) -> None:
        """Invalidate cached BM25 index for a specific user or globally."""
        if user_id is None:
            self._bm25_cache.clear()
            self.bm25 = None
            self.chunks = []
            self._corpus_tokenized = []
        else:
            to_del = [k for k in self._bm25_cache if k[0] == user_id]
            for k in to_del:
                del self._bm25_cache[k]
            if self.chunks and self.chunks[0].get("user_id") == user_id:
                self.bm25 = None
                self.chunks = []
                self._corpus_tokenized = []

    def index_corpus(self, chunks: list[dict[str, Any]], user_id: int | None = None) -> None:
        """Build BM25 index over the chunk collection, using cache when available."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.bm25 = None
            self._corpus_tokenized = []
            return

        uid = user_id if user_id is not None else self.chunks[0].get("user_id")
        fingerprint = self._compute_corpus_fingerprint(self.chunks)
        cache_key = (uid, fingerprint)

        if cache_key in self._bm25_cache:
            cached_bm25, cached_chunks, cached_tokenized = self._bm25_cache[cache_key]
            self.bm25 = cached_bm25
            self.chunks = cached_chunks
            self._corpus_tokenized = cached_tokenized
            logger.debug("Reused cached BM25 index for user %s", uid)
            return

        self._corpus_tokenized = [tokenize_text(c.get("content", "")) for c in self.chunks]
        if self._corpus_tokenized:
            self.bm25 = BM25Okapi(self._corpus_tokenized)
            self._bm25_cache[cache_key] = (self.bm25, self.chunks, self._corpus_tokenized)
            logger.info("Indexed %d chunks into BM25 corpus", len(self.chunks))
        else:
            self.bm25 = None

    def search(
        self,
        query: str,
        query_embedding: list[float],
        chunks: list[dict[str, Any]] | None = None,
        user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Perform hybrid retrieval using dense similarity and BM25 fused via RRF."""
        if chunks is not None:
            uid = user_id if user_id is not None else (chunks[0].get("user_id") if chunks else None)
            fp = self._compute_corpus_fingerprint(chunks)
            cache_key = (uid, fp)
            if self.bm25 is None or cache_key not in self._bm25_cache:
                self.index_corpus(chunks, user_id=uid)

        target_chunks = self.chunks
        if not target_chunks:
            logger.warning("No chunks available for hybrid search")
            return []

        # 1. Compute Dense Cosine Similarities
        valid_indices = [i for i, c in enumerate(target_chunks) if c.get("embedding") is not None]
        if not valid_indices:
            logger.warning("No chunks with embeddings found")
            return []

        query_vec = np.array(query_embedding, dtype=float)
        chunk_vecs = np.array([target_chunks[i]["embedding"] for i in valid_indices], dtype=float)
        dot_product = np.dot(chunk_vecs, query_vec)
        norm_chunks = np.linalg.norm(chunk_vecs, axis=1)
        norm_query = np.linalg.norm(query_vec)
        denominator = norm_chunks * norm_query
        denominator = np.where(denominator == 0, 1e-10, denominator)
        similarities = dot_product / denominator

        max_dense_score = float(max(similarities)) if len(similarities) > 0 else 0.0
        logger.debug(
            "Max dense similarity: %.4f (threshold: %.2f)",
            max_dense_score,
            self.similarity_threshold,
        )

        # Strict Abstention Gatekeeper: dense score must meet threshold
        if max_dense_score < self.similarity_threshold:
            logger.info(
                "Top dense score %.4f < %.2f; abstaining from retrieval",
                max_dense_score,
                self.similarity_threshold,
            )
            return []

        # Rank dense candidates meeting threshold
        dense_candidates = []
        for orig_idx, sim in zip(valid_indices, similarities, strict=False):
            if sim >= self.similarity_threshold:
                dense_candidates.append((orig_idx, float(sim)))
        dense_candidates.sort(key=lambda x: x[1], reverse=True)

        # 2. Compute BM25 Scores
        query_tokens = tokenize_text(query)
        bm25_candidates = []
        if self.bm25 is not None and query_tokens:
            bm25_scores = self.bm25.get_scores(query_tokens)
            for orig_idx, b_score in enumerate(bm25_scores):
                if b_score > 0.0:
                    bm25_candidates.append((orig_idx, float(b_score)))
            bm25_candidates.sort(key=lambda x: x[1], reverse=True)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[int, float] = defaultdict(float)
        dense_ranks: dict[int, int] = {}
        bm25_ranks: dict[int, int] = {}
        dense_scores_map: dict[int, float] = {}
        bm25_scores_map: dict[int, float] = {}

        for rank, (idx, score) in enumerate(dense_candidates, 1):
            rrf_scores[idx] += 1.0 / (self.rrf_k + rank)
            dense_ranks[idx] = rank
            dense_scores_map[idx] = score

        for rank, (idx, score) in enumerate(bm25_candidates, 1):
            rrf_scores[idx] += 1.0 / (self.rrf_k + rank)
            bm25_ranks[idx] = rank
            bm25_scores_map[idx] = score

        fused_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)

        results: list[dict[str, Any]] = []
        for idx in fused_indices[: self.top_k]:
            chunk_copy = dict(target_chunks[idx])
            chunk_copy["rrf_score"] = round(rrf_scores[idx], 6)
            chunk_copy["dense_score"] = round(dense_scores_map.get(idx, 0.0), 4)
            chunk_copy["bm25_score"] = round(bm25_scores_map.get(idx, 0.0), 4)
            chunk_copy["dense_rank"] = dense_ranks.get(idx)
            chunk_copy["bm25_rank"] = bm25_ranks.get(idx)
            chunk_copy["similarity_score"] = chunk_copy["dense_score"]
            results.append(chunk_copy)

        logger.info("Hybrid retrieval returned %d fused chunks", len(results))
        return results
