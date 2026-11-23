"""Hybrid retrieval module combining dense semantic search and BM25 via Reciprocal Rank Fusion."""

import hashlib
import logging
import re
import threading
from collections import OrderedDict, defaultdict
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Number of per-user BM25 indexes kept in memory (least recently used are dropped first).
DEFAULT_MAX_CACHE_ENTRIES = 8


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """Combines dense semantic retrieval and sparse BM25 via Reciprocal Rank Fusion (RRF).

    The retriever keeps no "current user" state. Every call to search() receives the chunks to
    search, so two users can never see each other's data. The only thing kept between calls is a
    small cache of BM25 indexes, keyed by (user id, corpus fingerprint), so that one user's index
    is never used for another user's chunks.
    """

    def __init__(
        self,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        rrf_k: int = 60,
        max_cache_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
    ):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.rrf_k = rrf_k
        self.max_cache_entries = max(1, max_cache_entries)
        # Least recently used entries are at the front, most recently used at the back.
        self._bm25_cache: OrderedDict[tuple[Any, str], BM25Okapi] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _compute_corpus_fingerprint(self, chunks: list[dict[str, Any]]) -> str:
        """Compute a deterministic hash of the chunk ids and text."""
        digest = hashlib.sha256()
        for chunk in chunks:
            doc_id = chunk.get("document_id", chunk.get("doc_id", ""))
            header = f"{doc_id}:{chunk.get('chunk_index', '')}:"
            digest.update(header.encode("utf-8"))
            digest.update(chunk.get("content", "").encode("utf-8"))
            digest.update(b"|")
        return digest.hexdigest()

    def _cache_get(self, cache_key: tuple[Any, str]) -> BM25Okapi | None:
        """Return a cached index and mark it as recently used."""
        with self._cache_lock:
            bm25 = self._bm25_cache.get(cache_key)
            if bm25 is not None:
                self._bm25_cache.move_to_end(cache_key)
            return bm25

    def _cache_put(self, cache_key: tuple[Any, str], bm25: BM25Okapi) -> None:
        """Store an index and evict the least recently used entries if the cache is full."""
        with self._cache_lock:
            self._bm25_cache[cache_key] = bm25
            self._bm25_cache.move_to_end(cache_key)
            while len(self._bm25_cache) > self.max_cache_entries:
                self._bm25_cache.popitem(last=False)

    def invalidate_cache(self, user_id: int | None = None) -> None:
        """Forget cached BM25 indexes for one user, or for everyone when user_id is None."""
        with self._cache_lock:
            if user_id is None:
                self._bm25_cache.clear()
                return
            stale_keys = [key for key in self._bm25_cache if key[0] == user_id]
            for key in stale_keys:
                del self._bm25_cache[key]

    def index_corpus(
        self, chunks: list[dict[str, Any]], user_id: int | None = None
    ) -> BM25Okapi | None:
        """Return the BM25 index for these chunks, building and caching it on a cache miss."""
        if not chunks:
            return None

        uid = user_id if user_id is not None else chunks[0].get("user_id")
        cache_key = (uid, self._compute_corpus_fingerprint(chunks))

        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug("Reused cached BM25 index for user %s", uid)
            return cached

        tokenized = [tokenize_text(chunk.get("content", "")) for chunk in chunks]
        bm25 = BM25Okapi(tokenized)
        self._cache_put(cache_key, bm25)
        logger.info("Indexed %d chunks into BM25 corpus", len(chunks))
        return bm25

    def search(
        self,
        query: str,
        query_embedding: list[float],
        chunks: list[dict[str, Any]],
        user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Perform hybrid retrieval over the given chunks using dense similarity and BM25."""
        if not chunks:
            logger.warning("No chunks available for hybrid search")
            return []

        bm25 = self.index_corpus(chunks, user_id=user_id)
        target_chunks = chunks

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
        if bm25 is not None and query_tokens:
            bm25_scores = bm25.get_scores(query_tokens)
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
