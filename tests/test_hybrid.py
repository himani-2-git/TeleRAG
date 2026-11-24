"""Unit tests for HybridRetriever and Reciprocal Rank Fusion."""

import unittest

from src.rag.hybrid import HybridRetriever, tokenize_text


class TestHybridRetriever(unittest.TestCase):
    """Test hybrid retrieval functionality and invariants."""

    def setUp(self):
        self.retriever = HybridRetriever(top_k=3, similarity_threshold=0.3, rrf_k=60)
        self.sample_chunks = [
            {
                "chunk_index": 0,
                "content": "Python is a high-level interpreted programming language with clear syntax.",
                "page": 1,
                "page_start": 1,
                "page_end": 1,
                "embedding": [0.9, 0.1, 0.0],
            },
            {
                "chunk_index": 1,
                "content": "Functions in Python are defined with the def keyword and can take arguments.",
                "page": 2,
                "page_start": 2,
                "page_end": 2,
                "embedding": [0.1, 0.9, 0.0],
            },
            {
                "chunk_index": 2,
                "content": "Lists and dictionaries are common mutable data structures in Python.",
                "page": 3,
                "page_start": 3,
                "page_end": 3,
                "embedding": [0.1, 0.2, 0.8],
            },
        ]
        self.retriever.index_corpus(self.sample_chunks)

    def test_tokenize_text(self):
        """Test lowercase alphanumeric tokenization."""
        tokens = tokenize_text("Hello, World! Python 3.10 is great.")
        self.assertEqual(tokens, ["hello", "world", "python", "3", "10", "is", "great"])

    def test_strict_abstention_invariant(self):
        """Verify pipeline immediately returns [] if max dense cosine similarity is below threshold."""
        # Query embedding orthogonal to all chunks -> cosine near 0
        orthogonal_embedding = [0.0, 0.0, 0.0]
        # Query has high lexical match ("Python", "language", "syntax")
        results = self.retriever.search(
            query="Python syntax programming language",
            query_embedding=orthogonal_embedding,
            chunks=self.sample_chunks,
        )
        self.assertEqual(
            results, [], "HybridRetriever must return [] when dense score is below threshold"
        )

    def test_hybrid_rrf_scoring(self):
        """Verify RRF fusion combines dense and BM25 signals."""
        query = "Python high-level programming language"
        query_embedding = [0.85, 0.15, 0.0]  # Very close to chunk 0
        results = self.retriever.search(
            query=query, query_embedding=query_embedding, chunks=self.sample_chunks
        )
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertEqual(top["chunk_index"], 0)
        self.assertIn("rrf_score", top)
        self.assertIn("dense_score", top)
        self.assertIn("bm25_score", top)
        self.assertGreater(top["rrf_score"], 0.0)
        self.assertGreaterEqual(top["dense_score"], 0.3)

    def test_empty_corpus(self):
        """Test search on empty corpus."""
        empty_retriever = HybridRetriever()
        results = empty_retriever.search(query="test", query_embedding=[0.1, 0.2, 0.3], chunks=[])
        self.assertEqual(results, [])


def make_user_chunks(document_id, text, user_id=None):
    """Build one single-chunk corpus for a fake user. All chunks share the same embedding."""
    chunk = {
        "document_id": document_id,
        "chunk_index": 0,
        "content": text,
        "filename": f"doc_{document_id}.txt",
        "embedding": [1.0, 0.0],
    }
    if user_id is not None:
        chunk["user_id"] = user_id
    return [chunk]


class TestUserIsolation(unittest.TestCase):
    """Regression tests: one user's chunks must never be returned to another user."""

    def setUp(self):
        self.user_1_chunks = make_user_chunks(1, "alpha private note for user one")
        self.user_2_chunks = make_user_chunks(2, "bravo private note for user two")

    def test_hybrid_search_does_not_leak_after_cache_hit(self):
        """user 1 -> user 2 -> user 1 must return user 1's own chunk the second time."""
        retriever = HybridRetriever(top_k=1, similarity_threshold=0.1)
        query_embedding = [1.0, 0.0]

        retriever.search("private note", query_embedding, self.user_1_chunks, user_id=1)
        retriever.search("private note", query_embedding, self.user_2_chunks, user_id=2)
        again = retriever.search("private note", query_embedding, self.user_1_chunks, user_id=1)

        self.assertEqual(len(again), 1)
        self.assertEqual(again[0]["content"], "alpha private note for user one")

    def test_document_retriever_does_not_leak_when_user_id_is_missing(self):
        """The real bot called retrieve() without a user id, so the leak must not depend on it."""
        from src.rag.retriever import DocumentRetriever

        retriever = DocumentRetriever(top_k=1, similarity_threshold=0.1, retrieval_mode="hybrid")
        query_embedding = [1.0, 0.0]

        retriever.retrieve("private note", query_embedding, self.user_1_chunks)
        retriever.retrieve("private note", query_embedding, self.user_2_chunks)
        again = retriever.retrieve("private note", query_embedding, self.user_1_chunks)

        self.assertEqual(again[0]["content"], "alpha private note for user one")

    def test_same_chunks_for_different_users_use_separate_cache_entries(self):
        """Cache keys include the user id, so users never share an index object."""
        retriever = HybridRetriever(top_k=1, similarity_threshold=0.1)
        shared_chunks = make_user_chunks(1, "shared text")

        retriever.search("shared", [1.0, 0.0], shared_chunks, user_id=1)
        retriever.search("shared", [1.0, 0.0], shared_chunks, user_id=2)

        self.assertEqual({key[0] for key in retriever._bm25_cache}, {1, 2})

    def test_cache_is_bounded(self):
        """The cache is a small LRU: the oldest user is evicted when it is full."""
        retriever = HybridRetriever(top_k=1, similarity_threshold=0.1, max_cache_entries=2)

        for user_id in (1, 2, 3):
            chunks = make_user_chunks(user_id, f"note for user {user_id}")
            retriever.search("note", [1.0, 0.0], chunks, user_id=user_id)

        self.assertEqual(len(retriever._bm25_cache), 2)
        cached_users = {key[0] for key in retriever._bm25_cache}
        self.assertEqual(cached_users, {2, 3})

    def test_invalidate_cache_only_removes_that_user(self):
        retriever = HybridRetriever(top_k=1, similarity_threshold=0.1)
        retriever.search("note", [1.0, 0.0], self.user_1_chunks, user_id=1)
        retriever.search("note", [1.0, 0.0], self.user_2_chunks, user_id=2)

        retriever.invalidate_cache(user_id=1)

        cached_users = {key[0] for key in retriever._bm25_cache}
        self.assertEqual(cached_users, {2})

    def test_document_retriever_invalidate_cache_forwards_user_id(self):
        from src.rag.retriever import DocumentRetriever

        retriever = DocumentRetriever(top_k=1, similarity_threshold=0.1, retrieval_mode="hybrid")
        retriever.retrieve("note", [1.0, 0.0], self.user_1_chunks, user_id=1)
        retriever.retrieve("note", [1.0, 0.0], self.user_2_chunks, user_id=2)

        retriever.invalidate_cache(1)

        cached_users = {key[0] for key in retriever._hybrid_retriever._bm25_cache}
        self.assertEqual(cached_users, {2})

    def test_pipeline_passes_user_id_to_retriever(self):
        """Without the user id the cache cannot tell users apart, so the pipeline must send it."""
        from unittest.mock import Mock

        from src.rag.pipeline import RAGPipeline

        retriever = Mock()
        retriever.retrieve.return_value = []
        embedder = Mock()
        embedder.generate_query_embedding.return_value = [1.0, 0.0]
        pipeline = RAGPipeline(retriever=retriever, llm_client=Mock(), embedding_generator=embedder)

        pipeline.node_retrieve(
            {"question": "what is in my note?", "user_id": 42, "user_chunks": self.user_1_chunks}
        )

        self.assertEqual(retriever.retrieve.call_args.kwargs["user_id"], 42)

    def test_two_users_searching_at_the_same_time_never_mix(self):
        """Run many searches from several threads; every answer must belong to its own user."""
        from concurrent.futures import ThreadPoolExecutor

        retriever = HybridRetriever(top_k=1, similarity_threshold=0.1, max_cache_entries=1)
        corpora = {1: self.user_1_chunks, 2: self.user_2_chunks}
        expected = {1: "alpha private note for user one", 2: "bravo private note for user two"}

        def search_as(user_id):
            results = retriever.search(
                "private note", [1.0, 0.0], corpora[user_id], user_id=user_id
            )
            return user_id, results[0]["content"]

        # max_cache_entries=1 forces constant eviction, which is the hardest case for the cache.
        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(search_as, [1, 2] * 100))

        for user_id, content in outcomes:
            self.assertEqual(content, expected[user_id])


if __name__ == "__main__":
    unittest.main()
