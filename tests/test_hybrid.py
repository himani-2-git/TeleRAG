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


if __name__ == "__main__":
    unittest.main()
