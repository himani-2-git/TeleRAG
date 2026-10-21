"""Unit tests for CrossEncoderReranker."""

import unittest
from unittest.mock import MagicMock, patch

from src.rag.reranker import CrossEncoderReranker


class TestCrossEncoderReranker(unittest.TestCase):
    """Test cross-encoder reranking behavior and edge cases."""

    def setUp(self):
        self.reranker = CrossEncoderReranker()
        self.sample_candidates = [
            {"chunk_index": 0, "content": "Python is dynamic.", "page": 1, "dense_score": 0.8},
            {
                "chunk_index": 1,
                "content": "Functions use def keyword.",
                "page": 2,
                "dense_score": 0.7,
            },
            {
                "chunk_index": 2,
                "content": "Lists are mutable sequences.",
                "page": 3,
                "dense_score": 0.6,
            },
        ]

    def test_empty_candidates(self):
        """Empty candidate list should immediately return []."""
        results = self.reranker.rerank(query="test query", candidates=[], top_k=5)
        self.assertEqual(results, [])

    def test_empty_query(self):
        """Whitespace query should return candidates up to top_k without error."""
        results = self.reranker.rerank(query="   ", candidates=self.sample_candidates, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_index"], 0)

    @patch.object(CrossEncoderReranker, "_get_model")
    def test_reranking_order_and_scores(self, mock_get_model):
        """Reranker should sort candidates descending by cross-encoder score."""
        mock_model = MagicMock()
        # Suppose candidate 1 gets highest score, candidate 2 gets second, candidate 0 gets lowest
        mock_model.predict.return_value = [0.15, 0.95, 0.45]
        mock_get_model.return_value = mock_model

        results = self.reranker.rerank(
            query="How to define functions in Python?", candidates=self.sample_candidates, top_k=2
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_index"], 1)
        self.assertEqual(results[0]["rerank_score"], 0.95)
        self.assertEqual(results[1]["chunk_index"], 2)
        self.assertEqual(results[1]["rerank_score"], 0.45)
        # Ensure candidate keys preserved
        self.assertEqual(results[0]["page"], 2)
        self.assertEqual(results[0]["dense_score"], 0.7)


if __name__ == "__main__":
    unittest.main()
