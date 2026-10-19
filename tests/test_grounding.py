"""Unit tests for grounding check module."""

import unittest
import numpy as np

from src.rag.grounding import (
    GroundingReport,
    cosine_similarity,
    grounding_report,
    strip_footers,
    split_sentences,
)


class FakeEmbedder:
    """Mock embedder returning deterministic vectors based on substring matches."""

    def __init__(self, mapping: dict[str, list[float]], default_dim: int = 4):
        self.mapping = mapping
        self.default_dim = default_dim

    def generate_embeddings(self, text: str) -> list[float]:
        for key, vec in self.mapping.items():
            if key.lower() in text.lower():
                return vec
        default_vec = [0.0] * self.default_dim
        default_vec[-1] = 1.0
        return default_vec


class TestGrounding(unittest.TestCase):
    """Test suite for grounding report generation."""

    def test_strip_footers(self):
        """Verify Sources footer and Answered-in footer are stripped cleanly."""
        raw = (
            "This is the actual answer sentence.\n\n"
            "Sources:\n"
            "[ci_fixture.pdf, p. 1] Some cited text.\n"
            "[ci_fixture.pdf, p. 2] Another citation.\n\n"
            "Answered in 18 s with qwen2.5:3b"
        )
        cleaned = strip_footers(raw)
        self.assertEqual(cleaned, "This is the actual answer sentence.")
        self.assertNotIn("Sources:", cleaned)
        self.assertNotIn("Answered in", cleaned)

    def test_strip_footers_no_footers(self):
        """Verify text without footers is preserved."""
        raw = "Just a direct answer without footers."
        self.assertEqual(strip_footers(raw), raw)

    def test_split_sentences(self):
        """Verify splitting into sentences."""
        text = "First sentence here. Second sentence follows! Third question?"
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "First sentence here.")
        self.assertEqual(sentences[1], "Second sentence follows!")
        self.assertEqual(sentences[2], "Third question?")

    def test_grounding_fully_supported(self):
        """Verify citation_coverage is 1.0 when all sentences match chunks."""
        embedder = FakeEmbedder(
            mapping={
                "alpha": [1.0, 0.0, 0.0, 0.0],
                "beta": [0.0, 1.0, 0.0, 0.0],
            }
        )
        chunks = [
            {"content": "Alpha contains important data."},
            {"content": "Beta contains secondary details."},
        ]
        answer = "Alpha is very important. Beta has secondary details."
        report = grounding_report(answer, chunks, embedder, threshold=0.45)

        self.assertTrue(report.applicable)
        self.assertEqual(report.citation_coverage, 1.0)
        self.assertEqual(report.supported_sentences, 2)
        self.assertEqual(report.total_sentences, 2)
        self.assertEqual(len(report.unsupported_sentences), 0)

    def test_grounding_unsupported(self):
        """Verify citation_coverage < 1.0 and unsupported sentences identified."""
        embedder = FakeEmbedder(
            mapping={
                "alpha": [1.0, 0.0, 0.0, 0.0],
                "gamma": [0.0, 0.0, 0.0, 1.0],  # orthogonal to chunk's alpha
            }
        )
        chunks = [{"content": "Alpha contains important data."}]
        answer = "Alpha is supported. Gamma is completely hallucinated."
        report = grounding_report(answer, chunks, embedder, threshold=0.45)

        self.assertTrue(report.applicable)
        self.assertEqual(report.total_sentences, 2)
        self.assertEqual(report.supported_sentences, 1)
        self.assertEqual(report.citation_coverage, 0.5)
        self.assertEqual(len(report.unsupported_sentences), 1)
        self.assertIn("Gamma", report.unsupported_sentences[0])

    def test_grounding_refusal_not_applicable(self):
        """Verify refusal answers are marked applicable=False with 1.0 coverage."""
        embedder = FakeEmbedder({})
        chunks = [{"content": "Some context."}]

        # INSUFFICIENT_CONTEXT marker
        report_marker = grounding_report("INSUFFICIENT_CONTEXT", chunks, embedder)
        self.assertFalse(report_marker.applicable)
        self.assertEqual(report_marker.citation_coverage, 1.0)
        self.assertEqual(report_marker.total_sentences, 0)

        # Friendly refusal message
        refusal = "🤔 I couldn't find relevant information in your documents to answer this question."
        report_msg = grounding_report(refusal, chunks, embedder)
        self.assertFalse(report_msg.applicable)
        self.assertEqual(report_msg.citation_coverage, 1.0)

        # Empty answer
        report_empty = grounding_report("", chunks, embedder)
        self.assertFalse(report_empty.applicable)
        self.assertEqual(report_empty.citation_coverage, 1.0)

    def test_grounding_footers_ignored(self):
        """Verify footers in the answer string do not count as sentences."""
        embedder = FakeEmbedder(
            mapping={"alpha": [1.0, 0.0, 0.0, 0.0]}
        )
        chunks = [{"content": "Alpha details."}]
        answer_with_footers = (
            "Alpha details are verified.\n\n"
            "Sources:\n"
            "[doc.pdf, p. 1] Some text\n\n"
            "Answered in 12 s with qwen2.5:3b"
        )
        report = grounding_report(answer_with_footers, chunks, embedder, threshold=0.45)

        self.assertTrue(report.applicable)
        self.assertEqual(report.total_sentences, 1)
        self.assertEqual(report.supported_sentences, 1)
        self.assertEqual(report.citation_coverage, 1.0)


if __name__ == "__main__":
    unittest.main()
