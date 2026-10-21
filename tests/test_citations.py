"""Unit tests for page-level citations and programmatic source verification."""

import unittest
from unittest.mock import Mock

from src.bot.handlers import BotHandlers
from src.rag.retriever import DocumentRetriever


class TestCitationsAndSources(unittest.TestCase):
    """Test citation formatting and programmatic source footer verification."""

    def setUp(self):
        self.retriever = DocumentRetriever(top_k=5, similarity_threshold=0.3)
        self.handlers = BotHandlers(
            pdf_processor=Mock(),
            text_chunker=Mock(),
            document_store=Mock(),
            embedding_generator=Mock(),
            retriever=Mock(),
            llm_client=Mock(),
            config=Mock(),
        )

    def test_format_context_with_page_span(self):
        """Test format_context produces [<filename>, p. X-Y] headers."""
        chunks = [
            {
                "filename": "thinkpython.pdf",
                "page_start": 42,
                "page_end": 43,
                "content": "Recursion occurs when a function calls itself.",
            },
            {
                "filename": "thinkpython.pdf",
                "page_start": 55,
                "page_end": 55,
                "content": "Strings are immutable sequences.",
            },
        ]
        context = self.retriever.format_context(chunks)
        self.assertIn(
            "[thinkpython.pdf, p. 42-43]\nRecursion occurs when a function calls itself.", context
        )
        self.assertIn("[thinkpython.pdf, p. 55]\nStrings are immutable sequences.", context)

    def test_format_context_fallback_single_page(self):
        """Test format_context fallback when only 'page' attribute is present."""
        chunks = [{"filename": "guide.pdf", "page": 12, "content": "Tuples are immutable lists."}]
        context = self.retriever.format_context(chunks)
        self.assertIn("[guide.pdf, p. 12]\nTuples are immutable lists.", context)

    def test_format_source_footer_deduplication(self):
        """Test _format_source_footer formats and deduplicates unique sources."""
        chunks = [
            {"filename": "doc_a.pdf", "page_start": 10, "page_end": 10},
            {"filename": "doc_a.pdf", "page_start": 10, "page_end": 10},  # duplicate
            {"filename": "doc_a.pdf", "page_start": 12, "page_end": 13},
            {"filename": "doc_b.pdf", "page_start": 5, "page_end": 5},
        ]
        footer = self.handlers._format_source_footer(chunks)
        expected_lines = [
            "\n\nSources:",
            "• doc_a.pdf (p. 10)",
            "• doc_a.pdf (p. 12-13)",
            "• doc_b.pdf (p. 5)",
        ]
        self.assertEqual(footer, "\n".join(expected_lines))

    def test_format_source_footer_empty(self):
        """Empty chunks should return an empty string footer."""
        self.assertEqual(self.handlers._format_source_footer([]), "")


if __name__ == "__main__":
    unittest.main()
