"""Tests for PDF processing functionality."""

import tempfile
import unittest

from pypdf import PdfWriter

from src.processors.pdf_processor import PDFProcessor
from src.processors.text_chunker import TextChunker


class TestPDFProcessor(unittest.TestCase):
    """Test PDF processor."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = PDFProcessor(max_file_size_mb=20)

    def test_clean_text(self):
        """Test text cleaning."""
        dirty_text = "Hello   World\n\n\n\nTest"
        clean = self.processor.clean_text(dirty_text)
        self.assertIn("Hello World", clean)
        self.assertNotIn("\n\n\n", clean)

    def test_validate_pdf_size(self):
        """Test file size validation."""
        # Test oversized file
        is_valid, error = self.processor.validate_pdf("dummy.pdf", 25 * 1024 * 1024)
        self.assertFalse(is_valid)
        self.assertIn("exceeds", error)

    def test_encrypted_pdf_handling(self):
        """Test that encrypted PDFs are rejected cleanly."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            pdf_path = f"{tmp_dir}/encrypted.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            writer.encrypt("test_password")
            with open(pdf_path, "wb") as f:
                writer.write(f)

            is_valid, error = self.processor.validate_pdf(pdf_path, 1024)
            self.assertFalse(is_valid)
            self.assertIn("Encrypted PDF", error)

            with self.assertLogs("src.processors.pdf_processor", level="ERROR"):
                with self.assertRaises(ValueError) as ctx:
                    self.processor.extract_text_with_pages(pdf_path)
            self.assertIn("Encrypted PDF", str(ctx.exception))

    def test_pdf_with_no_extractable_text(self):
        """Test that PDFs without extractable text raise a descriptive ValueError."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            pdf_path = f"{tmp_dir}/blank.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            with open(pdf_path, "wb") as f:
                writer.write(f)

            is_valid, error = self.processor.validate_pdf(pdf_path, 1024)
            self.assertTrue(is_valid)
            self.assertIsNone(error)

            with self.assertLogs("src.processors.pdf_processor", level="ERROR"):
                with self.assertRaises(ValueError) as ctx:
                    self.processor.extract_text_with_pages(pdf_path)
            self.assertIn("No text content could be extracted", str(ctx.exception))


class TestTextChunker(unittest.TestCase):
    """Test text chunker."""

    def setUp(self):
        """Set up test fixtures."""
        self.chunker = TextChunker(chunk_size=100, overlap=20)

    def test_create_chunks(self):
        """Test chunk creation."""
        text = "This is a test. " * 50
        chunks = self.chunker.create_chunks(text)

        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].chunk_index, 0)

    def test_empty_text(self):
        """Test empty text handling."""
        with self.assertLogs("src.processors.text_chunker", level="WARNING"):
            chunks = self.chunker.create_chunks("")
        self.assertEqual(len(chunks), 0)

    def test_validate_chunks(self):
        """Test chunk validation."""
        text = "Test text for chunking."
        chunks = self.chunker.create_chunks(text)
        is_valid = self.chunker.validate_chunks(chunks)
        self.assertTrue(is_valid)

    def test_create_chunks_with_pages(self):
        """Test page-aware chunking with page bounds."""
        pages_data = [
            (1, "First page content. " * 10),
            (2, "Second page content with more detailed information. " * 10),
            (3, "Third page conclusions and summary points. " * 10),
        ]
        chunks = self.chunker.create_chunks_with_pages(pages_data)
        self.assertGreater(len(chunks), 0)

        # Verify first chunk starts on page 1
        self.assertEqual(chunks[0].page_start, 1)
        self.assertEqual(chunks[0].page, 1)
        self.assertIsNotNone(chunks[0].page_end)

        # Verify last chunk ends on page 3
        self.assertEqual(chunks[-1].page_end, 3)

        # Verify all chunks have valid page intervals
        for chunk in chunks:
            self.assertIsNotNone(chunk.page_start)
            self.assertIsNotNone(chunk.page_end)
            self.assertLessEqual(chunk.page_start, chunk.page_end)


if __name__ == "__main__":
    unittest.main()
