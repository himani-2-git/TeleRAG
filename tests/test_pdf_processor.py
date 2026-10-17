"""Tests for PDF processing functionality."""

import unittest
import tempfile
import os
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
            (3, "Third page conclusions and summary points. " * 10)
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

if __name__ == '__main__':
    unittest.main()
