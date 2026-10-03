"""PDF processing module for text extraction and validation."""

import re
from typing import Optional, Tuple
from PyPDF2 import PdfReader
import logging

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF text extraction and validation."""
    
    def __init__(self, max_file_size_mb: int = 20):
        """Initialize PDF processor.
        
        Args:
            max_file_size_mb: Maximum allowed file size in megabytes
        """
        self.max_file_size_mb = max_file_size_mb
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
    
    def validate_pdf(self, file_path: str, file_size: int) -> Tuple[bool, Optional[str]]:
        """Validate PDF file size and format.
        
        Args:
            file_path: Path to the PDF file
            file_size: Size of the file in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        if file_size > self.max_file_size_bytes:
            return False, f"File size exceeds {self.max_file_size_mb}MB limit"
        
        # Check if it's a valid PDF
        try:
            with open(file_path, 'rb') as f:
                PdfReader(f)
            return True, None
        except Exception as e:
            logger.error(f"PDF validation failed: {e}")
            return False, "Invalid PDF format or corrupted file"
    
    def extract_text(self, file_path: str) -> str:
        """Extract text content from all PDF pages.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            Exception: If text extraction fails
        """
        try:
            text_content = []
            
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}: {e}")
                        continue
            
            if not text_content:
                raise Exception("No text content could be extracted from PDF")
            
            full_text = "\n\n".join(text_content)
            return self.clean_text(full_text)
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove special characters that might cause issues
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize horizontal whitespace (spaces/tabs) without stripping newlines
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove multiple consecutive newlines (collapse 3+ into 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
