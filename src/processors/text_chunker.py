"""Text chunking module for splitting documents into manageable pieces."""

from typing import List, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    chunk_index: int
    start_pos: int
    end_pos: int
    char_count: int

class TextChunker:
    """Handles text chunking with configurable size and overlap."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """Initialize text chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")
    
    def create_chunks(self, text: str) -> List[TextChunk]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to be chunked
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        chunks = []
        text_length = len(text)
        chunk_index = 0
        start_pos = 0
        
        while start_pos < text_length:
            # Calculate end position for this chunk
            end_pos = min(start_pos + self.chunk_size, text_length)
            
            # Try to break at sentence or word boundary if not at end
            if end_pos < text_length:
                end_pos = self._find_break_point(text, start_pos, end_pos)
            
            # Extract chunk content
            chunk_content = text[start_pos:end_pos].strip()
            
            # Only add non-empty chunks
            if chunk_content:
                chunk = TextChunk(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    char_count=len(chunk_content)
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move to next chunk with overlap
            start_pos = end_pos - self.overlap
            
            # Prevent infinite loop
            if start_pos <= chunks[-1].start_pos if chunks else False:
                start_pos = end_pos
        
        logger.info(f"Created {len(chunks)} chunks from text of length {text_length}")
        return chunks
    
    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """Find a good break point near the end position.
        
        Tries to break at sentence end, then paragraph, then word boundary.
        
        Args:
            text: Full text
            start: Start position of chunk
            end: Desired end position
            
        Returns:
            Adjusted end position
        """
        # Look back up to 100 characters for a good break point
        search_start = max(start, end - 100)
        search_text = text[search_start:end]
        
        # Try to find sentence end (. ! ?)
        for delimiter in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            pos = search_text.rfind(delimiter)
            if pos != -1:
                return search_start + pos + len(delimiter)
        
        # Try to find paragraph break
        pos = search_text.rfind('\n\n')
        if pos != -1:
            return search_start + pos + 2
        
        # Try to find any line break
        pos = search_text.rfind('\n')
        if pos != -1:
            return search_start + pos + 1
        
        # Try to find word boundary
        pos = search_text.rfind(' ')
        if pos != -1:
            return search_start + pos + 1
        
        # If no good break point found, use original end
        return end
    
    def validate_chunks(self, chunks: List[TextChunk]) -> bool:
        """Validate chunk quality and consistency.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            True if chunks are valid
        """
        if not chunks:
            return False
        
        for i, chunk in enumerate(chunks):
            # Check chunk has content
            if not chunk.content or not chunk.content.strip():
                logger.warning(f"Chunk {i} is empty")
                return False
            
            # Check chunk index is sequential
            if chunk.chunk_index != i:
                logger.warning(f"Chunk index mismatch at position {i}")
                return False
            
            # Check positions are valid
            if chunk.start_pos >= chunk.end_pos:
                logger.warning(f"Invalid positions in chunk {i}")
                return False
        
        return True
