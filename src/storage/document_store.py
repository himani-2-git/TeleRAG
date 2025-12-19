"""Document storage management with embeddings."""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional
import logging
from .database import Database

logger = logging.getLogger(__name__)

class DocumentStore:
    """Manages document storage with embeddings."""
    
    def __init__(self, db_path: str, embeddings_path: str):
        """Initialize document store.
        
        Args:
            db_path: Path to SQLite database
            embeddings_path: Path to embeddings storage directory
        """
        self.db = Database(db_path)
        self.embeddings_path = embeddings_path
        self._ensure_embeddings_directory()
    
    def _ensure_embeddings_directory(self):
        """Ensure embeddings directory exists."""
        if not os.path.exists(self.embeddings_path):
            os.makedirs(self.embeddings_path)
    
    def _calculate_file_hash(self, content: bytes) -> str:
        """Calculate SHA256 hash of file content.
        
        Args:
            content: File content as bytes
            
        Returns:
            Hex digest of hash
        """
        return hashlib.sha256(content).hexdigest()
    
    def _get_embedding_path(self, document_id: int, chunk_index: int) -> str:
        """Get path for storing chunk embedding.
        
        Args:
            document_id: Document ID
            chunk_index: Chunk index
            
        Returns:
            Path to embedding file
        """
        return os.path.join(self.embeddings_path, f"doc_{document_id}_chunk_{chunk_index}.json")
    
    def store_document(self, user_id: int, filename: str, file_content: bytes,
                      chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """Store document with chunks and embeddings.
        
        Args:
            user_id: Telegram user ID
            filename: Name of the PDF file
            file_content: Raw file content for hashing
            chunks: List of text chunks
            embeddings: List of embedding vectors
            
        Returns:
            Document ID
        """
        file_hash = self._calculate_file_hash(file_content)
        file_size = len(file_content)
        
        # Prepare chunks with embedding paths
        chunks_data = []
        for i, chunk in enumerate(chunks):
            chunks_data.append({
                'chunk_index': chunk['chunk_index'],
                'content': chunk['content'],
                'char_count': chunk['char_count'],
                'embedding_path': None  # Will be set after document is created
            })
        
        # Store in database
        document_id = self.db.store_document(user_id, filename, file_hash, 
                                            file_size, chunks_data)
        
        # Store embeddings
        for i, embedding in enumerate(embeddings):
            embedding_path = self._get_embedding_path(document_id, i)
            with open(embedding_path, 'w') as f:
                json.dump({'embedding': embedding}, f)
        
        logger.info(f"Stored document {document_id} with {len(embeddings)} embeddings")
        return document_id
    
    def list_documents(self, user_id: int) -> List[Dict[str, Any]]:
        """List all documents for a user.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            List of document metadata
        """
        return self.db.get_user_documents(user_id)
    
    def delete_document(self, user_id: int, filename: str) -> bool:
        """Delete a document and its embeddings.
        
        Args:
            user_id: Telegram user ID
            filename: Name of the file to delete
            
        Returns:
            True if document was deleted
        """
        # Get document info before deletion
        documents = self.db.get_user_documents(user_id)
        doc_to_delete = None
        
        for doc in documents:
            if doc['filename'] == filename:
                doc_to_delete = doc
                break
        
        if not doc_to_delete:
            return False
        
        # Delete embeddings
        document_id = doc_to_delete['id']
        chunk_count = doc_to_delete['chunk_count']
        
        for i in range(chunk_count):
            embedding_path = self._get_embedding_path(document_id, i)
            if os.path.exists(embedding_path):
                try:
                    os.remove(embedding_path)
                except Exception as e:
                    logger.warning(f"Failed to delete embedding file: {e}")
        
        # Delete from database
        return self.db.delete_document(user_id, filename)
    
    def clear_user_documents(self, user_id: int) -> int:
        """Delete all documents for a user.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            Number of documents deleted
        """
        # Get all documents before deletion
        documents = self.db.get_user_documents(user_id)
        
        # Delete all embeddings
        for doc in documents:
            document_id = doc['id']
            chunk_count = doc['chunk_count']
            
            for i in range(chunk_count):
                embedding_path = self._get_embedding_path(document_id, i)
                if os.path.exists(embedding_path):
                    try:
                        os.remove(embedding_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete embedding file: {e}")
        
        # Delete from database
        return self.db.clear_user_documents(user_id)
    
    def get_user_chunks_with_embeddings(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all chunks with their embeddings for a user.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            List of chunks with embeddings
        """
        chunks = self.db.get_all_user_chunks(user_id)
        
        # Load embeddings
        for chunk in chunks:
            embedding_path = self._get_embedding_path(chunk['document_id'], 
                                                     chunk['chunk_index'])
            
            if os.path.exists(embedding_path):
                try:
                    with open(embedding_path, 'r') as f:
                        data = json.load(f)
                        chunk['embedding'] = data['embedding']
                except Exception as e:
                    logger.warning(f"Failed to load embedding: {e}")
                    chunk['embedding'] = None
            else:
                chunk['embedding'] = None
        
        return chunks
