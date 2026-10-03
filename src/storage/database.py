"""Database management for document metadata and user associations."""

import sqlite3
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import json

logger = logging.getLogger(__name__)

class Database:
    """Manages SQLite database for document storage."""
    
    def __init__(self, db_path: str):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_directory()
        self._initialize_schema()
    
    def _ensure_directory(self):
        """Ensure database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection.
        
        Returns:
            SQLite connection object
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def _initialize_schema(self):
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    upload_date TIMESTAMP NOT NULL,
                    UNIQUE(user_id, filename)
                )
            """)
            
            # Document chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding_path TEXT,
                    char_count INTEGER NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    UNIQUE(document_id, chunk_index)
                )
            """)
            
            # Create indexes for efficient querying
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_user_id 
                ON documents(user_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
                ON document_chunks(document_id)
            """)
            
            conn.commit()
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise
        finally:
            conn.close()
    
    def store_document(self, user_id: int, filename: str, file_hash: str, 
                      file_size: int, chunks: List[Dict[str, Any]]) -> int:
        """Store document metadata and chunks.
        
        Args:
            user_id: Telegram user ID
            filename: Name of the PDF file
            file_hash: Hash of the file content
            file_size: Size of file in bytes
            chunks: List of chunk dictionaries
            
        Returns:
            Document ID
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Insert document metadata
            cursor.execute("""
                INSERT INTO documents (user_id, filename, file_hash, file_size, 
                                     chunk_count, upload_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, filename, file_hash, file_size, len(chunks), 
                  datetime.now().isoformat()))
            
            document_id = cursor.lastrowid
            
            # Insert chunks
            for chunk in chunks:
                cursor.execute("""
                    INSERT INTO document_chunks (document_id, chunk_index, content, 
                                                embedding_path, char_count)
                    VALUES (?, ?, ?, ?, ?)
                """, (document_id, chunk['chunk_index'], chunk['content'],
                      chunk.get('embedding_path'), chunk['char_count']))
            
            conn.commit()
            logger.info(f"Stored document {filename} with {len(chunks)} chunks")
            return document_id
            
        except sqlite3.IntegrityError:
            logger.error(f"Document {filename} already exists for user {user_id}")
            raise Exception(f"Document '{filename}' already exists")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store document: {e}")
            raise
        finally:
            conn.close()
    
    def get_user_documents(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all documents for a user.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            List of document dictionaries
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, filename, file_size, chunk_count, upload_date
                FROM documents
                WHERE user_id = ?
                ORDER BY upload_date DESC
            """, (user_id,))
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    'id': row['id'],
                    'filename': row['filename'],
                    'file_size': row['file_size'],
                    'chunk_count': row['chunk_count'],
                    'upload_date': row['upload_date']
                })
            
            return documents
            
        finally:
            conn.close()
    
    def get_document_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of chunk dictionaries
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, chunk_index, content, embedding_path, char_count
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY chunk_index
            """, (document_id,))
            
            chunks = []
            for row in cursor.fetchall():
                chunks.append({
                    'id': row['id'],
                    'chunk_index': row['chunk_index'],
                    'content': row['content'],
                    'embedding_path': row['embedding_path'],
                    'char_count': row['char_count']
                })
            
            return chunks
            
        finally:
            conn.close()
    
    def delete_document(self, user_id: int, filename: str) -> bool:
        """Delete a document and its chunks.
        
        Args:
            user_id: Telegram user ID
            filename: Name of the file to delete
            
        Returns:
            True if document was deleted
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM documents
                WHERE user_id = ? AND filename = ?
            """, (user_id, filename))
            
            conn.commit()
            deleted = cursor.rowcount > 0
            
            if deleted:
                logger.info(f"Deleted document {filename} for user {user_id}")
            
            return deleted
            
        finally:
            conn.close()
    
    def clear_user_documents(self, user_id: int) -> int:
        """Delete all documents for a user.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            Number of documents deleted
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM documents
                WHERE user_id = ?
            """, (user_id,))
            
            conn.commit()
            count = cursor.rowcount
            
            logger.info(f"Cleared {count} documents for user {user_id}")
            return count
            
        finally:
            conn.close()
    
    def get_all_user_chunks(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for all documents of a user.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            List of chunk dictionaries with document info
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT dc.id, dc.chunk_index, dc.content, dc.embedding_path,
                       d.id as document_id, d.filename
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.user_id = ?
                ORDER BY d.id, dc.chunk_index
            """, (user_id,))
            
            chunks = []
            for row in cursor.fetchall():
                chunks.append({
                    'id': row['id'],
                    'chunk_index': row['chunk_index'],
                    'content': row['content'],
                    'embedding_path': row['embedding_path'],
                    'document_id': row['document_id'],
                    'filename': row['filename']
                })
            
            return chunks
            
        finally:
            conn.close()
