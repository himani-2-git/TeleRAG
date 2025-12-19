"""Tests for document storage operations."""

import unittest
import tempfile
import os
import shutil
from src.storage.database import Database
from src.storage.document_store import DocumentStore

class TestDatabase(unittest.TestCase):
    """Test database operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.db = Database(self.db_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_store_document(self):
        """Test document storage."""
        chunks = [
            {'chunk_index': 0, 'content': 'Test chunk 1', 'char_count': 12},
            {'chunk_index': 1, 'content': 'Test chunk 2', 'char_count': 12}
        ]
        
        doc_id = self.db.store_document(
            user_id=123,
            filename="test.pdf",
            file_hash="abc123",
            file_size=1000,
            chunks=chunks
        )
        
        self.assertIsNotNone(doc_id)
    
    def test_get_user_documents(self):
        """Test retrieving user documents."""
        chunks = [{'chunk_index': 0, 'content': 'Test', 'char_count': 4}]
        self.db.store_document(123, "test.pdf", "hash", 1000, chunks)
        
        docs = self.db.get_user_documents(123)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]['filename'], "test.pdf")
    
    def test_delete_document(self):
        """Test document deletion."""
        chunks = [{'chunk_index': 0, 'content': 'Test', 'char_count': 4}]
        self.db.store_document(123, "test.pdf", "hash", 1000, chunks)
        
        deleted = self.db.delete_document(123, "test.pdf")
        self.assertTrue(deleted)
        
        docs = self.db.get_user_documents(123)
        self.assertEqual(len(docs), 0)
    
    def test_user_isolation(self):
        """Test user data isolation."""
        chunks = [{'chunk_index': 0, 'content': 'Test', 'char_count': 4}]
        self.db.store_document(123, "test1.pdf", "hash1", 1000, chunks)
        self.db.store_document(456, "test2.pdf", "hash2", 1000, chunks)
        
        docs_user1 = self.db.get_user_documents(123)
        docs_user2 = self.db.get_user_documents(456)
        
        self.assertEqual(len(docs_user1), 1)
        self.assertEqual(len(docs_user2), 1)
        self.assertNotEqual(docs_user1[0]['filename'], docs_user2[0]['filename'])

class TestDocumentStore(unittest.TestCase):
    """Test document store."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.embeddings_path = os.path.join(self.temp_dir, "embeddings")
        self.store = DocumentStore(self.db_path, self.embeddings_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_store_with_embeddings(self):
        """Test storing document with embeddings."""
        chunks = [
            {'chunk_index': 0, 'content': 'Test chunk', 'char_count': 10}
        ]
        embeddings = [[0.1, 0.2, 0.3]]
        
        doc_id = self.store.store_document(
            user_id=123,
            filename="test.pdf",
            file_content=b"test content",
            chunks=chunks,
            embeddings=embeddings
        )
        
        self.assertIsNotNone(doc_id)
        
        # Check embedding file exists
        embedding_file = os.path.join(self.embeddings_path, f"doc_{doc_id}_chunk_0.json")
        self.assertTrue(os.path.exists(embedding_file))

if __name__ == '__main__':
    unittest.main()
