"""Tests for document storage operations."""

import os
import shutil
import tempfile
import unittest

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
            {"chunk_index": 0, "content": "Test chunk 1", "char_count": 12},
            {"chunk_index": 1, "content": "Test chunk 2", "char_count": 12},
        ]

        doc_id = self.db.store_document(
            user_id=123, filename="test.pdf", file_hash="abc123", file_size=1000, chunks=chunks
        )

        self.assertIsNotNone(doc_id)

    def test_get_user_documents(self):
        """Test retrieving user documents."""
        chunks = [{"chunk_index": 0, "content": "Test", "char_count": 4}]
        self.db.store_document(123, "test.pdf", "hash", 1000, chunks)

        docs = self.db.get_user_documents(123)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["filename"], "test.pdf")

    def test_delete_document(self):
        """Test document deletion."""
        chunks = [{"chunk_index": 0, "content": "Test", "char_count": 4}]
        self.db.store_document(123, "test.pdf", "hash", 1000, chunks)

        deleted = self.db.delete_document(123, "test.pdf")
        self.assertTrue(deleted)

        docs = self.db.get_user_documents(123)
        self.assertEqual(len(docs), 0)

    def test_user_isolation(self):
        """Test user data isolation."""
        chunks = [{"chunk_index": 0, "content": "Test", "char_count": 4}]
        self.db.store_document(123, "test1.pdf", "hash1", 1000, chunks)
        self.db.store_document(456, "test2.pdf", "hash2", 1000, chunks)

        docs_user1 = self.db.get_user_documents(123)
        docs_user2 = self.db.get_user_documents(456)

        self.assertEqual(len(docs_user1), 1)
        self.assertEqual(len(docs_user2), 1)
        self.assertNotEqual(docs_user1[0]["filename"], docs_user2[0]["filename"])


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
        chunks = [{"chunk_index": 0, "content": "Test chunk", "char_count": 10}]
        embeddings = [[0.1, 0.2, 0.3]]

        doc_id = self.store.store_document(
            user_id=123,
            filename="test.pdf",
            file_content=b"test content",
            chunks=chunks,
            embeddings=embeddings,
        )

        self.assertIsNotNone(doc_id)

        # Check embedding file exists
        embedding_file = os.path.join(self.embeddings_path, f"{doc_id}.npy")
        self.assertTrue(os.path.exists(embedding_file))

    def test_store_and_retrieve_with_page_bounds(self):
        """Test storing and retrieving chunks with page, page_start, and page_end."""
        chunks = [
            {
                "chunk_index": 0,
                "content": "Page 1-2 chunk",
                "char_count": 15,
                "page": 1,
                "page_start": 1,
                "page_end": 2,
            },
            {
                "chunk_index": 1,
                "content": "Page 2 chunk",
                "char_count": 12,
                "page": 2,
                "page_start": 2,
                "page_end": 2,
            },
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        doc_id = self.store.store_document(
            user_id=123,
            filename="pages.pdf",
            file_content=b"pages pdf content",
            chunks=chunks,
            embeddings=embeddings,
        )
        self.assertIsNotNone(doc_id)

        user_chunks = self.store.get_user_chunks_with_embeddings(123)
        self.assertEqual(len(user_chunks), 2)
        self.assertEqual(user_chunks[0]["page_start"], 1)
        self.assertEqual(user_chunks[0]["page_end"], 2)
        self.assertEqual(user_chunks[0]["page"], 1)
        self.assertEqual(user_chunks[1]["page_start"], 2)
        self.assertEqual(user_chunks[1]["page_end"], 2)

    def test_schema_migration_backward_compatibility(self):
        """Test non-destructive migration on existing database lacking page columns."""
        import sqlite3

        legacy_db_path = os.path.join(self.temp_dir, "legacy.db")
        conn = sqlite3.connect(legacy_db_path)
        # Create old schema without page columns
        conn.execute("""
            CREATE TABLE documents (
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
        conn.execute("""
            CREATE TABLE document_chunks (
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
        conn.commit()
        conn.close()

        # Connect with Database class, which should trigger automatic migration
        Database(legacy_db_path)

        # Verify columns exist now
        conn2 = sqlite3.connect(legacy_db_path)
        cursor = conn2.cursor()
        cursor.execute("PRAGMA table_info(document_chunks)")
        cols = {row[1] for row in cursor.fetchall()}
        conn2.close()

        self.assertIn("page", cols)
        self.assertIn("page_start", cols)
        self.assertIn("page_end", cols)


class TestNumPyVectorStoreAndMigration(unittest.TestCase):
    """Test NumPyVectorStore npy format, migration script, and caching."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.embeddings_path = os.path.join(self.temp_dir, "embeddings")
        os.makedirs(self.embeddings_path, exist_ok=True)
        from src.storage.database import Database
        from src.storage.vector_store import NumPyVectorStore

        self.db = Database(self.db_path)
        self.vs = NumPyVectorStore(self.db_path, self.embeddings_path, db=self.db)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_numpy_vector_store_npy_format_and_load_once(self):
        """Test that NumPyVectorStore stores float32 .npy files and loads embeddings correctly."""
        import numpy as np

        chunks = [
            {"chunk_index": 0, "content": "Chunk 0", "char_count": 7, "filename": "test.pdf"},
            {"chunk_index": 1, "content": "Chunk 1", "char_count": 7, "filename": "test.pdf"},
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        self.vs.add(chunks, embeddings, user_id=1)

        doc_id = chunks[0]["document_id"]
        npy_path = os.path.join(self.embeddings_path, f"{doc_id}.npy")
        self.assertTrue(os.path.exists(npy_path))

        arr = np.load(npy_path)
        self.assertEqual(arr.dtype, np.float32)
        self.assertEqual(arr.shape, (2, 3))

        loaded = self.vs.get_user_chunks_with_embeddings(user_id=1)
        self.assertEqual(len(loaded), 2)
        self.assertAlmostEqual(loaded[0]["embedding"][0], 0.1, places=5)
        self.assertAlmostEqual(loaded[1]["embedding"][0], 0.4, places=5)

    def test_migrate_embeddings_script_idempotent(self):
        """Test scripts/migrate_embeddings.py idempotently converts JSON to .npy."""
        import json

        import numpy as np

        from scripts.migrate_embeddings import migrate_embeddings

        # Create mock legacy JSON files for doc 42
        for i, emb in enumerate([[1.0, 2.0], [3.0, 4.0]]):
            fn = os.path.join(self.embeddings_path, f"doc_42_chunk_{i}.json")
            with open(fn, "w") as f:
                json.dump({"embedding": emb}, f)

        migrated = migrate_embeddings(self.embeddings_path)
        self.assertEqual(migrated, 1)

        npy_path = os.path.join(self.embeddings_path, "42.npy")
        self.assertTrue(os.path.exists(npy_path))
        arr = np.load(npy_path)
        self.assertEqual(arr.shape, (2, 2))

        # Second run should skip (idempotent)
        second_run = migrate_embeddings(self.embeddings_path)
        self.assertEqual(second_run, 0)

    def test_bm25_cache_and_invalidation(self):
        """Test HybridRetriever BM25 caching per (user_id, corpus fingerprint) and invalidation."""
        from src.rag.hybrid import HybridRetriever

        retriever = HybridRetriever()

        chunks = [
            {"document_id": 1, "chunk_index": 0, "content": "alpha beta gamma", "user_id": 99},
            {"document_id": 1, "chunk_index": 1, "content": "delta epsilon zeta", "user_id": 99},
        ]
        retriever.index_corpus(chunks, user_id=99)
        self.assertIsNotNone(retriever.bm25)
        first_bm25 = retriever.bm25

        # Re-index same corpus -> cache hit, same instance
        retriever.index_corpus(chunks, user_id=99)
        self.assertIs(retriever.bm25, first_bm25)

        # Invalidate cache for user
        retriever.invalidate_cache(user_id=99)
        self.assertEqual(len(retriever._bm25_cache), 0)

    def test_cross_encoder_models_cached_by_name(self):
        """Test CrossEncoderReranker caches models keyed by model_name."""
        from unittest.mock import Mock, patch

        from src.rag.reranker import CrossEncoderReranker

        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_ce.side_effect = lambda name, device: Mock(name=name)
            m1 = CrossEncoderReranker._get_model("test-model-1")
            m2 = CrossEncoderReranker._get_model("test-model-2")
            m1_again = CrossEncoderReranker._get_model("test-model-1")

            self.assertIn("test-model-1", CrossEncoderReranker._models)
            self.assertIn("test-model-2", CrossEncoderReranker._models)
            self.assertIs(m1, m1_again)
            self.assertIsNot(m1, m2)


if __name__ == "__main__":
    unittest.main()
