"""Tests for VectorStore abstraction, NumPy backend, Chroma backend, and parity."""

import importlib.util
import os
import shutil
import tempfile
import unittest

from src.storage.vector_store import (
    ChromaVectorStore,
    NumPyVectorStore,
    get_vector_store,
)

HAS_CHROMADB = importlib.util.find_spec("chromadb") is not None


class TestNumPyVectorStore(unittest.TestCase):
    """Tests for NumPyVectorStore."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.embeddings_path = os.path.join(self.temp_dir, "embeddings")
        self.store = NumPyVectorStore(db_path=self.db_path, embeddings_path=self.embeddings_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_numpy_isolation(self):
        """User A data must never be visible to User B."""
        user_a = 101
        user_b = 202

        chunks_a = [
            {
                "content": "Alpha content about distributed consensus",
                "chunk_index": 0,
                "filename": "doc_a.txt",
            },
            {
                "content": "Beta content on replication logs",
                "chunk_index": 1,
                "filename": "doc_a.txt",
            },
        ]
        embeddings_a = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3]]
        self.store.add(chunks_a, embeddings_a, user_id=user_a)

        # User B queries with identical embedding
        query_emb = [0.1, 0.2, 0.3, 0.4]
        hits_b = self.store.search(query_emb, user_id=user_b, top_k=5)
        self.assertEqual(len(hits_b), 0)

        # User A gets results
        hits_a = self.store.search(query_emb, user_id=user_a, top_k=5)
        self.assertGreater(len(hits_a), 0)
        self.assertEqual(hits_a[0][0]["content"], "Alpha content about distributed consensus")

    def test_numpy_delete(self):
        """Deleting as user A only affects user A."""
        user_a = 101
        user_b = 202

        chunks_a = [{"content": "Doc A content", "chunk_index": 0, "filename": "doc_a.txt"}]
        chunks_b = [{"content": "Doc B content", "chunk_index": 0, "filename": "doc_b.txt"}]
        emb = [[0.5, 0.5, 0.5, 0.5]]

        self.store.add(chunks_a, emb, user_id=user_a)
        self.store.add(chunks_b, emb, user_id=user_b)

        # Delete user A's doc
        deleted = self.store.delete("doc_a.txt", user_id=user_a)
        self.assertTrue(deleted)

        self.assertEqual(len(self.store.search(emb[0], user_id=user_a)), 0)
        # User B remains intact
        hits_b = self.store.search(emb[0], user_id=user_b)
        self.assertEqual(len(hits_b), 1)
        self.assertEqual(hits_b[0][0]["content"], "Doc B content")

    def test_numpy_list(self):
        """Listing documents scopes correctly per user."""
        user_a = 101
        chunks_a = [{"content": "Doc A content", "chunk_index": 0, "filename": "doc_a.txt"}]
        self.store.add(chunks_a, [[0.1, 0.2]], user_id=user_a)

        docs_a = self.store.list(user_id=user_a)
        self.assertEqual(len(docs_a), 1)
        self.assertEqual(docs_a[0]["filename"], "doc_a.txt")

        docs_empty = self.store.list(user_id=999)
        self.assertEqual(len(docs_empty), 0)


@unittest.skipUnless(HAS_CHROMADB, "chromadb is not installed")
class TestChromaVectorStore(unittest.TestCase):
    """Tests for ChromaVectorStore."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_path = os.path.join(self.temp_dir, "chroma")
        self.store = ChromaVectorStore(
            persist_directory=self.chroma_path, collection_name="test_coll"
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_chroma_isolation(self):
        """User A data in Chroma must never be returned for User B."""
        user_a = 101
        user_b = 202

        chunks_a = [
            {
                "content": "Alpha content about distributed consensus",
                "chunk_index": 0,
                "filename": "doc_a.txt",
            },
            {
                "content": "Beta content on replication logs",
                "chunk_index": 1,
                "filename": "doc_a.txt",
            },
        ]
        embeddings_a = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3]]
        self.store.add(chunks_a, embeddings_a, user_id=user_a)

        query_emb = [0.1, 0.2, 0.3, 0.4]
        hits_b = self.store.search(query_emb, user_id=user_b, top_k=5)
        self.assertEqual(len(hits_b), 0)

        hits_a = self.store.search(query_emb, user_id=user_a, top_k=5)
        self.assertGreater(len(hits_a), 0)
        self.assertEqual(hits_a[0][0]["content"], "Alpha content about distributed consensus")

    def test_chroma_delete(self):
        """Deleting in Chroma as user A only affects user A."""
        user_a = 101
        user_b = 202

        chunks_a = [
            {
                "content": "Doc A content",
                "chunk_index": 0,
                "filename": "doc_a.txt",
                "document_id": 1,
            }
        ]
        chunks_b = [
            {
                "content": "Doc B content",
                "chunk_index": 0,
                "filename": "doc_b.txt",
                "document_id": 2,
            }
        ]
        emb = [[0.5, 0.5, 0.5, 0.5]]

        self.store.add(chunks_a, emb, user_id=user_a)
        self.store.add(chunks_b, emb, user_id=user_b)

        deleted = self.store.delete(1, user_id=user_a)
        self.assertTrue(deleted)

        hits_a = self.store.search(emb[0], user_id=user_a)
        self.assertEqual(len(hits_a), 0)

        hits_b = self.store.search(emb[0], user_id=user_b)
        self.assertEqual(len(hits_b), 1)
        self.assertEqual(hits_b[0][0]["content"], "Doc B content")

    def test_chroma_list(self):
        """Chroma list returns user document summaries."""
        user_a = 101
        chunks_a = [
            {"content": "Doc A1", "chunk_index": 0, "filename": "doc_a.txt", "document_id": 1},
            {"content": "Doc A2", "chunk_index": 1, "filename": "doc_a.txt", "document_id": 1},
        ]
        self.store.add(chunks_a, [[0.1, 0.2], [0.3, 0.4]], user_id=user_a)

        docs = self.store.list(user_id=user_a)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["filename"], "doc_a.txt")
        self.assertEqual(docs[0]["chunk_count"], 2)


@unittest.skipUnless(HAS_CHROMADB, "chromadb is not installed")
class TestVectorStoreParity(unittest.TestCase):
    """Tests cosine similarity parity between NumPy and Chroma backends on ci_fixture.pdf."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.emb_path = os.path.join(self.temp_dir, "emb")
        self.chroma_path = os.path.join(self.temp_dir, "chroma")

        self.numpy_store = NumPyVectorStore(db_path=self.db_path, embeddings_path=self.emb_path)
        self.chroma_store = ChromaVectorStore(
            persist_directory=self.chroma_path, collection_name="parity_coll"
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cosine_parity_on_ci_fixture(self):
        """Cosine similarities between query and chunks under numpy and chroma match within 1e-4."""
        from sentence_transformers import SentenceTransformer

        from src.processors.pdf_processor import PDFProcessor
        from src.processors.text_chunker import TextChunker

        pdf_path = "eval/fixtures/ci_fixture.pdf"
        self.assertTrue(os.path.exists(pdf_path), f"Missing {pdf_path}")

        pdf_processor = PDFProcessor()
        chunker = TextChunker(chunk_size=1000, overlap=200)
        pages_data = pdf_processor.extract_text_with_pages(pdf_path)
        chunks = chunker.create_chunks_with_pages(pages_data)
        self.assertGreater(len(chunks), 0)

        chunks_data = [
            {
                "chunk_index": i,
                "content": c.content,
                "char_count": c.char_count,
                "page": c.page,
                "document_id": 1,
                "filename": "ci_fixture.pdf",
                "source_type": "pdf",
            }
            for i, c in enumerate(chunks)
        ]

        # Generate real embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [c["content"] for c in chunks_data]
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()

        user_id = 999
        self.numpy_store.add(chunks_data, embeddings, user_id=user_id)
        self.chroma_store.add(chunks_data, embeddings, user_id=user_id)

        # Test with query from benchmark
        query = "What is the primary technique for high availability in distributed systems?"
        query_emb = model.encode(query, convert_to_numpy=True).tolist()

        top_k = 5
        numpy_results = self.numpy_store.search(query_emb, user_id=user_id, top_k=top_k)
        chroma_results = self.chroma_store.search(query_emb, user_id=user_id, top_k=top_k)

        self.assertEqual(len(numpy_results), top_k)
        self.assertEqual(len(chroma_results), top_k)

        for rank, ((np_chunk, np_sim), (ch_chunk, ch_sim)) in enumerate(
            zip(numpy_results, chroma_results, strict=False)
        ):
            diff = abs(np_sim - ch_sim)
            self.assertLess(
                diff,
                1e-4,
                f"Rank {rank} similarity mismatch: NumPy={np_sim:.6f}, Chroma={ch_sim:.6f}, diff={diff:.6f}",
            )
            self.assertEqual(
                np_chunk["chunk_index"],
                ch_chunk["chunk_index"],
                f"Rank {rank} chunk_index mismatch",
            )


class TestVectorStoreFallback(unittest.TestCase):
    """Test get_vector_store factory and fallback behavior."""

    def test_default_returns_numpy(self):
        store = get_vector_store("numpy")
        self.assertIsInstance(store, NumPyVectorStore)

    def test_fallback_when_chroma_missing(self):
        # Even if chroma is passed, if not installed, it falls back to NumPyVectorStore without crashing
        store = get_vector_store("chroma")
        if not HAS_CHROMADB:
            self.assertIsInstance(store, NumPyVectorStore)
        else:
            self.assertIsInstance(store, ChromaVectorStore)
