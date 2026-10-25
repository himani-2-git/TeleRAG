"""Document storage management with embeddings."""

import hashlib
import json
import logging
import os
from typing import Any

import numpy as np

from .database import Database
from .vector_store import ChromaVectorStore, NumPyVectorStore, VectorStore, get_vector_store

logger = logging.getLogger(__name__)


class DocumentStore:
    """Manages document storage with embeddings."""

    def __init__(
        self,
        db_path: str,
        embeddings_path: str,
        vector_backend: str = "numpy",
        chroma_path: str = "data/chroma",
        vector_store: VectorStore | None = None,
    ):
        """Initialize document store.

        Args:
            db_path: Path to SQLite database
            embeddings_path: Path to embeddings storage directory
            vector_backend: 'numpy' or 'chroma'
            chroma_path: Path to chroma persistent storage
            vector_store: Optional pre-configured VectorStore instance
        """
        self.db = Database(db_path)
        self.embeddings_path = embeddings_path
        self.vector_backend = vector_backend
        self.chroma_path = chroma_path
        self._ensure_embeddings_directory()
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = get_vector_store(
                backend=vector_backend,
                db_path=db_path,
                embeddings_path=embeddings_path,
                chroma_path=chroma_path,
                db=self.db,
            )

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

    def store_document(
        self,
        user_id: int,
        filename: str,
        file_content: bytes,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        source_type: str = "pdf",
    ) -> int:
        """Store document with chunks and embeddings.

        Args:
            user_id: Telegram user ID
            filename: Name of the document / file
            file_content: Raw file content for hashing
            chunks: List of text chunks
            embeddings: List of embedding vectors
            source_type: Source type ('pdf', 'markdown', 'text', 'web')

        Returns:
            Document ID
        """
        file_hash = self._calculate_file_hash(file_content)
        file_size = len(file_content)

        # Prepare chunks with embedding paths, page metadata, and section labels
        chunks_data = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                "chunk_index": chunk["chunk_index"]
                if isinstance(chunk, dict)
                else getattr(chunk, "chunk_index", i),
                "content": chunk["content"] if isinstance(chunk, dict) else chunk.content,
                "char_count": chunk["char_count"]
                if isinstance(chunk, dict)
                else getattr(chunk, "char_count", len(getattr(chunk, "content", ""))),
                "page": chunk.get("page")
                if isinstance(chunk, dict)
                else getattr(chunk, "page", None),
                "page_start": chunk.get("page_start")
                if isinstance(chunk, dict)
                else getattr(chunk, "page_start", None),
                "page_end": chunk.get("page_end")
                if isinstance(chunk, dict)
                else getattr(chunk, "page_end", None),
                "section_label": chunk.get("section_label")
                if isinstance(chunk, dict)
                else getattr(chunk, "section_label", None),
                "source_type": chunk.get("source_type")
                if isinstance(chunk, dict)
                else getattr(chunk, "source_type", source_type),
                "embedding_path": None,  # Will be set after document is created
            }
            chunks_data.append(chunk_dict)

        # Store in database
        document_id = self.db.store_document(
            user_id, filename, file_hash, file_size, chunks_data, source_type=source_type
        )

        # Store embeddings in single .npy file per document (float32)
        if embeddings:
            npy_path = os.path.join(self.embeddings_path, f"{document_id}.npy")
            np.save(npy_path, np.array(embeddings, dtype=np.float32))

        # Also sync to vector store
        for c in chunks_data:
            c["document_id"] = document_id
            c["filename"] = filename
        if not isinstance(self.vector_store, NumPyVectorStore):
            try:
                self.vector_store.add(chunks_data, embeddings, user_id)
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning(f"Failed to sync to vector_store: {e}")

        logger.info(f"Stored document {document_id} with {len(embeddings)} embeddings")
        return document_id

    def list_documents(self, user_id: int) -> list[dict[str, Any]]:
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
            if doc["filename"] == filename:
                doc_to_delete = doc
                break

        if not doc_to_delete:
            return False

        # Delete embeddings
        document_id = doc_to_delete["id"]
        chunk_count = doc_to_delete["chunk_count"]

        for p in (
            os.path.join(self.embeddings_path, f"{document_id}.npy"),
            os.path.join(self.embeddings_path, f"doc_{document_id}.npy"),
        ):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError as e:
                    logger.warning(f"Failed to delete embedding file: {e}")

        for i in range(chunk_count):
            embedding_path = self._get_embedding_path(document_id, i)
            if os.path.exists(embedding_path):
                try:
                    os.remove(embedding_path)
                except OSError as e:
                    logger.warning(f"Failed to delete legacy embedding file: {e}")

        # Delete from vector store
        vs_deleted = False
        try:
            vs_deleted = self.vector_store.delete(document_id, user_id)
        except (OSError, ValueError, RuntimeError, KeyError) as e:
            logger.warning(f"Failed to delete from vector_store: {e}")

        # Delete from database
        db_deleted = self.db.delete_document(user_id, filename)
        return vs_deleted or db_deleted

    def clear_user_documents(self, user_id: int) -> int:
        """Delete all documents for a user.

        Args:
            user_id: Telegram user ID

        Returns:
            Number of documents deleted
        """
        # Get all documents before deletion
        documents = self.db.get_user_documents(user_id)
        if not documents:
            return 0

        # Delete all embeddings
        for doc in documents:
            document_id = doc["id"]
            chunk_count = doc["chunk_count"]

            for p in (
                os.path.join(self.embeddings_path, f"{document_id}.npy"),
                os.path.join(self.embeddings_path, f"doc_{document_id}.npy"),
            ):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError as e:
                        logger.warning(f"Failed to delete embedding file: {e}")

            for i in range(chunk_count):
                embedding_path = self._get_embedding_path(document_id, i)
                if os.path.exists(embedding_path):
                    try:
                        os.remove(embedding_path)
                    except OSError as e:
                        logger.warning(f"Failed to delete legacy embedding file: {e}")

            try:
                self.vector_store.delete(document_id, user_id)
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                logger.warning(f"Failed to delete doc {document_id} from vector_store: {e}")

        # Delete from database
        db_count = self.db.clear_user_documents(user_id)
        return max(len(documents), db_count)

    def get_user_chunks_with_embeddings(self, user_id: int) -> list[dict[str, Any]]:
        """Get all chunks with their embeddings for a user.

        Args:
            user_id: Telegram user ID

        Returns:
            List of chunks with embeddings
        """
        if self.vector_backend == "chroma" and isinstance(self.vector_store, ChromaVectorStore):
            try:
                chroma_chunks = self.vector_store.get_user_chunks_with_embeddings(user_id)
                if chroma_chunks:
                    return chroma_chunks
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                logger.warning(f"Failed to fetch chunks from Chroma, falling back to SQLite: {e}")

        chunks = self.db.get_all_user_chunks(user_id)

        # Load embeddings per document once
        doc_embeddings: dict[int, np.ndarray | None] = {}
        for chunk in chunks:
            doc_id = chunk["document_id"]
            if doc_id not in doc_embeddings:
                npy_path = os.path.join(self.embeddings_path, f"{doc_id}.npy")
                alt_path = os.path.join(self.embeddings_path, f"doc_{doc_id}.npy")
                if os.path.exists(npy_path):
                    try:
                        doc_embeddings[doc_id] = np.load(npy_path)
                    except (OSError, ValueError) as e:
                        logger.warning(f"Failed to load embeddings from {npy_path}: {e}")
                        doc_embeddings[doc_id] = None
                elif os.path.exists(alt_path):
                    try:
                        doc_embeddings[doc_id] = np.load(alt_path)
                    except (OSError, ValueError) as e:
                        logger.warning(f"Failed to load embeddings from {alt_path}: {e}")
                        doc_embeddings[doc_id] = None
                else:
                    doc_embeddings[doc_id] = None

            emb_arr = doc_embeddings[doc_id]
            idx = chunk["chunk_index"]
            if emb_arr is not None and idx < len(emb_arr):
                chunk["embedding"] = emb_arr[idx].tolist()
            else:
                embedding_path = self._get_embedding_path(doc_id, idx)
                if os.path.exists(embedding_path):
                    try:
                        with open(embedding_path) as f:
                            data = json.load(f)
                            chunk["embedding"] = data["embedding"]
                    except (OSError, json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to load legacy embedding: {e}")
                        chunk["embedding"] = None
                else:
                    chunk["embedding"] = None

        return chunks
