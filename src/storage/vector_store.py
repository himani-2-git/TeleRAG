"""Vector store abstraction and implementations for NumPy and Chroma backends."""

from abc import ABC, abstractmethod
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .database import Database

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract interface for vector backends."""

    @abstractmethod
    def add(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        user_id: int
    ) -> None:
        """Add chunks and their embeddings for a specific user.

        Args:
            chunks: List of chunk dictionaries containing content and metadata.
            embeddings: Corresponding embedding vectors.
            user_id: Telegram user ID for isolation.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        user_id: int,
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search top-k most similar chunks for a specific user.

        Args:
            query_embedding: Query dense vector.
            user_id: Telegram user ID for scoping.
            top_k: Maximum number of results to return.

        Returns:
            List of (chunk_dict, cosine_similarity_score) tuples ordered descending by similarity.
        """
        pass

    @abstractmethod
    def delete(self, doc_id: Any, user_id: int) -> bool:
        """Delete chunks belonging to a document and user.

        Args:
            doc_id: Document ID or filename.
            user_id: Telegram user ID.

        Returns:
            True if deletion occurred, False otherwise.
        """
        pass

    @abstractmethod
    def list(self, user_id: int) -> List[Dict[str, Any]]:
        """List documents or chunk metadata belonging to a user.

        Args:
            user_id: Telegram user ID.

        Returns:
            List of document metadata dictionaries.
        """
        pass


class NumPyVectorStore(VectorStore):
    """Default vector store backed by SQLite metadata and JSON embedding files, computed via NumPy."""

    def __init__(
        self,
        db_path: str = "data/documents.db",
        embeddings_path: str = "data/embeddings",
        db: Optional[Database] = None
    ):
        self.db = db if db is not None else Database(db_path)
        self.embeddings_path = embeddings_path
        os.makedirs(self.embeddings_path, exist_ok=True)

    def _get_embedding_path(self, document_id: int, chunk_index: int) -> str:
        return os.path.join(self.embeddings_path, f"doc_{document_id}_chunk_{chunk_index}.json")

    def add(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        user_id: int
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )
        if not chunks:
            return

        first_doc_id = chunks[0].get("document_id") or chunks[0].get("doc_id")
        existing_docs = self.db.get_user_documents(user_id)
        doc_exists = any(d["id"] == first_doc_id for d in existing_docs) if first_doc_id else False

        if not doc_exists:
            filename = chunks[0].get("filename", "document.txt")
            source_type = chunks[0].get("source_type", "pdf")
            chunks_data = []
            for i, c in enumerate(chunks):
                content = c.get("content", "")
                chunks_data.append({
                    "chunk_index": c.get("chunk_index", i),
                    "content": content,
                    "char_count": c.get("char_count", len(content)),
                    "page": c.get("page"),
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                    "section_label": c.get("section_label"),
                    "source_type": c.get("source_type", source_type),
                    "embedding_path": None
                })
            doc_id = self.db.store_document(
                user_id=user_id,
                filename=filename,
                file_hash="",
                file_size=0,
                chunks=chunks_data,
                source_type=source_type
            )
            for i, c in enumerate(chunks):
                c["document_id"] = doc_id
        else:
            doc_id = first_doc_id

        for i, emb in enumerate(embeddings):
            chunk_idx = chunks[i].get("chunk_index", i)
            emb_path = self._get_embedding_path(doc_id, chunk_idx)
            with open(emb_path, "w") as f:
                json.dump({"embedding": emb}, f)

    def search(
        self,
        query_embedding: List[float],
        user_id: int,
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        chunks = self.get_user_chunks_with_embeddings(user_id)
        valid_chunks = [c for c in chunks if c.get("embedding") is not None]
        if not valid_chunks:
            return []

        query_vec = np.array(query_embedding, dtype=float)
        chunk_vecs = np.array([c["embedding"] for c in valid_chunks], dtype=float)
        dot_product = np.dot(chunk_vecs, query_vec)
        norm_chunks = np.linalg.norm(chunk_vecs, axis=1)
        norm_query = np.linalg.norm(query_vec)
        denominator = norm_chunks * norm_query
        denominator = np.where(denominator == 0, 1e-10, denominator)
        similarities = dot_product / denominator

        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        results: List[Tuple[Dict[str, Any], float]] = []
        for idx in ranked_indices:
            chunk_copy = dict(valid_chunks[idx])
            sim = float(similarities[idx])
            chunk_copy["similarity"] = sim
            chunk_copy["similarity_score"] = sim
            results.append((chunk_copy, sim))
        return results

    def delete(self, doc_id: Any, user_id: int) -> bool:
        docs = self.db.get_user_documents(user_id)
        matched = None
        for d in docs:
            if d["id"] == doc_id or str(d["id"]) == str(doc_id) or d["filename"] == str(doc_id):
                matched = d
                break
        if not matched:
            return False

        for i in range(matched.get("chunk_count", 0)):
            emb_path = self._get_embedding_path(matched["id"], i)
            if os.path.exists(emb_path):
                try:
                    os.remove(emb_path)
                except Exception as e:
                    logger.warning("Failed to delete embedding file %s: %s", emb_path, e)

        return self.db.delete_document(user_id, matched["filename"])

    def list(self, user_id: int) -> List[Dict[str, Any]]:
        return self.db.get_user_documents(user_id)

    def get_user_chunks_with_embeddings(self, user_id: int) -> List[Dict[str, Any]]:
        chunks = self.db.get_all_user_chunks(user_id)
        for chunk in chunks:
            emb_path = self._get_embedding_path(chunk["document_id"], chunk["chunk_index"])
            if os.path.exists(emb_path):
                try:
                    with open(emb_path, "r") as f:
                        data = json.load(f)
                        chunk["embedding"] = data["embedding"]
                except Exception as e:
                    logger.warning("Failed to load embedding: %s", e)
                    chunk["embedding"] = None
            else:
                chunk["embedding"] = None
        return chunks


class ChromaVectorStore(VectorStore):
    """Vector store backed by ChromaDB with cosine metric and per-user isolation."""

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "telerag_chunks"
    ):
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "Install it using: pip install -r requirements-chroma.txt"
            ) from e

        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        settings = chromadb.Settings(anonymized_telemetry=False)
        self.client = chromadb.PersistentClient(path=persist_directory, settings=settings)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        user_id: int
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )
        if not chunks:
            return

        ids = []
        docs = []
        metadatas = []

        first_doc_id = chunks[0].get("document_id") or chunks[0].get("doc_id") or 1

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            doc_id = chunk.get("document_id") or chunk.get("doc_id") or first_doc_id
            chunk_idx = chunk.get("chunk_index", i)
            chunk_id = f"user_{user_id}_doc_{doc_id}_chunk_{chunk_idx}"
            ids.append(chunk_id)

            content = chunk.get("content", "")
            docs.append(content)

            meta = {
                "user_id": int(user_id),
                "document_id": int(doc_id) if str(doc_id).isdigit() else str(doc_id),
                "chunk_index": int(chunk_idx),
                "filename": str(chunk.get("filename") or ""),
                "page": int(chunk.get("page") or 0) if chunk.get("page") is not None else -1,
                "page_start": int(chunk.get("page_start") or 0) if chunk.get("page_start") is not None else -1,
                "page_end": int(chunk.get("page_end") or 0) if chunk.get("page_end") is not None else -1,
                "section_label": str(chunk.get("section_label") or ""),
                "source_type": str(chunk.get("source_type") or "pdf"),
            }
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=docs,
            metadatas=metadatas
        )

    def search(
        self,
        query_embedding: List[float],
        user_id: int,
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        total_count = self.collection.count()
        if total_count == 0:
            return []

        n_results = min(top_k, total_count)
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"user_id": int(user_id)},
                include=["documents", "metadatas", "distances", "embeddings"]
            )
        except Exception as e:
            logger.warning("Chroma query failed: %s", e)
            return []

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        hits: List[Tuple[Dict[str, Any], float]] = []
        ids_list = results["ids"][0]
        docs_list = results.get("documents", [[]])[0] if results.get("documents") is not None else []
        metas_list = results.get("metadatas", [[]])[0] if results.get("metadatas") is not None else []
        dists_list = results.get("distances", [[]])[0] if results.get("distances") is not None else []
        embs_list = results.get("embeddings", [[]])[0] if results.get("embeddings") is not None else None

        for i in range(len(ids_list)):
            dist = dists_list[i]
            # Convert cosine distance (1 - cosine) to cosine similarity
            sim = max(-1.0, min(1.0, 1.0 - float(dist)))
            meta = metas_list[i] if metas_list is not None and i < len(metas_list) else {}
            doc_text = docs_list[i] if docs_list is not None and i < len(docs_list) else ""
            emb = embs_list[i] if embs_list is not None and i < len(embs_list) else None

            chunk_dict = {
                "content": doc_text,
                "document_id": meta.get("document_id"),
                "chunk_index": meta.get("chunk_index", 0),
                "filename": meta.get("filename", ""),
                "page": meta.get("page") if meta.get("page", -1) != -1 else None,
                "page_start": meta.get("page_start") if meta.get("page_start", -1) != -1 else None,
                "page_end": meta.get("page_end") if meta.get("page_end", -1) != -1 else None,
                "section_label": meta.get("section_label") if meta.get("section_label") else None,
                "source_type": meta.get("source_type", "pdf"),
                "similarity": sim,
                "similarity_score": sim,
                "embedding": emb
            }
            hits.append((chunk_dict, sim))

        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:top_k]

    def delete(self, doc_id: Any, user_id: int) -> bool:
        user_items = self.collection.get(where={"user_id": int(user_id)})
        if not user_items or not user_items.get("ids"):
            return False

        target_ids = []
        for chunk_id, meta in zip(user_items["ids"], user_items["metadatas"]):
            m_doc_id = meta.get("document_id")
            m_filename = meta.get("filename")
            if (m_doc_id == doc_id or 
                str(m_doc_id) == str(doc_id) or 
                m_filename == str(doc_id)):
                target_ids.append(chunk_id)

        if not target_ids:
            return False

        self.collection.delete(ids=target_ids)
        return True

    def list(self, user_id: int) -> List[Dict[str, Any]]:
        user_items = self.collection.get(where={"user_id": int(user_id)})
        if not user_items or not user_items.get("metadatas"):
            return []

        docs_map: Dict[Any, Dict[str, Any]] = {}
        for meta in user_items["metadatas"]:
            doc_id = meta.get("document_id")
            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "id": doc_id,
                    "filename": meta.get("filename", ""),
                    "source_type": meta.get("source_type", "pdf"),
                    "chunk_count": 0
                }
            docs_map[doc_id]["chunk_count"] += 1

        return list(docs_map.values())

    def get_user_chunks_with_embeddings(self, user_id: int) -> List[Dict[str, Any]]:
        user_items = self.collection.get(
            where={"user_id": int(user_id)},
            include=["documents", "metadatas", "embeddings"]
        )
        if not user_items or not user_items.get("ids"):
            return []

        chunks: List[Dict[str, Any]] = []
        for i, chunk_id in enumerate(user_items["ids"]):
            meta = user_items["metadatas"][i]
            doc = user_items["documents"][i]
            emb = user_items["embeddings"][i] if user_items.get("embeddings") is not None else None
            chunks.append({
                "content": doc,
                "document_id": meta.get("document_id"),
                "chunk_index": meta.get("chunk_index", 0),
                "filename": meta.get("filename", ""),
                "page": meta.get("page") if meta.get("page", -1) != -1 else None,
                "page_start": meta.get("page_start") if meta.get("page_start", -1) != -1 else None,
                "page_end": meta.get("page_end") if meta.get("page_end", -1) != -1 else None,
                "section_label": meta.get("section_label") if meta.get("section_label") else None,
                "source_type": meta.get("source_type", "pdf"),
                "embedding": emb
            })
        chunks.sort(key=lambda x: (x.get("document_id", 0), x.get("chunk_index", 0)))
        return chunks


def get_vector_store(
    backend: str = "numpy",
    db_path: str = "data/documents.db",
    embeddings_path: str = "data/embeddings",
    chroma_path: str = "data/chroma",
    db: Optional[Database] = None
) -> VectorStore:
    """Factory to retrieve vector store backend with graceful fallback.

    Args:
        backend: 'numpy' or 'chroma'.
        db_path: Path to SQLite DB.
        embeddings_path: Path to embeddings folder.
        chroma_path: Path to Chroma directory.
        db: Optional shared Database instance.

    Returns:
        VectorStore instance.
    """
    backend_norm = (backend or "numpy").strip().lower()
    if backend_norm == "chroma":
        try:
            import chromadb
            return ChromaVectorStore(persist_directory=chroma_path)
        except ImportError:
            logger.warning(
                "chromadb is requested (VECTOR_BACKEND=chroma) but not installed. "
                "Falling back to NumPyVectorStore."
            )
            return NumPyVectorStore(db_path=db_path, embeddings_path=embeddings_path, db=db)
    return NumPyVectorStore(db_path=db_path, embeddings_path=embeddings_path, db=db)
