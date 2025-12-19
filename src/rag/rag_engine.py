"""RAG Engine for retrieval and generation."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from .embeddings import EmbeddingEngine
from ..storage.database import DatabaseManager, DocumentChunk

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Query result structure."""
    chunks: List[DocumentChunk]
    similarity_scores: List[float]
    answer: str
    sources: List[str]

class RAGEngine:
    """Retrieval-Augmented Generation engine."""
    
    def __init__(self, db_manager: DatabaseManager, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG engine.
        
        Args:
            db_manager: Database manager instance
            embedding_model: Name of embedding model to use
        """
        self.db_manager = db_manager
        self.embedding_engine = EmbeddingEngine(embedding_model)
    
    def generate_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of embedding vectors
        """
        texts = [chunk['content'] for chunk in chunks]
        return self.embedding_engine.generate_embeddings(texts)
    
    def retrieve_context(self, user_id: int, question: str, top_k: int = 5, 
                        similarity_threshold: float = 0.3) -> Tuple[List[DocumentChunk], List[float]]:
        """
        Retrieve relevant context for a question.
        
        Args:
            user_id: Telegram user ID
            question: User question
            top_k: Number of top chunks to retrieve
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (relevant_chunks, similarity_scores)
        """
        try:
            # Generate embedding for the question
            question_embedding = self.embedding_engine.generate_single_embedding(question)
            
            if not question_embedding:
                logger.warning("Failed to generate question embedding")
                return [], []
            
            # Get all user chunks from database
            user_chunks = self.db_manager.get_user_chunks(user_id)
            
            if not user_chunks:
                logger.info(f"No chunks found for user {user_id}")
                return [], []
            
            # Extract embeddings from chunks
            chunk_embeddings = [chunk.embedding for chunk in user_chunks]
            
            # Find most similar chunks
            similar_indices = self.embedding_engine.find_most_similar(
                question_embedding, 
                chunk_embeddings, 
                top_k=top_k, 
                threshold=similarity_threshold
            )
            
            if not similar_indices:
                logger.info(f"No similar chunks found for question: {question}")
                return [], []
            
            # Get the relevant chunks and their scores
            relevant_chunks = []
            similarity_scores = []
            
            for idx, score in similar_indices:
                relevant_chunks.append(user_chunks[idx])
                similarity_scores.append(score)
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for user {user_id}")
            return relevant_chunks, similarity_scores
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return [], []
    
    def format_context_for_llm(self, chunks: List[DocumentChunk], max_context_length: int = 4000) -> str:
        """
        Format retrieved chunks into context for LLM.
        
        Args:
            chunks: List of relevant chunks
            max_context_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks):
            # Format chunk with source information
            chunk_text = f"[Source {i+1} - Page {chunk.page_number}]: {chunk.content}"
            
            # Check if adding this chunk would exceed the limit
            if current_length + len(chunk_text) > max_context_length:
                if current_length == 0:  # If even the first chunk is too long
                    # Truncate the first chunk
                    available_space = max_context_length - len(f"[Source 1 - Page {chunk.page_number}]: ") - 20
                    truncated_content = chunk.content[:available_space] + "..."
                    context_parts.append(f"[Source 1 - Page {chunk.page_number}]: {truncated_content}")
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text) + 2  # +2 for newlines
        
        return "\n\n".join(context_parts)
    
    def create_sources_list(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Create a list of source references.
        
        Args:
            chunks: List of relevant chunks
            
        Returns:
            List of source strings
        """
        sources = []
        seen_docs = set()
        
        for chunk in chunks:
            # Get document info from database
            documents = self.db_manager.get_user_documents(chunk.user_id)
            doc_info = None
            
            for doc in documents:
                if doc.document_id == chunk.document_id:
                    doc_info = doc
                    break
            
            if doc_info and doc_info.filename not in seen_docs:
                sources.append(f"{doc_info.filename} (Page {chunk.page_number})")
                seen_docs.add(doc_info.filename)
        
        return sources
    
    def rank_chunks_by_relevance(self, chunks: List[DocumentChunk], scores: List[float], 
                                question: str) -> List[Tuple[DocumentChunk, float]]:
        """
        Rank chunks by relevance considering both similarity score and content quality.
        
        Args:
            chunks: List of document chunks
            scores: List of similarity scores
            question: Original question
            
        Returns:
            List of (chunk, final_score) tuples sorted by relevance
        """
        ranked_chunks = []
        
        for chunk, score in zip(chunks, scores):
            # Base score is the similarity score
            final_score = score
            
            # Boost score for longer, more substantial chunks
            content_length_factor = min(len(chunk.content) / 500, 1.2)  # Cap at 20% boost
            final_score *= content_length_factor
            
            # Boost score if question keywords appear in chunk
            question_words = set(question.lower().split())
            chunk_words = set(chunk.content.lower().split())
            keyword_overlap = len(question_words.intersection(chunk_words)) / len(question_words)
            keyword_boost = 1 + (keyword_overlap * 0.3)  # Up to 30% boost
            final_score *= keyword_boost
            
            ranked_chunks.append((chunk, final_score))
        
        # Sort by final score (descending)
        ranked_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_chunks