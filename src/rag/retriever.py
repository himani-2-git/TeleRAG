"""Document retrieval using similarity search."""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Retrieves relevant document chunks using similarity search."""
    
    def __init__(self, top_k: int = 5, similarity_threshold: float = 0.7):
        """Initialize document retriever.
        
        Args:
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score threshold
        """
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def search_similar(self, query_embedding: List[float], 
                      chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search for similar chunks using cosine similarity.
        
        Args:
            query_embedding: Embedding vector for the query
            chunks: List of chunks with embeddings
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if not chunks:
            logger.warning("No chunks available for search")
            return []
        
        # Filter chunks that have embeddings
        valid_chunks = [c for c in chunks if c.get('embedding') is not None]
        
        if not valid_chunks:
            logger.warning("No chunks with embeddings found")
            return []
        
        # Calculate similarities
        similarities = self._calculate_similarities(query_embedding, valid_chunks)
        
        # Rank and filter results
        results = self._rank_results(valid_chunks, similarities)
        
        logger.info(f"Found {len(results)} relevant chunks")
        return results
    
    def _calculate_similarities(self, query_embedding: List[float], 
                               chunks: List[Dict[str, Any]]) -> List[float]:
        """Calculate cosine similarity between query and chunks.
        
        Args:
            query_embedding: Query embedding vector
            chunks: List of chunks with embeddings
            
        Returns:
            List of similarity scores
        """
        query_vec = np.array(query_embedding).reshape(1, -1)
        chunk_vecs = np.array([c['embedding'] for c in chunks])
        
        similarities = cosine_similarity(query_vec, chunk_vecs)[0]
        return similarities.tolist()
    
    def _rank_results(self, chunks: List[Dict[str, Any]], 
                     similarities: List[float]) -> List[Dict[str, Any]]:
        """Rank and filter results by similarity.
        
        Args:
            chunks: List of chunks
            similarities: List of similarity scores
            
        Returns:
            Ranked and filtered chunks with scores
        """
        # Log similarity scores for debugging
        max_score = max(similarities) if similarities else 0
        min_score = min(similarities) if similarities else 0
        logger.debug(f"Similarity scores - Max: {max_score:.3f}, Min: {min_score:.3f}, Threshold: {self.similarity_threshold}")
        
        # Combine chunks with scores
        results = []
        for chunk, score in zip(chunks, similarities):
            if score >= self.similarity_threshold:
                result = chunk.copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        # If no results meet threshold, return top results anyway
        if not results and chunks:
            logger.warning(f"No chunks met threshold {self.similarity_threshold}, returning top {self.top_k} anyway")
            for chunk, score in zip(chunks, similarities):
                result = chunk.copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top-k results
        return results[:self.top_k]
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            filename = chunk.get('filename', 'Unknown')
            content = chunk['content']
            score = chunk.get('similarity_score', 0)
            
            context_parts.append(
                f"[Source {i} - {filename} (relevance: {score:.2f})]\n{content}"
            )
        
        return "\n\n".join(context_parts)
