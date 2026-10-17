"""Document retrieval using similarity search."""

from typing import List, Dict, Any, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Retrieves relevant document chunks using similarity search or hybrid retrieval."""
    
    def __init__(self, top_k: int = 5, similarity_threshold: float = 0.3, retrieval_mode: str = "dense"):
        """Initialize document retriever.
        
        Args:
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score threshold
            retrieval_mode: Retrieval strategy ('dense', 'hybrid', 'hybrid_rerank')
        """
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.retrieval_mode = retrieval_mode
        self._hybrid_retriever = None
        
    def retrieve(self, query: str, query_embedding: List[float], 
                 chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks according to configured retrieval_mode.
        
        Args:
            query: Natural language query string
            query_embedding: Query embedding vector
            chunks: Document chunks
            
        Returns:
            List of relevant chunks meeting threshold
        """
        if self.retrieval_mode == "hybrid":
            from .hybrid import HybridRetriever
            if self._hybrid_retriever is None:
                self._hybrid_retriever = HybridRetriever(
                    top_k=self.top_k, 
                    similarity_threshold=self.similarity_threshold
                )
            return self._hybrid_retriever.search(query=query, query_embedding=query_embedding, chunks=chunks)
            
        elif self.retrieval_mode == "hybrid_rerank":
            from .hybrid import HybridRetriever
            from .reranker import CrossEncoderReranker
            if self._hybrid_retriever is None:
                self._hybrid_retriever = HybridRetriever(
                    top_k=20, 
                    similarity_threshold=self.similarity_threshold
                )
            candidates = self._hybrid_retriever.search(query=query, query_embedding=query_embedding, chunks=chunks)
            reranker = CrossEncoderReranker()
            return reranker.rerank(query=query, candidates=candidates, top_k=self.top_k)
            
        else:
            return self.search_similar(query_embedding, chunks)
    
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
        query_vec = np.array(query_embedding, dtype=float)
        chunk_vecs = np.array([c['embedding'] for c in chunks], dtype=float)
        dot_product = np.dot(chunk_vecs, query_vec)
        norm_chunks = np.linalg.norm(chunk_vecs, axis=1)
        norm_query = np.linalg.norm(query_vec)
        denominator = norm_chunks * norm_query
        denominator = np.where(denominator == 0, 1e-10, denominator)
        similarities = dot_product / denominator
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
        
        # Combine chunks with scores that meet or exceed the threshold
        results = []
        for chunk, score in zip(chunks, similarities):
            if score >= self.similarity_threshold:
                result = chunk.copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top-k results
        return results[:self.top_k]
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string with page-level citations.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        context_parts = []
        for chunk in chunks:
            filename = chunk.get('filename', 'Unknown')
            content = chunk.get('content', '')
            p_start = chunk.get('page_start')
            p_end = chunk.get('page_end')
            page = chunk.get('page')
            
            if p_start is not None and p_end is not None:
                page_str = f"p. {p_start}" if p_start == p_end else f"p. {p_start}-{p_end}"
            elif page is not None:
                page_str = f"p. {page}"
            else:
                page_str = "p. 1"
                
            context_parts.append(f"[{filename}, {page_str}]\n{content}")
        
        return "\n\n".join(context_parts)
