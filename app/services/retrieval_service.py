"""
Document retrieval service.
Handles semantic and text-based document retrieval.
"""

import logging
from typing import List, Dict, Any, Optional

from app.core.elastic import get_es_client
from app.core.embeddings import get_embeddings_client
from app.schemas.document import DocumentChunk

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Service for retrieving relevant documents.
    
    Provides both:
    - Semantic retrieval (vector similarity)
    - Lexical retrieval (text matching)
    """
    
    def __init__(self):
        """Initialize retrieval service with dependencies."""
        self.es_client = get_es_client()
        self.embeddings_client = get_embeddings_client()
    
    async def retrieve_semantic(
        self,
        query: str,
        top_k: int = 5
    ) -> List[DocumentChunk]:
        """
        Retrieve documents using semantic similarity.
        
        Process:
        1. Embed the query
        2. Search Elasticsearch using vector similarity
        3. Return top-k results
        
        Args:
            query: Query text
            top_k: Number of results to return
        
        Returns:
            List of relevant document chunks
        """
        try:
            logger.info(f"Semantic retrieval for query: {query}")
            
            # Generate query embedding
            query_embedding = self.embeddings_client.embed_text(query)
            
            # Search in Elasticsearch
            results = self.es_client.search_by_vector(
                embedding=query_embedding,
                top_k=top_k
            )
            
            # Convert to DocumentChunk objects
            chunks = [
                DocumentChunk(
                    doc_id=result["doc_id"],
                    text=result["text"],
                    score=result["score"],
                    metadata=result.get("metadata")
                )
                for result in results
            ]
            
            logger.info(f"Retrieved {len(chunks)} documents via semantic search")
            return chunks
        
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {str(e)}")
            raise
    
    async def retrieve_lexical(
        self,
        query: str,
        top_k: int = 5
    ) -> List[DocumentChunk]:
        """
        Retrieve documents using text matching (BM25).
        
        Args:
            query: Query text
            top_k: Number of results to return
        
        Returns:
            List of relevant document chunks
        """
        try:
            logger.info(f"Lexical retrieval for query: {query}")
            
            # Search using text matching
            results = self.es_client.search_text(
                query_text=query,
                top_k=top_k
            )
            
            # Convert to DocumentChunk objects
            chunks = [
                DocumentChunk(
                    doc_id=result["doc_id"],
                    text=result["text"],
                    score=result["score"],
                    metadata=result.get("metadata")
                )
                for result in results
            ]
            
            logger.info(f"Retrieved {len(chunks)} documents via lexical search")
            return chunks
        
        except Exception as e:
            logger.error(f"Error in lexical retrieval: {str(e)}")
            raise
    
    async def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.6
    ) -> List[DocumentChunk]:
        """
        Retrieve documents using hybrid (semantic + lexical) search.
        
        Combines results from both semantic and lexical search,
        weighted by semantic_weight.
        
        Args:
            query: Query text
            top_k: Number of results to return
            semantic_weight: Weight for semantic results (0.0-1.0)
        
        Returns:
            Combined list of relevant document chunks
        """
        try:
            logger.info(f"Hybrid retrieval for query: {query}")
            
            # Run both retrievals in parallel would be better, but for simplicity:
            semantic_results = await self.retrieve_semantic(query, top_k)
            lexical_results = await self.retrieve_lexical(query, top_k)
            
            # Combine and deduplicate by doc_id
            combined = {}
            
            # Add semantic results
            for chunk in semantic_results:
                combined[chunk.doc_id] = {
                    "chunk": chunk,
                    "semantic_score": chunk.score,
                    "lexical_score": 0.0
                }
            
            # Add/update lexical results
            for chunk in lexical_results:
                if chunk.doc_id in combined:
                    combined[chunk.doc_id]["lexical_score"] = chunk.score
                else:
                    combined[chunk.doc_id] = {
                        "chunk": chunk,
                        "semantic_score": 0.0,
                        "lexical_score": chunk.score
                    }
            
            # Calculate hybrid score
            results_with_hybrid_score = []
            for doc_id, data in combined.items():
                hybrid_score = (
                    data["semantic_score"] * semantic_weight +
                    data["lexical_score"] * (1 - semantic_weight)
                )
                chunk = data["chunk"]
                chunk.score = hybrid_score
                results_with_hybrid_score.append(chunk)
            
            # Sort by hybrid score and return top-k
            sorted_results = sorted(
                results_with_hybrid_score,
                key=lambda x: x.score,
                reverse=True
            )[:top_k]
            
            logger.info(f"Retrieved {len(sorted_results)} documents via hybrid search")
            return sorted_results
        
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            raise
