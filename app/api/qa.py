"""
FastAPI routes for QA and retrieval.
Provides HTTP endpoints for searching and answering questions.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from typing import List

from app.schemas.document import SearchQuery, DocumentChunk
from app.schemas.qa import QuestionRequest, QuestionResponse
from app.services.retrieval_service import RetrievalService
from app.services.agent_service import AgentService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["qa"])
retrieval_service = RetrievalService()
agent_service = AgentService()


@router.get(
    "/search",
    response_model=List[DocumentChunk],
    summary="Search documents",
    description="Find documents using semantic or lexical search"
)
async def search_documents(
    query: str,
    top_k: int = 5,
    search_type: str = "hybrid"
) -> List[DocumentChunk]:
    """
    Search for documents.
    
    - **query**: Search query text
    - **top_k**: Number of results to return (1-20)
    - **search_type**: "semantic", "lexical", or "hybrid"
    """
    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    if not (1 <= top_k <= 20):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k must be between 1 and 20"
        )
    
    try:
        if search_type == "semantic":
            results = await retrieval_service.retrieve_semantic(query, top_k)
        elif search_type == "lexical":
            results = await retrieval_service.retrieve_lexical(query, top_k)
        elif search_type == "hybrid":
            results = await retrieval_service.retrieve_hybrid(query, top_k)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        return results
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed"
        )


@router.post(
    "/qa",
    response_model=QuestionResponse,
    summary="Answer a question",
    description="Answer questions using RAG (Retrieval-Augmented Generation)"
)
async def answer_question(question_request: QuestionRequest) -> QuestionResponse:
    """
    Answer a natural language question using RAG.
    
    Process:
    1. Retrieve relevant documents
    2. Build context from documents
    3. Generate answer using LLM
    4. Return answer with sources and reasoning
    
    - **question**: The question to answer
    - **top_k**: Number of documents to retrieve
    - **include_sources**: Include source documents and reasoning in response
    """
    if not question_request.question or not question_request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )
    
    try:
        response = await agent_service.answer_question(question_request)
        return response
    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate answer"
        )


@router.get(
    "/qa/health",
    summary="Check QA service health",
    description="Verify that the QA pipeline is operational"
)
async def qa_health() -> dict:
    """
    Health check for QA service.
    
    Returns:
        Service status and configuration
    """
    try:
        from app.core.config import settings
        from app.core.elastic import get_es_client
        
        es = get_es_client()
        doc_count = es.get_document_count()
        
        return {
            "status": "healthy",
            "llm_provider": settings.llm_provider,
            "embeddings_provider": settings.embeddings_provider,
            "documents_available": doc_count > 0,
            "total_documents": doc_count
        }
    except Exception as e:
        logger.error(f"QA health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QA service unavailable"
        )
