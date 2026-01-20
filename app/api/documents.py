"""
FastAPI routes for document ingestion.
Provides HTTP endpoints for uploading and managing documents.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from typing import List

from app.schemas.document import DocumentCreate, DocumentResponse, SearchQuery
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])
ingestion_service = IngestionService()


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a single document",
    description="Upload and index a document with embedding generation"
)
async def ingest_document(document: DocumentCreate) -> DocumentResponse:
    """
    Ingest a single document.
    
    - **id**: Unique document identifier
    - **text**: Document content
    - **metadata**: Optional additional metadata
    """
    try:
        result = await ingestion_service.ingest_document(document)
        return result
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to ingest document"
        )


@router.post(
    "/batch",
    response_model=List[DocumentResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Ingest multiple documents",
    description="Batch upload and index documents"
)
async def ingest_documents_batch(documents: List[DocumentCreate]) -> List[DocumentResponse]:
    """
    Ingest multiple documents in batch.
    
    Processes documents sequentially, continuing on individual failures.
    """
    if not documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one document is required"
        )
    
    if len(documents) > 100:
        raise HTTPException(
            status_code=status.HTTP_413_PAYLOAD_TOO_LARGE,
            detail="Maximum 100 documents per batch"
        )
    
    try:
        results = await ingestion_service.ingest_documents(documents)
        if not results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to ingest any documents"
            )
        return results
    except Exception as e:
        logger.error(f"Error ingesting batch: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch ingestion failed"
        )


@router.get(
    "/health",
    summary="Check document service health",
    description="Verify that the document storage is accessible"
)
async def health_check() -> dict:
    """
    Health check endpoint.
    
    Returns:
        Status and document count
    """
    try:
        from app.core.elastic import get_es_client
        es = get_es_client()
        count = es.get_document_count()
        return {
            "status": "healthy",
            "documents_indexed": count
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document storage unavailable"
        )
