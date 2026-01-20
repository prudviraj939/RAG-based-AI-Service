"""
Document ingestion service.
Handles document upload, validation, and indexing with embeddings.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

from app.core.elastic import get_es_client
from app.core.embeddings import get_embeddings_client
from app.schemas.document import DocumentCreate, DocumentResponse

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Service for ingesting documents into the system.
    
    Flow:
    1. Validate document input
    2. Generate embedding from text
    3. Index in Elasticsearch with embedding
    4. Return confirmation
    """
    
    def __init__(self):
        """Initialize ingestion service with dependencies."""
        self.es_client = get_es_client()
        self.embeddings_client = get_embeddings_client()
    
    async def ingest_document(
        self,
        doc_create: DocumentCreate
    ) -> DocumentResponse:
        """
        Ingest a single document.
        
        Args:
            doc_create: Document creation request
        
        Returns:
            DocumentResponse with confirmation
        
        Raises:
            ValueError: If document validation fails
            Exception: If embedding or indexing fails
        """
        try:
            # Generate embedding for the document text
            logger.info(f"Generating embedding for document {doc_create.id}")
            embedding = self.embeddings_client.embed_text(doc_create.text)
            
            # Index the document
            logger.info(f"Indexing document {doc_create.id} in Elasticsearch")
            success = self.es_client.index_document(
                doc_id=doc_create.id,
                text=doc_create.text,
                embedding=embedding,
                metadata=doc_create.metadata or {}
            )
            
            if not success:
                raise Exception(f"Failed to index document {doc_create.id}")
            
            # Return success response
            response = DocumentResponse(
                id=doc_create.id,
                text=doc_create.text,
                metadata=doc_create.metadata or {},
                created_at=datetime.utcnow(),
                embedding_generated=True
            )
            
            logger.info(f"Successfully ingested document {doc_create.id}")
            return response
        
        except Exception as e:
            logger.error(f"Error ingesting document {doc_create.id}: {str(e)}")
            raise
    
    async def ingest_documents(
        self,
        documents: list[DocumentCreate]
    ) -> list[DocumentResponse]:
        """
        Ingest multiple documents in batch.
        
        Args:
            documents: List of documents to ingest
        
        Returns:
            List of DocumentResponse objects
        """
        results = []
        for doc in documents:
            try:
                result = await self.ingest_document(doc)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to ingest document {doc.id}: {str(e)}")
                # Continue with next document on failure
        
        return results
