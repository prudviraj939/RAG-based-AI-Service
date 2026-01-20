"""
Data schemas for QA (Question-Answering) operations.
Manages question input, retrieval context, and answer generation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class QuestionRequest(BaseModel):
    """Schema for incoming QA requests."""
    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    include_sources: bool = Field(default=True, description="Include source documents in response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the benefits of RAG systems?",
                "top_k": 5,
                "include_sources": True
            }
        }


class RetrievedDocument(BaseModel):
    """Schema for a single retrieved document."""
    doc_id: str
    text: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class QuestionResponse(BaseModel):
    """Schema for QA response."""
    question: str
    answer: str
    reasoning: Optional[str] = Field(None, description="Intermediate reasoning steps")
    retrieved_documents: List[RetrievedDocument] = Field(
        default_factory=list,
        description="Source documents used for answer generation"
    )
    model: str
    processing_time_ms: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is RAG?",
                "answer": "RAG (Retrieval-Augmented Generation) is a technique...",
                "reasoning": "Retrieved 3 relevant documents about RAG...",
                "retrieved_documents": [
                    {
                        "doc_id": "doc_001",
                        "text": "RAG is a technique that combines...",
                        "score": 0.95,
                        "metadata": {"source": "research_paper"}
                    }
                ],
                "model": "gpt-3.5-turbo",
                "processing_time_ms": 1250.5
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str
    code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
