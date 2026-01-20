"""
Data schemas for document management.
Provides validation and type safety for document operations.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class DocumentCreate(BaseModel):
    """Schema for creating/ingesting a document."""
    id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., min_length=1, description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Additional metadata (source, author, date, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_001",
                "text": "Python is a programming language...",
                "metadata": {
                    "source": "wikipedia",
                    "category": "programming"
                }
            }
        }


class DocumentResponse(BaseModel):
    """Schema for document response."""
    id: str
    text: str
    metadata: Dict[str, Any]
    created_at: datetime
    embedding_generated: bool

    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_001",
                "text": "Python is a programming language...",
                "metadata": {"source": "wikipedia"},
                "created_at": "2026-01-20T10:30:00",
                "embedding_generated": True
            }
        }


class SearchQuery(BaseModel):
    """Schema for search queries."""
    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How does Python handle memory?",
                "top_k": 5
            }
        }


class DocumentChunk(BaseModel):
    """Schema for document chunks retrieved from search."""
    doc_id: str
    text: str
    score: float = Field(..., description="Relevance score")
    metadata: Optional[Dict[str, Any]] = None
