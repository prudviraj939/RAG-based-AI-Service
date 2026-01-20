"""
Main FastAPI application entry point.
Configures and initializes the RAG service API.
"""

import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import settings, validate_settings
from app.api import documents, qa

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management.
    
    Startup: Validate configuration and initialize services
    Shutdown: Cleanup resources
    """
    # Startup
    logger.info("=== RAG Service Starting ===")
    
    # Validate settings
    validation_errors = validate_settings()
    if validation_errors:
        logger.error(f"Configuration errors: {validation_errors}")
        sys.exit(1)
    
    # Initialize services
    try:
        from app.core.elastic import get_es_client
        from app.core.embeddings import get_embeddings_client
        from app.core.llm import get_llm_client
        
        logger.info("Initializing Elasticsearch client...")
        es_client = get_es_client()
        logger.info(f"Connected to Elasticsearch - Documents indexed: {es_client.get_document_count()}")
        
        logger.info("Initializing embeddings client...")
        embeddings = get_embeddings_client()
        logger.info(f"Embeddings client ready - Provider: {settings.embeddings_provider}")
        
        logger.info("Initializing LLM client...")
        llm = get_llm_client()
        logger.info(f"LLM client ready - Provider: {settings.llm_provider}")
        
        logger.info("=== RAG Service Ready ===")
    
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        sys.exit(1)
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Service...")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="RAG-based AI Service with LangGraph and Vector Databases",
    lifespan=lifespan
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# Health check endpoint
@app.get(
    "/health",
    summary="API Health Check",
    description="Basic health check endpoint"
)
async def health() -> dict:
    """
    Check API health status.
    
    Returns basic system information.
    """
    try:
        from app.core.elastic import get_es_client
        es = get_es_client()
        return {
            "status": "healthy",
            "service": settings.api_title,
            "version": settings.api_version,
            "documents_indexed": es.get_document_count()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Root endpoint
@app.get(
    "/",
    summary="API Info",
    description="Get API information"
)
async def root() -> dict:
    """
    Get API information and available endpoints.
    """
    return {
        "title": settings.api_title,
        "version": settings.api_version,
        "endpoints": {
            "documents": "/documents - Document ingestion",
            "search": "/search - Document search",
            "qa": "/qa - Question answering",
            "health": "/health - Health check",
            "docs": "/docs - Interactive API documentation"
        }
    }


# Include routers
app.include_router(documents.router)
app.include_router(qa.router)


if __name__ == "__main__":
    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
