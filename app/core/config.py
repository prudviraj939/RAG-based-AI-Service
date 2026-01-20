"""
Configuration and environment setup.
Manages API keys, model parameters, and service initialization.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings from environment variables.
    Demonstrates production-ready configuration management.
    """
    
    # API Configuration
    api_title: str = "RAG-based AI Service"
    api_version: str = "0.1.0"
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    
    # LLM Configuration
    llm_provider: str = Field(default="openai", validation_alias="LLM_PROVIDER")  # "openai", "huggingface"
    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", validation_alias="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    
    # Hugging Face Configuration (alternative LLM)
    huggingface_api_key: Optional[str] = Field(default=None, validation_alias="HUGGINGFACE_API_KEY")
    huggingface_model: str = Field(default="mistralai/Mistral-7B-Instruct-v0.1", validation_alias="HUGGINGFACE_MODEL")
    
    # Embeddings Configuration
    embeddings_provider: str = Field(default="openai", validation_alias="EMBEDDINGS_PROVIDER")  # "openai", "huggingface"
    embeddings_model: str = Field(default="text-embedding-3-small", validation_alias="EMBEDDINGS_MODEL")
    embedding_dimension: int = Field(default=1536, validation_alias="EMBEDDING_DIMENSION")
    
    # Elasticsearch Configuration
    elasticsearch_host: str = Field(default="localhost", validation_alias="ES_HOST")
    elasticsearch_port: int = Field(default=9200, validation_alias="ES_PORT")
    elasticsearch_username: Optional[str] = Field(default=None, validation_alias="ES_USERNAME")
    elasticsearch_password: Optional[str] = Field(default=None, validation_alias="ES_PASSWORD")
    elasticsearch_index: str = Field(default="documents", validation_alias="ES_INDEX")
    elasticsearch_use_ssl: bool = Field(default=False, validation_alias="ES_USE_SSL")
    
    # Vector Store Configuration (for similarity search)
    vector_store_type: str = Field(default="faiss", validation_alias="VECTOR_STORE_TYPE")  # "faiss", "pinecone"
    vector_store_path: str = Field(default="./data/vector_store", validation_alias="VECTOR_STORE_PATH")
    
    # RAG Configuration
    retrieval_top_k: int = Field(default=5, validation_alias="RETRIEVAL_TOP_K")
    chunk_size: int = Field(default=512, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, validation_alias="CHUNK_OVERLAP")
    
    # Agent Configuration
    agent_timeout: int = Field(default=30, validation_alias="AGENT_TIMEOUT")
    agent_max_iterations: int = Field(default=10, validation_alias="AGENT_MAX_ITERATIONS")
    
    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()


def validate_settings() -> dict:
    """
    Validate that all required settings are configured.
    Returns validation errors if any.
    """
    errors = {}
    
    if settings.llm_provider == "openai" and not settings.openai_api_key:
        errors["openai_api_key"] = "OpenAI API key is required when LLM_PROVIDER=openai"
    
    if settings.embeddings_provider == "openai" and not settings.openai_api_key:
        errors["openai_api_key"] = "OpenAI API key is required for embeddings"
    
    if settings.llm_provider == "huggingface" and not settings.huggingface_api_key:
        errors["huggingface_api_key"] = "Hugging Face API key required when LLM_PROVIDER=huggingface"
    
    return errors
