"""
Text embeddings integration.
Supports multiple embedding providers (OpenAI, Hugging Face).
"""

import logging
from typing import List, Optional
from abc import ABC, abstractmethod
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingsProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass


class OpenAIEmbeddings(EmbeddingsProvider):
    """OpenAI embeddings provider using their API."""
    
    def __init__(self):
        """Initialize OpenAI embeddings client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embeddings_model
        logger.info(f"Initialized OpenAI embeddings with model: {self.model}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text using OpenAI."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            # Sort by index to ensure correct order
            embeddings = [None] * len(texts)
            for item in response.data:
                embeddings[item.index] = item.embedding
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise


class HuggingFaceEmbeddings(EmbeddingsProvider):
    """Hugging Face embeddings provider."""
    
    def __init__(self):
        """Initialize Hugging Face embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers package required: pip install sentence-transformers")
        
        self.model = SentenceTransformer(settings.huggingface_model)
        logger.info(f"Initialized Hugging Face embeddings with model: {settings.huggingface_model}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding using Hugging Face."""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using Hugging Face."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise


class EmbeddingsClient:
    """Factory and wrapper for embeddings providers."""
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize embeddings client with specified provider.
        
        Args:
            provider: Provider type ("openai" or "huggingface"). 
                      Defaults to settings.embeddings_provider
        """
        self.provider_type = provider or settings.embeddings_provider
        self.provider = self._create_provider()
    
    def _create_provider(self) -> EmbeddingsProvider:
        """Create the appropriate embeddings provider."""
        if self.provider_type == "openai":
            return OpenAIEmbeddings()
        elif self.provider_type == "huggingface":
            return HuggingFaceEmbeddings()
        else:
            raise ValueError(f"Unknown embeddings provider: {self.provider_type}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return self.provider.embed_text(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self.provider.embed_texts(texts)


# Global embeddings client instance
_embeddings_client: Optional[EmbeddingsClient] = None


def get_embeddings_client() -> EmbeddingsClient:
    """Get or create the embeddings client."""
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = EmbeddingsClient()
    return _embeddings_client
