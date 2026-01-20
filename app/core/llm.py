"""
LLM integration and client management.
Supports multiple LLM providers (OpenAI, Hugging Face).
"""

import logging
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def generate_with_context(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text with system and user prompts (for chat models)."""
        pass


class OpenAILLM(LLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.temperature = settings.openai_temperature
        logger.info(f"Initialized OpenAI LLM with model: {self.model}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate completion from prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or 512
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {str(e)}")
            raise
    
    def generate_with_context(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate with system and user prompts."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or 512
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {str(e)}")
            raise


class HuggingFaceLLM(LLMProvider):
    """Hugging Face LLM provider (using inference API)."""
    
    def __init__(self):
        """Initialize Hugging Face client."""
        if not settings.huggingface_api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable not set")
        
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("huggingface_hub package required: pip install huggingface_hub")
        
        self.client = InferenceClient(
            model=settings.huggingface_model,
            token=settings.huggingface_api_key
        )
        self.model = settings.huggingface_model
        logger.info(f"Initialized Hugging Face LLM with model: {self.model}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate completion from prompt."""
        try:
            response = self.client.text_generation(
                prompt,
                temperature=temperature or 0.7,
                max_new_tokens=max_tokens or 512
            )
            return response
        except Exception as e:
            logger.error(f"Error generating with Hugging Face: {str(e)}")
            raise
    
    def generate_with_context(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate with system and user prompts."""
        # For text generation models, combine prompts
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        return self.generate(combined_prompt, temperature, max_tokens)


class LLMClient:
    """Factory and wrapper for LLM providers."""
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            provider: Provider type ("openai" or "huggingface").
                     Defaults to settings.llm_provider
        """
        self.provider_type = provider or settings.llm_provider
        self.provider = self._create_provider()
    
    def _create_provider(self) -> LLMProvider:
        """Create the appropriate LLM provider."""
        if self.provider_type == "openai":
            return OpenAILLM()
        elif self.provider_type == "huggingface":
            return HuggingFaceLLM()
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider_type}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text from prompt."""
        return self.provider.generate(prompt, temperature, max_tokens)
    
    def generate_with_context(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate with system and user context."""
        return self.provider.generate_with_context(
            system_prompt,
            user_prompt,
            temperature,
            max_tokens
        )


# Global LLM client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
