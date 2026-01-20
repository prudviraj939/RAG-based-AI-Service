"""
Agent service for orchestrating RAG workflows.
Coordinates retrieval, reasoning, and answer generation.
"""

import logging
import time
from typing import Optional, List
from datetime import datetime

from app.core.llm import get_llm_client
from app.schemas.qa import QuestionRequest, QuestionResponse, RetrievedDocument
from app.services.retrieval_service import RetrievalService
from app.prompts.templates import get_qa_prompt, get_reasoning_prompt

logger = logging.getLogger(__name__)


class AgentService:
    """
    Orchestrates the RAG (Retrieval-Augmented Generation) workflow.
    
    Multi-step flow:
    1. Retrieve relevant documents based on question
    2. Build context from retrieved documents
    3. Generate answer using LLM with context
    4. Format response with sources
    """
    
    def __init__(self):
        """Initialize agent service with dependencies."""
        self.llm_client = get_llm_client()
        self.retrieval_service = RetrievalService()
    
    async def answer_question(
        self,
        question_request: QuestionRequest
    ) -> QuestionResponse:
        """
        Answer a question using RAG pipeline.
        
        Process:
        1. Retrieve relevant documents
        2. Build context string
        3. Generate answer with reasoning
        4. Format response
        
        Args:
            question_request: User question and parameters
        
        Returns:
            QuestionResponse with answer and sources
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing question: {question_request.question}")
            
            # Step 1: Retrieve documents
            logger.info("Step 1: Retrieving relevant documents")
            retrieved_docs = await self.retrieval_service.retrieve_hybrid(
                query=question_request.question,
                top_k=question_request.top_k,
                semantic_weight=0.7
            )
            
            if not retrieved_docs:
                logger.warning("No documents retrieved")
                return QuestionResponse(
                    question=question_request.question,
                    answer="No relevant documents found to answer your question.",
                    reasoning="No documents were retrieved for the given query.",
                    retrieved_documents=[],
                    model=self.llm_client.provider_type,
                    processing_time_ms=round((time.time() - start_time) * 1000, 2)
                )
            
            # Step 2: Build context
            logger.info("Step 2: Building context from retrieved documents")
            context = self._build_context(retrieved_docs)
            
            # Step 3: Generate answer
            logger.info("Step 3: Generating answer with LLM")
            answer = await self._generate_answer(
                question=question_request.question,
                context=context
            )
            
            # Step 4: Optional reasoning
            reasoning = None
            if question_request.include_sources:
                reasoning = await self._generate_reasoning(
                    question=question_request.question,
                    context=context,
                    answer=answer
                )
            
            # Step 5: Format response
            retrieved_docs_response = [
                RetrievedDocument(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=doc.score,
                    metadata=doc.metadata
                )
                for doc in retrieved_docs
            ] if question_request.include_sources else []
            
            response = QuestionResponse(
                question=question_request.question,
                answer=answer,
                reasoning=reasoning,
                retrieved_documents=retrieved_docs_response,
                model=self.llm_client.provider_type,
                processing_time_ms=round((time.time() - start_time) * 1000, 2)
            )
            
            logger.info(f"Successfully answered question in {response.processing_time_ms}ms")
            return response
        
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise
    
    def _build_context(self, documents: List) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            documents: List of retrieved document chunks
        
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"[Source {i}] (Score: {doc.score:.3f})\n{doc.text}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    async def _generate_answer(
        self,
        question: str,
        context: str
    ) -> str:
        """
        Generate answer using LLM with context.
        
        Args:
            question: User question
            context: Retrieved context
        
        Returns:
            Generated answer
        """
        system_prompt = get_qa_prompt()
        user_prompt = f"""Question: {question}

Context:
{context}

Please provide a clear, concise answer based on the context provided."""
        
        try:
            answer = self.llm_client.generate_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=512
            )
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    async def _generate_reasoning(
        self,
        question: str,
        context: str,
        answer: str
    ) -> str:
        """
        Generate reasoning explanation.
        
        Args:
            question: User question
            context: Retrieved context
            answer: Generated answer
        
        Returns:
            Reasoning explanation
        """
        system_prompt = get_reasoning_prompt()
        user_prompt = f"""Question: {question}

Retrieved context:
{context}

Generated answer:
{answer}

Provide a brief explanation of how the answer was derived from the context."""
        
        try:
            reasoning = self.llm_client.generate_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=256
            )
            return reasoning
        except Exception as e:
            logger.warning(f"Error generating reasoning: {str(e)}")
            return None
