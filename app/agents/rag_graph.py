"""
LangGraph-based RAG agent for multi-step reasoning.
Implements a stateful agentic workflow for question answering.

Architecture:
- Nodes: Distinct steps (retrieve, reason, generate)
- Edges: Conditional transitions between nodes
- State: Carries context through the workflow
"""

import logging
from typing import TypedDict, List, Annotated, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from app.core.llm import get_llm_client
from app.core.embeddings import get_embeddings_client
from app.core.elastic import get_es_client
from app.prompts.templates import get_qa_prompt, get_reasoning_prompt

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """State object for RAG workflow."""
    question: str
    context_docs: List[dict]
    reasoning_chain: List[str]
    answer: str
    sources_used: List[str]
    error: Optional[str]


class RAGAgent:
    """
    LangGraph-based RAG agent for multi-step question answering.
    
    Workflow:
    1. Retrieve - Fetch relevant documents
    2. Reason - Analyze relevance and build reasoning
    3. Generate - Create final answer
    4. Format - Package response with sources
    """
    
    def __init__(self):
        """Initialize RAG agent with dependencies."""
        self.llm = get_llm_client()
        self.embeddings = get_embeddings_client()
        self.es_client = get_es_client()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """
        Build the LangGraph workflow.
        
        Returns:
            Compiled graph executable
        """
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("format", self._format_node)
        
        # Add edges
        workflow.add_edge("retrieve", "reason")
        workflow.add_edge("reason", "generate")
        workflow.add_edge("generate", "format")
        workflow.add_edge("format", END)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Compile
        return workflow.compile()
    
    def _retrieve_node(self, state: RAGState) -> RAGState:
        """
        Retrieval node: Fetch relevant documents.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with retrieved documents
        """
        logger.info(f"Retrieve node: Processing question: {state['question']}")
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_text(state["question"])
            
            # Search in Elasticsearch
            results = self.es_client.search_by_vector(
                embedding=query_embedding,
                top_k=5
            )
            
            state["context_docs"] = results
            state["reasoning_chain"].append(
                f"Retrieved {len(results)} documents via semantic search"
            )
            
            logger.info(f"Retrieved {len(results)} documents")
            
        except Exception as e:
            logger.error(f"Error in retrieve node: {str(e)}")
            state["error"] = f"Retrieval failed: {str(e)}"
        
        return state
    
    def _reason_node(self, state: RAGState) -> RAGState:
        """
        Reasoning node: Analyze retrieved documents.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with reasoning trace
        """
        logger.info("Reason node: Analyzing documents")
        
        try:
            if not state["context_docs"]:
                state["reasoning_chain"].append("No documents available for reasoning")
                return state
            
            # Build context summary
            context_summary = []
            for i, doc in enumerate(state["context_docs"], 1):
                summary = f"Doc {i}: {doc['text'][:100]}... (relevance: {doc['score']:.2f})"
                context_summary.append(summary)
            
            state["reasoning_chain"].append(
                f"Analyzed {len(state['context_docs'])} documents for relevance"
            )
            state["reasoning_chain"].extend(context_summary)
            state["sources_used"] = [doc["doc_id"] for doc in state["context_docs"]]
            
            logger.info(f"Reasoning complete. Sources: {state['sources_used']}")
            
        except Exception as e:
            logger.error(f"Error in reason node: {str(e)}")
            state["error"] = f"Reasoning failed: {str(e)}"
        
        return state
    
    def _generate_node(self, state: RAGState) -> RAGState:
        """
        Generation node: Generate answer using LLM.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with generated answer
        """
        logger.info("Generate node: Creating answer")
        
        try:
            if state.get("error"):
                state["answer"] = f"Could not generate answer due to error: {state['error']}"
                return state
            
            # Build context string
            context = "\n\n---\n\n".join([
                f"[Source: {doc['doc_id']}]\n{doc['text']}"
                for doc in state["context_docs"]
            ])
            
            if not context:
                context = "No relevant documents found."
            
            # Generate answer
            system_prompt = get_qa_prompt()
            user_prompt = f"""Question: {state['question']}

Context:
{context}

Provide a clear, concise answer based on the context."""
            
            answer = self.llm.generate_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=512
            )
            
            state["answer"] = answer
            state["reasoning_chain"].append(f"Generated answer using LLM")
            
            logger.info("Answer generated successfully")
            
        except Exception as e:
            logger.error(f"Error in generate node: {str(e)}")
            state["error"] = f"Generation failed: {str(e)}"
            state["answer"] = "Error generating answer"
        
        return state
    
    def _format_node(self, state: RAGState) -> RAGState:
        """
        Format node: Prepare final response.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with formatted output
        """
        logger.info("Format node: Preparing response")
        
        state["reasoning_chain"].append(f"Formatted response with {len(state['sources_used'])} sources")
        
        return state
    
    async def run(self, question: str) -> dict:
        """
        Execute the RAG workflow.
        
        Args:
            question: User question
        
        Returns:
            Final state with answer and reasoning
        """
        logger.info(f"Starting RAG workflow for question: {question}")
        
        initial_state = RAGState(
            question=question,
            context_docs=[],
            reasoning_chain=[],
            answer="",
            sources_used=[],
            error=None
        )
        
        try:
            # Execute workflow
            final_state = self.graph.invoke(initial_state)
            
            logger.info("Workflow completed successfully")
            return final_state
        
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            initial_state["error"] = str(e)
            return initial_state
