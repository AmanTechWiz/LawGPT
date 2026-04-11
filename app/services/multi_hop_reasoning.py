"""
Multi-hop reasoning system for complex legal queries.

Provides iterative query decomposition, intermediate reasoning steps, and
comprehensive synthesis of complex legal questions requiring multiple reasoning hops.
"""

import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from ..core.config import settings
from ..core.database import db_manager
from .lightweight_llm_rag import lightweight_llm_rag
from .legal_tools import extract_legal_citations, classify_legal_text
from .legal_classifier import legal_classifier

logger = logging.getLogger(__name__)


class ReasoningComplexity(str, Enum):
    """Complexity levels for multi-hop reasoning"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ReasoningStepType(str, Enum):
    """Types of reasoning steps in multi-hop reasoning"""
    QUERY_DECOMPOSITION = "query_decomposition"
    SUB_QUERY_EXECUTION = "sub_query_execution"
    INFORMATION_SYNTHESIS = "information_synthesis"
    CONFLICT_RESOLUTION = "conflict_resolution"
    FINAL_SYNTHESIS = "final_synthesis"


@dataclass
class ReasoningStep:
    """Represents a single step in the multi-hop reasoning process"""
    step_id: str
    step_type: ReasoningStepType
    input_query: str
    output_result: str
    sources_used: List[Dict[str, Any]]
    confidence_score: float
    execution_time: float
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class ReasoningChain:
    """Represents a complete reasoning chain for a complex query"""
    chain_id: str
    original_query: str
    complexity_level: ReasoningComplexity
    steps: List[ReasoningStep]
    final_answer: str
    total_execution_time: float
    overall_confidence: float
    citations: List[str]
    metadata: Dict[str, Any]
    created_at: datetime


class QueryComplexityAnalyzer:
    """Analyzes query complexity for multi-hop reasoning using linguistic and legal indicators."""
    
    COMPLEXITY_INDICATORS = {
        "multiple_concepts": [
            "and", "also", "furthermore", "additionally", "moreover", 
            "in addition", "as well as", "along with", "together with"
        ],
        "conditional_reasoning": [
            "if", "when", "unless", "provided that", "in case", 
            "assuming", "supposing", "given that"
        ],
        "comparative_analysis": [
            "compare", "contrast", "difference", "similarity", 
            "versus", "vs", "between", "among"
        ],
        "causal_reasoning": [
            "because", "due to", "as a result", "consequently", 
            "therefore", "thus", "hence", "causes", "leads to"
        ],
        "legal_complexity": [
            "article", "section", "chapter", "provision", "clause",
            "amendment", "interpretation", "application", "scope",
            "jurisdiction", "precedent", "case law", "statute"
        ],
        "multi_document": [
            "across", "throughout", "in all", "various", "different",
            "multiple", "several", "both", "each", "respective"
        ]
    }
    
    @classmethod
    def analyze_complexity(cls, query: str) -> Tuple[ReasoningComplexity, Dict[str, Any]]:
        """
        Analyze query complexity and determine reasoning approach.
        
        Args:
            query: The legal query to analyze
            
        Returns:
            Tuple of complexity level and analysis details
        """
        query_lower = query.lower()
        complexity_score = 0
        detected_indicators = {}
        
        # Check for complexity indicators
        for category, indicators in cls.COMPLEXITY_INDICATORS.items():
            found_indicators = [ind for ind in indicators if ind in query_lower]
            if found_indicators:
                detected_indicators[category] = found_indicators
                complexity_score += len(found_indicators)
        
        # Additional complexity factors
        word_count = len(query.split())
        sentence_count = query.count('.') + query.count('?') + query.count('!')
        
        # Legal-specific complexity
        legal_terms = sum(1 for term in cls.COMPLEXITY_INDICATORS["legal_complexity"] 
                         if term in query_lower)
        
        # Adjust score based on length and structure
        if word_count > 50:
            complexity_score += 2
        if sentence_count > 2:
            complexity_score += 1
        if legal_terms > 3:
            complexity_score += 2
        
        # Determine complexity level
        if complexity_score >= 8:
            complexity_level = ReasoningComplexity.VERY_COMPLEX
        elif complexity_score >= 5:
            complexity_level = ReasoningComplexity.COMPLEX
        elif complexity_score >= 3:
            complexity_level = ReasoningComplexity.MODERATE
        else:
            complexity_level = ReasoningComplexity.SIMPLE
        
        analysis_details = {
            "complexity_score": complexity_score,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "legal_terms_count": legal_terms,
            "detected_indicators": detected_indicators,
            "requires_multi_hop": complexity_level in [ReasoningComplexity.COMPLEX, ReasoningComplexity.VERY_COMPLEX]
        }
        
        return complexity_level, analysis_details


class QueryDecomposer:
    """Decomposes complex queries into manageable sub-queries using OpenAI."""
    
    def __init__(self):
        self.openai_client = None
    
    async def initialize(self):
        """
        Initialize OpenAI client for query decomposition.
        
        Sets up the OpenAI client with API key from settings.
        """
        if not self.openai_client:
            import openai
            self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
    
    async def decompose_query(self, query: str, complexity_analysis: Dict[str, Any]) -> List[str]:
        """
        Decompose a complex query into sub-queries.
        
        Args:
            query: The original complex query
            complexity_analysis: Analysis results from complexity analyzer
            
        Returns:
            List of sub-queries for multi-hop reasoning
        """
        await self.initialize()
        
        try:
            prompt = f"""You are a legal research expert. Decompose the following complex legal query into 2-4 specific, focused sub-queries that can be answered independently and then synthesized into a direct, actionable answer.

Original Query: {query}

Complexity Analysis:
- Detected indicators: {complexity_analysis.get('detected_indicators', {})}
- Legal terms count: {complexity_analysis.get('legal_terms_count', 0)}

Guidelines:
1. Each sub-query should focus on a specific, actionable aspect
2. Sub-queries should be answerable from legal documents with specific provisions
3. Ensure sub-queries cover all aspects needed for a practical answer
4. Make sub-queries specific enough to find exact legal requirements or procedures
5. Focus on what the user needs to know to take action

Return only the sub-queries, one per line, without numbering or additional text."""

            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a legal research expert who specializes in breaking down complex legal questions into focused, actionable sub-queries that lead to practical answers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            sub_queries_text = response.choices[0].message.content.strip()
            sub_queries = [q.strip() for q in sub_queries_text.split('\n') if q.strip()]
            
            # Validate and clean sub-queries - filter out invalid or too-short queries
            validated_queries = []
            for sub_query in sub_queries:
                if len(sub_query) > 10 and '?' in sub_query:
                    validated_queries.append(sub_query)
            
            logger.info(f"Decomposed query into {len(validated_queries)} sub-queries")
            return validated_queries
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            # Fallback: return the original query as a single sub-query
            return [query]


class MultiHopReasoningEngine:
    """Main engine for multi-hop reasoning in legal queries with complexity analysis and synthesis."""
    
    def __init__(self):
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.query_decomposer = QueryDecomposer()
        self.reasoning_chains = {}  # In-memory storage for active chains
    
    async def process_complex_query(self, query: str, session_id: Optional[str] = None) -> ReasoningChain:
        """
        Process a complex query using multi-hop reasoning.
        
        Args:
            query: The complex legal query to process
            session_id: Optional session identifier
            
        Returns:
            Complete reasoning chain with final answer
        """
        start_time = time.time()
        chain_id = str(uuid.uuid4())
        
        try:
            complexity_level, analysis_details = self.complexity_analyzer.analyze_complexity(query)
            
            logger.info(f"Processing complex query with {complexity_level} complexity")
            
            if complexity_level in [ReasoningComplexity.COMPLEX, ReasoningComplexity.VERY_COMPLEX]:
                sub_queries = await self.query_decomposer.decompose_query(query, analysis_details)
            else:
                sub_queries = [query]  # Simple queries don't need decomposition
            
            reasoning_steps = []
            all_sources = []
            all_citations = []
            confidence_scores = []
            
            for i, sub_query in enumerate(sub_queries):
                step_start = time.time()
                step_id = f"{chain_id}_step_{i+1}"
                
                step_result = await self._execute_reasoning_step(
                    step_id, sub_query, i, len(sub_queries)
                )
                
                reasoning_steps.append(step_result)
                all_sources.extend(step_result.sources_used)
                all_citations.extend(extract_legal_citations(step_result.output_result))
                confidence_scores.append(step_result.confidence_score)
                
                logger.info(f"Completed reasoning step {i+1}/{len(sub_queries)}")
            
            synthesis_start = time.time()
            final_answer = await self._synthesize_final_answer(
                query, reasoning_steps, all_sources
            )
            synthesis_time = time.time() - synthesis_start
            synthesis_step = ReasoningStep(
                step_id=f"{chain_id}_synthesis",
                step_type=ReasoningStepType.FINAL_SYNTHESIS,
                input_query="Final synthesis of all reasoning steps",
                output_result=final_answer,
                sources_used=all_sources,
                confidence_score=sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
                execution_time=synthesis_time,
                metadata={"synthesis_type": "multi_step_integration"},
                timestamp=datetime.now()
            )
            
            reasoning_steps.append(synthesis_step)
            
            total_time = time.time() - start_time
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            reasoning_chain = ReasoningChain(
                chain_id=chain_id,
                original_query=query,
                complexity_level=complexity_level,
                steps=reasoning_steps,
                final_answer=final_answer,
                total_execution_time=total_time,
                overall_confidence=overall_confidence,
                citations=list(set(all_citations)),
                metadata={
                    "analysis_details": analysis_details,
                    "sub_queries_count": len(sub_queries),
                    "session_id": session_id
                },
                created_at=datetime.now()
            )
            
            self.reasoning_chains[chain_id] = reasoning_chain
            
            logger.info(f"Multi-hop reasoning completed in {total_time:.2f}s with {overall_confidence:.2f} confidence")
            return reasoning_chain
            
        except Exception as e:
            logger.error(f"Multi-hop reasoning failed: {e}")
            return ReasoningChain(
                chain_id=chain_id,
                original_query=query,
                complexity_level=complexity_level,
                steps=[],
                final_answer=f"Error in multi-hop reasoning: {str(e)}",
                total_execution_time=time.time() - start_time,
                overall_confidence=0.0,
                citations=[],
                metadata={"error": str(e)},
                created_at=datetime.now()
            )
    
    async def _execute_reasoning_step(self, step_id: str, sub_query: str, 
                                    step_index: int, total_steps: int) -> ReasoningStep:
        """
        Execute a single reasoning step.
        
        Args:
            step_id: Unique identifier for the step
            sub_query: Query to process in this step
            step_index: Index of current step
            total_steps: Total number of steps
            
        Returns:
            ReasoningStep: Completed reasoning step with results
        """
        step_start = time.time()
        
        try:
            # Use RAG to answer sub-query
            rag_result = await lightweight_llm_rag.query(
                query=sub_query,
                top_k=8,
                similarity_threshold=0.25
            )
            
            response = rag_result.get("response", "")
            sources = rag_result.get("sources", [])
            
            # Calculate confidence based on source quality and relevance
            confidence = self._calculate_step_confidence(sources, response)
            
            # Classify the legal domain for this step
            domain_result = legal_classifier.classify(sub_query)
            
            step_type = ReasoningStepType.SUB_QUERY_EXECUTION
            if step_index == 0:
                step_type = ReasoningStepType.QUERY_DECOMPOSITION
            
            execution_time = time.time() - step_start
            
            return ReasoningStep(
                step_id=step_id,
                step_type=step_type,
                input_query=sub_query,
                output_result=response,
                sources_used=sources,
                confidence_score=confidence,
                execution_time=execution_time,
                metadata={
                    "step_index": step_index,
                    "total_steps": total_steps,
                    "domain": domain_result.get("category", "Other"),
                    "domain_confidence": domain_result.get("confidence", 0.0)
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Reasoning step execution failed: {e}")
            return ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.SUB_QUERY_EXECUTION,
                input_query=sub_query,
                output_result=f"Error in step execution: {str(e)}",
                sources_used=[],
                confidence_score=0.0,
                execution_time=time.time() - step_start,
                metadata={"error": str(e)},
                timestamp=datetime.now()
            )
    
    def _calculate_step_confidence(self, sources: List[Dict[str, Any]], response: str) -> float:
        """
        Calculate confidence score for a reasoning step.
        
        Args:
            sources: List of sources used in the step
            response: Generated response text
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        if not sources or not response:
            return 0.0
        
        # Base confidence from source similarity scores
        avg_similarity = sum(source.get('score', 0) for source in sources) / len(sources)
        
        # Adjust based on response quality indicators
        quality_indicators = {
            "has_citations": len(extract_legal_citations(response)) > 0,
            "sufficient_length": len(response.split()) > 20,
            "has_legal_terms": any(term in response.lower() for term in 
                                 ["article", "section", "law", "legal", "provision"]),
            "multiple_sources": len(sources) > 1
        }
        
        quality_bonus = sum(quality_indicators.values()) * 0.1
        
        confidence = min(1.0, avg_similarity + quality_bonus)
        return confidence
    
    async def _synthesize_final_answer(self, original_query: str, 
                                     reasoning_steps: List[ReasoningStep],
                                     all_sources: List[Dict[str, Any]]) -> str:
        """
        Synthesize final answer from all reasoning steps.
        
        Args:
            original_query: The original complex query
            reasoning_steps: List of completed reasoning steps
            all_sources: All sources used across steps
            
        Returns:
            str: Synthesized final answer
        """
        try:
            await self.query_decomposer.initialize()
            
            # Prepare context from all reasoning steps
            step_contexts = []
            for i, step in enumerate(reasoning_steps):
                step_context = f"Step {i+1}: {step.input_query}\nAnswer: {step.output_result}\n"
                step_contexts.append(step_context)
            
            context_text = "\n".join(step_contexts)
            
            prompt = f"""You are a legal research expert. Synthesize the following reasoning steps into a direct, actionable answer to the original query.

Original Query: {original_query}

Reasoning Steps and Answers:
{context_text}

Synthesis Guidelines:
1. Lead with the direct answer to the original question
2. Integrate key findings from all steps into a practical response
3. Focus on actionable information and specific legal provisions
4. Use bullet points for multiple related points
5. Include exact legal citations for immediate verification
6. Highlight practical implications or requirements
7. Keep the response concise and immediately useful
8. Avoid lengthy explanations unless specifically needed

Synthesized Answer:"""

            response = self.query_decomposer.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a legal research expert who specializes in synthesizing complex multi-step legal analysis into direct, actionable answers that immediately address the user's question."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Final synthesis failed: {e}")
            # Fallback: concatenate step results
            fallback_answer = "Based on the analysis of multiple aspects of your query:\n\n"
            for i, step in enumerate(reasoning_steps):
                fallback_answer += f"{i+1}. {step.output_result}\n\n"
            return fallback_answer
    
    def get_reasoning_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        """Retrieve a reasoning chain by ID"""
        return self.reasoning_chains.get(chain_id)
    
    def get_reasoning_chains_by_session(self, session_id: str) -> List[ReasoningChain]:
        """Get all reasoning chains for a session"""
        return [chain for chain in self.reasoning_chains.values() 
                if chain.metadata.get("session_id") == session_id]


# Global instance
multi_hop_reasoning_engine = MultiHopReasoningEngine()
