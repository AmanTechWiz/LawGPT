"""
Adaptive RAG Orchestrator for intelligent query processing.

Coordinates the entire adaptive RAG pipeline: query classification, dynamic retrieval,
adaptive generation, and response formatting based on query intent.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .query_intent_classifier import QueryIntentClassifier, QueryIntent, IntentClassification
from .prompt_templates import PromptTemplateManager
from .lightweight_llm_rag import lightweight_llm_rag
from .enhanced_citation_formatter import enhanced_citation_formatter
from .cache import rag_cache
from .hallucination_validator import hallucination_validator
from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveRAGResponse:
    """Response from the adaptive RAG system."""
    response: str
    sources: List[Dict[str, Any]]
    intent: QueryIntent
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]


class AdaptiveRAGOrchestrator:
    """Orchestrates the adaptive RAG pipeline with intent-based processing."""
    
    def __init__(self):
        self.intent_classifier = QueryIntentClassifier()
        self.prompt_manager = PromptTemplateManager()
        self.initialized = False
    
    async def initialize(self):
        """Initialize the orchestrator and its components."""
        if self.initialized:
            return
        
        try:
            await self.intent_classifier.initialize()
            await lightweight_llm_rag.initialize()
            self.initialized = True
            logger.info("Adaptive RAG Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Adaptive RAG Orchestrator: {e}")
            raise
    
    async def process_query(self, query: str, user_preferences: Optional[Dict[str, Any]] = None) -> AdaptiveRAGResponse:
        """
        Process a query through the adaptive RAG pipeline.
        
        Args:
            query: The user query to process
            user_preferences: Optional user preferences for processing
            
        Returns:
            AdaptiveRAGResponse: Complete response with metadata
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Step 1: Classify query intent to determine processing strategy
            intent_classification = await self.intent_classifier.classify_intent(query)
            logger.info(f"Query classified as: {intent_classification.intent.value} (confidence: {intent_classification.confidence:.2f})")
            
            # Step 2: Check cache first to avoid expensive processing
            cache_key = f"adaptive:{intent_classification.intent.value}:{hash(query)}"
            cached_result = await rag_cache.get_rag_query(query, f"adaptive_{intent_classification.intent.value}")
            if cached_result:
                logger.info("Returning cached adaptive result")
                return self._format_cached_response(cached_result, intent_classification)
            
            # Step 3: Dynamic retrieval based on intent and user preferences
            retrieval_count = self._get_retrieval_count(intent_classification, user_preferences)
            relevant_docs = await self._perform_dynamic_retrieval(
                query, retrieval_count, intent_classification
            )
            
            if not relevant_docs:
                return AdaptiveRAGResponse(
                    response="This information is not available in the provided legal documents.",
                    sources=[],
                    intent=intent_classification.intent,
                    confidence=intent_classification.confidence,
                    processing_time=time.time() - start_time,
                    metadata={"error": "no_relevant_documents"}
                )
            
            # Step 4: Generate response using intent-specific prompts and parameters
            response = await self._generate_adaptive_response(
                query, relevant_docs, intent_classification
            )
            
            # Step 5: Post-process response for quality and length appropriateness
            response = await self._post_process_response(
                response, intent_classification, user_preferences
            )
            
            # Step 6: Prepare enhanced sources with better citation formatting
            enhanced_sources = self._prepare_enhanced_sources(relevant_docs, intent_classification)
            
            # Step 7: Prepare final response with comprehensive metadata
            result = AdaptiveRAGResponse(
                response=response,
                sources=enhanced_sources,
                intent=intent_classification.intent,
                confidence=intent_classification.confidence,
                processing_time=time.time() - start_time,
                metadata={
                    "intent_classification": intent_classification.reasoning,
                    "retrieval_count": len(relevant_docs),
                    "generation_parameters": self.prompt_manager.get_generation_parameters(intent_classification.intent),
                    "citation_summary": enhanced_citation_formatter.generate_citation_summary(enhanced_sources)
                }
            )
            
            # Cache the result for future similar queries
            await rag_cache.cache_rag_query(query, result.__dict__, f"adaptive_{intent_classification.intent.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Adaptive RAG processing failed: {e}")
            return AdaptiveRAGResponse(
                response=f"An error occurred while processing your query: {str(e)}",
                sources=[],
                intent=QueryIntent.FACTUAL,
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _get_retrieval_count(self, intent_classification: IntentClassification, 
                           user_preferences: Optional[Dict[str, Any]]) -> int:
        """Determine the number of documents to retrieve based on intent and preferences."""
        base_count = intent_classification.suggested_retrieval_count
        
        # Apply user preferences to adjust retrieval count
        if user_preferences:
            if "retrieval_count" in user_preferences:
                return user_preferences["retrieval_count"]
            if "response_length" in user_preferences:
                # Adjust retrieval count based on desired response length
                length_multiplier = {
                    "short": 0.7,
                    "normal": 1.0,
                    "detailed": 1.5
                }.get(user_preferences["response_length"], 1.0)
                base_count = int(base_count * length_multiplier)
        
        return max(2, min(base_count, 15))  # Enforce reasonable bounds (2-15 documents)
    
    async def _perform_dynamic_retrieval(self, query: str, retrieval_count: int, 
                                       intent_classification: IntentClassification) -> List[Dict[str, Any]]:
        """Perform retrieval with parameters adjusted for the query intent."""
        try:
            # Adjust similarity threshold based on intent (analytical queries need lower threshold)
            similarity_threshold = self._get_similarity_threshold(intent_classification.intent)
            
            # Use the existing RAG system for retrieval with intent-specific parameters
            return await lightweight_llm_rag.retrieve_documents(
                query=query,
                top_k=retrieval_count,
                algorithm="hybrid",
                similarity_threshold=similarity_threshold,
            )
            
        except Exception as e:
            logger.error(f"Dynamic retrieval failed: {e}")
            return []
    
    def _get_similarity_threshold(self, intent: QueryIntent) -> float:
        """Get similarity threshold based on query intent."""
        thresholds = {
            QueryIntent.DEFINITION: 0.3,  # Higher precision for definitions
            QueryIntent.LIST: 0.25,      # Standard precision for lists
            QueryIntent.EXPLANATION: 0.2, # Lower threshold for comprehensive explanations
            QueryIntent.COMPARATIVE: 0.15, # Very low threshold for comparative analysis
            QueryIntent.PROCEDURAL: 0.25, # Standard precision for procedures
            QueryIntent.ANALYTICAL: 0.15, # Low threshold for analytical queries
            QueryIntent.INTERPRETATIVE: 0.2, # Lower threshold for interpretations
            QueryIntent.FACTUAL: 0.25     # Standard precision for factual queries
        }
        return thresholds.get(intent, 0.25)
    
    async def _generate_adaptive_response(self, query: str, relevant_docs: List[Dict[str, Any]], 
                                        intent_classification: IntentClassification) -> str:
        """Generate response using intent-specific prompt templates and parameters."""
        try:
            # Prepare structured context and detect potential conflicts
            context = self._prepare_structured_context(relevant_docs)
            conflict_info = self._detect_conflicts(relevant_docs)
            
            # Get intent-specific prompt template
            prompt_data = self.prompt_manager.generate_prompt(
                intent_classification.intent,
                query,
                context,
                conflict_info
            )
            
            # Get generation parameters optimized for the query intent
            gen_params = self.prompt_manager.get_generation_parameters(intent_classification.intent)
            
            # Generate response using OpenAI with intent-specific parameters
            import openai
            openai_client = openai.OpenAI(api_key=settings.openai_api_key)
            
            response = openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": prompt_data["system"]},
                    {"role": "user", "content": prompt_data["user"]}
                ],
                max_tokens=gen_params["max_tokens"],
                temperature=gen_params["temperature"]
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # Validate response to prevent hallucinated legal content
            should_reject, rejection_reason = hallucination_validator.should_reject_response(
                raw_response, context, query
            )
            
            if should_reject:
                logger.warning(f"Response rejected due to hallucination: {rejection_reason}")
                safe_response = hallucination_validator.get_safe_response(query, context)
                return safe_response
            
            # Format response based on intent-specific requirements
            return self._format_response(raw_response, intent_classification.intent)
            
        except Exception as e:
            logger.error(f"Adaptive response generation failed: {e}")
            raise
    
    def _prepare_structured_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Prepare structured context from relevant documents."""
        context = ""
        
        # Group documents by source for better organization and readability
        sources = {}
        for doc in relevant_docs:
            source = doc['metadata'].get('source', 'Unknown')
            if source not in sources:
                sources[source] = []
            sources[source].append(doc)
        
        for source, docs in sources.items():
            context += f"=== {source} ===\n"
            for i, doc in enumerate(docs, 1):
                title = doc['metadata'].get('title', f'Section {i}')
                similarity = doc['metadata'].get('similarity_score', 0)
                citations = doc['metadata'].get('legal_citations', [])
                
                # Format citations for display
                citation_text = ""
                if citations:
                    citation_text = f" [Citations: {', '.join(citations)}]"
                
                cleaned_content = self._clean_context_content(doc['content'])
                
                context += f"\n[{title}]{citation_text} (Relevance: {similarity:.2f})\n"
                context += f"{cleaned_content}\n"
            context += "\n"
        
        return context
    
    def _clean_context_content(self, content: str) -> str:
        """Clean content for better context presentation."""
        if not content:
            return content
        
        # Basic cleaning - remove excessive whitespace and normalize formatting
        import re
        cleaned = re.sub(r'\s+', ' ', content)
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        return cleaned.strip()
    
    def _detect_conflicts(self, docs: List[Dict[str, Any]]) -> str:
        """Detect potential conflicts between documents."""
        if len(docs) < 2:
            return ""
        
        # Simple conflict detection based on shared legal citations
        citation_groups = {}
        for doc in docs:
            citations = doc['metadata'].get('legal_citations', [])
            for citation in citations:
                if citation not in citation_groups:
                    citation_groups[citation] = []
                citation_groups[citation].append(doc)
        
        # Identify citations referenced by multiple documents
        conflicts = []
        for citation, citation_docs in citation_groups.items():
            if len(citation_docs) > 1:
                conflicts.append(f"Multiple documents reference {citation}")
        
        if conflicts:
            return f"\n\nPOTENTIAL CONFLICTS DETECTED:\n" + "\n".join(f"- {conflict}" for conflict in conflicts)
        
        return ""
    
    def _format_response(self, response: str, intent: QueryIntent) -> str:
        """Format response based on intent-specific requirements."""
        if not response:
            return response
        
        # Basic formatting - normalize whitespace and line breaks
        import re
        formatted = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)
        formatted = re.sub(r'^\s+', '', formatted, flags=re.MULTILINE)
        
        # Apply intent-specific formatting rules
        if intent == QueryIntent.LIST:
            # Ensure proper list formatting with numbered items
            formatted = re.sub(r'^(\d+\.)', r'\n\1', formatted, flags=re.MULTILINE)
        elif intent == QueryIntent.COMPARATIVE:
            # Ensure proper comparison formatting with clear sections
            formatted = re.sub(r'\n([A-Z][^:]+:)', r'\n\n\1', formatted)
        
        return formatted.strip()
    
    def _prepare_enhanced_sources(self, sources: List[Dict[str, Any]], 
                                 intent_classification: IntentClassification) -> List[Dict[str, Any]]:
        """
        Prepare sources with enhanced metadata and citation formatting.
        
        Args:
            sources: List of document sources
            intent_classification: Query intent classification
            
        Returns:
            List[Dict[str, Any]]: Enhanced sources with better metadata
        """
        try:
            # Use enhanced citation formatter for better source presentation
            enhanced_sources = enhanced_citation_formatter.format_context_sources(sources)
            
            # Add intent-specific metadata to each source
            for source in enhanced_sources:
                metadata = source.get('metadata', {})
                
                # Add intent classification information
                metadata['query_intent'] = intent_classification.intent.value
                metadata['intent_confidence'] = intent_classification.confidence
                
                # Add enhanced display information for better UX
                if metadata.get('enhanced', False):
                    metadata['display_title'] = f"{metadata.get('title', 'Untitled')} ({metadata.get('chunk_info', '')})"
                else:
                    metadata['display_title'] = metadata.get('title', 'Untitled')
            
            return enhanced_sources
            
        except Exception as e:
            logger.error(f"Error preparing enhanced sources: {e}")
            # Fallback to original sources
            return sources
    
    async def _post_process_response(self, response: str, intent_classification: IntentClassification, 
                                   user_preferences: Optional[Dict[str, Any]]) -> str:
        """Post-process response for quality and length appropriateness."""
        if not response:
            return response
        
        # Check response length appropriateness for the query intent
        word_count = len(response.split())
        expected_range = self._get_expected_word_range(intent_classification.intent)
        
        if word_count < expected_range[0] and intent_classification.confidence > 0.7:
            # Response too short for the intent type
            logger.warning(f"Response too short for {intent_classification.intent.value} intent: {word_count} words")
        elif word_count > expected_range[1] and intent_classification.intent in [QueryIntent.DEFINITION, QueryIntent.FACTUAL]:
            # Response too long for concise intents
            logger.warning(f"Response too long for {intent_classification.intent.value} intent: {word_count} words")
            # Truncate if significantly too long
            if word_count > expected_range[1] * 1.5:
                sentences = response.split('. ')
                truncated = '. '.join(sentences[:3]) + '.'
                return truncated
        
        return response
    
    def _get_expected_word_range(self, intent: QueryIntent) -> Tuple[int, int]:
        """Get expected word count range for different intents."""
        ranges = {
            QueryIntent.DEFINITION: (20, 100),
            QueryIntent.LIST: (50, 200),
            QueryIntent.EXPLANATION: (150, 500),
            QueryIntent.COMPARATIVE: (200, 600),
            QueryIntent.PROCEDURAL: (100, 400),
            QueryIntent.ANALYTICAL: (300, 800),
            QueryIntent.INTERPRETATIVE: (200, 500),
            QueryIntent.FACTUAL: (30, 150)
        }
        return ranges.get(intent, (50, 300))
    
    def _format_cached_response(self, cached_result: Dict[str, Any], 
                               intent_classification: IntentClassification) -> AdaptiveRAGResponse:
        """Format a cached result into an AdaptiveRAGResponse."""
        return AdaptiveRAGResponse(
            response=cached_result.get("response", ""),
            sources=cached_result.get("sources", []),
            intent=intent_classification.intent,
            confidence=intent_classification.confidence,
            processing_time=cached_result.get("processing_time", 0),
            metadata=cached_result.get("metadata", {})
        )


# Global instance
adaptive_rag_orchestrator = AdaptiveRAGOrchestrator()
