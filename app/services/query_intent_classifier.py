"""
Query Intent Classification system for adaptive RAG.

Provides intelligent classification of user queries into specific intent types
to enable adaptive retrieval, generation, and response formatting.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from ..core.config import settings

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Types of query intents for adaptive RAG processing"""
    DEFINITION = "definition"
    LIST = "list"
    EXPLANATION = "explanation"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    ANALYTICAL = "analytical"
    FACTUAL = "factual"
    INTERPRETATIVE = "interpretative"


@dataclass
class IntentClassification:
    """Result of query intent classification"""
    intent: QueryIntent
    confidence: float
    reasoning: str
    suggested_retrieval_count: int
    suggested_max_tokens: int
    suggested_temperature: float


class QueryIntentClassifier:
    """Advanced query intent classifier using pattern matching and LLM-based classification."""
    
    def __init__(self):
        self.initialized = False
        self.openai_client = None
        
        # Pattern-based classification rules
        self.intent_patterns = {
            QueryIntent.DEFINITION: [
                r"what\s+is\s+",
                r"define\s+",
                r"definition\s+of\s+",
                r"meaning\s+of\s+",
                r"what\s+does\s+.*\s+mean",
                r"explain\s+the\s+concept\s+of",
                r"what\s+is\s+.*\s+defined\s+as"
            ],
            QueryIntent.LIST: [
                r"list\s+",
                r"enumerate\s+",
                r"what\s+are\s+the\s+",
                r"name\s+the\s+",
                r"identify\s+the\s+",
                r"provide\s+a\s+list\s+of",
                r"what\s+are\s+all\s+the\s+",
                r"give\s+me\s+all\s+the\s+"
            ],
            QueryIntent.EXPLANATION: [
                r"explain\s+",
                r"how\s+does\s+",
                r"why\s+",
                r"describe\s+",
                r"elaborate\s+on\s+",
                r"provide\s+details\s+about",
                r"tell\s+me\s+about\s+",
                r"can\s+you\s+explain"
            ],
            QueryIntent.COMPARATIVE: [
                r"compare\s+",
                r"difference\s+between\s+",
                r"similarity\s+between\s+",
                r"versus\s+",
                r"vs\s+",
                r"contrast\s+",
                r"distinguish\s+between\s+",
                r"how\s+do\s+.*\s+and\s+.*\s+differ"
            ],
            QueryIntent.PROCEDURAL: [
                r"how\s+to\s+",
                r"steps\s+to\s+",
                r"process\s+for\s+",
                r"procedure\s+for\s+",
                r"workflow\s+for\s+",
                r"method\s+to\s+",
                r"way\s+to\s+",
                r"guide\s+to\s+"
            ],
            QueryIntent.ANALYTICAL: [
                r"analyze\s+",
                r"evaluate\s+",
                r"assess\s+",
                r"examine\s+",
                r"critique\s+",
                r"review\s+",
                r"investigate\s+",
                r"study\s+"
            ],
            QueryIntent.INTERPRETATIVE: [
                r"interpret\s+",
                r"interpretation\s+of\s+",
                r"how\s+to\s+interpret\s+",
                r"what\s+does\s+.*\s+imply",
                r"implications\s+of\s+",
                r"significance\s+of\s+",
                r"meaning\s+and\s+scope\s+of"
            ]
        }
        
        # Intent-specific processing parameters
        self.intent_parameters = {
            QueryIntent.DEFINITION: {
                "retrieval_count": 3,
                "max_tokens": 200,
                "temperature": 0.1,
                "response_style": "concise"
            },
            QueryIntent.LIST: {
                "retrieval_count": 5,
                "max_tokens": 300,
                "temperature": 0.2,
                "response_style": "structured"
            },
            QueryIntent.EXPLANATION: {
                "retrieval_count": 8,
                "max_tokens": 600,
                "temperature": 0.3,
                "response_style": "detailed"
            },
            QueryIntent.COMPARATIVE: {
                "retrieval_count": 10,
                "max_tokens": 800,
                "temperature": 0.2,
                "response_style": "analytical"
            },
            QueryIntent.PROCEDURAL: {
                "retrieval_count": 6,
                "max_tokens": 500,
                "temperature": 0.1,
                "response_style": "step_by_step"
            },
            QueryIntent.ANALYTICAL: {
                "retrieval_count": 12,
                "max_tokens": 1000,
                "temperature": 0.3,
                "response_style": "comprehensive"
            },
            QueryIntent.INTERPRETATIVE: {
                "retrieval_count": 8,
                "max_tokens": 700,
                "temperature": 0.2,
                "response_style": "interpretive"
            },
            QueryIntent.FACTUAL: {
                "retrieval_count": 4,
                "max_tokens": 250,
                "temperature": 0.1,
                "response_style": "direct"
            }
        }
    
    async def initialize(self):
        """Initialize the classifier with OpenAI client if available."""
        if self.initialized:
            return
            
        try:
            if settings.openai_api_key:
                import openai
                self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
                logger.info("Query Intent Classifier initialized with OpenAI support")
            else:
                logger.info("Query Intent Classifier initialized with pattern-based classification only")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Query Intent Classifier: {e}")
            self.initialized = True  # Still allow pattern-based classification
    
    async def classify_intent(self, query: str) -> IntentClassification:
        """
        Classify the intent of a query using pattern matching and optional LLM enhancement.
        
        Args:
            query: The user query to classify
            
        Returns:
            IntentClassification: Classification result with parameters
        """
        if not self.initialized:
            await self.initialize()
        
        query_lower = query.lower().strip()
        
        # First, try pattern-based classification
        pattern_result = self._classify_by_patterns(query_lower)
        
        # If we have OpenAI client, enhance with LLM-based classification
        if self.openai_client:
            llm_result = await self._classify_with_llm(query)
            
            # Combine results with LLM taking precedence for high confidence
            if llm_result.confidence > 0.8:
                final_intent = llm_result.intent
                final_confidence = llm_result.confidence
                final_reasoning = f"LLM classification: {llm_result.reasoning}"
            else:
                # Use pattern-based result but adjust confidence based on LLM
                final_intent = pattern_result.intent
                final_confidence = (pattern_result.confidence + llm_result.confidence) / 2
                final_reasoning = f"Pattern-based: {pattern_result.reasoning}. LLM validation: {llm_result.reasoning}"
        else:
            final_intent = pattern_result.intent
            final_confidence = pattern_result.confidence
            final_reasoning = pattern_result.reasoning
        
        # Get parameters for the classified intent
        params = self.intent_parameters.get(final_intent, self.intent_parameters[QueryIntent.FACTUAL])
        
        return IntentClassification(
            intent=final_intent,
            confidence=final_confidence,
            reasoning=final_reasoning,
            suggested_retrieval_count=params["retrieval_count"],
            suggested_max_tokens=params["max_tokens"],
            suggested_temperature=params["temperature"]
        )
    
    def _classify_by_patterns(self, query_lower: str) -> IntentClassification:
        """Classify query intent using pattern matching."""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    score += 1
                    matched_patterns.append(pattern)
            
            if score > 0:
                intent_scores[intent] = {
                    'score': score,
                    'patterns': matched_patterns
                }
        
        if intent_scores:
            # Get the intent with highest score
            best_intent = max(intent_scores.items(), key=lambda x: x[1]['score'])[0]
            best_score = intent_scores[best_intent]['score']
            matched_patterns = intent_scores[best_intent]['patterns']
            
            # Calculate confidence based on score and pattern specificity
            confidence = min(0.9, best_score * 0.2 + 0.3)
            
            reasoning = f"Matched {best_score} pattern(s): {', '.join(matched_patterns[:2])}"
        else:
            # Default to factual if no patterns match
            best_intent = QueryIntent.FACTUAL
            confidence = 0.5
            reasoning = "No specific patterns matched, defaulting to factual query"
        
        return IntentClassification(
            intent=best_intent,
            confidence=confidence,
            reasoning=reasoning,
            suggested_retrieval_count=self.intent_parameters[best_intent]["retrieval_count"],
            suggested_max_tokens=self.intent_parameters[best_intent]["max_tokens"],
            suggested_temperature=self.intent_parameters[best_intent]["temperature"]
        )
    
    async def _classify_with_llm(self, query: str) -> IntentClassification:
        """Classify query intent using LLM-based zero-shot classification."""
        try:
            prompt = f"""Classify the following legal query into one of these intent categories:

1. DEFINITION - Asking for a definition or meaning of a legal term/concept
2. LIST - Requesting a list or enumeration of items
3. EXPLANATION - Asking for detailed explanation or description
4. COMPARATIVE - Comparing two or more legal concepts/provisions
5. PROCEDURAL - Asking about procedures, steps, or processes
6. ANALYTICAL - Requesting analysis, evaluation, or assessment
7. INTERPRETATIVE - Asking for interpretation or implications
8. FACTUAL - Simple factual question

Query: "{query}"

Respond with only the category name and a brief reason (max 20 words). Format: CATEGORY: reason"""

            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a legal query classifier. Classify queries into the most appropriate intent category."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse the response
            if ":" in result:
                intent_str, reason = result.split(":", 1)
                intent_str = intent_str.strip().upper()
                reason = reason.strip()
            else:
                intent_str = result.strip().upper()
                reason = "LLM classification"
            
            # Map to enum
            intent_mapping = {
                "DEFINITION": QueryIntent.DEFINITION,
                "LIST": QueryIntent.LIST,
                "EXPLANATION": QueryIntent.EXPLANATION,
                "COMPARATIVE": QueryIntent.COMPARATIVE,
                "PROCEDURAL": QueryIntent.PROCEDURAL,
                "ANALYTICAL": QueryIntent.ANALYTICAL,
                "INTERPRETATIVE": QueryIntent.INTERPRETATIVE,
                "FACTUAL": QueryIntent.FACTUAL
            }
            
            intent = intent_mapping.get(intent_str, QueryIntent.FACTUAL)
            confidence = 0.8 if intent_str in intent_mapping else 0.5
            
            return IntentClassification(
                intent=intent,
                confidence=confidence,
                reasoning=reason,
                suggested_retrieval_count=self.intent_parameters[intent]["retrieval_count"],
                suggested_max_tokens=self.intent_parameters[intent]["max_tokens"],
                suggested_temperature=self.intent_parameters[intent]["temperature"]
            )
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Fallback to pattern-based classification
            return self._classify_by_patterns(query.lower().strip())
    
    def get_intent_parameters(self, intent: QueryIntent) -> Dict[str, Any]:
        """Get processing parameters for a specific intent."""
        return self.intent_parameters.get(intent, self.intent_parameters[QueryIntent.FACTUAL])
    
    def get_supported_intents(self) -> List[QueryIntent]:
        """Get list of all supported query intents."""
        return list(QueryIntent)


# Global instance
query_intent_classifier = QueryIntentClassifier()
