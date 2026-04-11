"""
Query complexity detection and routing system.

Automatically detects complex queries that require multi-hop reasoning and
routes them to the appropriate processing pipeline.
"""

import logging
import re
from typing import Dict, Any, Tuple, List
from enum import Enum

from .multi_hop_reasoning import ReasoningComplexity, QueryComplexityAnalyzer

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of legal queries based on complexity and structure"""
    SIMPLE_FACTUAL = "simple_factual"
    DEFINITION = "definition"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    MULTI_ASPECT = "multi_aspect"
    PROCEDURAL = "procedural"
    INTERPRETATIVE = "interpretative"


class QueryComplexityDetector:
    """Advanced query complexity detection with legal domain awareness and routing."""
    
    def __init__(self):
        self.complexity_analyzer = QueryComplexityAnalyzer()
        
        # Legal-specific complexity patterns (optimized to prevent catastrophic backtracking)
        self.legal_complexity_patterns = {
            "multi_article": [
                r"article\s+\d+[^a-z]*article\s+\d+",
                r"section\s+\d+[^a-z]*section\s+\d+",
                r"chapter\s+\d+[^a-z]*chapter\s+\d+"
            ],
            "conditional_reasoning": [
                r"if[^a-z]{1,50}then",
                r"when[^a-z]{1,50}shall",
                r"unless[^a-z]{1,50}provided",
                r"in\s+case[^a-z]{1,50}where"
            ],
            "comparative_analysis": [
                r"compare[^a-z]{1,50}with",
                r"difference\s+between",
                r"similarity[^a-z]{1,50}and",
                r"versus[^a-z]{1,50}versus"
            ],
            "causal_chain": [
                r"because[^a-z]{1,50}therefore",
                r"due\s+to[^a-z]{1,50}as\s+a\s+result",
                r"leads\s+to[^a-z]{1,50}consequently"
            ],
            "multi_document": [
                r"across[^a-z]{1,50}documents",
                r"throughout[^a-z]{1,50}chapters",
                r"in\s+all[^a-z]{1,50}sections",
                r"various[^a-z]{1,50}provisions"
            ],
            "procedural_complexity": [
                r"step\s+by\s+step",
                r"process[^a-z]{1,50}procedure",
                r"workflow[^a-z]{1,50}sequence",
                r"timeline[^a-z]{1,50}stages"
            ]
        }
        
        # Query type classification patterns
        self.query_type_patterns = {
            QueryType.SIMPLE_FACTUAL: [
                r"what\s+is",
                r"define",
                r"explain\s+the\s+meaning",
                r"what\s+does.*?mean"
            ],
            QueryType.DEFINITION: [
                r"definition\s+of",
                r"meaning\s+of",
                r"what\s+is.*?defined\s+as"
            ],
            QueryType.COMPARATIVE: [
                r"compare.*?and",
                r"difference\s+between",
                r"similarity.*?and",
                r"versus.*?vs"
            ],
            QueryType.ANALYTICAL: [
                r"analyze.*?implications",
                r"evaluate.*?effectiveness",
                r"assess.*?impact",
                r"examine.*?consequences"
            ],
            QueryType.MULTI_ASPECT: [
                r"all\s+aspects\s+of",
                r"various.*?elements",
                r"different.*?components",
                r"multiple.*?factors"
            ],
            QueryType.PROCEDURAL: [
                r"how\s+to.*?process",
                r"steps\s+involved",
                r"procedure\s+for",
                r"workflow.*?sequence"
            ],
            QueryType.INTERPRETATIVE: [
                r"interpretation\s+of",
                r"how\s+to\s+interpret",
                r"meaning\s+and\s+scope",
                r"application\s+of"
            ]
        }
    
    def detect_complexity_and_type(self, query: str) -> Tuple[ReasoningComplexity, QueryType, Dict[str, Any]]:
        """
        Detect query complexity and type with detailed analysis.
        
        Args:
            query: The legal query to analyze
            
        Returns:
            Tuple of (complexity_level, query_type, analysis_details)
        """
        query_lower = query.lower().strip()
        
        # Get base complexity analysis
        complexity_level, base_analysis = self.complexity_analyzer.analyze_complexity(query)
        
        # Detect legal-specific complexity patterns
        legal_patterns_found = self._detect_legal_patterns(query_lower)
        
        # Classify query type
        query_type = self._classify_query_type(query_lower)
        
        # Calculate enhanced complexity score
        enhanced_score = self._calculate_enhanced_complexity_score(
            base_analysis, legal_patterns_found, query_type
        )
        
        # Determine final complexity level
        final_complexity = self._determine_final_complexity(enhanced_score, legal_patterns_found)
        
        # Prepare detailed analysis
        analysis_details = {
            **base_analysis,
            "legal_patterns_found": legal_patterns_found,
            "query_type": query_type.value,
            "enhanced_complexity_score": enhanced_score,
            "final_complexity_level": final_complexity.value,
            "requires_multi_hop": final_complexity in [ReasoningComplexity.COMPLEX, ReasoningComplexity.VERY_COMPLEX],
            "recommended_approach": self._get_recommended_approach(final_complexity, query_type)
        }
        
        logger.info(f"Query analysis: {final_complexity.value} complexity, {query_type.value} type")
        return final_complexity, query_type, analysis_details
    
    def _detect_legal_patterns(self, query_lower: str) -> Dict[str, List[str]]:
        """Detect legal-specific complexity patterns in the query"""
        patterns_found = {}
        
        for pattern_category, patterns in self.legal_complexity_patterns.items():
            found_patterns = []
            for pattern in patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                if matches:
                    found_patterns.extend(matches)
            
            if found_patterns:
                patterns_found[pattern_category] = found_patterns
        
        return patterns_found
    
    def _classify_query_type(self, query_lower: str) -> QueryType:
        """Classify the type of legal query"""
        type_scores = {}
        
        for query_type, patterns in self.query_type_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    score += 1
            type_scores[query_type] = score
        
        # Return the type with highest score, default to SIMPLE_FACTUAL
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        return QueryType.SIMPLE_FACTUAL
    
    def _calculate_enhanced_complexity_score(self, base_analysis: Dict[str, Any], 
                                           legal_patterns: Dict[str, List[str]], 
                                           query_type: QueryType) -> float:
        """Calculate enhanced complexity score considering legal patterns and query type"""
        base_score = base_analysis.get("complexity_score", 0)
        
        # Add points for legal complexity patterns
        legal_complexity_bonus = 0
        for pattern_category, patterns in legal_patterns.items():
            legal_complexity_bonus += len(patterns) * 2
        
        # Add points based on query type complexity
        type_complexity_bonus = {
            QueryType.SIMPLE_FACTUAL: 0,
            QueryType.DEFINITION: 1,
            QueryType.COMPARATIVE: 3,
            QueryType.ANALYTICAL: 4,
            QueryType.MULTI_ASPECT: 5,
            QueryType.PROCEDURAL: 3,
            QueryType.INTERPRETATIVE: 4
        }.get(query_type, 0)
        
        # Add points for query length and structure
        length_bonus = 0
        word_count = base_analysis.get("word_count", 0)
        if word_count > 100:
            length_bonus = 3
        elif word_count > 50:
            length_bonus = 2
        elif word_count > 30:
            length_bonus = 1
        
        enhanced_score = base_score + legal_complexity_bonus + type_complexity_bonus + length_bonus
        return enhanced_score
    
    def _determine_final_complexity(self, enhanced_score: float, 
                                  legal_patterns: Dict[str, List[str]]) -> ReasoningComplexity:
        """Determine final complexity level based on enhanced analysis"""
        # Adjust thresholds based on legal pattern complexity
        if "multi_document" in legal_patterns or "causal_chain" in legal_patterns:
            # Very complex legal reasoning required
            if enhanced_score >= 6:
                return ReasoningComplexity.VERY_COMPLEX
            elif enhanced_score >= 4:
                return ReasoningComplexity.COMPLEX
        elif "comparative_analysis" in legal_patterns or "multi_article" in legal_patterns:
            # Complex analysis required
            if enhanced_score >= 7:
                return ReasoningComplexity.VERY_COMPLEX
            elif enhanced_score >= 5:
                return ReasoningComplexity.COMPLEX
        else:
            # Standard complexity assessment
            if enhanced_score >= 8:
                return ReasoningComplexity.VERY_COMPLEX
            elif enhanced_score >= 5:
                return ReasoningComplexity.COMPLEX
            elif enhanced_score >= 3:
                return ReasoningComplexity.MODERATE
        
        return ReasoningComplexity.SIMPLE
    
    def _get_recommended_approach(self, complexity: ReasoningComplexity, 
                                query_type: QueryType) -> str:
        """Get recommended processing approach based on complexity and type"""
        if complexity == ReasoningComplexity.VERY_COMPLEX:
            return "multi_hop_reasoning_with_iterative_refinement"
        elif complexity == ReasoningComplexity.COMPLEX:
            if query_type in [QueryType.COMPARATIVE, QueryType.ANALYTICAL, QueryType.MULTI_ASPECT]:
                return "multi_hop_reasoning"
            else:
                return "enhanced_rag_with_agent"
        elif complexity == ReasoningComplexity.MODERATE:
            if query_type in [QueryType.PROCEDURAL, QueryType.INTERPRETATIVE]:
                return "enhanced_rag"
            else:
                return "standard_rag"
        else:
            return "standard_rag"
    
    def should_use_multi_hop_reasoning(self, query: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if a query should use multi-hop reasoning.
        
        Args:
            query: The legal query to evaluate
            
        Returns:
            Tuple of (should_use_multi_hop, analysis_details)
        """
        complexity, query_type, analysis = self.detect_complexity_and_type(query)
        
        should_use = analysis.get("requires_multi_hop", False)
        
        return should_use, analysis


# Global instance
query_complexity_detector = QueryComplexityDetector()
