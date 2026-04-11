"""
Legal analysis tools for citation extraction and text classification.

Provides pattern-based legal citation extraction, domain classification using
machine learning with keyword fallback, and comprehensive response analysis.
"""

import re
import logging
from typing import List, Dict, Any, Set
from .legal_classifier import legal_classifier

logger = logging.getLogger(__name__)


class LegalCitationExtractor:
    """
    Extracts legal citations from text using pattern matching.
    
    Provides methods to identify and extract various types of legal citations
    including case law, statutes, and legal references.
    """
    
    # Legal citation patterns for different types of legal references
    CITATION_PATTERNS = [
        # Case law citations (e.g., "123 U.S. 456")
        r"\b\d+\s+[A-Z][A-Za-z\.]*(?:\s*[A-Z][A-Za-z\.]*)*\s+\d+\b",
        
        # Full case citations (e.g., "Smith v. Jones, 123 F.2d 456")
        r"\b[A-Z][A-Za-z\.&\-\s]+\sv\.\s[A-Z][A-Za-z\.&\-\s]+,\s*\d+\s+[A-Z][A-Za-z\.]*(?:\s*[A-Z][A-Za-z\.]*)*\s+\d+\b",
        
        # Year-based citations (e.g., "2023 (Supreme Court)")
        r"\b\d{4}\b\s*\([A-Za-z\s\.]+\)",
        
        # Act citations (e.g., "Civil Rights Act 1964")
        r"\b[A-Z][A-Za-z\s]+\sAct\s\d{4}\b",
        
        # Section references (e.g., "Section 123A")
        r"\bSection\s+\d+[A-Za-z\-]*\b",
        
        # Article references (e.g., "Article 14")
        r"\bArticle\s+\d+[A-Za-z\-]*\b",
    ]
    
    @classmethod
    def extract_citations(cls, text: str) -> List[str]:
        """
        Extract legal citations from the given text.
        
        Args:
            text: Text to extract citations from
            
        Returns:
            List[str]: List of unique legal citations found in the text
        """
        if not text or not text.strip():
            return []
        
        citations: List[str] = []
        
        for pattern in cls.CITATION_PATTERNS:
            try:
                matches = re.findall(pattern, text)
                citations.extend(matches)
            except re.error as e:
                logger.warning(f"Invalid regex pattern {pattern}: {e}")
                continue
        
        return cls._remove_duplicates(citations)
    
    @staticmethod
    def _remove_duplicates(citations: List[str]) -> List[str]:
        """
        Remove duplicate citations while preserving order.
        
        Args:
            citations: List of citations that may contain duplicates
            
        Returns:
            List[str]: List of unique citations
        """
        seen: Set[str] = set()
        unique_citations: List[str] = []
        
        for citation in citations:
            if citation not in seen:
                seen.add(citation)
                unique_citations.append(citation)
        
        return unique_citations


class LegalTextClassifier:
    """Classifies legal text into domains using ML models with keyword fallback."""
    
    # Keywords for different legal domains
    CONSTITUTIONAL_KEYWORDS = [
        "fundamental right", "article 14", "constitution", "writ", 
        "supreme court of india", "bill of rights", "amendment"
    ]
    
    CRIMINAL_KEYWORDS = [
        "ipc", "indian penal code", "mens rea", "actus reus", 
        "offence", "punishment", "crime", "homicide", "murder", "theft"
    ]
    
    CONTRACT_KEYWORDS = [
        "contract", "agreement", "consideration", "offer", 
        "acceptance", "breach", "damages", "indemnity"
    ]
    
    @classmethod
    def classify_text(cls, text: str) -> str:
        """
        Classify legal text into domains using ML model with keyword fallback.
        
        Args:
            text: Text to classify
            
        Returns:
            str: Legal domain classification (Constitutional, Criminal, Contract, Other)
        """
        if not text or not text.strip():
            return "Other"
        
        try:
            # Try ML-based classification first
            result = legal_classifier.classify(text)
            return result["category"]
        except Exception as e:
            logger.warning(f"ML classification failed, using keyword fallback: {e}")
            return cls._classify_by_keywords(text)
    
    @classmethod
    def _classify_by_keywords(cls, text: str) -> str:
        """
        Classify text using keyword matching as fallback.
        
        Args:
            text: Text to classify
            
        Returns:
            str: Legal domain based on keyword matching
        """
        text_lower = text.lower()
        
        if cls._contains_any_keyword(text_lower, cls.CONSTITUTIONAL_KEYWORDS):
            return "Constitutional"
        
        if cls._contains_any_keyword(text_lower, cls.CRIMINAL_KEYWORDS):
            return "Criminal"
        
        if cls._contains_any_keyword(text_lower, cls.CONTRACT_KEYWORDS):
            return "Contract"
        
        return "Other"
    
    @staticmethod
    def _contains_any_keyword(text: str, keywords: List[str]) -> bool:
        """
        Check if text contains any of the specified keywords.
        
        Args:
            text: Text to search in
            keywords: List of keywords to search for
            
        Returns:
            bool: True if any keyword is found, False otherwise
        """
        return any(keyword in text for keyword in keywords)


class LegalResponseAnalyzer:
    """Analyzes legal responses for citations and domain classification."""
    
    def __init__(self):
        self.citation_extractor = LegalCitationExtractor()
        self.text_classifier = LegalTextClassifier()
    
    def analyze_response(
        self, 
        query: str, 
        response_text: str, 
        context_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze a legal response for citations and domain classification.
        
        Args:
            query: Original legal query
            response_text: Generated response text
            context_documents: Context documents used for the response
            
        Returns:
            Dict[str, Any]: Analysis results including citations and domain
        """
        try:
            # Extract citations from response
            citations = self.citation_extractor.extract_citations(response_text)
            
            # Combine query and context for domain classification
            combined_text = self._combine_text_for_classification(
                query, response_text, context_documents
            )
            
            # Classify the combined text
            domain = self.text_classifier.classify_text(combined_text)
            
            return {
                "citations": citations,
                "domain": domain
            }
            
        except Exception as e:
            logger.error(f"Error analyzing legal response: {e}")
            return {
                "citations": [],
                "domain": "Other"
            }
    
    def _combine_text_for_classification(
        self, 
        query: str, 
        response_text: str, 
        context_documents: List[Dict[str, Any]]
    ) -> str:
        """
        Combine query, response, and context for domain classification.
        
        Args:
            query: Original query
            response_text: Response text
            context_documents: Context documents
            
        Returns:
            str: Combined text for classification
        """
        text_parts = [query, response_text]
        
        # Add content from context documents
        for doc in context_documents:
            content = doc.get("content", "")
            if content:
                text_parts.append(content)
        
        return " ".join(text_parts)


# Global instances for backward compatibility
legal_citation_extractor = LegalCitationExtractor()
legal_text_classifier = LegalTextClassifier()
legal_response_analyzer = LegalResponseAnalyzer()


# Backward compatibility functions
def extract_legal_citations(text: str) -> List[str]:
    """Extract legal citations from text (backward compatibility)."""
    return legal_citation_extractor.extract_citations(text)


def classify_legal_text(text: str) -> str:
    """Classify legal text into domains (backward compatibility)."""
    return legal_text_classifier.classify_text(text)


def analyze_legal_response(
    query: str, 
    response_text: str, 
    context_docs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze legal response for citations and domain (backward compatibility)."""
    return legal_response_analyzer.analyze_response(query, response_text, context_docs)


