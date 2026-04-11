"""
Hallucination validation service to prevent false legal citations and information.

Provides validation mechanisms to detect and prevent hallucinated legal content,
citations, and references that are not present in the source documents.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of hallucination validation."""
    is_valid: bool
    confidence: float
    issues: List[str]
    suggestions: List[str]


class HallucinationValidator:
    """Validates responses to prevent hallucinated legal content."""
    
    def __init__(self):
        self.legal_citation_patterns = [
            r'Article\s+\d+',
            r'Section\s+\d+',
            r'Chapter\s+\d+',
            r'Part\s+\d+',
            r'Clause\s+\d+',
            r'Paragraph\s+\d+',
            r'Subsection\s+\d+',
            r'\(\d+\)',
            r'\(\w+\)',
        ]
        
        self.document_name_patterns = [
            r'UN Convention Against Corruption',
            r'UNCAC',
            r'United Nations Convention Against Corruption',
            r'Convention Against Corruption',
        ]
    
    def validate_response(self, response: str, context: str, query: str) -> ValidationResult:
        """
        Validate a response for potential hallucination.
        
        Args:
            response: The generated response to validate
            context: The source context used for generation
            query: The original query
            
        Returns:
            ValidationResult: Validation result with issues and suggestions
        """
        issues = []
        suggestions = []
        confidence = 1.0
        
        # Run multiple validation checks to detect different types of hallucinations
        citation_issues = self._validate_citations(response, context)
        issues.extend(citation_issues)
        
        doc_issues = self._validate_document_references(response, context, query)
        issues.extend(doc_issues)
        
        content_issues = self._validate_legal_content(response, context)
        issues.extend(content_issues)
        
        intent_issues = self._validate_query_intent(response, query)
        issues.extend(intent_issues)
        
        # Calculate confidence based on number of issues found (each issue reduces confidence by 20%)
        if issues:
            confidence = max(0.0, 1.0 - (len(issues) * 0.2))
            suggestions.append("Review the response for accuracy and ensure all citations are from the provided context")
            suggestions.append("If information is not available in the context, state this clearly rather than generating content")
        
        is_valid = len(issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions
        )
    
    def _validate_citations(self, response: str, context: str) -> List[str]:
        """Validate that all citations in response exist in context."""
        issues = []
        
        # Find all citations in response
        response_citations = []
        for pattern in self.legal_citation_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            response_citations.extend(matches)
        
        # Verify each citation exists in the provided context to prevent hallucination
        for citation in response_citations:
            if citation.lower() not in context.lower():
                issues.append(f"Citation '{citation}' not found in provided context")
        
        return issues
    
    def _validate_document_references(self, response: str, context: str, query: str) -> List[str]:
        """Validate that document references match what's available in context."""
        issues = []
        
        # Check if response mentions documents not in context
        for doc_pattern in self.document_name_patterns:
            if re.search(doc_pattern, response, re.IGNORECASE):
                # Check if this document is mentioned in context
                if not re.search(doc_pattern, context, re.IGNORECASE):
                    issues.append(f"Document '{doc_pattern}' mentioned in response but not found in context")
        
        # Special validation for UNCAC document requests
        if "convention against corruption" in query.lower():
            if not any(pattern.lower() in context.lower() for pattern in self.document_name_patterns):
                issues.append("Query requests UNCAC information but document not available in context")
        
        return issues
    
    def _validate_legal_content(self, response: str, context: str) -> List[str]:
        """Validate that legal content is grounded in context."""
        issues = []
        
        # Check for specific legal terms that might be hallucinated
        legal_terms = [
            "preventive measures",
            "anti-corruption",
            "money laundering",
            "whistleblower",
            "public procurement",
            "transparency",
        ]
        
        # Check if legal terms mentioned in response are grounded in context
        for term in legal_terms:
            if term.lower() in response.lower():
                if term.lower() not in context.lower():
                    issues.append(f"Legal term '{term}' mentioned in response but not found in context")
        
        return issues
    
    def _validate_query_intent(self, response: str, query: str) -> List[str]:
        """Validate that response matches query intent."""
        issues = []
        
        # Validate that list requests are properly formatted with enumeration
        if any(word in query.lower() for word in ["list", "enumerate", "what are", "name the"]):
            if not any(indicator in response.lower() for indicator in ["1.", "2.", "3.", "-", "•", "first", "second", "third"]):
                issues.append("Query requests a list but response doesn't provide structured enumeration")
        
        # Validate that definition requests are adequately detailed
        if any(word in query.lower() for word in ["what is", "define", "definition", "meaning"]):
            if "not available" not in response.lower() and "not found" not in response.lower():
                if len(response.split()) < 10:
                    issues.append("Query requests definition but response may be too brief")
        
        return issues
    
    def should_reject_response(self, response: str, context: str, query: str) -> Tuple[bool, str]:
        """
        Determine if a response should be rejected due to hallucination.
        
        Returns:
            Tuple of (should_reject, reason)
        """
        validation = self.validate_response(response, context, query)
        
        # Reject if confidence is too low (below 30%)
        if validation.confidence < 0.3:
            return True, f"Response rejected due to low confidence ({validation.confidence:.2f}). Issues: {', '.join(validation.issues)}"
        
        # Reject if critical issues found (citations not in context)
        critical_issues = [issue for issue in validation.issues if "not found in context" in issue]
        if critical_issues:
            return True, f"Response rejected due to critical issues: {', '.join(critical_issues)}"
        
        return False, "Response accepted"
    
    def get_safe_response(self, query: str, context: str) -> str:
        """
        Generate a safe response when original response is rejected.
        
        Args:
            query: The original query
            context: The available context
            
        Returns:
            Safe response that doesn't hallucinate
        """
        # Check if context has sufficient content for safe response
        if not context or len(context.strip()) < 50:
            return "This information is not available in the provided legal documents."
        
        # Handle specific UNCAC document requests with appropriate response
        if "convention against corruption" in query.lower():
            return "The UN Convention Against Corruption (UNCAC) is not available in the provided legal documents."
        
        return "The requested information is not available in the provided legal documents. Please check if the relevant legal document has been uploaded to the system."


# Global instance
hallucination_validator = HallucinationValidator()
