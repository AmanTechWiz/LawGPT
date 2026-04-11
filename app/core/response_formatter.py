"""
Response formatting utilities for API responses and text content cleaning.

Provides text cleaning, legal response formatting, and error response formatting
with specialized handling for legal references and document structure.
"""

import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """
    Utility class for formatting API responses and cleaning text content.
    
    Provides methods for cleaning and formatting text responses to improve
    readability and consistency across the application.
    """
    
    @staticmethod
    def clean_text_for_display(text: str) -> str:
        """
        Clean and format text for clean display by removing escape characters and formatting.
        
        Args:
            text: Raw text to clean
            
        Returns:
            str: Cleaned and formatted text
        """
        if not text:
            return text
        
        cleaned_text = text.replace('\\n', '\n')
        cleaned_text = cleaned_text.replace('\\"', '"')
        cleaned_text = cleaned_text.replace('\\\\', '\\')
        
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
        
        cleaned_text = re.sub(r'\n[ \t]+', '\n', cleaned_text)
        cleaned_text = re.sub(r'[ \t]+\n', '\n', cleaned_text)
        
        cleaned_text = re.sub(r'([.!?])([A-Z])', r'\1 \2', cleaned_text)
        cleaned_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_text)
        
        cleaned_text = re.sub(r'Article\s+(\d+)', r'Article \1', cleaned_text)
        cleaned_text = re.sub(r'Section\s+(\d+)', r'Section \1', cleaned_text)
        cleaned_text = re.sub(r'Chapter\s+(\d+)', r'Chapter \1', cleaned_text)
        
        cleaned_text = re.sub(r'[.]{3,}', '...', cleaned_text)
        cleaned_text = re.sub(r'[-]{3,}', '---', cleaned_text)
        
        cleaned_text = re.sub(r'\n(\d+\.)', r'\n\n\1', cleaned_text)
        cleaned_text = re.sub(r'\n([-\*])', r'\n\n\1', cleaned_text)
        
        cleaned_text = re.sub(r'\n([A-Z][A-Z\s]+:)\n', r'\n\n**\1**\n', cleaned_text)
        cleaned_text = re.sub(r'\n(\d+\.\s*[A-Z][^:]+:)\n', r'\n\n\1\n', cleaned_text)
        
        cleaned_text = re.sub(r'\b(Article|Section|Chapter)\s+(\d+)\b', r'**\1 \2**', cleaned_text)
        
        cleaned_text = re.sub(r'([.!?])\s*\n([A-Z])', r'\1\n\n\2', cleaned_text)
        
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        cleaned_text = re.sub(r'^\s+', '', cleaned_text, flags=re.MULTILINE)
        
        return cleaned_text.strip()
    
    @staticmethod
    def format_legal_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a legal research response for API output.
        
        Args:
            response_data: Raw response data from the RAG system
            
        Returns:
            Dict[str, Any]: Formatted response data
        """
        try:
            formatted_response = ResponseFormatter.clean_text_for_display(
                response_data.get("response", "")
            )
            
            return {
                "response": formatted_response,
                "query": response_data.get("query", ""),
                "context": response_data.get("sources", []),
                "metadata": response_data.get("metadata", {}),
                "source": response_data.get("source", "rag_engine"),
                "response_time_ms": int(response_data.get("processing_time", 0) * 1000)
            }
        except Exception as e:
            logger.error(f"Error formatting legal response: {e}")
            return response_data
    
    @staticmethod
    def format_agent_response(agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a legal agent response for API output.
        
        Args:
            agent_data: Raw response data from the legal agent
            
        Returns:
            Dict[str, Any]: Formatted response data
        """
        try:
            formatted_response = ResponseFormatter.clean_text_for_display(
                agent_data.get("response", "")
            )
            
            return {
                "response": formatted_response,
                "query": agent_data.get("query", ""),
                "context": agent_data.get("sources", []),
                "metadata": {
                    "algorithm": "langchain_agent",
                    "citations": agent_data.get("citations", []),
                    "domain": agent_data.get("domain", "Other"),
                    "confidence": agent_data.get("confidence", 0.0),
                    "tools_used": agent_data.get("tools_used", [])
                },
                "source": "legal_agent",
                "response_time_ms": 0
            }
        except Exception as e:
            logger.error(f"Error formatting agent response: {e}")
            return agent_data
    
    @staticmethod
    def clean_context_content(content: str) -> str:
        """
        Clean content for better context presentation in RAG responses.
        
        Args:
            content: Raw content to clean
            
        Returns:
            str: Cleaned content
        """
        if not content:
            return content
        
        # Normalize whitespace
        cleaned_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        cleaned_content = re.sub(r'[ \t]+', ' ', cleaned_content)
        cleaned_content = re.sub(r'\n[ \t]+', '\n', cleaned_content)
        cleaned_content = re.sub(r'[ \t]+\n', '\n', cleaned_content)
        
        # Fix sentence spacing
        cleaned_content = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_content)
        cleaned_content = re.sub(r'([.!?])([A-Z])', r'\1 \2', cleaned_content)
        
        # Format legal references
        cleaned_content = re.sub(r'Article\s+(\d+)', r'Article \1', cleaned_content)
        cleaned_content = re.sub(r'Section\s+(\d+)', r'Section \1', cleaned_content)
        cleaned_content = re.sub(r'Chapter\s+(\d+)', r'Chapter \1', cleaned_content)
        
        # Normalize punctuation
        cleaned_content = re.sub(r'[.]{3,}', '...', cleaned_content)
        cleaned_content = re.sub(r'[-]{3,}', '---', cleaned_content)
        
        return cleaned_content.strip()
    
    @staticmethod
    def format_error_response(error_message: str, error_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Format an error response for API output.
        
        Args:
            error_message: Error message to include
            error_code: Optional error code
            
        Returns:
            Dict[str, Any]: Formatted error response
        """
        response = {
            "error": "An error occurred while processing your request",
            "detail": error_message
        }
        
        if error_code:
            response["error_code"] = error_code
        
        return response


# Global formatter instance
response_formatter = ResponseFormatter()
