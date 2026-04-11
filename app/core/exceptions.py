"""
Custom exception classes for specific error handling scenarios.

Defines specialized exceptions for document processing, embedding generation,
database operations, cache management, and legal agent errors with context information.
"""

from typing import Optional, Dict, Any


class LegalResearchException(Exception):
    """
    Base exception for legal research application errors.
    
    Provides common error handling with optional error codes and context.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}


class DocumentProcessingError(LegalResearchException):
    """
    Exception for document processing failures.
    
    Raised during PDF ingestion, text extraction, or embedding generation.
    """
    
    def __init__(
        self, 
        message: str, 
        document_id: Optional[str] = None,
        processing_stage: Optional[str] = None
    ):
        context = {}
        if document_id:
            context["document_id"] = document_id
        if processing_stage:
            context["processing_stage"] = processing_stage
        
        super().__init__(
            message=message,
            error_code="DOCUMENT_PROCESSING_ERROR",
            context=context
        )


class EmbeddingGenerationError(LegalResearchException):
    """
    Exception for vector embedding generation failures.
    
    Raised during OpenAI API calls or database storage issues.
    """
    
    def __init__(
        self, 
        message: str, 
        text_length: Optional[int] = None,
        model_name: Optional[str] = None
    ):
        context = {}
        if text_length:
            context["text_length"] = text_length
        if model_name:
            context["model_name"] = model_name
        
        super().__init__(
            message=message,
            error_code="EMBEDDING_GENERATION_ERROR",
            context=context
        )


class DatabaseConnectionError(LegalResearchException):
    """
    Exception for database operation failures.
    
    Raised for connection issues, query failures, or initialization problems.
    """
    
    def __init__(
        self, 
        message: str, 
        operation: Optional[str] = None,
        connection_pool_size: Optional[int] = None
    ):
        context = {}
        if operation:
            context["operation"] = operation
        if connection_pool_size:
            context["connection_pool_size"] = connection_pool_size
        
        super().__init__(
            message=message,
            error_code="DATABASE_CONNECTION_ERROR",
            context=context
        )


class CacheOperationError(LegalResearchException):
    """
    Exception for cache operation failures.
    
    Raised for Redis connection issues, key generation, or serialization errors.
    """
    
    def __init__(
        self, 
        message: str, 
        cache_key: Optional[str] = None,
        operation_type: Optional[str] = None
    ):
        context = {}
        if cache_key:
            context["cache_key"] = cache_key
        if operation_type:
            context["operation_type"] = operation_type
        
        super().__init__(
            message=message,
            error_code="CACHE_OPERATION_ERROR",
            context=context
        )


class LegalAgentError(LegalResearchException):
    """
    Exception for legal agent failures.
    
    Raised for LangChain agent failures, tool execution errors, or initialization issues.
    """
    
    def __init__(
        self, 
        message: str, 
        agent_tool: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        context = {}
        if agent_tool:
            context["agent_tool"] = agent_tool
        if session_id:
            context["session_id"] = session_id
        
        super().__init__(
            message=message,
            error_code="LEGAL_AGENT_ERROR",
            context=context
        )


class QueryProcessingError(LegalResearchException):
    """
    Exception for query processing failures.
    
    Raised during query validation, embedding generation, or response generation.
    """
    
    def __init__(
        self, 
        message: str, 
        query: Optional[str] = None,
        algorithm: Optional[str] = None
    ):
        context = {}
        if query:
            context["query"] = query[:100] + "..." if len(query) > 100 else query
        if algorithm:
            context["algorithm"] = algorithm
        
        super().__init__(
            message=message,
            error_code="QUERY_PROCESSING_ERROR",
            context=context
        )


class RateLimitExceededError(LegalResearchException):
    """
    Exception for rate limit violations.
    
    Raised when API rate limits or request throttling limits are exceeded.
    """
    
    def __init__(
        self, 
        message: str, 
        client_ip: Optional[str] = None,
        request_count: Optional[int] = None,
        limit: Optional[int] = None
    ):
        context = {}
        if client_ip:
            context["client_ip"] = client_ip
        if request_count:
            context["request_count"] = request_count
        if limit:
            context["limit"] = limit
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            context=context
        )


class ConfigurationError(LegalResearchException):
    """
    Exception for configuration errors.
    
    Raised for missing environment variables, invalid settings, or validation failures.
    """
    
    def __init__(
        self, 
        message: str, 
        setting_name: Optional[str] = None,
        expected_type: Optional[str] = None
    ):
        context = {}
        if setting_name:
            context["setting_name"] = setting_name
        if expected_type:
            context["expected_type"] = expected_type
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            context=context
        )
