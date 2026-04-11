"""
Core package for the legal research assistant.

This package contains core functionality including configuration,
database management, exceptions, utilities, and lifecycle management.
"""

from .config import settings, get_settings
from .database import db_manager, get_database_manager, init_database
from .exceptions import (
    LegalResearchException,
    DocumentProcessingError,
    EmbeddingGenerationError,
    DatabaseConnectionError,
    CacheOperationError,
    LegalAgentError,
    QueryProcessingError,
    RateLimitExceededError,
    ConfigurationError
)
from .rate_limiter import rate_limiter, RateLimiter
from .response_formatter import response_formatter, ResponseFormatter
from .utils import (
    text_processor,
    validation_utils,
    data_transformer,
    TextProcessor,
    PerformanceTimer,
    ValidationUtils,
    DataTransformer,
    time_operation
)

__all__ = [
    # Configuration
    "settings",
    "get_settings",
    
    # Database
    "db_manager",
    "get_database_manager", 
    "init_database",
    
    # Exceptions
    "LegalResearchException",
    "DocumentProcessingError",
    "EmbeddingGenerationError",
    "DatabaseConnectionError",
    "CacheOperationError",
    "LegalAgentError",
    "QueryProcessingError",
    "RateLimitExceededError",
    "ConfigurationError",
    
    # Rate limiting
    "rate_limiter",
    "RateLimiter",
    
    # Response formatting
    "response_formatter",
    "ResponseFormatter",
    
    # Lifecycle management (imported locally to avoid circular imports)
    
    # Utilities
    "text_processor",
    "validation_utils",
    "data_transformer",
    "TextProcessor",
    "PerformanceTimer",
    "ValidationUtils",
    "DataTransformer",
    "time_operation"
] 