"""
Models package for the legal research assistant.

This package contains all data models including database models,
request/response models, and related enumerations.
"""

from .document import (
    Base,
    LegalDocument,
    LegalConversationHistory,
    DocumentProcessingStatus
)

from .requests import (
    LegalQueryRequest,
    LegalQueryResponse,
    LegalAgentResponse,
    PDFIngestionRequest,
    PDFIngestionResponse,
    HealthCheckResponse,
    ServiceInfoResponse,
    LegalResearchAlgorithm
)

__all__ = [
    # Database models
    "Base",
    "LegalDocument", 
    "LegalConversationHistory",
    "DocumentProcessingStatus",
    
    # Request/Response models
    "LegalQueryRequest",
    "LegalQueryResponse", 
    "LegalAgentResponse",
    "PDFIngestionRequest",
    "PDFIngestionResponse",
    "HealthCheckResponse",
    "ServiceInfoResponse",
    "LegalResearchAlgorithm"
] 