"""
Pydantic models for API request validation and response formatting.

Provides structured models for legal query requests, responses, PDF ingestion,
health checks, and service information with automatic validation and serialization.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from enum import Enum


class LegalResearchAlgorithm(str, Enum):
    HYBRID = "hybrid"
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    

class LegalQueryRequest(BaseModel):
    
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=10000,
        description="The legal research query to process"
    )
    top_k: int = Field(
        default=5, 
        ge=1, 
        le=50,
        description="Number of top relevant documents to retrieve"
    )
    use_agent: bool = Field(
        default=False,
        description="Whether to use the LangChain agent for enhanced processing"
    )
    algorithm: LegalResearchAlgorithm = Field(
        default=LegalResearchAlgorithm.HYBRID,
        description="Algorithm to use for document retrieval"
    )
    similarity_threshold: float = Field(
        default=0.3, 
        ge=0.0, 
        le=1.0,
        description="Minimum similarity threshold for document relevance"
    )
    enable_multi_hop_reasoning: bool = Field(
        default=True,
        description="Whether to enable multi-hop reasoning for complex queries"
    )
    force_multi_hop: bool = Field(
        default=False,
        description="Force multi-hop reasoning even for simple queries"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for tracking reasoning chains"
    )
    text_only: bool = Field(
        default=False,
        description="Return only the response text instead of full JSON structure"
    )
    
    @field_validator('query')
    def validate_query_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query cannot be empty or contain only whitespace")
        return value.strip()
    
    @field_validator('similarity_threshold')
    def validate_similarity_threshold(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        return value


class LegalQueryResponse(BaseModel):
    
    response: str = Field(..., description="The generated legal research response")
    query: str = Field(..., description="The original query that was processed")
    context: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Source documents and context used for the response"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the response"
    )
    source: str = Field(..., description="Source of the response (rag_engine or legal_agent)")
    response_time_ms: int = Field(..., description="Response time in milliseconds")


class LegalAgentResponse(BaseModel):
    
    response: str = Field(..., description="The legal research response")
    citations: List[str] = Field(
        default_factory=list,
        description="Extracted legal citations from the response"
    )
    domain: str = Field(..., description="Legal domain classification")
    confidence: float = Field(..., description="Confidence score for the classification")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Source documents used in the analysis"
    )
    tools_used: List[str] = Field(
        default_factory=list,
        description="Tools used by the agent during processing"
    )


class PDFIngestionRequest(BaseModel):
    
    source: str = Field(
        default="uploaded-pdf",
        description="Source identifier for the uploaded documents"
    )
    
    @field_validator('source')
    def validate_source(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Source cannot be empty")
        return value.strip()


class PDFIngestionResponse(BaseModel):
    
    message: str = Field(..., description="Status message about the ingestion")
    document_ids: List[str] = Field(
        default_factory=list,
        description="IDs of successfully ingested documents"
    )
    status: str = Field(..., description="Overall status of the ingestion operation")


class HealthCheckResponse(BaseModel):
    
    status: str = Field(..., description="Overall system health status")
    timestamp: float = Field(..., description="Unix timestamp of the health check")
    database: str = Field(..., description="Database connection status")
    services: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of various system services"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if health check failed"
    )


class ServiceInfoResponse(BaseModel):
    
    service: str = Field(..., description="Name of the service")
    version: str = Field(..., description="Service version")
    status: str = Field(..., description="Current service status")
    endpoints: Dict[str, str] = Field(
        default_factory=dict,
        description="Available API endpoints"
    )


class ReasoningStepResponse(BaseModel):
    step_id: str = Field(..., description="Unique identifier for the reasoning step")
    step_type: str = Field(..., description="Type of reasoning step")
    input_query: str = Field(..., description="Input query for this step")
    output_result: str = Field(..., description="Output result from this step")
    confidence_score: float = Field(..., description="Confidence score for this step")
    execution_time: float = Field(..., description="Execution time in seconds")
    sources_used: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sources used in this step"
    )


class MultiHopReasoningResponse(BaseModel):
    chain_id: str = Field(..., description="Unique identifier for the reasoning chain")
    original_query: str = Field(..., description="The original complex query")
    complexity_level: str = Field(..., description="Detected complexity level")
    final_answer: str = Field(..., description="Final synthesized answer")
    reasoning_steps: List[ReasoningStepResponse] = Field(
        default_factory=list,
        description="Individual reasoning steps"
    )
    total_execution_time: float = Field(..., description="Total execution time in seconds")
    overall_confidence: float = Field(..., description="Overall confidence score")
    citations: List[str] = Field(
        default_factory=list,
        description="All legal citations found"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the reasoning process"
    )
    source: str = Field(default="multi_hop_reasoning", description="Source of the response")


class ReasoningChainRequest(BaseModel):
    chain_id: Optional[str] = Field(
        default=None,
        description="Specific reasoning chain ID to retrieve"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID to get all chains for"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of chains to return"
    )
