"""
API endpoint definitions for legal research operations.

Provides RESTful endpoints for query processing, document ingestion, health checks,
and streaming responses with rate limiting and comprehensive error handling.
"""

import time
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Body, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio

from ..core.config import settings
from ..core.database import db_manager
from ..core.rate_limiter import rate_limiter
from ..core.response_formatter import response_formatter
from ..core.exceptions import RateLimitExceededError, QueryProcessingError
from ..models.requests import (
    LegalQueryRequest,
    LegalQueryResponse,
    PDFIngestionResponse,
    HealthCheckResponse,
    ServiceInfoResponse,
    MultiHopReasoningResponse,
    ReasoningStepResponse
)
from ..services.lightweight_llm_rag import lightweight_llm_rag
from ..services.legal_tools import legal_response_analyzer
from ..services.pdf_ingestion import pdf_ingestion_service
from ..services.langchain_agent import langchain_legal_agent
from ..services.multi_hop_reasoning import multi_hop_reasoning_engine
from ..services.query_complexity_detector import query_complexity_detector
from ..services.adaptive_rag_orchestrator import adaptive_rag_orchestrator
from ..services.feedback_system import feedback_system, UserFeedback, FeedbackType
from ..services.session_reset import reset_session_state
from ..services.cache import rag_cache

logger = logging.getLogger(__name__)

# Create router for API endpoints
router = APIRouter()


@router.get("/", response_model=ServiceInfoResponse)
async def get_service_info():
    """
    Get basic service information and available endpoints.
    
    Returns:
        ServiceInfoResponse: Service information including version and endpoints
    """
    return ServiceInfoResponse(
        service="Legal Research Assistant",
        version="1.0.0",
        status="running",
        endpoints={
            "query": "/query (JSON by default, text with text_only=true)",
            "query-json": "/query-json (always JSON)",
            "query-text": "/query-text (text only, deprecated)",
            "stream": "/stream", 
            "ingest": "/ingest-pdfs",
            "health": "/health",
            "docs": "/docs",
            "session-reset": "/session/reset (POST, requires ALLOW_SESSION_RESET=true)",
        }
    )


@router.post("/session/reset")
async def reset_session():
    """
    Truncate RAG tables and flush Redis / in-memory caches.
    Disabled by default; set ALLOW_SESSION_RESET=true to enable.
    """
    if not settings.allow_session_reset:
        raise HTTPException(
            status_code=403,
            detail="Session reset is disabled. Set ALLOW_SESSION_RESET=true to enable.",
        )
    return await reset_session_state()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Perform comprehensive health check of the system.
    
    Returns:
        HealthCheckResponse: Health status of all system components
    """
    try:
        database_healthy = await db_manager.health_check()
        
        return HealthCheckResponse(
            status="healthy" if database_healthy else "degraded",
            timestamp=time.time(),
            database="connected" if database_healthy else "disconnected",
            services={
                "rag_engine": "available",
                "legal_tools": "available",
                "pdf_ingestion": "available"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=time.time(),
            database="unknown",
            services={},
            error=str(e)
        )


@router.post("/query")
async def process_legal_query(request: Request, payload: LegalQueryRequest = Body(...)):
    """
    Process a legal research query using RAG or agent-based approach.
    
    Args:
        request: FastAPI request object for client IP
        payload: Legal query request with parameters
        
    Returns:
        LegalQueryResponse or str: Processed legal research response (JSON by default, text if text_only=True)
        
    Raises:
        HTTPException: If rate limit exceeded or processing fails
    """
    client_ip = request.client.host
    
    try:
        # Validate input
        if not payload.query or len(payload.query.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Query must be at least 3 characters long"
            )
        if len(payload.query) > 10000:
            raise HTTPException(
                status_code=400,
                detail="Query too long. Maximum 10000 characters allowed"
            )
        
        # Check rate limit
        rate_limiter.check_and_record_request(client_ip)
    except RateLimitExceededError as e:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "detail": e.message,
                "retry_after": e.context.get("reset_time", 0) - time.time()
            }
        )
    
    try:
        should_use_multi_hop, complexity_analysis = query_complexity_detector.should_use_multi_hop_reasoning(payload.query)
        
        # Route to appropriate processing pipeline based on complexity and user preferences
        if (payload.enable_multi_hop_reasoning and should_use_multi_hop) or payload.force_multi_hop:
            result = await _process_multi_hop_query(payload, complexity_analysis)
        elif payload.use_agent:
            result = await _process_agent_query(payload)
        else:
            result = await _process_rag_query(payload)
        
        if payload.text_only:
            if hasattr(result, 'response'):
                return {"response": result.response}
            elif hasattr(result, 'final_answer'):
                return {"response": result.final_answer}
            else:
                return {"response": str(result)}
        
        return result
            
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=response_formatter.format_error_response(
                str(e), 
                "QUERY_PROCESSING_ERROR"
            )
        )


@router.post("/query-text")
async def process_text_only_query(payload: LegalQueryRequest = Body(...)):
    """
    Process a legal query and return only the formatted response text.
    This endpoint is deprecated - use /query with text_only=true instead.
    
    Args:
        payload: Legal query request
        
    Returns:
        dict: Simple response with formatted text
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        result = await lightweight_llm_rag.query(
            query=payload.query,
            top_k=payload.top_k,
            use_agent=payload.use_agent,
            algorithm=payload.algorithm.value,
            similarity_threshold=payload.similarity_threshold,
        )
        
        formatted_response = response_formatter.clean_text_for_display(
            result.get("response", "")
        )
        
        return {"response": formatted_response}
        
    except Exception as e:
        logger.error(f"Text query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=response_formatter.format_error_response(
                str(e),
                "TEXT_QUERY_ERROR"
            )
        )


@router.post("/query-json", response_model=LegalQueryResponse)
async def process_json_query(request: Request, payload: LegalQueryRequest = Body(...)):
    """
    Process a legal query and return the full JSON response structure.
    This endpoint always returns JSON format regardless of text_only parameter.
    
    Args:
        request: FastAPI request object for client IP
        payload: Legal query request with parameters
        
    Returns:
        LegalQueryResponse: Full structured legal research response
        
    Raises:
        HTTPException: If rate limit exceeded or processing fails
    """
    # Override text_only to always return JSON
    payload.text_only = False
    return await process_legal_query(request, payload)


@router.post("/stream")
async def stream_legal_query(payload: LegalQueryRequest = Body(...)):
    """
    Stream legal research results with Server-Sent Events.
    
    Args:
        payload: Legal query request
        
    Returns:
        StreamingResponse: SSE stream with query results
    """
    async def generate_stream():
        try:
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Starting legal research...'})}\n\n"
            
            if payload.use_agent:
                result = await langchain_legal_agent.research(
                    query=payload.query,
                    session_id=f"stream_session_{int(time.time())}"
                )
                text = result.response
                citations = result.citations
                domain = result.domain
            else:
                result = await lightweight_llm_rag.query(
                    query=payload.query,
                    top_k=min(payload.top_k, 3),
                    use_agent=False,
                    algorithm="hybrid",
                    similarity_threshold=payload.similarity_threshold,
                )
                text = result.get("response", "")
                citations = []
                domain = "Other"
            
            yield f"data: {json.dumps({'status': 'streaming', 'message': 'Streaming response...'})}\n\n"
            
            # Stream text in chunks
            chunk_size = 100
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                yield f"data: {json.dumps({'delta': chunk, 'type': 'content'})}\n\n"
                await asyncio.sleep(0.05)
            
            # Send metadata
            yield f"data: {json.dumps({'type': 'metadata', 'citations': citations, 'domain': domain})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done', 'status': 'completed'})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )


@router.post("/ingest-pdfs", response_model=PDFIngestionResponse)
async def ingest_pdf_documents(files: List[UploadFile] = File(...)):
    """
    Ingest PDF documents into the legal knowledge base.
    
    Args:
        files: List of uploaded PDF files
        
    Returns:
        PDFIngestionResponse: Results of PDF ingestion
        
    Raises:
        HTTPException: If ingestion fails
    """
    try:
        temp_paths = []
        
        # Save uploaded files temporarily
        for file in files:
            if not file.filename.endswith('.pdf'):
                continue
                
            temp_path = f"/tmp/{file.filename}"
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            temp_paths.append(temp_path)
        
        # Ingest PDFs
        document_ids = await pdf_ingestion_service.ingest_multiple_pdfs(
            temp_paths, 
            source="uploaded-pdf"
        )
        
        # Clean up temporary files
        import os
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass

        if document_ids:
            await rag_cache.invalidate_rag_cache()

        return PDFIngestionResponse(
            message=f"Successfully ingested {len(document_ids)} PDF files",
            document_ids=document_ids,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"PDF ingestion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=response_formatter.format_error_response(
                str(e),
                "PDF_INGESTION_ERROR"
            )
        )


async def _process_agent_query(payload: LegalQueryRequest) -> LegalQueryResponse:
    """
    Process query using the LangChain legal agent.
    
    Args:
        payload: Legal query request
        
    Returns:
        LegalQueryResponse: Agent-based response
    """
    agent_result = await langchain_legal_agent.research(
        query=payload.query,
        session_id=f"session_{int(time.time())}"
    )
    
    return LegalQueryResponse(
        response=agent_result.response,
        query=payload.query,
        context=agent_result.sources,
        metadata={
            "algorithm": "langchain_agent",
            "citations": agent_result.citations,
            "domain": agent_result.domain,
            "confidence": agent_result.confidence,
            "tools_used": agent_result.tools_used
        },
        source="legal_agent",
        response_time_ms=0
    )


async def _process_rag_query(payload: LegalQueryRequest) -> LegalQueryResponse:
    """
    Process query using the RAG engine.
    
    Args:
        payload: Legal query request
        
    Returns:
        LegalQueryResponse: RAG-based response
    """
    result = await lightweight_llm_rag.query(
        query=payload.query,
        top_k=payload.top_k,
        use_agent=payload.use_agent,
        algorithm=payload.algorithm.value,
        similarity_threshold=payload.similarity_threshold,
    )
    
    # Analyze the response for legal annotations
    annotations = legal_response_analyzer.analyze_response(
        payload.query, 
        result.get("response", ""), 
        result.get("sources", [])
    )
    
    # Format the response
    formatted_response = response_formatter.clean_text_for_display(
        result.get("response", "")
    )
    
    return LegalQueryResponse(
        response=formatted_response,
        query=payload.query,
        context=result.get("sources", []),
        metadata={
            "algorithm": payload.algorithm.value,
            "citations": annotations.get("citations", []),
            "domain": annotations.get("domain", "Other")
        },
        source="rag_engine",
        response_time_ms=int(result.get("processing_time", 0) * 1000)
    )


async def _process_multi_hop_query(payload: LegalQueryRequest, complexity_analysis: Dict[str, Any]) -> MultiHopReasoningResponse:
    """
    Process query using multi-hop reasoning engine.
    
    Args:
        payload: Legal query request
        complexity_analysis: Query complexity analysis results
        
    Returns:
        MultiHopReasoningResponse: Multi-hop reasoning response
    """
    try:
        # Process with multi-hop reasoning engine
        reasoning_chain = await multi_hop_reasoning_engine.process_complex_query(
            query=payload.query,
            session_id=payload.session_id
        )
        
        
        # Convert reasoning steps to response format
        reasoning_steps = []
        for step in reasoning_chain.steps:
            reasoning_steps.append(ReasoningStepResponse(
                step_id=step.step_id,
                step_type=step.step_type.value,
                input_query=step.input_query,
                output_result=step.output_result,
                confidence_score=step.confidence_score,
                execution_time=step.execution_time,
                sources_used=step.sources_used
            ))
        
        return MultiHopReasoningResponse(
            chain_id=reasoning_chain.chain_id,
            original_query=reasoning_chain.original_query,
            complexity_level=reasoning_chain.complexity_level.value,
            final_answer=reasoning_chain.final_answer,
            reasoning_steps=reasoning_steps,
            total_execution_time=reasoning_chain.total_execution_time,
            overall_confidence=reasoning_chain.overall_confidence,
            citations=reasoning_chain.citations,
            metadata={
                **reasoning_chain.metadata,
                "complexity_analysis": complexity_analysis
            }
        )
        
    except Exception as e:
        logger.error(f"Multi-hop reasoning failed: {e}")
        # Return error response
        return MultiHopReasoningResponse(
            chain_id="error",
            original_query=payload.query,
            complexity_level="error",
            final_answer=f"Error in multi-hop reasoning: {str(e)}",
            reasoning_steps=[],
            total_execution_time=0.0,
            overall_confidence=0.0,
            citations=[],
            metadata={"error": str(e)}
        )




# ============================================================================
# ADAPTIVE RAG ENDPOINTS
# ============================================================================

@router.post("/adaptive-query")
async def process_adaptive_query(request: Request, payload: LegalQueryRequest = Body(...)):
    """
    Process query using the adaptive RAG system with intent-based processing.
    
    This endpoint automatically classifies query intent and adapts retrieval,
    generation, and response formatting accordingly.
    """
    try:
        # Check rate limiting
        client_ip = request.client.host
        if rate_limiter.is_rate_limit_exceeded(client_ip):
            raise RateLimitExceededError(
                "Rate limit exceeded for adaptive queries. Please try again later.",
                retry_after=60
            )
        
        # Process with adaptive RAG
        result = await adaptive_rag_orchestrator.process_query(
            query=payload.query,
            user_preferences={
                "response_length": payload.response_length.value if hasattr(payload, 'response_length') else "normal",
                "retrieval_count": payload.top_k
            }
        )
        
        # Format response
        return {
            "response": result.response,
            "query": payload.query,
            "context": result.sources,
            "metadata": {
                **result.metadata,
                "intent": result.intent.value,
                "confidence": result.confidence,
                "processing_time_ms": int(result.processing_time * 1000)
            },
            "source": "adaptive_rag",
            "response_time_ms": int(result.processing_time * 1000)
        }
        
    except RateLimitExceededError as e:
        raise HTTPException(
            status_code=429,
            detail=response_formatter.format_error_response(
                str(e), 
                "RATE_LIMIT_EXCEEDED",
                retry_after=e.context.get("retry_after", 60)
            )
        )
    except Exception as e:
        logger.error(f"Adaptive query processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=response_formatter.format_error_response(
                str(e), 
                "ADAPTIVE_QUERY_PROCESSING_ERROR"
            )
        )


@router.post("/feedback")
async def submit_feedback(feedback_data: Dict[str, Any] = Body(...)):
    """
    Submit user feedback for system improvement.
    
    Expected feedback_data:
    - query: str
    - response: str
    - intent_classified: str
    - feedback_type: str (rating, intent_correction, response_quality, etc.)
    - rating: int (1-5, optional)
    - correction: str (optional)
    - comments: str (optional)
    - user_id: str (optional)
    - session_id: str (optional)
    """
    try:
        import uuid
        
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            query=feedback_data.get("query", ""),
            response=feedback_data.get("response", ""),
            intent_classified=feedback_data.get("intent_classified", ""),
            feedback_type=FeedbackType(feedback_data.get("feedback_type", "rating")),
            rating=feedback_data.get("rating"),
            correction=feedback_data.get("correction"),
            comments=feedback_data.get("comments"),
            user_id=feedback_data.get("user_id"),
            session_id=feedback_data.get("session_id")
        )
        
        success = await feedback_system.submit_feedback(feedback)
        
        if success:
            return {"message": "Feedback submitted successfully", "feedback_id": feedback.feedback_id}
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to submit feedback"
            )
            
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=response_formatter.format_error_response(
                str(e), 
                "FEEDBACK_SUBMISSION_ERROR"
            )
        )


@router.get("/feedback/metrics")
async def get_feedback_metrics(days: int = 30):
    """
    Get aggregated feedback metrics and system performance analysis.
    
    Args:
        days: Number of days to analyze (default: 30)
    """
    try:
        metrics = await feedback_system.get_feedback_metrics(days=days)
        
        if metrics:
            return {
                "metrics": {
                    "intent_accuracy": metrics.intent_accuracy,
                    "average_rating": metrics.average_rating,
                    "response_quality_score": metrics.response_quality_score,
                    "citation_accuracy": metrics.citation_accuracy,
                    "total_feedback_count": metrics.total_feedback_count,
                    "improvement_suggestions": metrics.improvement_suggestions
                },
                "analysis_period_days": days
            }
        else:
            return {"message": "No feedback data available for the specified period"}
            
    except Exception as e:
        logger.error(f"Failed to get feedback metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=response_formatter.format_error_response(
                str(e), 
                "FEEDBACK_METRICS_ERROR"
            )
        )


@router.get("/feedback/analysis")
async def get_intent_performance_analysis(days: int = 30):
    """
    Get detailed performance analysis by intent type.
    
    Args:
        days: Number of days to analyze (default: 30)
    """
    try:
        analysis = await feedback_system.get_intent_performance_analysis(days=days)
        
        return {
            "intent_performance": analysis,
            "analysis_period_days": days
        }
        
    except Exception as e:
        logger.error(f"Failed to get intent performance analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=response_formatter.format_error_response(
                str(e), 
                "INTENT_ANALYSIS_ERROR"
            )
        )


@router.get("/feedback/recent")
async def get_recent_feedback(limit: int = 50):
    """
    Get recent feedback for review and analysis.
    
    Args:
        limit: Maximum number of feedback items to return (default: 50)
    """
    try:
        feedback_items = await feedback_system.get_recent_feedback(limit=limit)
        
        return {
            "recent_feedback": feedback_items,
            "count": len(feedback_items)
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail=response_formatter.format_error_response(
                str(e), 
                "RECENT_FEEDBACK_ERROR"
            )
        )
