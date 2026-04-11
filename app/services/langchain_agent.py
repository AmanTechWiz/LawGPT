
"""
Intelligent legal research agent using LangChain tools and OpenAI models.

Provides automated legal research with citation extraction, domain classification,
and enhanced response generation with fallback mechanisms for reliability.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Type
from datetime import datetime

from pydantic import BaseModel, Field

from ..core.config import settings
from ..core.database import db_manager
from .legal_tools import extract_legal_citations, classify_legal_text
from .legal_classifier import legal_classifier
from .lightweight_llm_rag import lightweight_llm_rag

logger = logging.getLogger(__name__)

try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain.tools import BaseTool
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    from langchain_core.messages import BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available, using fallback implementation")
    LANGCHAIN_AVAILABLE = False

class LegalResearchInput(BaseModel):
    """Input for legal research"""
    query: str = Field(..., description="The legal research query")
    context: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None

class LegalResearchOutput(BaseModel):
    """Output for legal research"""
    response: str = Field(..., description="The legal research response")
    citations: List[str] = Field(default_factory=list, description="Extracted legal citations")
    domain: str = Field(..., description="Legal domain classification")
    confidence: float = Field(..., description="Confidence score")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents used")
    tools_used: List[str] = Field(default_factory=list, description="Tools used by the agent")

if LANGCHAIN_AVAILABLE:
    class LegalCitationTool(BaseTool):
        """Tool for extracting legal citations from text"""
        
        name: str = "extract_legal_citations"
        description: str = "Extract legal citations from text. Use this to identify case law, statutes, and legal references."
        
        def _run(self, text: str) -> str:
            """Extract legal citations from text"""
            try:
                citations = extract_legal_citations(text)
                if citations:
                    return f"Found legal citations: {', '.join(citations)}"
                else:
                    return "No legal citations found in the text."
            except Exception as e:
                logger.error(f"Citation extraction failed: {e}")
                return f"Error extracting citations: {str(e)}"
        
        async def _arun(self, text: str) -> str:
            """Async version of citation extraction"""
            return self._run(text)

    class LegalClassificationTool(BaseTool):
        """Tool for classifying legal text into domains"""
        
        name: str = "classify_legal_domain"
        description: str = "Classify legal text into domains: Constitutional Law, Criminal Law, Contract Law, or Other."
        
        def _run(self, text: str) -> str:
            """Classify legal text"""
            try:
                result = legal_classifier.classify(text)
                return f"Legal domain: {result['category']} (confidence: {result['confidence']:.2f})"
            except Exception as e:
                logger.error(f"Legal classification failed: {e}")
                return f"Error in classification: {str(e)}"
        
        async def _arun(self, text: str) -> str:
            """Async version of classification"""
            return self._run(text)

    class LegalResearchTool(BaseTool):
        """Tool for performing legal research using RAG"""
        
        name: str = "legal_research"
        description: str = "Search legal documents and retrieve relevant information for legal research queries."
        
        def _run(self, query: str) -> str:
            """Perform legal research (synchronous wrapper)"""
            try:
                # For synchronous calls, we'll use a simple approach
                return f"Legal research query: {query}. Please use the async version for full functionality."
            except Exception as e:
                logger.error(f"Legal research failed: {e}")
                return f"Error in legal research: {str(e)}"
        
        async def _arun(self, query: str) -> str:
            """Async version of legal research"""
            try:
                result = await lightweight_llm_rag.query(query, top_k=5, similarity_threshold=0.3)
                return f"Legal research results: {result.get('response', 'No results found')}"
            except Exception as e:
                logger.error(f"Legal research failed: {e}")
                return f"Error in legal research: {str(e)}"

class LangChainLegalAgent:
    """LangChain-based legal research agent with tool integration and fallback support."""
    
    def __init__(self):
        self.llm = None
        self.tools = []
        self.agent_executor = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the LangChain legal research agent"""
        if self.initialized:
            return
        
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, using fallback implementation")
            self.initialized = True
            return
        
        try:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is required but not provided")
            
            self.llm = ChatOpenAI(
                model=settings.openai_model,
                temperature=0.1,
                api_key=settings.openai_api_key
            )
            
            self.tools = [
                LegalCitationTool(),
                LegalClassificationTool(),
                LegalResearchTool()
            ]
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a legal research assistant focused on delivering direct, actionable information. Your role is to:

                    1. Provide immediate, specific answers to legal questions
                    2. Use tools to find relevant legal provisions quickly
                    3. Extract and cite specific legal references
                    4. Focus on practical, actionable information

                    Response Guidelines:
                    - Lead with the direct answer to the question
                    - Provide specific legal provisions with exact citations
                    - Use bullet points for multiple related points
                    - Focus on what the user can do or what applies
                    - Avoid lengthy explanations unless specifically requested

                    Available tools:
                    - legal_research: Search legal documents
                    - extract_legal_citations: Extract citations from text
                    - classify_legal_domain: Classify legal text into domains"""),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            agent = create_openai_tools_agent(self.llm, self.tools, prompt)
            self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
            
            self.initialized = True
            logger.info("LangChain Legal Research Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain Legal Research Agent: {e}")
            raise
    
    async def research(self, query: str, session_id: Optional[str] = None) -> LegalResearchOutput:
        """Perform legal research using the LangChain agent"""
        if not self.initialized:
            await self.initialize()
        
        if not LANGCHAIN_AVAILABLE or not self.agent_executor:
            return await self._fallback_research(query, session_id)
        
        try:
            result = self.agent_executor.invoke({"input": query})
            agent_response = result["output"]
            
            rag_result = await lightweight_llm_rag.query(query, top_k=5, similarity_threshold=0.3)
            sources = rag_result.get("sources", [])
            
            citations = extract_legal_citations(agent_response)
            
            domain_result = legal_classifier.classify(query)
            domain = domain_result["category"]
            confidence = domain_result["confidence"]
            
            tools_used = ["legal_research", "extract_legal_citations", "classify_legal_domain"]
            
            output = LegalResearchOutput(
                response=agent_response,
                citations=citations,
                domain=domain,
                confidence=confidence,
                sources=sources,
                tools_used=tools_used
            )
            
            return output
            
        except Exception as e:
            logger.error(f"LangChain agent research failed: {e}")
            return await self._fallback_research(query, session_id)
    
    async def _fallback_research(self, query: str, session_id: Optional[str] = None) -> LegalResearchOutput:
        """Fallback research implementation when LangChain is not available"""
        try:
            rag_result = await lightweight_llm_rag.query(query, top_k=5, similarity_threshold=0.3)
            response = rag_result.get("response", "")
            sources = rag_result.get("sources", [])
            
            citations = extract_legal_citations(response)
            
            domain_result = legal_classifier.classify(query)
            domain = domain_result["category"]
            confidence = domain_result["confidence"]
            
            enhanced_response = await self._enhance_response(query, response, domain, citations)
            
            output = LegalResearchOutput(
                response=enhanced_response,
                citations=citations,
                domain=domain,
                confidence=confidence,
                sources=sources,
                tools_used=["fallback_research"]
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Fallback research failed: {e}")
            return LegalResearchOutput(
                response=f"Error in legal research: {str(e)}",
                citations=[],
                domain="Other",
                confidence=0.0,
                sources=[],
                tools_used=[]
            )
    
    async def _enhance_response(self, query: str, response: str, domain: str, citations: List[str]) -> str:
        """Enhance the response with legal analysis"""
        try:
            domain_context = {
                "Constitutional Law": "This response relates to constitutional law and fundamental rights.",
                "Criminal Law": "This response relates to criminal law and legal procedures.",
                "Contract Law": "This response relates to contract law and agreements.",
                "Other": "This response relates to general legal matters."
            }
            
            enhanced = f"{domain_context.get(domain, '')}\n\n{response}"
            
            if citations:
                enhanced += f"\n\nLegal Citations Found: {', '.join(citations)}"
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Response enhancement failed: {e}")
            return response

langchain_legal_agent = LangChainLegalAgent()
