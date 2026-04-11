"""
Adaptive prompt templates for different query intents.

Provides specialized prompt templates tailored for each query type with
specific instructions for response format, length, and style.
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum

from .query_intent_classifier import QueryIntent

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Template for generating prompts based on query intent."""
    
    def __init__(self, intent: QueryIntent, system_prompt: str, user_template: str, 
                 response_guidelines: str, max_tokens: int, temperature: float):
        self.intent = intent
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.response_guidelines = response_guidelines
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def generate_prompt(self, query: str, context: str, conflict_info: str = "") -> Dict[str, str]:
        """Generate the complete prompt for the given query and context."""
        # Format the user template with query, context, and guidelines
        user_content = self.user_template.format(
            query=query,
            context=context,
            conflict_info=conflict_info,
            guidelines=self.response_guidelines
        )
        
        return {
            "system": self.system_prompt,
            "user": user_content
        }


class PromptTemplateManager:
    """Manages prompt templates for different query intents."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[QueryIntent, PromptTemplate]:
        """Initialize all prompt templates for different query intents."""
        templates = {}
        
        # Definition Template - optimized for concise, precise definitions
        templates[QueryIntent.DEFINITION] = PromptTemplate(
            intent=QueryIntent.DEFINITION,
            system_prompt="""You are a legal research assistant specializing in providing precise, concise definitions. Your role is to deliver clear, accurate definitions of legal terms and concepts with exact citations.

CRITICAL ANTI-HALLUCINATION RULES:
1. ONLY use information that is explicitly present in the provided context
2. NEVER invent, assume, or generate legal citations that are not in the context
3. If the requested information is not in the context, respond: "This information is not available in the provided legal documents"
4. NEVER provide specific article numbers, section references, or citations unless they appear exactly in the context
5. If context is insufficient, clearly state what is missing rather than making assumptions""",
            user_template="""{guidelines}

Context from Legal Documents:
{context}{conflict_info}

Legal Question: {query}""",
            response_guidelines="""RESPONSE REQUIREMENTS:
- Provide a clear, concise definition (2-3 sentences maximum)
- Include ONLY the exact legal source and article/section reference that appears in the context
- Use precise legal language
- Avoid lengthy explanations or background information
- If information is not in context, state: "This information is not available in the provided legal documents"
- Format: Definition + Citation (only if citation exists in context)""",
            max_tokens=200,
            temperature=0.1
        )
        
        # List Template - optimized for structured enumerations
        templates[QueryIntent.LIST] = PromptTemplate(
            intent=QueryIntent.LIST,
            system_prompt="""You are a legal research assistant specializing in providing structured lists and enumerations. Your role is to deliver clear, organized lists with proper legal citations.

            CRITICAL ANTI-HALLUCINATION RULES:
            1. ONLY list items that are explicitly present in the provided context
            2. NEVER invent, assume, or generate legal citations that are not in the context
            3. If the requested information is not in the context, respond: "This information is not available in the provided legal documents"
            4. NEVER provide specific article numbers, section references, or citations unless they appear exactly in the context
            5. If context is insufficient, clearly state what is missing rather than making assumptions
            6. If you cannot find the specific document requested (e.g., "UN Convention Against Corruption"), state that the document is not available""",
            user_template="""{guidelines}

Context from Legal Documents:
{context}{conflict_info}

Legal Question: {query}""",
            response_guidelines="""RESPONSE REQUIREMENTS:
- Provide a clear, numbered or bulleted list ONLY if items are found in the context
- Each item should be concise but complete
- Include ONLY specific legal citations that appear exactly in the context
- Use consistent formatting
- Maximum 8-10 items unless specifically requested otherwise
- If no relevant items found in context, respond: "This information is not available in the provided legal documents"
- Format: Numbered list with citations (only if citations exist in context)""",
            max_tokens=300,
            temperature=0.2
        )
        
        # Explanation Template - optimized for detailed, comprehensive explanations
        templates[QueryIntent.EXPLANATION] = PromptTemplate(
            intent=QueryIntent.EXPLANATION,
            system_prompt="""You are a legal research assistant specializing in providing detailed explanations. Your role is to deliver comprehensive, well-structured explanations with thorough legal analysis.""",
            user_template="""{guidelines}

Context from Legal Documents:
{context}{conflict_info}

Legal Question: {query}""",
            response_guidelines="""RESPONSE REQUIREMENTS:
- Provide a detailed, comprehensive explanation
- Include background context and legal framework
- Explain key concepts and their relationships
- Use clear headings and structure
- Include extensive legal citations
- Provide practical implications where relevant
- Format: Structured explanation with clear sections""",
            max_tokens=600,
            temperature=0.3
        )
        
        # Comparative Template - optimized for structured comparisons
        templates[QueryIntent.COMPARATIVE] = PromptTemplate(
            intent=QueryIntent.COMPARATIVE,
            system_prompt="""You are a legal research assistant specializing in comparative analysis. Your role is to deliver structured comparisons with clear distinctions and similarities.""",
            user_template="""{guidelines}

Context from Legal Documents:
{context}{conflict_info}

Legal Question: {query}""",
            response_guidelines="""RESPONSE REQUIREMENTS:
- Provide a structured comparison with clear sections
- Identify key similarities and differences
- Use side-by-side or tabular format when appropriate
- Include specific legal citations for each comparison point
- Highlight practical implications of differences
- Conclude with summary of key distinctions
- Format: Comparative analysis with clear structure""",
            max_tokens=800,
            temperature=0.2
        )
        
        # Procedural Template - optimized for step-by-step procedures
        templates[QueryIntent.PROCEDURAL] = PromptTemplate(
            intent=QueryIntent.PROCEDURAL,
            system_prompt="""You are a legal research assistant specializing in procedural guidance. Your role is to deliver clear, step-by-step procedures with specific legal requirements.""",
            user_template="""{guidelines}

Context from Legal Documents:
{context}{conflict_info}

Legal Question: {query}""",
            response_guidelines="""RESPONSE REQUIREMENTS:
- Provide clear, sequential steps
- Include specific legal requirements for each step
- Mention deadlines, forms, or documentation needed
- Highlight potential pitfalls or common mistakes
- Include relevant legal citations
- Use numbered steps with clear action items
- Format: Step-by-step procedure with legal requirements""",
            max_tokens=500,
            temperature=0.1
        )
        
        # Analytical Template - optimized for comprehensive analysis
        templates[QueryIntent.ANALYTICAL] = PromptTemplate(
            intent=QueryIntent.ANALYTICAL,
            system_prompt="""You are a legal research assistant specializing in analytical assessment. Your role is to deliver comprehensive analysis with critical evaluation and insights.""",
            user_template="""{guidelines}

Context from Legal Documents:
{context}{conflict_info}

Legal Question: {query}""",
            response_guidelines="""RESPONSE REQUIREMENTS:
- Provide comprehensive analysis with multiple perspectives
- Include critical evaluation and assessment
- Identify strengths, weaknesses, and implications
- Consider practical applications and real-world scenarios
- Include extensive legal citations and references
- Provide well-reasoned conclusions
- Use structured analysis format with clear sections
- Format: Comprehensive analysis with critical insights""",
            max_tokens=1000,
            temperature=0.3
        )
        
        # Interpretative Template - optimized for legal interpretation
        templates[QueryIntent.INTERPRETATIVE] = PromptTemplate(
            intent=QueryIntent.INTERPRETATIVE,
            system_prompt="""You are a legal research assistant specializing in legal interpretation. Your role is to deliver nuanced interpretations with careful analysis of legal language and implications.""",
            user_template="""{guidelines}

Context from Legal Documents:
{context}{conflict_info}

Legal Question: {query}""",
            response_guidelines="""RESPONSE REQUIREMENTS:
- Provide careful interpretation of legal language
- Analyze different possible meanings and interpretations
- Consider context and legal principles
- Discuss implications and practical applications
- Include relevant legal citations and precedents
- Address potential ambiguities or uncertainties
- Format: Interpretative analysis with multiple perspectives""",
            max_tokens=700,
            temperature=0.2
        )
        
        # Factual Template (default) - optimized for direct factual answers
        templates[QueryIntent.FACTUAL] = PromptTemplate(
            intent=QueryIntent.FACTUAL,
            system_prompt="""You are a legal research assistant specializing in factual information. Your role is to deliver direct, accurate answers with proper legal citations.""",
            user_template="""{guidelines}

Context from Legal Documents:
{context}{conflict_info}

Legal Question: {query}""",
            response_guidelines="""RESPONSE REQUIREMENTS:
- Provide direct, factual answers
- Include specific legal citations
- Be concise but complete
- Focus on the specific question asked
- Avoid unnecessary elaboration
- Format: Direct answer with citation""",
            max_tokens=250,
            temperature=0.1
        )
        
        return templates
    
    def get_template(self, intent: QueryIntent) -> PromptTemplate:
        """Get the prompt template for a specific intent, fallback to factual if not found."""
        return self.templates.get(intent, self.templates[QueryIntent.FACTUAL])
    
    def generate_prompt(self, intent: QueryIntent, query: str, context: str, 
                       conflict_info: str = "") -> Dict[str, str]:
        """Generate a prompt for the given intent, query, and context."""
        # Get the appropriate template and generate the complete prompt
        template = self.get_template(intent)
        return template.generate_prompt(query, context, conflict_info)
    
    def get_generation_parameters(self, intent: QueryIntent) -> Dict[str, Any]:
        """Get generation parameters optimized for a specific intent."""
        template = self.get_template(intent)
        return {
            "max_tokens": template.max_tokens,
            "temperature": template.temperature
        }
    
    def get_supported_intents(self) -> list:
        """Get list of all supported query intents."""
        return list(self.templates.keys())


# Global instance
prompt_template_manager = PromptTemplateManager()
