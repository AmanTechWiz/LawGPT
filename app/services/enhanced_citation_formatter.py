"""
Enhanced citation formatter for legal documents.

Provides intelligent citation formatting using enhanced metadata
for better legal research and document referencing.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedCitationFormatter:
    """Enhanced citation formatter using improved metadata structure."""
    
    def __init__(self):
        self.citation_templates = {
            "charter": "{title}, Article {article}",
            "declaration": "{title}, Article {article}",
            "convention": "{title}, Article {article}",
            "constitution": "{title}, Article {article}",
            "treaty": "{title}, Article {article}",
            "protocol": "{title}, Article {article}",
            "act": "{title}, Section {article}",
            "legal_document": "{title}, Article {article}"
        }
    
    def format_citation(self, document: Dict[str, Any], article: str) -> str:
        """
        Format a citation using enhanced metadata.
        
        Args:
            document: Document metadata
            article: Article or section number
            
        Returns:
            str: Formatted citation
        """
        try:
            metadata = document.get('metadata', {})
            title = metadata.get('title', 'Unknown Document')
            doc_type = metadata.get('document_type', 'legal_document')
            
            # Get citation template
            template = self.citation_templates.get(doc_type, self.citation_templates['legal_document'])
            
            # Format the citation
            citation = template.format(
                title=title,
                article=article
            )
            
            return citation
            
        except Exception as e:
            logger.error(f"Error formatting citation: {e}")
            return f"Article {article}"
    
    def format_document_citation(self, document: Dict[str, Any]) -> str:
        """
        Format a general document citation.
        
        Args:
            document: Document metadata
            
        Returns:
            str: Formatted document citation
        """
        try:
            metadata = document.get('metadata', {})
            title = metadata.get('title', 'Unknown Document')
            filename = metadata.get('filename', '')
            
            # Create a clean document citation
            if filename and filename != title:
                return f"{title} ({filename})"
            else:
                return title
                
        except Exception as e:
            logger.error(f"Error formatting document citation: {e}")
            return "Unknown Document"
    
    def extract_enhanced_citations(self, sources: List[Dict[str, Any]]) -> List[str]:
        """
        Extract and format citations from document sources.
        
        Args:
            sources: List of document sources with metadata
            
        Returns:
            List[str]: List of formatted citations
        """
        citations = []
        
        for source in sources:
            try:
                metadata = source.get('metadata', {})
                legal_citations = metadata.get('legal_citations', [])
                
                # Format each citation
                for article in legal_citations:
                    citation = self.format_citation(source, article)
                    if citation not in citations:
                        citations.append(citation)
                
            except Exception as e:
                logger.error(f"Error extracting citations from source: {e}")
                continue
        
        return citations
    
    def format_context_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format context sources with enhanced metadata display.
        
        Args:
            sources: List of document sources
            
        Returns:
            List[Dict[str, Any]]: Formatted sources with enhanced metadata
        """
        formatted_sources = []
        
        for source in sources:
            try:
                metadata = source.get('metadata', {})
                
                # Create enhanced source display
                enhanced_source = {
                    "content": source.get('content', ''),
                    "metadata": {
                        "id": metadata.get('id'),
                        "title": metadata.get('title', 'Untitled'),
                        "filename": metadata.get('filename', ''),
                        "document_type": metadata.get('document_type', 'legal_document'),
                        "chunk_info": f"Part {metadata.get('chunk_index', 0) + 1} of {metadata.get('total_chunks', 1)}",
                        "legal_citations": metadata.get('legal_citations', []),
                        "citation_count": metadata.get('citation_count', 0),
                        "key_articles": metadata.get('key_articles', []),
                        "document_id": metadata.get('document_id', ''),
                        "similarity_score": source.get('score', 0.0),
                        "enhanced": metadata.get('enhanced', False)
                    }
                }
                
                formatted_sources.append(enhanced_source)
                
            except Exception as e:
                logger.error(f"Error formatting source: {e}")
                # Fallback to original source
                formatted_sources.append(source)
        
        return formatted_sources
    
    def generate_citation_summary(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of citations and document information.
        
        Args:
            sources: List of document sources
            
        Returns:
            Dict[str, Any]: Citation summary
        """
        try:
            total_sources = len(sources)
            total_citations = 0
            document_types = set()
            document_titles = set()
            all_citations = set()
            
            for source in sources:
                metadata = source.get('metadata', {})
                
                # Count citations
                legal_citations = metadata.get('legal_citations', [])
                total_citations += len(legal_citations)
                all_citations.update(legal_citations)
                
                # Collect document info
                doc_type = metadata.get('document_type', 'legal_document')
                title = metadata.get('title', 'Untitled')
                document_types.add(doc_type)
                document_titles.add(title)
            
            return {
                "total_sources": total_sources,
                "total_citations": total_citations,
                "unique_citations": len(all_citations),
                "document_types": list(document_types),
                "document_titles": list(document_titles),
                "citation_density": total_citations / total_sources if total_sources > 0 else 0,
                "enhanced_metadata_available": any(
                    source.get('metadata', {}).get('enhanced', False) 
                    for source in sources
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating citation summary: {e}")
            return {
                "total_sources": len(sources),
                "total_citations": 0,
                "unique_citations": 0,
                "document_types": [],
                "document_titles": [],
                "citation_density": 0,
                "enhanced_metadata_available": False
            }


# Global instance
enhanced_citation_formatter = EnhancedCitationFormatter()
