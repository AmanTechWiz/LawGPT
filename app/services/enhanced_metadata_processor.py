"""
Enhanced metadata processor for legal documents.

Provides intelligent metadata extraction, document title detection,
citation enhancement, and structured metadata management for better
legal research and citation accuracy.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class DocumentTitleExtractor:
    """Extracts meaningful titles from legal documents."""
    
    def __init__(self):
        self.title_patterns = [
            # UN Charter patterns
            r"(?:Charter of the United Nations|UN Charter|United Nations Charter)",
            r"(?:Universal Declaration of Human Rights|UDHR)",
            r"(?:Convention on the Rights of the Child|CRC)",
            r"(?:International Covenant on Civil and Political Rights|ICCPR)",
            r"(?:International Covenant on Economic, Social and Cultural Rights|ICESCR)",
            
            # General legal document patterns
            r"(?:Constitution of [^,]+)",
            r"(?:Convention on [^,]+)",
            r"(?:Declaration of [^,]+)",
            r"(?:Protocol on [^,]+)",
            r"(?:Treaty of [^,]+)",
            r"(?:Agreement on [^,]+)",
            r"(?:Act of [^,]+)",
            r"(?:Law on [^,]+)",
            r"(?:Statute of [^,]+)",
            
            # Court documents
            r"(?:Supreme Court|High Court|Appellate Court).*?(?:Decision|Judgment|Opinion)",
            r"(?:Case No\.|Docket No\.|File No\.)\s*[\d\-]+",
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.title_patterns]
    
    def extract_title(self, content: str, filename: str) -> str:
        """
        Extract a meaningful title from document content and filename.
        
        Args:
            content: Document content
            filename: Original filename
            
        Returns:
            str: Extracted or generated title
        """
        # Try to extract from content first
        for pattern in self.compiled_patterns:
            match = pattern.search(content)
            if match:
                return match.group(0).strip()
        
        # Try to extract from filename
        if filename:
            # Remove extension and clean up
            base_name = Path(filename).stem
            # Convert underscores and hyphens to spaces
            clean_name = re.sub(r'[_-]+', ' ', base_name)
            # Capitalize words
            clean_name = ' '.join(word.capitalize() for word in clean_name.split())
            if clean_name and len(clean_name) > 3:
                return clean_name
        
        # Fallback to generic title
        return "Legal Document"


class LegalCitationExtractor:
    """Enhanced legal citation extraction and processing."""
    
    def __init__(self):
        self.article_patterns = [
            r"Article\s+(\d+(?:[a-z]|\.[0-9]+)?)",
            r"Art\.\s*(\d+(?:[a-z]|\.[0-9]+)?)",
            r"Section\s+(\d+(?:[a-z]|\.[0-9]+)?)",
            r"Sec\.\s*(\d+(?:[a-z]|\.[0-9]+)?)",
            r"Chapter\s+(\d+(?:[a-z]|\.[0-9]+)?)",
            r"Ch\.\s*(\d+(?:[a-z]|\.[0-9]+)?)",
            r"Paragraph\s+(\d+(?:[a-z]|\.[0-9]+)?)",
            r"Para\.\s*(\d+(?:[a-z]|\.[0-9]+)?)",
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.article_patterns]
    
    def extract_citations(self, content: str) -> List[str]:
        """
        Extract legal citations from document content.
        
        Args:
            content: Document content
            
        Returns:
            List[str]: List of extracted citations
        """
        citations = set()
        
        for pattern in self.compiled_patterns:
            matches = pattern.findall(content)
            citations.update(matches)
        
        return sorted(list(citations), key=lambda x: (len(x), x))


class DocumentMetadataEnhancer:
    """Enhances document metadata for better citation and retrieval."""
    
    def __init__(self):
        self.title_extractor = DocumentTitleExtractor()
        self.citation_extractor = LegalCitationExtractor()
    
    def enhance_metadata(
        self, 
        content: str, 
        filename: str, 
        chunk_index: int, 
        total_chunks: int,
        file_path: str,
        source: str = "uploaded-pdf"
    ) -> Dict[str, Any]:
        """
        Enhance document metadata with better structure and information.
        
        Args:
            content: Document content
            filename: Original filename
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks
            file_path: Full file path
            source: Source identifier
            
        Returns:
            Dict[str, Any]: Enhanced metadata
        """
        # Extract title
        title = self.title_extractor.extract_title(content, filename)
        
        # Extract citations
        citations = self.citation_extractor.extract_citations(content)
        
        # Generate document ID
        doc_id = self._generate_document_id(filename, file_path)
        
        # Determine document type
        doc_type = self._determine_document_type(filename, content)
        
        # Extract key information
        key_info = self._extract_key_information(content, title)
        
        # Create enhanced metadata structure
        enhanced_metadata = {
            # Basic document info
            "document_id": doc_id,
            "title": title,
            "filename": filename,
            "file_path": file_path,
            "source": source,
            "document_type": doc_type,
            
            # Chunk information
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "chunk_size": len(content),
            "chunk_ratio": f"{chunk_index + 1}/{total_chunks}",
            
            # Legal information
            "legal_citations": citations,
            "citation_count": len(citations),
            "key_articles": citations[:5],  # Top 5 most relevant articles
            
            # Content analysis
            "word_count": len(content.split()),
            "character_count": len(content),
            "has_legal_citations": len(citations) > 0,
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            
            # Key information
            "key_information": key_info,
            
            # Processing metadata
            "processed_at": datetime.utcnow().isoformat(),
            "metadata_version": "2.0",
            "enhanced": True
        }
        
        return enhanced_metadata
    
    def _generate_document_id(self, filename: str, file_path: str) -> str:
        """Generate a unique document ID."""
        # Use filename and path to create a consistent ID
        base_string = f"{filename}_{file_path}"
        return hashlib.md5(base_string.encode()).hexdigest()[:12]
    
    def _determine_document_type(self, filename: str, content: str) -> str:
        """Determine the type of legal document."""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        if "charter" in filename_lower or "charter" in content_lower:
            return "charter"
        elif "declaration" in filename_lower or "declaration" in content_lower:
            return "declaration"
        elif "convention" in filename_lower or "convention" in content_lower:
            return "convention"
        elif "constitution" in filename_lower or "constitution" in content_lower:
            return "constitution"
        elif "treaty" in filename_lower or "treaty" in content_lower:
            return "treaty"
        elif "protocol" in filename_lower or "protocol" in content_lower:
            return "protocol"
        elif "act" in filename_lower or "act" in content_lower:
            return "act"
        else:
            return "legal_document"
    
    def _extract_key_information(self, content: str, title: str) -> Dict[str, Any]:
        """Extract key information from the document."""
        key_info = {
            "document_title": title,
            "is_international": any(keyword in content.lower() for keyword in [
                "united nations", "international", "global", "worldwide"
            ]),
            "is_human_rights": any(keyword in content.lower() for keyword in [
                "human rights", "fundamental rights", "civil rights", "political rights"
            ]),
            "is_peace_security": any(keyword in content.lower() for keyword in [
                "peace", "security", "conflict", "dispute", "aggression"
            ]),
            "has_articles": "article" in content.lower(),
            "has_chapters": "chapter" in content.lower(),
            "has_sections": "section" in content.lower(),
        }
        
        return key_info


class MetadataProcessor:
    """Main metadata processor for document ingestion."""
    
    def __init__(self):
        self.enhancer = DocumentMetadataEnhancer()
    
    def process_document_metadata(
        self,
        content: str,
        filename: str,
        chunk_index: int,
        total_chunks: int,
        file_path: str,
        source: str = "uploaded-pdf"
    ) -> Dict[str, Any]:
        """
        Process and enhance document metadata.
        
        Args:
            content: Document content
            filename: Original filename
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks
            file_path: Full file path
            source: Source identifier
            
        Returns:
            Dict[str, Any]: Processed metadata
        """
        try:
            enhanced_metadata = self.enhancer.enhance_metadata(
                content, filename, chunk_index, total_chunks, file_path, source
            )
            
            logger.info(f"Enhanced metadata for {filename} chunk {chunk_index + 1}/{total_chunks}")
            return enhanced_metadata
            
        except Exception as e:
            logger.error(f"Error processing metadata for {filename}: {e}")
            # Return basic metadata as fallback
            return {
                "title": filename,
                "filename": filename,
                "file_path": file_path,
                "source": source,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "document_type": "legal_document",
                "legal_citations": [],
                "enhanced": False,
                "error": str(e)
            }


# Global instance
metadata_processor = MetadataProcessor()
