"""
Reusable utility functions and classes for common operations.

Provides text processing, performance timing, validation utilities, and data
transformation functions to eliminate code duplication across the application.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union
from functools import wraps
import asyncio
import time

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Text processing utilities for legal documents.
    
    Provides methods for cleaning, formatting, and validating text content.
    """
    
    @staticmethod
    def clean_whitespace(text: str) -> str:
        """
        Clean and normalize whitespace in text.
        
        Args:
            text: Text to clean
            
        Returns:
            str: Text with normalized whitespace
        """
        if not text:
            return text
        
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
        cleaned_text = re.sub(r'\n[ \t]+', '\n', cleaned_text)
        cleaned_text = re.sub(r'[ \t]+\n', '\n', cleaned_text)
        
        return cleaned_text.strip()
    
    @staticmethod
    def fix_sentence_spacing(text: str) -> str:
        """
        Fix spacing between sentences and words.
        
        Args:
            text: Text to fix
            
        Returns:
            str: Text with proper sentence spacing
        """
        if not text:
            return text
        
        fixed_text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        fixed_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', fixed_text)
        fixed_text = re.sub(r'([a-z])(\d)', r'\1 \2', fixed_text)
        fixed_text = re.sub(r'(\d)([A-Z])', r'\1 \2', fixed_text)
        
        return fixed_text
    
    @staticmethod
    def format_legal_references(text: str) -> str:
        """
        Format legal references for consistency.
        
        Args:
            text: Text containing legal references
            
        Returns:
            str: Text with formatted legal references
        """
        if not text:
            return text
        
        formatted_text = re.sub(r'Article\s+(\d+)', r'Article \1', text)
        formatted_text = re.sub(r'Section\s+(\d+)', r'Section \1', formatted_text)
        formatted_text = re.sub(r'Chapter\s+(\d+)', r'Chapter \1', formatted_text)
        
        return formatted_text
    
    @staticmethod
    def normalize_punctuation(text: str) -> str:
        """
        Normalize punctuation marks.
        
        Args:
            text: Text to normalize
            
        Returns:
            str: Text with normalized punctuation
        """
        if not text:
            return text
        
        normalized_text = re.sub(r'[.]{3,}', '...', text)
        normalized_text = re.sub(r'[-]{3,}', '---', normalized_text)
        
        return normalized_text
    
    @staticmethod
    def remove_hyphenated_line_breaks(text: str) -> str:
        """
        Remove hyphenated line breaks that split words.
        
        Args:
            text: Text to process
            
        Returns:
            str: Text with hyphenated line breaks removed
        """
        if not text:
            return text
        
        return re.sub(r'-\s*\n\s*', '', text)
    
    @staticmethod
    def remove_page_numbers(text: str) -> str:
        """
        Remove page numbers and headers from text.
        
        Args:
            text: Text to clean
            
        Returns:
            str: Text with page numbers removed
        """
        if not text:
            return text
        
        cleaned_text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'^\d+\s*$', '', cleaned_text, flags=re.MULTILINE)
        
        return cleaned_text
    
    @classmethod
    def clean_text_comprehensive(cls, text: str) -> str:
        """
        Apply comprehensive text cleaning using all available methods.
        
        Args:
            text: Text to clean
            
        Returns:
            str: Comprehensively cleaned text
        """
        if not text:
            return text
        
        # Apply all cleaning methods in order
        cleaned_text = text
        cleaned_text = cls.remove_hyphenated_line_breaks(cleaned_text)
        cleaned_text = cls.remove_page_numbers(cleaned_text)
        cleaned_text = cls.clean_whitespace(cleaned_text)
        cleaned_text = cls.fix_sentence_spacing(cleaned_text)
        cleaned_text = cls.format_legal_references(cleaned_text)
        cleaned_text = cls.normalize_punctuation(cleaned_text)
        
        return cleaned_text


class PerformanceTimer:
    """
    Performance timing utilities.
    
    Provides context managers and decorators for measuring operation duration.
    """
    
    def __init__(self, operation_name: str, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize the performance timer.
        
        Args:
            operation_name: Name of the operation being timed
            logger_instance: Logger instance to use (defaults to module logger)
        """
        self.operation_name = operation_name
        self.logger = logger_instance or logger
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        """Start timing the operation."""
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log the duration."""
        self.end_time = time.time()
        duration = self.get_duration()
        self.logger.info(f"{self.operation_name} completed in {duration:.3f} seconds")
    
    def get_duration(self) -> float:
        """
        Get the duration of the operation.
        
        Returns:
            float: Duration in seconds
        """
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time or time.time()
        return end_time - self.start_time
    
    def get_duration_ms(self) -> int:
        """
        Get the duration of the operation in milliseconds.
        
        Returns:
            int: Duration in milliseconds
        """
        return int(self.get_duration() * 1000)


def time_operation(operation_name: str, logger_instance: Optional[logging.Logger] = None):
    """
    Decorator to time function execution.
    
    Args:
        operation_name: Name of the operation
        logger_instance: Logger instance to use
        
    Returns:
        Decorated function with timing
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with PerformanceTimer(operation_name, logger_instance):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with PerformanceTimer(operation_name, logger_instance):
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class ValidationUtils:
    """
    Data validation utilities.
    
    Provides validation methods for ensuring data integrity.
    """
    
    @staticmethod
    def validate_text_length(text: str, min_length: int = 1, max_length: int = 10000) -> bool:
        """
        Validate text length is within specified bounds.
        
        Args:
            text: Text to validate
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            
        Returns:
            bool: True if text length is valid, False otherwise
        """
        if not text:
            return min_length == 0
        
        text_length = len(text.strip())
        return min_length <= text_length <= max_length
    
    @staticmethod
    def validate_similarity_threshold(threshold: float) -> bool:
        """
        Validate similarity threshold is within valid range.
        
        Args:
            threshold: Threshold value to validate
            
        Returns:
            bool: True if threshold is valid, False otherwise
        """
        return 0.0 <= threshold <= 1.0
    
    @staticmethod
    def validate_top_k_value(top_k: int, min_value: int = 1, max_value: int = 50) -> bool:
        """
        Validate top_k parameter is within valid range.
        
        Args:
            top_k: Top-k value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            bool: True if top_k is valid, False otherwise
        """
        return min_value <= top_k <= max_value


class DataTransformer:
    """
    Data transformation utilities.
    
    Provides methods for converting between different data formats and structures.
    """
    
    @staticmethod
    def convert_to_dictionary_list(data: List[Any], key_field: str = "id") -> Dict[str, Any]:
        """
        Convert a list of objects to a dictionary keyed by specified field.
        
        Args:
            data: List of objects to convert
            key_field: Field to use as dictionary key
            
        Returns:
            Dict[str, Any]: Dictionary with objects keyed by specified field
        """
        result = {}
        for item in data:
            if hasattr(item, key_field):
                key = getattr(item, key_field)
                result[key] = item
            elif isinstance(item, dict) and key_field in item:
                result[item[key_field]] = item
        
        return result
    
    @staticmethod
    def extract_field_values(data: List[Dict[str, Any]], field_name: str) -> List[Any]:
        """
        Extract values of a specific field from a list of dictionaries.
        
        Args:
            data: List of dictionaries
            field_name: Name of field to extract
            
        Returns:
            List[Any]: List of field values
        """
        return [item.get(field_name) for item in data if field_name in item]
    
    @staticmethod
    def group_by_field(data: List[Dict[str, Any]], field_name: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group a list of dictionaries by a specific field.
        
        Args:
            data: List of dictionaries to group
            field_name: Field to group by
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Grouped data
        """
        groups = {}
        for item in data:
            key = item.get(field_name, "unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        
        return groups


# Global utility instances
text_processor = TextProcessor()
validation_utils = ValidationUtils()
data_transformer = DataTransformer()
