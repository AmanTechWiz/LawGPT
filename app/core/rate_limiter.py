"""
Rate limiting functionality to prevent abuse and ensure fair API usage.

Implements sliding window rate limiting with configurable request limits,
automatic cleanup of expired requests, and comprehensive rate limit information.
"""

import time
import logging
from typing import Dict, List
from collections import defaultdict
from .exceptions import RateLimitExceededError

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter implementation using sliding window approach.
    
    Tracks requests per client IP address and enforces rate limits
    to prevent abuse and ensure fair usage of API resources.
    """
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed per window
            window_seconds: Time window in seconds for rate limiting
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_timestamps: Dict[str, List[float]] = defaultdict(list)
        self._lock = None  # Will be set to asyncio.Lock when needed
    
    def _cleanup_old_requests(self, client_ip: str, current_time: float) -> None:
        """
        Remove old requests outside the current window.
        
        Args:
            client_ip: Client IP address
            current_time: Current timestamp
        """
        cutoff_time = current_time - self.window_seconds
        self.request_timestamps[client_ip] = [
            timestamp for timestamp in self.request_timestamps[client_ip]
            if timestamp > cutoff_time
        ]
    
    def is_rate_limit_exceeded(self, client_ip: str) -> bool:
        """
        Check if the client has exceeded the rate limit.
        
        Args:
            client_ip: Client IP address to check
            
        Returns:
            bool: True if rate limit is exceeded, False otherwise
        """
        current_time = time.time()
        self._cleanup_old_requests(client_ip, current_time)
        
        request_count = len(self.request_timestamps[client_ip])
        return request_count >= self.max_requests
    
    def record_request(self, client_ip: str) -> None:
        """
        Record a new request for the client.
        
        Args:
            client_ip: Client IP address making the request
        """
        current_time = time.time()
        self.request_timestamps[client_ip].append(current_time)
        logger.debug(f"Recorded request for client {client_ip}")
    
    def get_remaining_requests(self, client_ip: str) -> int:
        """
        Get the number of remaining requests for the client.
        
        Args:
            client_ip: Client IP address to check
            
        Returns:
            int: Number of remaining requests in the current window
        """
        current_time = time.time()
        self._cleanup_old_requests(client_ip, current_time)
        
        request_count = len(self.request_timestamps[client_ip])
        return max(0, self.max_requests - request_count)
    
    def get_reset_time(self, client_ip: str) -> float:
        """
        Get the time when the rate limit will reset for the client.
        
        Args:
            client_ip: Client IP address to check
            
        Returns:
            float: Timestamp when the rate limit resets
        """
        if not self.request_timestamps[client_ip]:
            return time.time()
        
        oldest_request = min(self.request_timestamps[client_ip])
        return oldest_request + self.window_seconds
    
    def check_and_record_request(self, client_ip: str) -> None:
        """
        Check rate limit and record the request if allowed.
        
        Args:
            client_ip: Client IP address making the request
            
        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        if self.is_rate_limit_exceeded(client_ip):
            reset_time = self.get_reset_time(client_ip)
            remaining_time = reset_time - time.time()
            
            raise RateLimitExceededError(
                message=f"Rate limit exceeded. Try again in {remaining_time:.0f} seconds.",
                client_ip=client_ip,
                request_count=len(self.request_timestamps[client_ip]),
                limit=self.max_requests
            )
        
        self.record_request(client_ip)
    
    def get_rate_limit_info(self, client_ip: str) -> Dict[str, any]:
        """
        Get comprehensive rate limit information for the client.
        
        Args:
            client_ip: Client IP address to check
            
        Returns:
            Dict[str, any]: Rate limit information including remaining requests and reset time
        """
        current_time = time.time()
        self._cleanup_old_requests(client_ip, current_time)
        
        request_count = len(self.request_timestamps[client_ip])
        remaining_requests = self.get_remaining_requests(client_ip)
        reset_time = self.get_reset_time(client_ip)
        
        return {
            "client_ip": client_ip,
            "request_count": request_count,
            "remaining_requests": remaining_requests,
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "reset_time": reset_time,
            "is_limited": request_count >= self.max_requests
        }
    
    def clear_client_requests(self, client_ip: str) -> None:
        """
        Clear all requests for a specific client.
        
        Args:
            client_ip: Client IP address to clear
        """
        if client_ip in self.request_timestamps:
            del self.request_timestamps[client_ip]
            logger.info(f"Cleared rate limit data for client {client_ip}")
    
    def clear_all_requests(self) -> None:
        """Clear all rate limit data for all clients."""
        self.request_timestamps.clear()
        logger.info("Cleared all rate limit data")


# Global rate limiter instance
rate_limiter = RateLimiter()
