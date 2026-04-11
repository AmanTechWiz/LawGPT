"""
API package for the legal research assistant.

This package contains all API-related modules including endpoints,
middleware, and request/response handling.
"""

from .endpoints import router

__all__ = ["router"]
