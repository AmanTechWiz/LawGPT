"""
Centralized configuration management for RAG system (Groq + Local Embeddings)

- Uses environment variables
- Supports Docker
- No OpenAI dependency
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration"""

    # -------------------------------
    # App Config
    # -------------------------------
    app_name: str = "Legal RAG System"
    app_version: str = "2.0.0"
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 8000

    # -------------------------------
    # Database
    # -------------------------------
    database_url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", "")
    )

    db_pool_min_size: int = 5
    db_pool_max_size: int = 20
    db_command_timeout: int = 60

    # -------------------------------
    # Redis
    # -------------------------------
    redis_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("REDIS_URL", None)
    )

    # -------------------------------
    # 🔥 GROQ CONFIG (MAIN LLM)
    # -------------------------------
    groq_api_key: str = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )

    groq_model: str = Field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama3-70b-8192")
    )

    # -------------------------------
    # RAG SETTINGS
    # -------------------------------
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.25
    rag_response_length: str = "normal"
    # Total character budget for all retrieved chunks passed to the LLM (was 500/chunk, which hid answers).
    rag_max_total_context_chars: int = 12000

    # -------------------------------
    # Cache
    # -------------------------------
    cache_ttl_seconds: int = 300

    # -------------------------------
    # Session reset (dev / demos)
    # -------------------------------
    # When True, POST /session/reset truncates RAG tables and flushes Redis + memory cache.
    allow_session_reset: bool = False

    # -------------------------------
    # Text Processing
    # -------------------------------
    max_text_length: int = 8000
    pdf_chunk_size: int = 1800
    pdf_chunk_overlap: int = 300

    # -------------------------------
    # CORS
    # -------------------------------
    cors_origins: str = Field(
        default_factory=lambda: os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://localhost:5173,http://localhost"
        )
    )

    # -------------------------------
    # Limits
    # -------------------------------
    rate_limit_requests: int = 10
    websocket_max_connections: int = 1000

    # -------------------------------
    # Logging
    # -------------------------------
    log_level: str = "INFO"

    # -------------------------------
    # Validators
    # -------------------------------

    @field_validator("database_url")
    @classmethod
    def validate_database(cls, v):
        if not v:
            raise ValueError("DATABASE_URL is required")
        return v

    @field_validator("groq_api_key")
    @classmethod
    def validate_groq(cls, v):
        if not v:
            raise ValueError("GROQ_API_KEY is required")
        return v

    @field_validator("max_text_length")
    @classmethod
    def validate_text_length(cls, v):
        if v < 1000:
            raise ValueError("max_text_length must be >= 1000")
        return v

    # -------------------------------
    # Pydantic config
    # -------------------------------
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()