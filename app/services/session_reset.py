"""
Truncate RAG-related database tables and flush caches for a clean session.

Used when ALLOW_SESSION_RESET is enabled (typically local dev / demos).
"""

import logging
from typing import List

import asyncpg

from ..core.database import db_manager
from ..services.cache import cache

logger = logging.getLogger(__name__)

# Child tables first (FK order); missing tables are skipped.
_RESET_TABLES_ORDER: List[str] = [
    "reasoning_steps",
    "reasoning_chains",
    "reasoning_sessions",
    "user_feedback",
    "feedback_metrics",
    "conversation_history",
    "documents",
]


async def reset_session_state() -> dict:
    """Truncate all RAG-related tables and flush Redis + in-memory cache."""
    truncated: List[str] = []

    async with db_manager.get_connection() as conn:
        for table in _RESET_TABLES_ORDER:
            try:
                await conn.execute(
                    f'TRUNCATE TABLE "{table}" RESTART IDENTITY CASCADE'
                )
                truncated.append(table)
            except asyncpg.UndefinedTableError:
                logger.debug("Session reset: table %s does not exist", table)
            except Exception as e:
                logger.warning("Session reset: could not truncate %s: %s", table, e)

    await cache.flush_session_caches()

    return {
        "status": "ok",
        "truncated_tables": truncated,
        "cache_flushed": True,
    }
