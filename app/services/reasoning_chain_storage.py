"""
Reasoning chain storage and retrieval system.

Provides persistent storage for multi-hop reasoning chains, intermediate results,
and reasoning history with efficient querying and retrieval capabilities.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import asdict

from ..core.database import db_manager
from .multi_hop_reasoning import ReasoningChain, ReasoningStep, ReasoningComplexity

logger = logging.getLogger(__name__)


class ReasoningChainStorage:
    """Persistent storage for reasoning chains and intermediate results with efficient querying."""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize storage tables for reasoning chains"""
        if self.initialized:
            return
        
        try:
            async with db_manager.get_connection() as conn:
                # Create reasoning_chains table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS reasoning_chains (
                        chain_id VARCHAR(255) PRIMARY KEY,
                        original_query TEXT NOT NULL,
                        complexity_level VARCHAR(50) NOT NULL,
                        final_answer TEXT NOT NULL,
                        total_execution_time FLOAT NOT NULL,
                        overall_confidence FLOAT NOT NULL,
                        citations JSONB,
                        metadata JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # Create reasoning_steps table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS reasoning_steps (
                        step_id VARCHAR(255) PRIMARY KEY,
                        chain_id VARCHAR(255) REFERENCES reasoning_chains(chain_id) ON DELETE CASCADE,
                        step_type VARCHAR(50) NOT NULL,
                        input_query TEXT NOT NULL,
                        output_result TEXT NOT NULL,
                        sources_used JSONB,
                        confidence_score FLOAT NOT NULL,
                        execution_time FLOAT NOT NULL,
                        metadata JSONB,
                        step_order INTEGER NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # Create indexes for efficient querying
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_reasoning_chains_created_at 
                    ON reasoning_chains(created_at)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_reasoning_chains_complexity 
                    ON reasoning_chains(complexity_level)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_reasoning_steps_chain_id 
                    ON reasoning_steps(chain_id)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_reasoning_steps_step_order 
                    ON reasoning_steps(chain_id, step_order)
                """)
                
                # Create reasoning_sessions table for session-based tracking
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS reasoning_sessions (
                        session_id VARCHAR(255) PRIMARY KEY,
                        user_id VARCHAR(255),
                        chain_ids JSONB,
                        total_queries INTEGER DEFAULT 0,
                        total_execution_time FLOAT DEFAULT 0.0,
                        average_confidence FLOAT DEFAULT 0.0,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                self.initialized = True
                logger.info("Reasoning chain storage initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize reasoning chain storage: {e}")
            raise
    
    async def store_reasoning_chain(self, reasoning_chain: ReasoningChain) -> bool:
        """
        Store a complete reasoning chain in the database.
        
        Args:
            reasoning_chain: The reasoning chain to store
            
        Returns:
            True if successful, False otherwise
        """
        await self.initialize()
        
        try:
            async with db_manager.get_connection() as conn:
                async with conn.transaction():
                    # Store main reasoning chain
                    await conn.execute("""
                        INSERT INTO reasoning_chains (
                            chain_id, original_query, complexity_level, final_answer,
                            total_execution_time, overall_confidence, citations, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (chain_id) DO UPDATE SET
                            final_answer = EXCLUDED.final_answer,
                            total_execution_time = EXCLUDED.total_execution_time,
                            overall_confidence = EXCLUDED.overall_confidence,
                            citations = EXCLUDED.citations,
                            metadata = EXCLUDED.metadata,
                            updated_at = NOW()
                    """, 
                    reasoning_chain.chain_id,
                    reasoning_chain.original_query,
                    reasoning_chain.complexity_level.value,
                    reasoning_chain.final_answer,
                    reasoning_chain.total_execution_time,
                    reasoning_chain.overall_confidence,
                    json.dumps(reasoning_chain.citations),
                    json.dumps(reasoning_chain.metadata)
                    )
                    
                    # Store reasoning steps
                    for i, step in enumerate(reasoning_chain.steps):
                        await conn.execute("""
                            INSERT INTO reasoning_steps (
                                step_id, chain_id, step_type, input_query, output_result,
                                sources_used, confidence_score, execution_time, metadata, step_order
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                            ON CONFLICT (step_id) DO UPDATE SET
                                output_result = EXCLUDED.output_result,
                                sources_used = EXCLUDED.sources_used,
                                confidence_score = EXCLUDED.confidence_score,
                                execution_time = EXCLUDED.execution_time,
                                metadata = EXCLUDED.metadata
                        """,
                        step.step_id,
                        reasoning_chain.chain_id,
                        step.step_type.value,
                        step.input_query,
                        step.output_result,
                        json.dumps(step.sources_used),
                        step.confidence_score,
                        step.execution_time,
                        json.dumps(step.metadata),
                        i
                        )
                    
                    # Update session tracking if session_id exists
                    session_id = reasoning_chain.metadata.get("session_id")
                    if session_id:
                        await self._update_session_tracking(conn, session_id, reasoning_chain)
                    
                    logger.info(f"Stored reasoning chain {reasoning_chain.chain_id} with {len(reasoning_chain.steps)} steps")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to store reasoning chain: {e}")
            return False
    
    async def retrieve_reasoning_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        """
        Retrieve a reasoning chain by ID.
        
        Args:
            chain_id: The ID of the reasoning chain to retrieve
            
        Returns:
            ReasoningChain if found, None otherwise
        """
        await self.initialize()
        
        try:
            async with db_manager.get_connection() as conn:
                # Get main chain data
                chain_row = await conn.fetchrow("""
                    SELECT * FROM reasoning_chains WHERE chain_id = $1
                """, chain_id)
                
                if not chain_row:
                    return None
                
                # Get reasoning steps
                step_rows = await conn.fetch("""
                    SELECT * FROM reasoning_steps 
                    WHERE chain_id = $1 
                    ORDER BY step_order
                """, chain_id)
                
                # Reconstruct reasoning steps
                steps = []
                for step_row in step_rows:
                    # Handle step_type conversion from string to enum
                    step_type_value = step_row['step_type']
                    if isinstance(step_type_value, str):
                        try:
                            from .multi_hop_reasoning import ReasoningStepType
                            step_type_value = ReasoningStepType(step_type_value)
                        except ValueError:
                            # If not a valid enum value, use a default
                            step_type_value = ReasoningStepType.QUERY_DECOMPOSITION
                    
                    step = ReasoningStep(
                        step_id=step_row['step_id'],
                        step_type=step_type_value,
                        input_query=step_row['input_query'],
                        output_result=step_row['output_result'],
                        sources_used=json.loads(step_row['sources_used']) if step_row['sources_used'] else [],
                        confidence_score=step_row['confidence_score'],
                        execution_time=step_row['execution_time'],
                        metadata=json.loads(step_row['metadata']) if step_row['metadata'] else {},
                        timestamp=step_row['created_at']
                    )
                    steps.append(step)
                
                # Reconstruct reasoning chain
                # Handle complexity_level conversion from string to enum
                complexity_level_value = chain_row['complexity_level']
                if isinstance(complexity_level_value, str):
                    try:
                        complexity_level_value = ReasoningComplexity(complexity_level_value)
                    except ValueError:
                        # If not a valid enum value, use a default
                        complexity_level_value = ReasoningComplexity.SIMPLE
                else:
                    complexity_level_value = ReasoningComplexity(complexity_level_value)
                
                reasoning_chain = ReasoningChain(
                    chain_id=chain_row['chain_id'],
                    original_query=chain_row['original_query'],
                    complexity_level=complexity_level_value,
                    steps=steps,
                    final_answer=chain_row['final_answer'],
                    total_execution_time=chain_row['total_execution_time'],
                    overall_confidence=chain_row['overall_confidence'],
                    citations=json.loads(chain_row['citations']) if chain_row['citations'] else [],
                    metadata=json.loads(chain_row['metadata']) if chain_row['metadata'] else {},
                    created_at=chain_row['created_at']
                )
                
                return reasoning_chain
                
        except Exception as e:
            logger.error(f"Failed to retrieve reasoning chain {chain_id}: {e}")
            return None
    
    async def get_reasoning_chains_by_session(self, session_id: str, 
                                            limit: int = 50) -> List[ReasoningChain]:
        """
        Get reasoning chains for a specific session.
        
        Args:
            session_id: The session ID
            limit: Maximum number of chains to return
            
        Returns:
            List of reasoning chains for the session
        """
        await self.initialize()
        
        try:
            async with db_manager.get_connection() as conn:
                # Get chain IDs for the session
                session_row = await conn.fetchrow("""
                    SELECT chain_ids FROM reasoning_sessions WHERE session_id = $1
                """, session_id)
                
                if not session_row or not session_row['chain_ids']:
                    return []
                
                chain_ids = json.loads(session_row['chain_ids'])
                
                # Retrieve chains
                chains = []
                for chain_id in chain_ids[:limit]:
                    chain = await self.retrieve_reasoning_chain(chain_id)
                    if chain:
                        chains.append(chain)
                
                return chains
                
        except Exception as e:
            logger.error(f"Failed to get reasoning chains for session {session_id}: {e}")
            return []
    
    async def search_similar_reasoning_chains(self, query: str, 
                                            complexity_level: Optional[ReasoningComplexity] = None,
                                            limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar reasoning chains based on query similarity.
        
        Args:
            query: The query to search for
            complexity_level: Optional complexity level filter
            limit: Maximum number of results
            
        Returns:
            List of similar reasoning chains with similarity scores
        """
        await self.initialize()
        
        try:
            async with db_manager.get_connection() as conn:
                # Use full-text search on original queries
                search_query = """
                    SELECT chain_id, original_query, complexity_level, 
                           final_answer, overall_confidence, created_at,
                           ts_rank(to_tsvector('english', original_query), 
                                   plainto_tsquery('english', $1)) as similarity_score
                    FROM reasoning_chains
                    WHERE to_tsvector('english', original_query) @@ plainto_tsquery('english', $1)
                """
                
                params = [query]
                if complexity_level:
                    search_query += " AND complexity_level = $2"
                    params.append(complexity_level.value)
                
                search_query += " ORDER BY similarity_score DESC LIMIT $3"
                params.append(limit)
                
                results = await conn.fetch(search_query, *params)
                
                similar_chains = []
                for row in results:
                    similar_chains.append({
                        "chain_id": row['chain_id'],
                        "original_query": row['original_query'],
                        "complexity_level": row['complexity_level'],
                        "final_answer": row['final_answer'],
                        "overall_confidence": row['overall_confidence'],
                        "similarity_score": float(row['similarity_score']),
                        "created_at": row['created_at']
                    })
                
                return similar_chains
                
        except Exception as e:
            logger.error(f"Failed to search similar reasoning chains: {e}")
            return []
    
    async def get_reasoning_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get reasoning statistics for the specified time period.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with reasoning statistics
        """
        await self.initialize()
        
        try:
            async with db_manager.get_connection() as conn:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # Get basic statistics
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_chains,
                        AVG(total_execution_time) as avg_execution_time,
                        AVG(overall_confidence) as avg_confidence,
                        COUNT(CASE WHEN complexity_level = 'simple' THEN 1 END) as simple_count,
                        COUNT(CASE WHEN complexity_level = 'moderate' THEN 1 END) as moderate_count,
                        COUNT(CASE WHEN complexity_level = 'complex' THEN 1 END) as complex_count,
                        COUNT(CASE WHEN complexity_level = 'very_complex' THEN 1 END) as very_complex_count
                    FROM reasoning_chains
                    WHERE created_at >= $1
                """, cutoff_date)
                
                # Get step statistics
                step_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_steps,
                        AVG(confidence_score) as avg_step_confidence,
                        AVG(execution_time) as avg_step_time
                    FROM reasoning_steps rs
                    JOIN reasoning_chains rc ON rs.chain_id = rc.chain_id
                    WHERE rc.created_at >= $1
                """, cutoff_date)
                
                return {
                    "period_days": days,
                    "total_reasoning_chains": stats['total_chains'],
                    "average_execution_time": float(stats['avg_execution_time']) if stats['avg_execution_time'] else 0.0,
                    "average_confidence": float(stats['avg_confidence']) if stats['avg_confidence'] else 0.0,
                    "complexity_distribution": {
                        "simple": stats['simple_count'],
                        "moderate": stats['moderate_count'],
                        "complex": stats['complex_count'],
                        "very_complex": stats['very_complex_count']
                    },
                    "step_statistics": {
                        "total_steps": step_stats['total_steps'],
                        "average_step_confidence": float(step_stats['avg_step_confidence']) if step_stats['avg_step_confidence'] else 0.0,
                        "average_step_time": float(step_stats['avg_step_time']) if step_stats['avg_step_time'] else 0.0
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get reasoning statistics: {e}")
            return {}
    
    async def cleanup_old_chains(self, days_to_keep: int = 90) -> int:
        """
        Clean up old reasoning chains to manage storage.
        
        Args:
            days_to_keep: Number of days of chains to keep
            
        Returns:
            Number of chains deleted
        """
        await self.initialize()
        
        try:
            async with db_manager.get_connection() as conn:
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                # Delete old chains (cascade will handle steps)
                result = await conn.execute("""
                    DELETE FROM reasoning_chains WHERE created_at < $1
                """, cutoff_date)
                
                deleted_count = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
                logger.info(f"Cleaned up {deleted_count} old reasoning chains")
                
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old reasoning chains: {e}")
            return 0
    
    async def _update_session_tracking(self, conn, session_id: str, reasoning_chain: ReasoningChain):
        """Update session tracking with new reasoning chain"""
        try:
            # Get or create session
            session_row = await conn.fetchrow("""
                SELECT chain_ids, total_queries, total_execution_time, average_confidence
                FROM reasoning_sessions WHERE session_id = $1
            """, session_id)
            
            if session_row:
                # Update existing session
                chain_ids = json.loads(session_row['chain_ids']) if session_row['chain_ids'] else []
                chain_ids.append(reasoning_chain.chain_id)
                
                total_queries = session_row['total_queries'] + 1
                total_time = session_row['total_execution_time'] + reasoning_chain.total_execution_time
                avg_confidence = (session_row['average_confidence'] * session_row['total_queries'] + 
                                reasoning_chain.overall_confidence) / total_queries
                
                await conn.execute("""
                    UPDATE reasoning_sessions SET
                        chain_ids = $1,
                        total_queries = $2,
                        total_execution_time = $3,
                        average_confidence = $4,
                        last_activity = NOW()
                    WHERE session_id = $5
                """, json.dumps(chain_ids), total_queries, total_time, avg_confidence, session_id)
            else:
                # Create new session
                await conn.execute("""
                    INSERT INTO reasoning_sessions (
                        session_id, chain_ids, total_queries, 
                        total_execution_time, average_confidence
                    ) VALUES ($1, $2, $3, $4, $5)
                """, session_id, json.dumps([reasoning_chain.chain_id]), 1, 
                   reasoning_chain.total_execution_time, reasoning_chain.overall_confidence)
                
        except Exception as e:
            logger.error(f"Failed to update session tracking: {e}")


# Global instance
reasoning_chain_storage = ReasoningChainStorage()
