"""
User feedback and iterative improvement system for adaptive RAG.

Enables feedback collection, analysis, and system adaptation based on user ratings
and response quality metrics.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

from ..core.database import db_manager

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of user feedback"""
    RATING = "rating"
    INTENT_CORRECTION = "intent_correction"
    RESPONSE_QUALITY = "response_quality"
    LENGTH_APPROPRIATENESS = "length_appropriateness"
    CITATION_ACCURACY = "citation_accuracy"


@dataclass
class UserFeedback:
    """User feedback data structure"""
    feedback_id: str
    query: str
    response: str
    intent_classified: str
    feedback_type: FeedbackType
    rating: Optional[int] = None  # 1-5 scale
    correction: Optional[str] = None  # Corrected intent or response
    comments: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class FeedbackMetrics:
    """Aggregated feedback metrics"""
    intent_accuracy: float
    average_rating: float
    response_quality_score: float
    citation_accuracy: float
    total_feedback_count: int
    improvement_suggestions: List[str]


class FeedbackSystem:
    """System for collecting, analyzing, and acting on user feedback."""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize the feedback system and create necessary database tables."""
        if self.initialized:
            return
        
        try:
            await self._create_feedback_tables()
            self.initialized = True
            logger.info("Feedback System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Feedback System: {e}")
            raise
    
    async def _create_feedback_tables(self):
        """Create database tables for feedback storage."""
        async with db_manager.get_connection() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    feedback_id VARCHAR(255) PRIMARY KEY,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    intent_classified VARCHAR(50) NOT NULL,
                    feedback_type VARCHAR(50) NOT NULL,
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                    correction TEXT,
                    comments TEXT,
                    user_id VARCHAR(255),
                    session_id VARCHAR(255),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_metrics (
                    metric_id SERIAL PRIMARY KEY,
                    intent_accuracy FLOAT,
                    average_rating FLOAT,
                    response_quality_score FLOAT,
                    citation_accuracy FLOAT,
                    total_feedback_count INTEGER,
                    improvement_suggestions JSONB,
                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_intent ON user_feedback(intent_classified);
                CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON user_feedback(timestamp);
                CREATE INDEX IF NOT EXISTS idx_feedback_type ON user_feedback(feedback_type);
            """)
    
    async def submit_feedback(self, feedback: UserFeedback) -> bool:
        """Submit user feedback for analysis and improvement."""
        if not self.initialized:
            await self.initialize()
        
        try:
            async with db_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO user_feedback (
                        feedback_id, query, response, intent_classified, feedback_type,
                        rating, correction, comments, user_id, session_id, timestamp
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, 
                feedback.feedback_id, feedback.query, feedback.response,
                feedback.intent_classified, feedback.feedback_type.value,
                feedback.rating, feedback.correction, feedback.comments,
                feedback.user_id, feedback.session_id, feedback.timestamp
                )
            
            logger.info(f"Feedback submitted: {feedback.feedback_type.value}")
            
            # Trigger metrics recalculation if significant feedback
            await self._check_and_recalculate_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            return False
    
    async def get_feedback_metrics(self, days: int = 30) -> Optional[FeedbackMetrics]:
        """Get aggregated feedback metrics for the specified period."""
        if not self.initialized:
            await self.initialize()
        
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            async with db_manager.get_connection() as conn:
                # Get intent accuracy
                intent_result = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(CASE WHEN correction IS NOT NULL THEN 1 END) as corrections
                    FROM user_feedback 
                    WHERE feedback_type = 'intent_correction' 
                    AND timestamp >= $1
                """, start_date)
                
                intent_accuracy = 0.0
                if intent_result['total'] > 0:
                    intent_accuracy = 1.0 - (intent_result['corrections'] / intent_result['total'])
                
                # Get average rating
                rating_result = await conn.fetchrow("""
                    SELECT AVG(rating) as avg_rating, COUNT(*) as count
                    FROM user_feedback 
                    WHERE feedback_type = 'rating' 
                    AND rating IS NOT NULL 
                    AND timestamp >= $1
                """, start_date)
                
                average_rating = rating_result['avg_rating'] or 0.0
                
                # Get response quality score
                quality_result = await conn.fetchrow("""
                    SELECT AVG(rating) as avg_quality, COUNT(*) as count
                    FROM user_feedback 
                    WHERE feedback_type = 'response_quality' 
                    AND rating IS NOT NULL 
                    AND timestamp >= $1
                """, start_date)
                
                response_quality_score = quality_result['avg_quality'] or 0.0
                
                # Get citation accuracy
                citation_result = await conn.fetchrow("""
                    SELECT AVG(rating) as avg_citation, COUNT(*) as count
                    FROM user_feedback 
                    WHERE feedback_type = 'citation_accuracy' 
                    AND rating IS NOT NULL 
                    AND timestamp >= $1
                """, start_date)
                
                citation_accuracy = citation_result['avg_citation'] or 0.0
                
                # Get total feedback count
                total_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM user_feedback WHERE timestamp >= $1
                """, start_date)
                
                # Generate improvement suggestions
                suggestions = await self._generate_improvement_suggestions(conn, start_date)
                
                return FeedbackMetrics(
                    intent_accuracy=intent_accuracy,
                    average_rating=average_rating,
                    response_quality_score=response_quality_score,
                    citation_accuracy=citation_accuracy,
                    total_feedback_count=total_count,
                    improvement_suggestions=suggestions
                )
                
        except Exception as e:
            logger.error(f"Failed to get feedback metrics: {e}")
            return None
    
    async def _generate_improvement_suggestions(self, conn, start_date: datetime) -> List[str]:
        """Generate improvement suggestions based on feedback analysis."""
        suggestions = []
        
        try:
            # Check for low-rated intents
            low_rated_intents = await conn.fetch("""
                SELECT intent_classified, AVG(rating) as avg_rating, COUNT(*) as count
                FROM user_feedback 
                WHERE feedback_type = 'rating' 
                AND rating IS NOT NULL 
                AND timestamp >= $1
                GROUP BY intent_classified
                HAVING AVG(rating) < 3.0 AND COUNT(*) >= 3
            """, start_date)
            
            for intent_data in low_rated_intents:
                suggestions.append(
                    f"Consider improving {intent_data['intent_classified']} intent classification "
                    f"(avg rating: {intent_data['avg_rating']:.1f})"
                )
            
            # Check for frequent intent corrections
            frequent_corrections = await conn.fetch("""
                SELECT correction, COUNT(*) as count
                FROM user_feedback 
                WHERE feedback_type = 'intent_correction' 
                AND correction IS NOT NULL 
                AND timestamp >= $1
                GROUP BY correction
                HAVING COUNT(*) >= 2
                ORDER BY COUNT(*) DESC
                LIMIT 3
            """, start_date)
            
            for correction_data in frequent_corrections:
                suggestions.append(
                    f"Common intent correction: {correction_data['correction']} "
                    f"({correction_data['count']} times) - review classification patterns"
                )
            
            # Check for response quality issues
            quality_issues = await conn.fetch("""
                SELECT intent_classified, AVG(rating) as avg_quality
                FROM user_feedback 
                WHERE feedback_type = 'response_quality' 
                AND rating IS NOT NULL 
                AND timestamp >= $1
                GROUP BY intent_classified
                HAVING AVG(rating) < 3.0
            """, start_date)
            
            for quality_data in quality_issues:
                suggestions.append(
                    f"Improve response quality for {quality_data['intent_classified']} "
                    f"(avg quality: {quality_data['avg_quality']:.1f})"
                )
            
        except Exception as e:
            logger.error(f"Failed to generate improvement suggestions: {e}")
        
        return suggestions
    
    async def _check_and_recalculate_metrics(self):
        """Check if metrics should be recalculated based on new feedback."""
        try:
            async with db_manager.get_connection() as conn:
                # Check if we have enough new feedback to warrant recalculation
                recent_feedback = await conn.fetchval("""
                    SELECT COUNT(*) FROM user_feedback 
                    WHERE timestamp >= NOW() - INTERVAL '1 hour'
                """)
                
                if recent_feedback >= 5:  # Recalculate if 5+ new feedback items
                    await self._recalculate_and_store_metrics(conn)
                    
        except Exception as e:
            logger.error(f"Failed to check and recalculate metrics: {e}")
    
    async def _recalculate_and_store_metrics(self, conn):
        """Recalculate and store current metrics."""
        try:
            metrics = await self.get_feedback_metrics(days=30)
            if metrics:
                await conn.execute("""
                    INSERT INTO feedback_metrics (
                        intent_accuracy, average_rating, response_quality_score,
                        citation_accuracy, total_feedback_count, improvement_suggestions
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, 
                metrics.intent_accuracy, metrics.average_rating, metrics.response_quality_score,
                metrics.citation_accuracy, metrics.total_feedback_count, 
                json.dumps(metrics.improvement_suggestions)
                )
                
                logger.info("Feedback metrics recalculated and stored")
                
        except Exception as e:
            logger.error(f"Failed to recalculate and store metrics: {e}")
    
    async def get_intent_performance_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Get detailed performance analysis by intent type."""
        if not self.initialized:
            await self.initialize()
        
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            async with db_manager.get_connection() as conn:
                results = await conn.fetch("""
                    SELECT 
                        intent_classified,
                        COUNT(*) as total_queries,
                        AVG(CASE WHEN feedback_type = 'rating' THEN rating END) as avg_rating,
                        COUNT(CASE WHEN feedback_type = 'intent_correction' AND correction IS NOT NULL THEN 1 END) as corrections,
                        COUNT(CASE WHEN feedback_type = 'response_quality' THEN 1 END) as quality_feedback
                    FROM user_feedback 
                    WHERE timestamp >= $1
                    GROUP BY intent_classified
                    ORDER BY total_queries DESC
                """, start_date)
                
                analysis = {}
                for row in results:
                    intent = row['intent_classified']
                    analysis[intent] = {
                        'total_queries': row['total_queries'],
                        'avg_rating': float(row['avg_rating']) if row['avg_rating'] else 0.0,
                        'corrections': row['corrections'],
                        'quality_feedback_count': row['quality_feedback'],
                        'accuracy': 1.0 - (row['corrections'] / row['total_queries']) if row['total_queries'] > 0 else 1.0
                    }
                
                return analysis
                
        except Exception as e:
            logger.error(f"Failed to get intent performance analysis: {e}")
            return {}
    
    async def get_recent_feedback(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent feedback for review."""
        if not self.initialized:
            await self.initialize()
        
        try:
            async with db_manager.get_connection() as conn:
                results = await conn.fetch("""
                    SELECT * FROM user_feedback 
                    ORDER BY timestamp DESC 
                    LIMIT $1
                """, limit)
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Failed to get recent feedback: {e}")
            return []


# Global instance
feedback_system = FeedbackSystem()
