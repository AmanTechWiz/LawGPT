"""
Intelligent caching service with Redis backend and memory fallback.

Provides multi-tier caching with automatic failover, specialized RAG query caching,
and decorators for transparent cache management with TTL support.
"""

import redis
import json
import hashlib
import asyncio
import logging
from typing import Any, Optional, Callable, Dict
from functools import wraps
from collections import OrderedDict
import time

from ..core.config import settings

logger = logging.getLogger(__name__)


class MemoryCache:
    """In-memory cache with LRU eviction and TTL support."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl_map = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self.cache:
                ttl = self.ttl_map.get(key)
                if ttl and time.time() > ttl:
                    del self.cache[key]
                    del self.ttl_map[key]
                    return None
                
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None, expire: int = None) -> bool:
        async with self._lock:
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                if oldest_key in self.ttl_map:
                    del self.ttl_map[oldest_key]
            
            self.cache[key] = value
            if ttl:
                self.ttl_map[key] = time.time() + ttl
            
            return True
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.ttl_map:
                    del self.ttl_map[key]
                return True
            return False
    
    async def clear(self) -> None:
        async with self._lock:
            self.cache.clear()
            self.ttl_map.clear()

class SmartCache:
    """Multi-tier cache with Redis backend and memory fallback."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache = MemoryCache(max_size=1000)
        self._lock = asyncio.Lock()
        self._redis_available = None
        self._last_check_time = 0
        self._check_interval = 30 
    
    async def initialize(self):
        if not settings.redis_url:
            logger.info("Redis not configured, using memory cache only")
            self._redis_available = False
            self.redis_client = None
            return
            
        if self.redis_client is None:
            async with self._lock:
                if self.redis_client is None:
                    try:
                        self.redis_client = redis.from_url(
                            settings.redis_url,
                            max_connections=settings.redis_max_connections,
                            decode_responses=True,
                            socket_connect_timeout=2,
                            socket_timeout=2,
                            retry_on_timeout=False  
                        )
                        
                        self.redis_client.ping()
                        logger.info("Redis cache initialized successfully")
                        self._redis_available = True
                        
                    except Exception as e:
                        self._redis_available = False
                        logger.info(f"Redis unavailable ({str(e)}), using memory cache only")
                        self.redis_client = None
    
    def _generate_cache_key(self, prefix: str, identifier: str) -> str:
        if not identifier:
            identifier = "empty"
        
        if len(identifier) > settings.cache_max_query_length:
            identifier = hashlib.sha256(identifier.encode()).hexdigest()
        return f"{prefix}:{identifier}"
    
    async def _should_attempt_redis(self) -> bool:
        if self._redis_available is False:
            return False
            
        if self.redis_client is None:
            return False
            
        current_time = time.time()
        if self._redis_available is False and current_time - self._last_check_time > self._check_interval:
            self._last_check_time = current_time
            await self.initialize()
            
        return self._redis_available
    
    async def get(self, key: str) -> Optional[Any]:
        if await self._should_attempt_redis():
            try:
                cached_value = self.redis_client.get(key)
                if cached_value:
                    return json.loads(cached_value)
            except Exception as e:
                pass
        
        return await self.memory_cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = None, expire: int = None) -> bool:
        ttl = expire or ttl or settings.cache_ttl_seconds
        success = False
        
        if await self._should_attempt_redis():
            try:
                serialized_value = json.dumps(value, default=str)
                result = self.redis_client.setex(key, ttl, serialized_value)
                success = bool(result)
            except Exception as e:
                pass
        
        await self.memory_cache.set(key, value, ttl)
        
        return success or True
    
    async def delete(self, key: str) -> bool:
        success = False
        
        if await self._should_attempt_redis():
            try:
                result = self.redis_client.delete(key)
                success = bool(result)
            except Exception as e:
                pass
        
        memory_success = await self.memory_cache.delete(key)
        
        return success or memory_success
    
    async def delete_pattern(self, pattern: str) -> int:
        deleted = 0
        
        if await self._should_attempt_redis():
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted = self.redis_client.delete(*keys)
                    logger.info(f"Deleted {deleted} cache keys matching pattern: {pattern}")
            except Exception as e:
                logger.warning(f"Cache pattern delete failed for pattern {pattern}: {e}")
        
        await self.memory_cache.clear()
        
        return deleted
    
    async def clear_pattern(self, pattern: str) -> int:
        return await self.delete_pattern(pattern)
    
    async def get_or_set(self, key: str, factory: Callable, ttl: int = None) -> Any:
        cached_result = await self.get(key)
        if cached_result is not None:
            return cached_result
        
        try:
            if asyncio.iscoroutinefunction(factory):
                result = await factory()
            else:
                result = factory()
            
            await self.set(key, result, ttl)
            return result
            
        except Exception as e:
            logger.error(f"Factory function failed for cache key {key}: {e}")
            raise
    
    async def increment(self, key: str, amount: int = 1, ttl: int = None) -> int:
        if await self._should_attempt_redis():
            try:
                pipe = self.redis_client.pipeline()
                pipe.incr(key, amount)
                if ttl:
                    pipe.expire(key, ttl)
                results = pipe.execute()
                return results[0]
            except Exception as e:
                logger.warning(f"Cache increment failed for key {key}: {e}")
        
        current = await self.memory_cache.get(key) or 0
        new_value = int(current) + amount
        await self.memory_cache.set(key, new_value, ttl)
        return new_value
    
    async def health_check(self) -> bool:
        if await self._should_attempt_redis():
            try:
                self.redis_client.ping()
                return True
            except Exception:
                pass
        
        return True
    
    async def flush_session_caches(self) -> None:
        """Clear in-memory cache and flush Redis DB (single-app Redis only)."""
        await self.memory_cache.clear()
        if await self._should_attempt_redis() and self.redis_client:
            try:
                self.redis_client.flushdb()
                logger.info("Redis FLUSHDB completed for session reset")
            except Exception as e:
                logger.warning("Redis flushdb failed: %s", e)
                for pattern in ("rag_query:*", "embedding:*", "rag:*"):
                    try:
                        await self.delete_pattern(pattern)
                    except Exception as pe:
                        logger.warning("Cache pattern delete failed %s: %s", pattern, pe)

    async def get_stats(self) -> Dict[str, Any]:
        stats = {
            "backend": "redis" if self._redis_available else "memory",
            "status": "available",
            "memory_cache_size": len(self.memory_cache.cache)
        }
        
        if await self._should_attempt_redis():
            try:
                info = self.redis_client.info()
                stats.update({
                    "backend": "redis",
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                    "hit_rate": self._calculate_hit_rate(
                        info.get("keyspace_hits", 0),
                        info.get("keyspace_misses", 0)
                    )
                })
            except Exception as e:
                pass
        
        return stats
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0

cache = SmartCache()

def cache_result(prefix: str, ttl: int = None, key_func: Optional[Callable] = None):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if key_func:
                cache_key_suffix = key_func(*args, **kwargs)
            else:
                key_parts = [str(arg) for arg in args] + [f"{k}={v}" for k, v in kwargs.items()]
                cache_key_suffix = "|".join(key_parts)
            
            cache_key = cache._generate_cache_key(prefix, cache_key_suffix)
            
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator

class RAGCache:
    """Specialized cache for RAG queries and embeddings."""
    
    def __init__(self, cache_instance: SmartCache):
        self.cache = cache_instance
    
    async def cache_rag_query(self, query: str, result: Dict[str, Any], algorithm: str = "hybrid", ttl: int = None) -> bool:
        cache_key = f"rag_query:{algorithm}:{hashlib.sha256(query.encode()).hexdigest()}"
        return await self.cache.set(cache_key, result, ttl)
    
    async def get_rag_query(self, query: str, algorithm: str = "hybrid") -> Optional[Dict[str, Any]]:
        # Handle None or empty queries
        if not query:
            query = "empty"
        
        cache_key = f"rag_query:{algorithm}:{hashlib.sha256(query.encode()).hexdigest()}"
        return await self.cache.get(cache_key)
    
    async def cache_embedding(self, text: str, embedding: list, ttl: int = 3600) -> bool:
        cache_key = self.cache._generate_cache_key("embedding", text)
        return await self.cache.set(cache_key, embedding, ttl)
    
    async def get_embedding(self, text: str) -> Optional[list]:
        cache_key = self.cache._generate_cache_key("embedding", text)
        return await self.cache.get(cache_key)
    
    async def invalidate_rag_cache(self) -> int:
        # Keys are rag_query:<algorithm>:<hash> and embedding:<id>
        patterns = ["rag_query:*", "embedding:*"]
        total_deleted = 0

        for pattern in patterns:
            deleted = await self.cache.delete_pattern(pattern)
            total_deleted += deleted

        return total_deleted

rag_cache = RAGCache(cache) 