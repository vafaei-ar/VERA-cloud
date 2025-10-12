"""
Redis Cache Service for VERA Cloud
Handles caching of TTS audio, responses, and session data
"""

import asyncio
import logging
import json
import pickle
from typing import Optional, Dict, Any, List
import redis.asyncio as redis
from datetime import timedelta

logger = logging.getLogger(__name__)

class RedisCacheService:
    def __init__(self, connection_string: str, ttl_seconds: int = 3600):
        self.redis = redis.from_url(connection_string)
        self.ttl = timedelta(seconds=ttl_seconds)
        self.default_ttl = ttl_seconds
    
    async def get_tts_audio(self, text: str, voice: str, rate: float) -> Optional[bytes]:
        """Get cached TTS audio"""
        try:
            key = f"tts:{hash(text)}:{voice}:{rate}"
            cached_audio = await self.redis.get(key)
            if cached_audio:
                logger.debug(f"Cache hit for TTS: {text[:30]}...")
                return cached_audio
            return None
        except Exception as e:
            logger.error(f"TTS cache get failed: {e}")
            return None
    
    async def set_tts_audio(self, text: str, voice: str, rate: float, audio_data: bytes) -> bool:
        """Cache TTS audio"""
        try:
            key = f"tts:{hash(text)}:{voice}:{rate}"
            await self.redis.setex(key, self.default_ttl, audio_data)
            logger.debug(f"Cached TTS audio: {text[:30]}...")
            return True
        except Exception as e:
            logger.error(f"TTS cache set failed: {e}")
            return False
    
    async def get_rag_response(self, user_input: str, context_hash: str) -> Optional[Dict]:
        """Get cached RAG response"""
        try:
            key = f"rag:{hash(user_input)}:{context_hash}"
            cached_response = await self.redis.get(key)
            if cached_response:
                logger.debug(f"Cache hit for RAG response: {user_input[:30]}...")
                return json.loads(cached_response)
            return None
        except Exception as e:
            logger.error(f"RAG cache get failed: {e}")
            return None
    
    async def set_rag_response(self, user_input: str, context_hash: str, response: Dict) -> bool:
        """Cache RAG response"""
        try:
            key = f"rag:{hash(user_input)}:{context_hash}"
            await self.redis.setex(key, self.default_ttl, json.dumps(response))
            logger.debug(f"Cached RAG response: {user_input[:30]}...")
            return True
        except Exception as e:
            logger.error(f"RAG cache set failed: {e}")
            return False
    
    async def get_session_data(self, session_id: str) -> Optional[Dict]:
        """Get cached session data"""
        try:
            key = f"session:{session_id}"
            cached_data = await self.redis.get(key)
            if cached_data:
                logger.debug(f"Cache hit for session: {session_id}")
                return pickle.loads(cached_data)
            return None
        except Exception as e:
            logger.error(f"Session cache get failed: {e}")
            return None
    
    async def set_session_data(self, session_id: str, data: Dict) -> bool:
        """Cache session data"""
        try:
            key = f"session:{session_id}"
            await self.redis.setex(key, self.default_ttl, pickle.dumps(data))
            logger.debug(f"Cached session data: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Session cache set failed: {e}")
            return False
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding"""
        try:
            key = f"embedding:{hash(text)}"
            cached_embedding = await self.redis.get(key)
            if cached_embedding:
                logger.debug(f"Cache hit for embedding: {text[:30]}...")
                return pickle.loads(cached_embedding)
            return None
        except Exception as e:
            logger.error(f"Embedding cache get failed: {e}")
            return None
    
    async def set_embedding(self, text: str, embedding: List[float]) -> bool:
        """Cache embedding"""
        try:
            key = f"embedding:{hash(text)}"
            await self.redis.setex(key, self.default_ttl * 24, pickle.dumps(embedding))  # Longer TTL for embeddings
            logger.debug(f"Cached embedding: {text[:30]}...")
            return True
        except Exception as e:
            logger.error(f"Embedding cache set failed: {e}")
            return False
    
    async def get_search_results(self, query: str, filters: str = None) -> Optional[List[Dict]]:
        """Get cached search results"""
        try:
            key = f"search:{hash(query)}:{hash(filters or '')}"
            cached_results = await self.redis.get(key)
            if cached_results:
                logger.debug(f"Cache hit for search: {query[:30]}...")
                return pickle.loads(cached_results)
            return None
        except Exception as e:
            logger.error(f"Search cache get failed: {e}")
            return None
    
    async def set_search_results(self, query: str, filters: str, results: List[Dict]) -> bool:
        """Cache search results"""
        try:
            key = f"search:{hash(query)}:{hash(filters or '')}"
            await self.redis.setex(key, self.default_ttl, pickle.dumps(results))
            logger.debug(f"Cached search results: {query[:30]}...")
            return True
        except Exception as e:
            logger.error(f"Search cache set failed: {e}")
            return False
    
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate all cache entries for a session"""
        try:
            pattern = f"session:{session_id}*"
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries for session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Session cache invalidation failed: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = await self.redis.info()
            return {
                "used_memory": info.get("used_memory_human", "0B"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info)
            }
        except Exception as e:
            logger.error(f"Cache stats failed: {e}")
            return {}
    
    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate"""
        try:
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            total = hits + misses
            return (hits / total * 100) if total > 0 else 0.0
        except Exception:
            return 0.0
    
    async def clear_cache(self, pattern: str = "*") -> bool:
        """Clear cache entries matching pattern"""
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries matching: {pattern}")
            return True
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False
    
    async def close(self):
        """Close Redis connection"""
        try:
            await self.redis.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Redis close failed: {e}")
