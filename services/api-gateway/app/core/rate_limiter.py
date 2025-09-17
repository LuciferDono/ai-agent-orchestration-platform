"""
Redis-based rate limiter for API Gateway
Implements sliding window rate limiting with different time windows
"""

import asyncio
import time
from typing import Optional, Dict, Any
import json

import redis.asyncio as redis
import structlog

logger = structlog.get_logger()


class RateLimiter:
    """
    Redis-based rate limiter using sliding window algorithm
    """
    
    def __init__(
        self,
        redis_url: str,
        default_limit: int = 100,
        window_size: int = 60,
        namespace: str = "rate_limit"
    ):
        """
        Initialize rate limiter
        
        Args:
            redis_url: Redis connection URL
            default_limit: Default requests per window
            window_size: Window size in seconds
            namespace: Redis key namespace
        """
        self.redis_url = redis_url
        self.default_limit = default_limit
        self.window_size = window_size
        self.namespace = namespace
        self._redis_client: Optional[redis.Redis] = None
        
        # Rate limiting scripts (loaded once for efficiency)
        self._sliding_window_script = """
        local key = KEYS[1]
        local window_size = tonumber(ARGV[1])
        local limit = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])
        
        -- Clean expired entries
        local expired_score = current_time - window_size
        redis.call('ZREMRANGEBYSCORE', key, '-inf', expired_score)
        
        -- Count current requests in window
        local current_count = redis.call('ZCARD', key)
        
        if current_count < limit then
            -- Add current request
            redis.call('ZADD', key, current_time, current_time)
            redis.call('EXPIRE', key, window_size)
            return {1, limit - current_count - 1}
        else
            return {0, 0}
        end
        """
    
    async def _get_redis(self) -> redis.Redis:
        """Get Redis client, creating if necessary"""
        if self._redis_client is None:
            self._redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                health_check_interval=30
            )
        return self._redis_client
    
    async def is_allowed(
        self,
        identifier: str,
        limit: Optional[int] = None,
        window_size: Optional[int] = None
    ) -> bool:
        """
        Check if request is allowed under rate limit
        
        Args:
            identifier: Unique identifier for the client (IP, user ID, etc.)
            limit: Custom limit for this check
            window_size: Custom window size for this check
            
        Returns:
            True if request is allowed, False otherwise
        """
        limit = limit or self.default_limit
        window_size = window_size or self.window_size
        
        try:
            client = await self._get_redis()
            key = f"{self.namespace}:{identifier}"
            current_time = time.time()
            
            # Use Lua script for atomic operation
            result = await client.eval(
                self._sliding_window_script,
                1,  # Number of keys
                key,
                window_size,
                limit,
                current_time
            )
            
            allowed = bool(result[0])
            remaining = int(result[1])
            
            if not allowed:
                logger.warning(
                    "Rate limit exceeded",
                    identifier=identifier,
                    limit=limit,
                    window_size=window_size
                )
            
            return allowed
            
        except Exception as e:
            logger.error(
                "Rate limiter error",
                identifier=identifier,
                error=str(e)
            )
            # Fail open - allow request if rate limiter fails
            return True
    
    async def get_limit_info(
        self,
        identifier: str,
        limit: Optional[int] = None,
        window_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get rate limit information for an identifier
        
        Args:
            identifier: Unique identifier for the client
            limit: Custom limit
            window_size: Custom window size
            
        Returns:
            Dictionary with rate limit information
        """
        limit = limit or self.default_limit
        window_size = window_size or self.window_size
        
        try:
            client = await self._get_redis()
            key = f"{self.namespace}:{identifier}"
            current_time = time.time()
            
            # Clean expired entries
            expired_score = current_time - window_size
            await client.zremrangebyscore(key, '-inf', expired_score)
            
            # Get current count
            current_count = await client.zcard(key)
            
            # Get oldest request time in current window
            oldest_requests = await client.zrange(key, 0, 0, withscores=True)
            oldest_time = oldest_requests[0][1] if oldest_requests else None
            
            # Calculate reset time
            reset_time = None
            if oldest_time:
                reset_time = oldest_time + window_size
            
            return {
                "limit": limit,
                "remaining": max(0, limit - current_count),
                "used": current_count,
                "window_size": window_size,
                "reset_time": reset_time,
                "current_time": current_time
            }
            
        except Exception as e:
            logger.error(
                "Error getting rate limit info",
                identifier=identifier,
                error=str(e)
            )
            return {
                "limit": limit,
                "remaining": limit,
                "used": 0,
                "window_size": window_size,
                "reset_time": None,
                "current_time": time.time(),
                "error": str(e)
            }
    
    async def reset_limit(self, identifier: str) -> bool:
        """
        Reset rate limit for an identifier
        
        Args:
            identifier: Unique identifier to reset
            
        Returns:
            True if reset successful, False otherwise
        """
        try:
            client = await self._get_redis()
            key = f"{self.namespace}:{identifier}"
            
            await client.delete(key)
            
            logger.info(
                "Rate limit reset",
                identifier=identifier
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Error resetting rate limit",
                identifier=identifier,
                error=str(e)
            )
            return False
    
    async def add_tokens(
        self,
        identifier: str,
        tokens: int,
        limit: Optional[int] = None
    ) -> bool:
        """
        Add tokens (reduce usage) for an identifier
        This can be used to implement token bucket-like behavior
        
        Args:
            identifier: Unique identifier
            tokens: Number of tokens to add (reduce usage by this amount)
            limit: Custom limit
            
        Returns:
            True if successful, False otherwise
        """
        limit = limit or self.default_limit
        
        try:
            client = await self._get_redis()
            key = f"{self.namespace}:{identifier}"
            current_time = time.time()
            
            # Remove oldest requests to effectively add tokens
            for _ in range(tokens):
                await client.zpopmin(key, 1)
            
            logger.debug(
                "Tokens added",
                identifier=identifier,
                tokens=tokens
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Error adding tokens",
                identifier=identifier,
                tokens=tokens,
                error=str(e)
            )
            return False
    
    async def get_all_limits(self) -> Dict[str, Dict[str, Any]]:
        """
        Get rate limit information for all tracked identifiers
        
        Returns:
            Dictionary mapping identifiers to their rate limit info
        """
        try:
            client = await self._get_redis()
            pattern = f"{self.namespace}:*"
            keys = await client.keys(pattern)
            
            results = {}
            for key in keys:
                identifier = key.replace(f"{self.namespace}:", "")
                results[identifier] = await self.get_limit_info(identifier)
            
            return results
            
        except Exception as e:
            logger.error(
                "Error getting all limits",
                error=str(e)
            )
            return {}
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired rate limit entries
        
        Returns:
            Number of keys cleaned up
        """
        try:
            client = await self._get_redis()
            pattern = f"{self.namespace}:*"
            keys = await client.keys(pattern)
            
            current_time = time.time()
            expired_score = current_time - self.window_size
            cleaned_count = 0
            
            for key in keys:
                # Remove expired entries
                removed = await client.zremrangebyscore(key, '-inf', expired_score)
                
                # If key is empty, delete it
                if await client.zcard(key) == 0:
                    await client.delete(key)
                    cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(
                    "Rate limiter cleanup completed",
                    cleaned_keys=cleaned_count,
                    total_keys=len(keys)
                )
            
            return cleaned_count
            
        except Exception as e:
            logger.error(
                "Error during cleanup",
                error=str(e)
            )
            return 0
    
    async def close(self):
        """Close Redis connection"""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None
            logger.info("Rate limiter Redis connection closed")


class MultiWindowRateLimiter:
    """
    Rate limiter with multiple time windows (minute, hour, day)
    """
    
    def __init__(
        self,
        redis_url: str,
        limits: Dict[str, int],
        namespace: str = "multi_rate_limit"
    ):
        """
        Initialize multi-window rate limiter
        
        Args:
            redis_url: Redis connection URL
            limits: Dictionary mapping window names to limits
                   e.g., {"minute": 100, "hour": 1000, "day": 10000}
            namespace: Redis key namespace
        """
        self.limits = limits
        self.namespace = namespace
        
        # Create individual rate limiters for each window
        self.limiters = {}
        window_sizes = {
            "minute": 60,
            "hour": 3600,
            "day": 86400
        }
        
        for window_name, limit in limits.items():
            if window_name in window_sizes:
                self.limiters[window_name] = RateLimiter(
                    redis_url=redis_url,
                    default_limit=limit,
                    window_size=window_sizes[window_name],
                    namespace=f"{namespace}_{window_name}"
                )
    
    async def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed across all windows
        
        Args:
            identifier: Unique identifier for the client
            
        Returns:
            True if allowed in all windows, False otherwise
        """
        results = await asyncio.gather(
            *[limiter.is_allowed(identifier) for limiter in self.limiters.values()],
            return_exceptions=True
        )
        
        # All windows must allow the request
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                window_name = list(self.limiters.keys())[i]
                logger.error(
                    "Error checking rate limit",
                    window=window_name,
                    identifier=identifier,
                    error=str(result)
                )
                continue  # Fail open for this window
            elif not result:
                return False
        
        return True
    
    async def get_limit_info(self, identifier: str) -> Dict[str, Dict[str, Any]]:
        """
        Get rate limit information for all windows
        
        Args:
            identifier: Unique identifier
            
        Returns:
            Dictionary mapping window names to their limit info
        """
        results = {}
        
        for window_name, limiter in self.limiters.items():
            try:
                results[window_name] = await limiter.get_limit_info(identifier)
            except Exception as e:
                logger.error(
                    "Error getting limit info",
                    window=window_name,
                    identifier=identifier,
                    error=str(e)
                )
                results[window_name] = {"error": str(e)}
        
        return results
    
    async def close(self):
        """Close all rate limiter connections"""
        await asyncio.gather(
            *[limiter.close() for limiter in self.limiters.values()],
            return_exceptions=True
        )