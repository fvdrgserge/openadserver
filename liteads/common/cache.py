"""
Redis cache client with async support.

Provides caching, feature storage, and frequency control capabilities.
"""

from __future__ import annotations

from typing import Any

import redis.asyncio as redis
from redis.asyncio import ConnectionPool, Redis

from liteads.common.config import get_settings
from liteads.common.exceptions import CacheError
from liteads.common.logger import get_logger
from liteads.common.utils import json_dumps, json_loads

logger = get_logger(__name__)


class RedisClient:
    """
    Async Redis client wrapper.

    Provides high-level caching operations with JSON serialization.
    """

    def __init__(self) -> None:
        self._pool: ConnectionPool | None = None
        self._client: Redis | None = None

    @property
    def client(self) -> Redis:
        """Get the Redis client."""
        if self._client is None:
            raise RuntimeError("Redis not initialized. Call connect() first.")
        return self._client

    async def connect(self) -> None:
        """Initialize Redis connection pool."""
        settings = get_settings()

        self._pool = ConnectionPool.from_url(
            settings.redis.url,
            max_connections=settings.redis.pool_size,
            decode_responses=True,
        )
        self._client = Redis(connection_pool=self._pool)

        # Test connection
        await self._client.ping()

        logger.info(
            "Redis connected",
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db,
        )

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        logger.info("Redis connection closed")

    async def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            await self.client.ping()
            return True
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return False

    # ==================== Basic Operations ====================

    async def get(self, key: str) -> str | None:
        """Get a string value."""
        return await self.client.get(key)

    async def set(
        self,
        key: str,
        value: str,
        ttl: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """
        Set a string value.

        Args:
            key: Redis key.
            value: String value.
            ttl: Time to live in seconds.
            nx: Only set if key doesn't exist.
            xx: Only set if key exists.
        """
        result = await self.client.set(key, value, ex=ttl, nx=nx, xx=xx)
        return bool(result)

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        if not keys:
            return 0
        return await self.client.delete(*keys)

    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        if not keys:
            return 0
        return await self.client.exists(*keys)

    async def expire(self, key: str, ttl: int) -> bool:
        """Set key expiration."""
        return await self.client.expire(key, ttl)

    async def ttl(self, key: str) -> int:
        """Get key time to live."""
        return await self.client.ttl(key)

    # ==================== JSON Operations ====================

    async def get_json(self, key: str) -> Any:
        """Get and deserialize JSON value."""
        value = await self.get(key)
        if value is None:
            return None
        try:
            return json_loads(value)
        except Exception as e:
            logger.warning("Failed to parse JSON", key=key, error=str(e))
            return None

    async def set_json(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Serialize and set JSON value."""
        try:
            json_str = json_dumps(value)
            return await self.set(key, json_str, ttl=ttl)
        except Exception as e:
            raise CacheError(f"Failed to serialize JSON: {e}")

    # ==================== Hash Operations ====================

    async def hget(self, key: str, field: str) -> str | None:
        """Get hash field value."""
        return await self.client.hget(key, field)

    async def hset(self, key: str, field: str, value: str) -> int:
        """Set hash field value."""
        return await self.client.hset(key, field, value)

    async def hmget(self, key: str, *fields: str) -> list[str | None]:
        """Get multiple hash field values."""
        if not fields:
            return []
        return await self.client.hmget(key, list(fields))

    async def hmset(self, key: str, mapping: dict[str, str]) -> bool:
        """Set multiple hash field values."""
        if not mapping:
            return True
        result = await self.client.hset(key, mapping=mapping)
        return result >= 0

    async def hgetall(self, key: str) -> dict[str, str]:
        """Get all hash field values."""
        return await self.client.hgetall(key)

    async def hdel(self, key: str, *fields: str) -> int:
        """Delete hash fields."""
        if not fields:
            return 0
        return await self.client.hdel(key, *fields)

    async def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        """Increment hash field by integer."""
        return await self.client.hincrby(key, field, amount)

    async def hincrbyfloat(self, key: str, field: str, amount: float) -> float:
        """Increment hash field by float."""
        return await self.client.hincrbyfloat(key, field, amount)

    # ==================== Counter Operations ====================

    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment counter."""
        return await self.client.incrby(key, amount)

    async def decr(self, key: str, amount: int = 1) -> int:
        """Decrement counter."""
        return await self.client.decrby(key, amount)

    # ==================== Sorted Set Operations ====================

    async def zadd(
        self,
        key: str,
        mapping: dict[str, float],
        nx: bool = False,
        xx: bool = False,
    ) -> int:
        """Add members to sorted set."""
        return await self.client.zadd(key, mapping, nx=nx, xx=xx)

    async def zrem(self, key: str, *members: str) -> int:
        """Remove members from sorted set."""
        if not members:
            return 0
        return await self.client.zrem(key, *members)

    async def zscore(self, key: str, member: str) -> float | None:
        """Get member score."""
        return await self.client.zscore(key, member)

    async def zrange(
        self,
        key: str,
        start: int = 0,
        end: int = -1,
        withscores: bool = False,
    ) -> list[Any]:
        """Get members in range by index."""
        return await self.client.zrange(key, start, end, withscores=withscores)

    async def zrevrange(
        self,
        key: str,
        start: int = 0,
        end: int = -1,
        withscores: bool = False,
    ) -> list[Any]:
        """Get members in range by index (descending)."""
        return await self.client.zrevrange(key, start, end, withscores=withscores)

    # ==================== Set Operations ====================

    async def sadd(self, key: str, *members: str) -> int:
        """Add members to set."""
        if not members:
            return 0
        return await self.client.sadd(key, *members)

    async def srem(self, key: str, *members: str) -> int:
        """Remove members from set."""
        if not members:
            return 0
        return await self.client.srem(key, *members)

    async def sismember(self, key: str, member: str) -> bool:
        """Check if member exists in set."""
        return await self.client.sismember(key, member)

    async def smembers(self, key: str) -> set[str]:
        """Get all members of set."""
        return await self.client.smembers(key)

    # ==================== Pipeline Operations ====================

    def pipeline(self) -> Any:
        """Get a pipeline for batch operations."""
        return self.client.pipeline()

    # ==================== Pub/Sub Operations ====================

    async def publish(self, channel: str, message: str) -> int:
        """Publish message to channel."""
        return await self.client.publish(channel, message)


# Global Redis client instance
redis_client = RedisClient()


# ==================== Key Builders ====================


class CacheKeys:
    """Redis key builders for different data types."""

    # Ad campaign cache
    @staticmethod
    def campaign(campaign_id: int) -> str:
        return f"ad:campaign:{campaign_id}"

    @staticmethod
    def campaign_creative(campaign_id: int, creative_id: int) -> str:
        return f"ad:creative:{campaign_id}:{creative_id}"

    # User features
    @staticmethod
    def user_feature(user_id: str) -> str:
        return f"user:feature:{user_id}"

    # Frequency control
    @staticmethod
    def freq_daily(user_id: str, campaign_id: int, date: str) -> str:
        return f"freq:daily:{user_id}:{campaign_id}:{date}"

    @staticmethod
    def freq_hourly(user_id: str, campaign_id: int, hour: str) -> str:
        return f"freq:hourly:{user_id}:{campaign_id}:{hour}"

    # Statistics
    @staticmethod
    def stat_hourly(campaign_id: int, hour: str) -> str:
        return f"stat:{campaign_id}:{hour}"

    # Active ads
    @staticmethod
    def active_ads() -> str:
        return "ads:active"

    # Model cache
    @staticmethod
    def model(model_name: str, version: str) -> str:
        return f"model:{model_name}:{version}"

    # Rate limiting
    @staticmethod
    def rate_limit(client_id: str, endpoint: str) -> str:
        return f"ratelimit:{client_id}:{endpoint}"
