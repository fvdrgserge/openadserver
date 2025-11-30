"""
Cache-based fast retrieval.

Retrieves ads from pre-computed cache for high-performance scenarios.
"""

from typing import Any

from liteads.common.cache import redis_client
from liteads.common.logger import get_logger
from liteads.common.utils import json_loads
from liteads.rec_engine.retrieval.base import BaseRetrieval
from liteads.schemas.internal import AdCandidate, UserContext

logger = get_logger(__name__)


class CacheRetrieval(BaseRetrieval):
    """
    Fast retrieval from pre-computed cache.

    Used for:
    - Hot ads that are frequently served
    - Pre-computed user-ad matches
    - Fallback when database is slow
    """

    def __init__(self, cache_prefix: str = "retrieval"):
        self.cache_prefix = cache_prefix

    async def retrieve(
        self,
        user_context: UserContext,
        slot_id: str,
        limit: int = 100,
        **kwargs: Any,
    ) -> list[AdCandidate]:
        """
        Retrieve candidates from cache.

        Tries multiple cache keys:
        1. User-specific recommendations
        2. Slot-specific hot ads
        3. Global hot ads
        """
        candidates: list[AdCandidate] = []

        # Try user-specific cache
        if user_context.user_id:
            user_key = f"{self.cache_prefix}:user:{user_context.user_id}"
            user_candidates = await self._get_cached_candidates(user_key, limit)
            candidates.extend(user_candidates)

        # Try slot-specific hot ads
        if len(candidates) < limit:
            slot_key = f"{self.cache_prefix}:slot:{slot_id}"
            slot_candidates = await self._get_cached_candidates(
                slot_key, limit - len(candidates)
            )
            candidates.extend(slot_candidates)

        # Try global hot ads
        if len(candidates) < limit:
            global_key = f"{self.cache_prefix}:hot"
            hot_candidates = await self._get_cached_candidates(
                global_key, limit - len(candidates)
            )
            candidates.extend(hot_candidates)

        logger.debug(f"Cache retrieval returned {len(candidates)} candidates")
        return candidates

    async def _get_cached_candidates(
        self, key: str, limit: int
    ) -> list[AdCandidate]:
        """Get candidates from a specific cache key."""
        try:
            cached = await redis_client.get(key)
            if not cached:
                return []

            data = json_loads(cached)
            candidates = []

            for item in data[:limit]:
                candidate = AdCandidate(
                    campaign_id=item["campaign_id"],
                    creative_id=item["creative_id"],
                    advertiser_id=item.get("advertiser_id", 0),
                    bid=item.get("bid", 1.0),
                    bid_type=item.get("bid_type", 1),
                    title=item.get("title"),
                    description=item.get("description"),
                    image_url=item.get("image_url"),
                    video_url=item.get("video_url"),
                    landing_url=item.get("landing_url", ""),
                    creative_type=item.get("creative_type", 1),
                    width=item.get("width"),
                    height=item.get("height"),
                )
                candidates.append(candidate)

            return candidates

        except Exception as e:
            logger.warning(f"Cache retrieval error for {key}: {e}")
            return []

    async def refresh(self) -> None:
        """Refresh cache - implemented by external job."""
        logger.info("Cache retrieval refresh requested (handled by external job)")

    async def set_hot_ads(
        self,
        candidates: list[AdCandidate],
        slot_id: str | None = None,
        ttl: int = 300,
    ) -> None:
        """
        Set hot ads in cache.

        Called by external job to pre-compute hot ads.
        """
        data = [
            {
                "campaign_id": c.campaign_id,
                "creative_id": c.creative_id,
                "advertiser_id": c.advertiser_id,
                "bid": c.bid,
                "bid_type": c.bid_type,
                "title": c.title,
                "description": c.description,
                "image_url": c.image_url,
                "video_url": c.video_url,
                "landing_url": c.landing_url,
                "creative_type": c.creative_type,
                "width": c.width,
                "height": c.height,
            }
            for c in candidates
        ]

        if slot_id:
            key = f"{self.cache_prefix}:slot:{slot_id}"
        else:
            key = f"{self.cache_prefix}:hot"

        from liteads.common.utils import json_dumps
        await redis_client.set(key, json_dumps(data), ttl=ttl)
        logger.info(f"Set {len(candidates)} hot ads in cache: {key}")
