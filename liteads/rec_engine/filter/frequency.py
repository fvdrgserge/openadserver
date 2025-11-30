"""
Frequency filter for controlling ad exposure per user.
"""

from typing import Any

from liteads.common.cache import CacheKeys, redis_client
from liteads.common.config import get_settings
from liteads.common.logger import get_logger
from liteads.common.utils import current_date, current_hour
from liteads.rec_engine.filter.base import BaseFilter
from liteads.schemas.internal import AdCandidate, FrequencyInfo, UserContext

logger = get_logger(__name__)


class FrequencyFilter(BaseFilter):
    """
    Filter candidates by frequency cap.

    Prevents showing the same ad too many times to the same user.
    Supports both daily and hourly caps.
    """

    def __init__(
        self,
        default_daily_cap: int | None = None,
        default_hourly_cap: int | None = None,
    ):
        """
        Initialize frequency filter.

        Args:
            default_daily_cap: Default daily frequency cap
            default_hourly_cap: Default hourly frequency cap
        """
        settings = get_settings()
        self.default_daily_cap = default_daily_cap or settings.frequency.default_daily_cap
        self.default_hourly_cap = default_hourly_cap or settings.frequency.default_hourly_cap

    async def filter(
        self,
        candidates: list[AdCandidate],
        user_context: UserContext,
        **kwargs: Any,
    ) -> list[AdCandidate]:
        """Filter candidates by frequency cap."""
        if not candidates:
            return []

        # Skip if no user ID (can't do frequency control)
        if not user_context.user_id:
            return candidates

        # Batch get frequency info
        campaign_ids = list(set(c.campaign_id for c in candidates))
        freq_infos = await self._get_frequency_batch(
            user_context.user_id, campaign_ids
        )

        # Filter
        result = []
        for candidate in candidates:
            freq_info = freq_infos.get(candidate.campaign_id)
            if freq_info and not freq_info.is_capped:
                result.append(candidate)

        filtered_count = len(candidates) - len(result)
        if filtered_count > 0:
            logger.debug(
                f"Frequency filter removed {filtered_count} candidates",
                user_id=user_context.user_id,
            )

        return result

    async def filter_single(
        self,
        candidate: AdCandidate,
        user_context: UserContext,
        **kwargs: Any,
    ) -> bool:
        """Check if single candidate passes frequency filter."""
        if not user_context.user_id:
            return True

        freq_info = await self._get_frequency(
            user_context.user_id, candidate.campaign_id
        )
        return not freq_info.is_capped

    async def _get_frequency_batch(
        self,
        user_id: str,
        campaign_ids: list[int],
    ) -> dict[int, FrequencyInfo]:
        """Get frequency info for multiple campaigns."""
        result: dict[int, FrequencyInfo] = {}
        today = current_date()
        hour = current_hour()

        # Build keys
        daily_keys = [
            CacheKeys.freq_daily(user_id, cid, today) for cid in campaign_ids
        ]
        hourly_keys = [
            CacheKeys.freq_hourly(user_id, cid, hour) for cid in campaign_ids
        ]

        try:
            # Batch get from Redis
            pipeline = redis_client.pipeline()
            for key in daily_keys + hourly_keys:
                pipeline.get(key)

            values = await pipeline.execute()
            daily_values = values[: len(campaign_ids)]
            hourly_values = values[len(campaign_ids) :]

            for i, campaign_id in enumerate(campaign_ids):
                daily_count = int(daily_values[i]) if daily_values[i] else 0
                hourly_count = int(hourly_values[i]) if hourly_values[i] else 0

                result[campaign_id] = FrequencyInfo(
                    user_id=user_id,
                    campaign_id=campaign_id,
                    daily_count=daily_count,
                    hourly_count=hourly_count,
                    daily_cap=self.default_daily_cap,
                    hourly_cap=self.default_hourly_cap,
                )

        except Exception as e:
            logger.warning(f"Failed to get frequency from cache: {e}")
            # Return default (not capped)
            for campaign_id in campaign_ids:
                result[campaign_id] = FrequencyInfo(
                    user_id=user_id,
                    campaign_id=campaign_id,
                    daily_count=0,
                    hourly_count=0,
                    daily_cap=self.default_daily_cap,
                    hourly_cap=self.default_hourly_cap,
                )

        return result

    async def _get_frequency(
        self,
        user_id: str,
        campaign_id: int,
    ) -> FrequencyInfo:
        """Get frequency info for a single campaign."""
        result = await self._get_frequency_batch(user_id, [campaign_id])
        return result.get(
            campaign_id,
            FrequencyInfo(user_id=user_id, campaign_id=campaign_id),
        )

    async def increment(
        self,
        user_id: str,
        campaign_id: int,
    ) -> None:
        """
        Increment frequency counter after ad impression.

        Called after ad is shown to user.
        """
        today = current_date()
        hour = current_hour()

        daily_key = CacheKeys.freq_daily(user_id, campaign_id, today)
        hourly_key = CacheKeys.freq_hourly(user_id, campaign_id, hour)

        try:
            pipeline = redis_client.pipeline()
            pipeline.incr(daily_key)
            pipeline.expire(daily_key, 86400)  # 24 hours
            pipeline.incr(hourly_key)
            pipeline.expire(hourly_key, 3600)  # 1 hour
            await pipeline.execute()
        except Exception as e:
            logger.error(
                f"Failed to increment frequency for user {user_id}, "
                f"campaign {campaign_id}: {e}"
            )

    async def reset(
        self,
        user_id: str,
        campaign_id: int | None = None,
    ) -> None:
        """
        Reset frequency counter for user.

        Args:
            user_id: User identifier
            campaign_id: Optional campaign ID. If None, resets all.
        """
        today = current_date()
        hour = current_hour()

        if campaign_id:
            keys = [
                CacheKeys.freq_daily(user_id, campaign_id, today),
                CacheKeys.freq_hourly(user_id, campaign_id, hour),
            ]
        else:
            # Reset all (need pattern match - expensive)
            logger.warning("Resetting all frequency for user is expensive")
            return

        await redis_client.delete(*keys)
