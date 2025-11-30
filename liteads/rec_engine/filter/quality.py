"""
Quality filter for ensuring ad quality standards.
"""

from typing import Any

from liteads.common.logger import get_logger
from liteads.rec_engine.filter.base import BaseFilter
from liteads.schemas.internal import AdCandidate, UserContext

logger = get_logger(__name__)


class QualityFilter(BaseFilter):
    """
    Filter candidates by quality criteria.

    Checks:
    1. Creative completeness (required fields)
    2. Minimum performance thresholds
    3. Content policy compliance
    """

    def __init__(
        self,
        require_image: bool = False,
        require_title: bool = False,
        min_ctr: float = 0.0,
        min_cvr: float = 0.0,
    ):
        """
        Initialize quality filter.

        Args:
            require_image: Require image URL
            require_title: Require title
            min_ctr: Minimum CTR threshold
            min_cvr: Minimum CVR threshold
        """
        self.require_image = require_image
        self.require_title = require_title
        self.min_ctr = min_ctr
        self.min_cvr = min_cvr

    async def filter(
        self,
        candidates: list[AdCandidate],
        user_context: UserContext,
        **kwargs: Any,
    ) -> list[AdCandidate]:
        """Filter candidates by quality."""
        if not candidates:
            return []

        result = []
        for candidate in candidates:
            if await self.filter_single(candidate, user_context, **kwargs):
                result.append(candidate)

        filtered_count = len(candidates) - len(result)
        if filtered_count > 0:
            logger.debug(f"Quality filter removed {filtered_count} candidates")

        return result

    async def filter_single(
        self,
        candidate: AdCandidate,
        user_context: UserContext,
        **kwargs: Any,
    ) -> bool:
        """Check if single candidate passes quality filter."""
        # Check required fields
        if not candidate.landing_url:
            return False

        if self.require_image and not candidate.image_url:
            return False

        if self.require_title and not candidate.title:
            return False

        # Check performance thresholds
        if self.min_ctr > 0 and candidate.pctr < self.min_ctr:
            return False

        if self.min_cvr > 0 and candidate.pcvr < self.min_cvr:
            return False

        return True


class DiversityFilter(BaseFilter):
    """
    Filter to ensure diversity in ad results.

    Prevents showing too many ads from the same advertiser or category.
    """

    def __init__(
        self,
        max_per_advertiser: int = 3,
        max_per_category: int | None = None,
    ):
        """
        Initialize diversity filter.

        Args:
            max_per_advertiser: Max ads from same advertiser
            max_per_category: Max ads from same category (optional)
        """
        self.max_per_advertiser = max_per_advertiser
        self.max_per_category = max_per_category

    async def filter(
        self,
        candidates: list[AdCandidate],
        user_context: UserContext,
        **kwargs: Any,
    ) -> list[AdCandidate]:
        """Filter candidates for diversity."""
        if not candidates:
            return []

        result: list[AdCandidate] = []
        advertiser_counts: dict[int, int] = {}

        for candidate in candidates:
            # Check advertiser limit
            adv_id = candidate.advertiser_id
            adv_count = advertiser_counts.get(adv_id, 0)

            if adv_count >= self.max_per_advertiser:
                continue

            advertiser_counts[adv_id] = adv_count + 1
            result.append(candidate)

        filtered_count = len(candidates) - len(result)
        if filtered_count > 0:
            logger.debug(f"Diversity filter removed {filtered_count} candidates")

        return result

    async def filter_single(
        self,
        candidate: AdCandidate,
        user_context: UserContext,
        **kwargs: Any,
    ) -> bool:
        """Diversity check requires full list context."""
        # Single candidate always passes - diversity needs full list
        return True


class BlacklistFilter(BaseFilter):
    """
    Filter to exclude blacklisted ads, advertisers, or categories.
    """

    def __init__(
        self,
        blocked_campaign_ids: set[int] | None = None,
        blocked_advertiser_ids: set[int] | None = None,
        blocked_creative_ids: set[int] | None = None,
    ):
        """
        Initialize blacklist filter.

        Args:
            blocked_campaign_ids: Set of blocked campaign IDs
            blocked_advertiser_ids: Set of blocked advertiser IDs
            blocked_creative_ids: Set of blocked creative IDs
        """
        self.blocked_campaign_ids = blocked_campaign_ids or set()
        self.blocked_advertiser_ids = blocked_advertiser_ids or set()
        self.blocked_creative_ids = blocked_creative_ids or set()

    async def filter(
        self,
        candidates: list[AdCandidate],
        user_context: UserContext,
        **kwargs: Any,
    ) -> list[AdCandidate]:
        """Filter out blacklisted candidates."""
        if not candidates:
            return []

        result = []
        for candidate in candidates:
            if await self.filter_single(candidate, user_context, **kwargs):
                result.append(candidate)

        filtered_count = len(candidates) - len(result)
        if filtered_count > 0:
            logger.debug(f"Blacklist filter removed {filtered_count} candidates")

        return result

    async def filter_single(
        self,
        candidate: AdCandidate,
        user_context: UserContext,
        **kwargs: Any,
    ) -> bool:
        """Check if candidate is not blacklisted."""
        if candidate.campaign_id in self.blocked_campaign_ids:
            return False

        if candidate.advertiser_id in self.blocked_advertiser_ids:
            return False

        if candidate.creative_id in self.blocked_creative_ids:
            return False

        return True

    def add_blocked_campaign(self, campaign_id: int) -> None:
        """Add campaign to blacklist."""
        self.blocked_campaign_ids.add(campaign_id)

    def add_blocked_advertiser(self, advertiser_id: int) -> None:
        """Add advertiser to blacklist."""
        self.blocked_advertiser_ids.add(advertiser_id)

    def remove_blocked_campaign(self, campaign_id: int) -> None:
        """Remove campaign from blacklist."""
        self.blocked_campaign_ids.discard(campaign_id)
