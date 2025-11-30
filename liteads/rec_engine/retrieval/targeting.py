"""
Targeting-based retrieval.

Retrieves ads based on targeting rules matching user attributes.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from liteads.common.cache import CacheKeys, redis_client
from liteads.common.logger import get_logger
from liteads.common.utils import json_dumps, json_loads
from liteads.models import Campaign, Creative, Status, TargetingRule
from liteads.rec_engine.retrieval.base import BaseRetrieval
from liteads.schemas.internal import AdCandidate, UserContext

logger = get_logger(__name__)


class TargetingRetrieval(BaseRetrieval):
    """
    Retrieval based on targeting rules.

    Matches user attributes against campaign targeting rules to find
    eligible ads.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self._cache_ttl = 300  # 5 minutes

    async def retrieve(
        self,
        user_context: UserContext,
        slot_id: str,
        limit: int = 100,
        **kwargs: Any,
    ) -> list[AdCandidate]:
        """
        Retrieve candidates matching targeting rules.

        Flow:
        1. Get all active campaigns with creatives
        2. For each campaign, check targeting rules
        3. Return matching candidates
        """
        # Get active campaigns (with caching)
        campaigns = await self._get_active_campaigns()

        if not campaigns:
            logger.debug("No active campaigns found")
            return []

        candidates: list[AdCandidate] = []

        for campaign_data in campaigns:
            # Check if campaign matches user targeting
            if not self._match_targeting(campaign_data, user_context):
                continue

            # Create candidate for each creative
            for creative_data in campaign_data.get("creatives", []):
                candidate = AdCandidate(
                    campaign_id=campaign_data["id"],
                    creative_id=creative_data["id"],
                    advertiser_id=campaign_data["advertiser_id"],
                    bid=campaign_data["bid_amount"],
                    bid_type=campaign_data["bid_type"],
                    title=creative_data.get("title"),
                    description=creative_data.get("description"),
                    image_url=creative_data.get("image_url"),
                    video_url=creative_data.get("video_url"),
                    landing_url=creative_data.get("landing_url", ""),
                    creative_type=creative_data.get("creative_type", 1),
                    width=creative_data.get("width"),
                    height=creative_data.get("height"),
                )
                candidates.append(candidate)

                if len(candidates) >= limit:
                    break

            if len(candidates) >= limit:
                break

        logger.debug(f"Retrieved {len(candidates)} candidates from targeting")
        return candidates

    async def _get_active_campaigns(self) -> list[dict[str, Any]]:
        """Get all active campaigns with creatives and targeting rules."""
        # Try cache first
        cache_key = CacheKeys.active_ads()
        cached = await redis_client.get(cache_key)

        if cached:
            try:
                return json_loads(cached)
            except Exception:
                pass

        # Query from database
        stmt = (
            select(Campaign)
            .where(Campaign.status == Status.ACTIVE)
            .limit(1000)
        )

        result = await self.session.execute(stmt)
        campaigns = result.scalars().all()

        campaign_list: list[dict[str, Any]] = []

        for campaign in campaigns:
            if not campaign.is_active:
                continue

            campaign_data: dict[str, Any] = {
                "id": campaign.id,
                "advertiser_id": campaign.advertiser_id,
                "name": campaign.name,
                "bid_type": campaign.bid_type,
                "bid_amount": float(campaign.bid_amount),
                "budget_daily": float(campaign.budget_daily) if campaign.budget_daily else None,
                "budget_total": float(campaign.budget_total) if campaign.budget_total else None,
                "spent_today": float(campaign.spent_today),
                "spent_total": float(campaign.spent_total),
                "freq_cap_daily": campaign.freq_cap_daily,
                "freq_cap_hourly": campaign.freq_cap_hourly,
                "creatives": [],
                "targeting_rules": [],
            }

            # Add creatives
            for creative in campaign.creatives:
                if creative.status == Status.ACTIVE:
                    campaign_data["creatives"].append({
                        "id": creative.id,
                        "title": creative.title,
                        "description": creative.description,
                        "image_url": creative.image_url,
                        "video_url": creative.video_url,
                        "landing_url": creative.landing_url,
                        "creative_type": creative.creative_type,
                        "width": creative.width,
                        "height": creative.height,
                    })

            # Add targeting rules
            for rule in campaign.targeting_rules:
                campaign_data["targeting_rules"].append({
                    "rule_type": rule.rule_type,
                    "rule_value": rule.rule_value,
                    "is_include": rule.is_include,
                })

            if campaign_data["creatives"]:  # Only add if has active creatives
                campaign_list.append(campaign_data)

        # Cache the result
        if campaign_list:
            await redis_client.set(
                cache_key,
                json_dumps(campaign_list),
                ttl=self._cache_ttl,
            )

        return campaign_list

    def _match_targeting(
        self,
        campaign_data: dict[str, Any],
        user_context: UserContext,
    ) -> bool:
        """Check if user matches campaign targeting rules."""
        targeting_rules = campaign_data.get("targeting_rules", [])

        if not targeting_rules:
            return True  # No targeting = match all

        for rule in targeting_rules:
            rule_type = rule["rule_type"]
            rule_value = rule["rule_value"]
            is_include = rule["is_include"]

            matched = self._match_rule(rule_type, rule_value, user_context)

            # Include rule: must match
            # Exclude rule: must not match
            if is_include and not matched:
                return False
            if not is_include and matched:
                return False

        return True

    def _match_rule(
        self,
        rule_type: str,
        rule_value: dict[str, Any],
        user_context: UserContext,
    ) -> bool:
        """Match a single targeting rule against user context."""
        if rule_type == "age":
            if user_context.age is None:
                return True  # Unknown age matches
            min_age = rule_value.get("min", 0)
            max_age = rule_value.get("max", 999)
            return min_age <= user_context.age <= max_age

        elif rule_type == "gender":
            if user_context.gender is None:
                return True
            values = rule_value.get("values", [])
            return user_context.gender.lower() in [v.lower() for v in values]

        elif rule_type == "geo":
            countries = rule_value.get("countries", [])
            cities = rule_value.get("cities", [])

            if countries and user_context.country:
                if user_context.country.upper() not in [c.upper() for c in countries]:
                    return False

            if cities and user_context.city:
                if user_context.city.lower() not in [c.lower() for c in cities]:
                    return False

            return True

        elif rule_type == "device":
            device_types = rule_value.get("types", [])
            # Simplified device type detection
            if user_context.device_model:
                model_lower = user_context.device_model.lower()
                if "tablet" in model_lower or "pad" in model_lower:
                    device_type = "tablet"
                else:
                    device_type = "phone"

                if device_types and device_type not in device_types:
                    return False

            return True

        elif rule_type == "os":
            os_values = rule_value.get("values", [])
            if os_values and user_context.os:
                if user_context.os.lower() not in [v.lower() for v in os_values]:
                    return False
            return True

        elif rule_type == "interest":
            interests = rule_value.get("values", [])
            if interests and user_context.interests:
                # Match if user has any of the target interests
                user_interests_lower = [i.lower() for i in user_context.interests]
                target_interests_lower = [i.lower() for i in interests]
                if not any(i in user_interests_lower for i in target_interests_lower):
                    return False
            return True

        elif rule_type == "app_category":
            categories = rule_value.get("values", [])
            if categories and user_context.app_categories:
                user_cats_lower = [c.lower() for c in user_context.app_categories]
                target_cats_lower = [c.lower() for c in categories]
                if not any(c in user_cats_lower for c in target_cats_lower):
                    return False
            return True

        # Unknown rule type - default match
        return True

    async def refresh(self) -> None:
        """Clear cache to force refresh."""
        cache_key = CacheKeys.active_ads()
        await redis_client.delete(cache_key)
        logger.info("Targeting retrieval cache refreshed")
