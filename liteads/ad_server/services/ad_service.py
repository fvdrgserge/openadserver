"""
Ad serving service.

Handles the core ad serving logic using the recommendation engine.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from liteads.common.config import get_settings
from liteads.common.logger import get_logger
from liteads.common.utils import hash_user_id
from liteads.rec_engine import RecommendationConfig, RecommendationEngine
from liteads.rec_engine.ranking.bidding import RankingStrategy
from liteads.schemas.internal import AdCandidate, UserContext
from liteads.schemas.request import AdRequest

logger = get_logger(__name__)


class AdService:
    """Ad serving service using recommendation engine."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.settings = get_settings()
        self._engine: RecommendationEngine | None = None

    @property
    def engine(self) -> RecommendationEngine:
        """Lazy initialization of recommendation engine."""
        if self._engine is None:
            config = RecommendationConfig(
                max_retrieval=100,
                enable_budget_filter=True,
                enable_frequency_filter=True,
                enable_quality_filter=True,
                enable_ml_prediction=self.settings.ad_serving.enable_ml_prediction,
                fallback_ctr=0.01,
                fallback_cvr=0.001,
                ranking_strategy=RankingStrategy.ECPM,
                enable_diversity_rerank=True,
                enable_exploration=True,
                exploration_epsilon=0.1,
            )
            self._engine = RecommendationEngine(
                session=self.session,
                config=config,
            )
        return self._engine

    async def serve_ads(
        self,
        request: AdRequest,
        request_id: str,
    ) -> list[AdCandidate]:
        """
        Main ad serving method.

        Uses the recommendation engine to:
        1. Retrieve candidate ads
        2. Filter by targeting, budget, frequency
        3. Predict CTR/CVR
        4. Rank and re-rank
        5. Return top candidates
        """
        # Build user context
        user_context = self._build_user_context(request)

        # Get recommendations
        candidates, metrics = await self.engine.recommend(
            user_context=user_context,
            slot_id=request.slot_id,
            num_ads=request.num_ads,
        )

        logger.info(
            "Ad serving completed",
            request_id=request_id,
            retrieval_count=metrics.retrieval_count,
            final_count=metrics.final_count,
            total_ms=round(metrics.total_ms, 2),
        )

        return candidates

    def _build_user_context(self, request: AdRequest) -> UserContext:
        """Build user context from request."""
        ctx = UserContext(
            user_id=request.user_id,
            user_hash=hash_user_id(request.user_id) if request.user_id else 0,
        )

        # Device info
        if request.device:
            ctx.os = request.device.os
            ctx.os_version = request.device.os_version or ""
            ctx.device_model = request.device.model or ""
            ctx.device_brand = request.device.brand or ""

        # Geo info
        if request.geo:
            ctx.ip = request.geo.ip or ""
            ctx.country = request.geo.country or ""
            ctx.region = request.geo.region or ""
            ctx.city = request.geo.city or ""
            ctx.latitude = request.geo.latitude
            ctx.longitude = request.geo.longitude

        # Context info
        if request.context:
            ctx.app_id = request.context.app_id or ""
            ctx.app_name = request.context.app_name or ""
            ctx.network = request.context.network or ""
            ctx.carrier = request.context.carrier or ""

        # User features
        if request.user_features:
            ctx.age = request.user_features.age
            ctx.gender = request.user_features.gender
            ctx.interests = request.user_features.interests or []
            ctx.app_categories = request.user_features.app_categories or []
            ctx.custom_features = request.user_features.custom or {}

        return ctx

    async def refresh_cache(self) -> None:
        """Refresh recommendation engine cache."""
        await self.engine.refresh_cache()
