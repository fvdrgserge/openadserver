"""
Recommendation Engine - Main orchestrator for ad recommendation.

Coordinates retrieval, filtering, prediction, ranking, and re-ranking.
"""

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from liteads.common.logger import get_logger
from liteads.common.utils import Timer
from liteads.rec_engine.filter.base import BaseFilter, CompositeFilter
from liteads.rec_engine.filter.budget import BudgetFilter
from liteads.rec_engine.filter.frequency import FrequencyFilter
from liteads.rec_engine.filter.quality import DiversityFilter, QualityFilter
from liteads.rec_engine.ranking.bidding import Bidding, RankingStrategy
from liteads.rec_engine.ranking.predictor import BasePredictor, StatisticalPredictor
from liteads.rec_engine.ranking.reranker import (
    BaseReranker,
    CompositeReranker,
    DiversityReranker,
    ExplorationReranker,
)
from liteads.rec_engine.retrieval.base import BaseRetrieval
from liteads.rec_engine.retrieval.targeting import TargetingRetrieval
from liteads.schemas.internal import AdCandidate, UserContext

logger = get_logger(__name__)


@dataclass
class RecommendationMetrics:
    """Metrics for a recommendation request."""

    retrieval_count: int = 0
    post_filter_count: int = 0
    post_ranking_count: int = 0
    final_count: int = 0

    retrieval_ms: float = 0.0
    filter_ms: float = 0.0
    prediction_ms: float = 0.0
    ranking_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class RecommendationConfig:
    """Configuration for recommendation engine."""

    # Retrieval
    max_retrieval: int = 100

    # Filtering
    enable_budget_filter: bool = True
    enable_frequency_filter: bool = True
    enable_quality_filter: bool = True

    # Prediction
    enable_ml_prediction: bool = False
    fallback_ctr: float = 0.01
    fallback_cvr: float = 0.001

    # Ranking
    ranking_strategy: RankingStrategy = RankingStrategy.ECPM
    min_ecpm: float = 0.01

    # Re-ranking
    enable_diversity_rerank: bool = True
    enable_exploration: bool = True
    exploration_epsilon: float = 0.1
    diversity_lambda: float = 0.7


class RecommendationEngine:
    """
    Main recommendation engine.

    Pipeline:
    1. Retrieval - Get candidate ads
    2. Filtering - Remove ineligible ads
    3. Prediction - Predict CTR/CVR
    4. Ranking - Calculate scores and sort
    5. Re-ranking - Apply business rules and diversity
    """

    def __init__(
        self,
        session: AsyncSession,
        config: RecommendationConfig | None = None,
        retrieval: BaseRetrieval | None = None,
        filters: list[BaseFilter] | None = None,
        predictor: BasePredictor | None = None,
        bidding: Bidding | None = None,
        rerankers: list[BaseReranker] | None = None,
    ):
        """
        Initialize recommendation engine.

        Args:
            session: Database session
            config: Engine configuration
            retrieval: Custom retrieval module
            filters: Custom filter list
            predictor: Custom predictor
            bidding: Custom bidding module
            rerankers: Custom re-ranker list
        """
        self.session = session
        self.config = config or RecommendationConfig()

        # Initialize components
        self.retrieval = retrieval or TargetingRetrieval(session)
        self.filters = filters or self._create_default_filters()
        self.predictor = predictor or self._create_default_predictor()
        self.bidding = bidding or Bidding(
            strategy=self.config.ranking_strategy,
            min_ecpm=self.config.min_ecpm,
        )
        self.rerankers = rerankers or self._create_default_rerankers()

        # Composite components
        self._filter_chain = CompositeFilter(self.filters) if self.filters else None
        self._reranker_chain = (
            CompositeReranker(self.rerankers) if self.rerankers else None
        )

    def _create_default_filters(self) -> list[BaseFilter]:
        """Create default filter chain."""
        filters: list[BaseFilter] = []

        if self.config.enable_budget_filter:
            filters.append(BudgetFilter())

        if self.config.enable_frequency_filter:
            filters.append(FrequencyFilter())

        if self.config.enable_quality_filter:
            filters.append(QualityFilter())

        return filters

    def _create_default_predictor(self) -> BasePredictor:
        """Create default predictor."""
        return StatisticalPredictor(
            default_ctr=self.config.fallback_ctr,
            default_cvr=self.config.fallback_cvr,
        )

    def _create_default_rerankers(self) -> list[BaseReranker]:
        """Create default re-ranker chain."""
        rerankers: list[BaseReranker] = []

        if self.config.enable_diversity_rerank:
            rerankers.append(
                DiversityReranker(lambda_param=self.config.diversity_lambda)
            )

        if self.config.enable_exploration:
            rerankers.append(
                ExplorationReranker(epsilon=self.config.exploration_epsilon)
            )

        return rerankers

    async def recommend(
        self,
        user_context: UserContext,
        slot_id: str,
        num_ads: int = 1,
        **kwargs: Any,
    ) -> tuple[list[AdCandidate], RecommendationMetrics]:
        """
        Get ad recommendations for user.

        Args:
            user_context: User context information
            slot_id: Ad slot identifier
            num_ads: Number of ads to return
            **kwargs: Additional parameters

        Returns:
            Tuple of (recommended ads, metrics)
        """
        metrics = RecommendationMetrics()

        with Timer() as total_timer:
            # 1. Retrieval
            with Timer() as retrieval_timer:
                candidates = await self.retrieval.retrieve(
                    user_context=user_context,
                    slot_id=slot_id,
                    limit=self.config.max_retrieval,
                )
            metrics.retrieval_ms = retrieval_timer.elapsed_ms
            metrics.retrieval_count = len(candidates)

            logger.debug(
                f"Retrieved {len(candidates)} candidates",
                slot_id=slot_id,
            )

            if not candidates:
                return [], metrics

            # 2. Filtering
            with Timer() as filter_timer:
                if self._filter_chain:
                    candidates = await self._filter_chain.filter(
                        candidates, user_context
                    )
            metrics.filter_ms = filter_timer.elapsed_ms
            metrics.post_filter_count = len(candidates)

            logger.debug(f"After filtering: {len(candidates)} candidates")

            if not candidates:
                return [], metrics

            # 3. Prediction
            with Timer() as prediction_timer:
                predictions = await self.predictor.predict_batch(
                    user_context, candidates
                )
                for candidate, pred in zip(candidates, predictions):
                    candidate.pctr = pred.pctr
                    candidate.pcvr = pred.pcvr
            metrics.prediction_ms = prediction_timer.elapsed_ms

            # 4. Ranking
            with Timer() as ranking_timer:
                candidates = self.bidding.rank(candidates)
            metrics.ranking_ms = ranking_timer.elapsed_ms
            metrics.post_ranking_count = len(candidates)

            logger.debug(f"After ranking: top score = {candidates[0].score:.4f}")

            # 5. Re-ranking
            with Timer() as rerank_timer:
                if self._reranker_chain:
                    candidates = self._reranker_chain.rerank(
                        candidates,
                        user_context,
                        num_results=num_ads * 2,  # Get more for diversity
                    )
            metrics.rerank_ms = rerank_timer.elapsed_ms

            # Final selection
            final_candidates = candidates[:num_ads]
            metrics.final_count = len(final_candidates)

        metrics.total_ms = total_timer.elapsed_ms

        logger.info(
            "Recommendation completed",
            retrieval=metrics.retrieval_count,
            final=metrics.final_count,
            total_ms=round(metrics.total_ms, 2),
        )

        return final_candidates, metrics

    async def refresh_cache(self) -> None:
        """Refresh all caches."""
        await self.retrieval.refresh()
        logger.info("Recommendation engine cache refreshed")


def create_engine(
    session: AsyncSession,
    enable_ml: bool = False,
    ranking_strategy: RankingStrategy = RankingStrategy.ECPM,
) -> RecommendationEngine:
    """
    Factory function to create recommendation engine.

    Args:
        session: Database session
        enable_ml: Whether to enable ML prediction
        ranking_strategy: Ranking strategy to use

    Returns:
        Configured RecommendationEngine
    """
    config = RecommendationConfig(
        enable_ml_prediction=enable_ml,
        ranking_strategy=ranking_strategy,
    )

    return RecommendationEngine(session=session, config=config)
