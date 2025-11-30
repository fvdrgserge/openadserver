"""
Tests for recommendation engine.
"""

import pytest

from liteads.rec_engine import RecommendationConfig, RecommendationMetrics
from liteads.rec_engine.filter.base import PassThroughFilter
from liteads.rec_engine.ranking.bidding import Bidding, RankingStrategy
from liteads.rec_engine.ranking.predictor import StatisticalPredictor
from liteads.rec_engine.ranking.reranker import DiversityReranker
from liteads.schemas.internal import AdCandidate, UserContext


@pytest.fixture
def sample_candidates() -> list[AdCandidate]:
    """Create sample ad candidates."""
    return [
        AdCandidate(
            campaign_id=1,
            creative_id=101,
            advertiser_id=1,
            bid=5.0,
            bid_type=1,
            title="Ad 1",
            landing_url="https://example.com/1",
            pctr=0.02,
        ),
        AdCandidate(
            campaign_id=2,
            creative_id=201,
            advertiser_id=2,
            bid=3.0,
            bid_type=1,
            title="Ad 2",
            landing_url="https://example.com/2",
            pctr=0.03,
        ),
        AdCandidate(
            campaign_id=3,
            creative_id=301,
            advertiser_id=1,
            bid=4.0,
            bid_type=1,
            title="Ad 3",
            landing_url="https://example.com/3",
            pctr=0.015,
        ),
    ]


@pytest.fixture
def sample_user_context() -> UserContext:
    """Create sample user context."""
    return UserContext(
        user_id="test_user_123",
        os="android",
        os_version="13.0",
        country="CN",
        city="shanghai",
        age=25,
        gender="male",
    )


class TestBidding:
    """Tests for bidding module."""

    def test_calculate_ecpm_cpm(self, sample_candidates: list[AdCandidate]) -> None:
        """Test eCPM calculation for CPM bid type."""
        bidding = Bidding()
        candidate = sample_candidates[0]
        candidate.bid_type = 1  # CPM

        ecpm = bidding.calculate_ecpm(candidate)
        assert ecpm == candidate.bid

    def test_calculate_ecpm_cpc(self, sample_candidates: list[AdCandidate]) -> None:
        """Test eCPM calculation for CPC bid type."""
        bidding = Bidding()
        candidate = sample_candidates[0]
        candidate.bid_type = 2  # CPC
        candidate.pctr = 0.02

        ecpm = bidding.calculate_ecpm(candidate)
        expected = candidate.bid * candidate.pctr * 1000
        assert abs(ecpm - expected) < 0.01

    def test_rank_candidates(self, sample_candidates: list[AdCandidate]) -> None:
        """Test candidate ranking."""
        bidding = Bidding(strategy=RankingStrategy.ECPM)
        ranked = bidding.rank(sample_candidates)

        assert len(ranked) == len(sample_candidates)
        # Should be sorted by score descending
        for i in range(len(ranked) - 1):
            assert ranked[i].score >= ranked[i + 1].score


class TestPredictor:
    """Tests for predictor module."""

    @pytest.mark.asyncio
    async def test_statistical_predictor(
        self,
        sample_user_context: UserContext,
        sample_candidates: list[AdCandidate],
    ) -> None:
        """Test statistical predictor."""
        predictor = StatisticalPredictor(default_ctr=0.01, default_cvr=0.001)

        results = await predictor.predict_batch(sample_user_context, sample_candidates)

        assert len(results) == len(sample_candidates)
        for result in results:
            assert result.pctr > 0
            assert result.pcvr > 0
            assert result.model_version == "statistical_v1"


class TestReranker:
    """Tests for reranker module."""

    def test_diversity_reranker(
        self,
        sample_user_context: UserContext,
        sample_candidates: list[AdCandidate],
    ) -> None:
        """Test diversity reranker."""
        reranker = DiversityReranker(max_per_advertiser=1)

        # Set scores for ranking
        for i, c in enumerate(sample_candidates):
            c.score = 100 - i * 10

        reranked = reranker.rerank(
            sample_candidates,
            sample_user_context,
            num_results=3,
        )

        # Should limit to 1 per advertiser
        advertiser_ids = [c.advertiser_id for c in reranked]
        # Check max per advertiser is respected
        from collections import Counter
        counts = Counter(advertiser_ids)
        assert all(count <= 1 for count in counts.values())


class TestRecommendationConfig:
    """Tests for recommendation config."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RecommendationConfig()

        assert config.max_retrieval == 100
        assert config.enable_budget_filter is True
        assert config.enable_frequency_filter is True
        assert config.fallback_ctr == 0.01

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = RecommendationConfig(
            max_retrieval=50,
            enable_ml_prediction=True,
            ranking_strategy=RankingStrategy.REVENUE,
        )

        assert config.max_retrieval == 50
        assert config.enable_ml_prediction is True
        assert config.ranking_strategy == RankingStrategy.REVENUE


class TestRecommendationMetrics:
    """Tests for recommendation metrics."""

    def test_metrics_initialization(self) -> None:
        """Test metrics default values."""
        metrics = RecommendationMetrics()

        assert metrics.retrieval_count == 0
        assert metrics.final_count == 0
        assert metrics.total_ms == 0.0

    def test_metrics_update(self) -> None:
        """Test metrics can be updated."""
        metrics = RecommendationMetrics()
        metrics.retrieval_count = 100
        metrics.post_filter_count = 80
        metrics.final_count = 3
        metrics.total_ms = 25.5

        assert metrics.retrieval_count == 100
        assert metrics.final_count == 3
