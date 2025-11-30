"""
Tests for filter modules.
"""

import pytest

from liteads.rec_engine.filter.base import CompositeFilter, PassThroughFilter
from liteads.rec_engine.filter.quality import BlacklistFilter, DiversityFilter, QualityFilter
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
            image_url="https://example.com/img1.jpg",
            landing_url="https://example.com/1",
        ),
        AdCandidate(
            campaign_id=2,
            creative_id=201,
            advertiser_id=2,
            bid=3.0,
            bid_type=1,
            title="",  # Empty title
            landing_url="https://example.com/2",
        ),
        AdCandidate(
            campaign_id=3,
            creative_id=301,
            advertiser_id=1,
            bid=4.0,
            bid_type=1,
            title="Ad 3",
            landing_url="",  # Empty landing URL
        ),
    ]


@pytest.fixture
def sample_user_context() -> UserContext:
    """Create sample user context."""
    return UserContext(user_id="test_user_123")


class TestPassThroughFilter:
    """Tests for pass-through filter."""

    @pytest.mark.asyncio
    async def test_passes_all(
        self,
        sample_candidates: list[AdCandidate],
        sample_user_context: UserContext,
    ) -> None:
        """Test that pass-through filter passes all candidates."""
        filter = PassThroughFilter()
        result = await filter.filter(sample_candidates, sample_user_context)
        assert len(result) == len(sample_candidates)


class TestQualityFilter:
    """Tests for quality filter."""

    @pytest.mark.asyncio
    async def test_filters_missing_landing_url(
        self,
        sample_candidates: list[AdCandidate],
        sample_user_context: UserContext,
    ) -> None:
        """Test filtering candidates without landing URL."""
        filter = QualityFilter()
        result = await filter.filter(sample_candidates, sample_user_context)

        # Should filter out candidate with empty landing_url
        assert len(result) == 2
        assert all(c.landing_url for c in result)

    @pytest.mark.asyncio
    async def test_require_title(
        self,
        sample_candidates: list[AdCandidate],
        sample_user_context: UserContext,
    ) -> None:
        """Test filtering candidates without title when required."""
        filter = QualityFilter(require_title=True)
        result = await filter.filter(sample_candidates, sample_user_context)

        # Should filter out candidates without title
        assert all(c.title for c in result)

    @pytest.mark.asyncio
    async def test_require_image(
        self,
        sample_candidates: list[AdCandidate],
        sample_user_context: UserContext,
    ) -> None:
        """Test filtering candidates without image when required."""
        filter = QualityFilter(require_image=True)
        result = await filter.filter(sample_candidates, sample_user_context)

        # Should only pass candidates with image
        assert all(c.image_url for c in result)


class TestDiversityFilter:
    """Tests for diversity filter."""

    @pytest.mark.asyncio
    async def test_max_per_advertiser(
        self,
        sample_user_context: UserContext,
    ) -> None:
        """Test max ads per advertiser limit."""
        candidates = [
            AdCandidate(
                campaign_id=i,
                creative_id=i * 100,
                advertiser_id=1,  # All same advertiser
                bid=5.0,
                bid_type=1,
                landing_url=f"https://example.com/{i}",
            )
            for i in range(5)
        ]

        filter = DiversityFilter(max_per_advertiser=2)
        result = await filter.filter(candidates, sample_user_context)

        assert len(result) == 2


class TestBlacklistFilter:
    """Tests for blacklist filter."""

    @pytest.mark.asyncio
    async def test_blocks_campaigns(
        self,
        sample_candidates: list[AdCandidate],
        sample_user_context: UserContext,
    ) -> None:
        """Test blocking specific campaigns."""
        filter = BlacklistFilter(blocked_campaign_ids={1, 2})
        result = await filter.filter(sample_candidates, sample_user_context)

        # Should filter out campaigns 1 and 2
        campaign_ids = {c.campaign_id for c in result}
        assert 1 not in campaign_ids
        assert 2 not in campaign_ids

    @pytest.mark.asyncio
    async def test_blocks_advertisers(
        self,
        sample_candidates: list[AdCandidate],
        sample_user_context: UserContext,
    ) -> None:
        """Test blocking specific advertisers."""
        filter = BlacklistFilter(blocked_advertiser_ids={1})
        result = await filter.filter(sample_candidates, sample_user_context)

        # Should filter out all ads from advertiser 1
        assert all(c.advertiser_id != 1 for c in result)


class TestCompositeFilter:
    """Tests for composite filter."""

    @pytest.mark.asyncio
    async def test_chains_filters(
        self,
        sample_candidates: list[AdCandidate],
        sample_user_context: UserContext,
    ) -> None:
        """Test that composite filter chains multiple filters."""
        filter1 = QualityFilter()  # Filters empty landing_url
        filter2 = BlacklistFilter(blocked_campaign_ids={1})

        composite = CompositeFilter([filter1, filter2])
        result = await composite.filter(sample_candidates, sample_user_context)

        # Should apply both filters
        assert all(c.landing_url for c in result)
        assert all(c.campaign_id != 1 for c in result)

    @pytest.mark.asyncio
    async def test_empty_result_stops_early(
        self,
        sample_user_context: UserContext,
    ) -> None:
        """Test that composite filter stops when no candidates remain."""
        candidates = [
            AdCandidate(
                campaign_id=1,
                creative_id=101,
                advertiser_id=1,
                bid=5.0,
                bid_type=1,
                landing_url="",  # Will be filtered
            )
        ]

        filter1 = QualityFilter()
        filter2 = PassThroughFilter()  # Should not be called

        composite = CompositeFilter([filter1, filter2])
        result = await composite.filter(candidates, sample_user_context)

        assert len(result) == 0
