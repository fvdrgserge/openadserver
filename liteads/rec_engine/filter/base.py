"""
Base filter class for candidate filtering.
"""

from abc import ABC, abstractmethod
from typing import Any

from liteads.schemas.internal import AdCandidate, UserContext


class BaseFilter(ABC):
    """
    Abstract base class for ad filters.

    Filters remove ineligible candidates from the recommendation pipeline.
    """

    @abstractmethod
    async def filter(
        self,
        candidates: list[AdCandidate],
        user_context: UserContext,
        **kwargs: Any,
    ) -> list[AdCandidate]:
        """
        Filter candidates based on specific criteria.

        Args:
            candidates: List of ad candidates to filter
            user_context: User context information
            **kwargs: Additional filter parameters

        Returns:
            Filtered list of candidates
        """
        pass

    @abstractmethod
    async def filter_single(
        self,
        candidate: AdCandidate,
        user_context: UserContext,
        **kwargs: Any,
    ) -> bool:
        """
        Check if a single candidate passes the filter.

        Args:
            candidate: Single ad candidate
            user_context: User context information
            **kwargs: Additional filter parameters

        Returns:
            True if candidate passes filter, False otherwise
        """
        pass


class CompositeFilter(BaseFilter):
    """
    Composite filter that chains multiple filters.

    Applies filters in sequence, passing results from one to the next.
    """

    def __init__(self, filters: list[BaseFilter]):
        self.filters = filters

    async def filter(
        self,
        candidates: list[AdCandidate],
        user_context: UserContext,
        **kwargs: Any,
    ) -> list[AdCandidate]:
        """Apply all filters in sequence."""
        result = candidates

        for f in self.filters:
            if not result:
                break
            result = await f.filter(result, user_context, **kwargs)

        return result

    async def filter_single(
        self,
        candidate: AdCandidate,
        user_context: UserContext,
        **kwargs: Any,
    ) -> bool:
        """Check if candidate passes all filters."""
        for f in self.filters:
            if not await f.filter_single(candidate, user_context, **kwargs):
                return False
        return True


class PassThroughFilter(BaseFilter):
    """Filter that passes all candidates through (for testing)."""

    async def filter(
        self,
        candidates: list[AdCandidate],
        user_context: UserContext,
        **kwargs: Any,
    ) -> list[AdCandidate]:
        return candidates

    async def filter_single(
        self,
        candidate: AdCandidate,
        user_context: UserContext,
        **kwargs: Any,
    ) -> bool:
        return True
