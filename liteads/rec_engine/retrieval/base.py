"""
Base retrieval class for candidate generation.
"""

from abc import ABC, abstractmethod
from typing import Any

from liteads.schemas.internal import AdCandidate, UserContext


class BaseRetrieval(ABC):
    """
    Abstract base class for ad retrieval strategies.

    Retrieval is the first stage of the recommendation pipeline,
    responsible for generating a set of candidate ads from the full inventory.
    """

    @abstractmethod
    async def retrieve(
        self,
        user_context: UserContext,
        slot_id: str,
        limit: int = 100,
        **kwargs: Any,
    ) -> list[AdCandidate]:
        """
        Retrieve candidate ads for the given user and slot.

        Args:
            user_context: User context information
            slot_id: Ad slot identifier
            limit: Maximum number of candidates to retrieve
            **kwargs: Additional retrieval parameters

        Returns:
            List of ad candidates
        """
        pass

    @abstractmethod
    async def refresh(self) -> None:
        """Refresh retrieval index/cache."""
        pass


class CompositeRetrieval(BaseRetrieval):
    """
    Composite retrieval that combines multiple retrieval strategies.

    Merges results from multiple retrievers and deduplicates.
    """

    def __init__(self, retrievers: list[BaseRetrieval]):
        self.retrievers = retrievers

    async def retrieve(
        self,
        user_context: UserContext,
        slot_id: str,
        limit: int = 100,
        **kwargs: Any,
    ) -> list[AdCandidate]:
        """Retrieve candidates from all retrievers and merge."""
        all_candidates: list[AdCandidate] = []
        seen_ids: set[tuple[int, int]] = set()

        for retriever in self.retrievers:
            candidates = await retriever.retrieve(
                user_context=user_context,
                slot_id=slot_id,
                limit=limit,
                **kwargs,
            )

            for candidate in candidates:
                key = (candidate.campaign_id, candidate.creative_id)
                if key not in seen_ids:
                    seen_ids.add(key)
                    all_candidates.append(candidate)

        return all_candidates[:limit]

    async def refresh(self) -> None:
        """Refresh all retrievers."""
        for retriever in self.retrievers:
            await retriever.refresh()
