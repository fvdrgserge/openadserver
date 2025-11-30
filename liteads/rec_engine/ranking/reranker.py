"""
Re-ranking module for post-processing ranked candidates.

Applies business rules and optimization after initial ranking.
"""

from abc import ABC, abstractmethod
import random
from typing import Any

from liteads.common.logger import get_logger
from liteads.schemas.internal import AdCandidate, UserContext

logger = get_logger(__name__)


class BaseReranker(ABC):
    """Abstract base class for re-rankers."""

    @abstractmethod
    def rerank(
        self,
        candidates: list[AdCandidate],
        user_context: UserContext,
        **kwargs: Any,
    ) -> list[AdCandidate]:
        """Re-rank candidates."""
        pass


class DiversityReranker(BaseReranker):
    """
    Re-ranker for diversity in ad results.

    Uses Maximal Marginal Relevance (MMR) to balance
    relevance and diversity.
    """

    def __init__(
        self,
        lambda_param: float = 0.7,
        max_per_advertiser: int = 2,
    ):
        """
        Initialize diversity re-ranker.

        Args:
            lambda_param: Balance between relevance (1) and diversity (0)
            max_per_advertiser: Max ads from same advertiser in results
        """
        self.lambda_param = lambda_param
        self.max_per_advertiser = max_per_advertiser

    def rerank(
        self,
        candidates: list[AdCandidate],
        user_context: UserContext,
        num_results: int = 10,
        **kwargs: Any,
    ) -> list[AdCandidate]:
        """
        Re-rank using MMR-like algorithm.

        Selects ads that are both high-scoring and diverse.
        """
        if not candidates or num_results <= 0:
            return []

        result: list[AdCandidate] = []
        remaining = list(candidates)
        advertiser_counts: dict[int, int] = {}

        while len(result) < num_results and remaining:
            best_idx = -1
            best_score = -float("inf")

            for i, candidate in enumerate(remaining):
                # Check advertiser limit
                adv_count = advertiser_counts.get(candidate.advertiser_id, 0)
                if adv_count >= self.max_per_advertiser:
                    continue

                # Calculate MMR score
                relevance = candidate.score

                # Diversity: penalty for similarity to already selected
                diversity_penalty = 0.0
                for selected in result:
                    if selected.advertiser_id == candidate.advertiser_id:
                        diversity_penalty += 0.5
                    if selected.creative_type == candidate.creative_type:
                        diversity_penalty += 0.2

                mmr_score = (
                    self.lambda_param * relevance
                    - (1 - self.lambda_param) * diversity_penalty
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected = remaining.pop(best_idx)
                result.append(selected)
                advertiser_counts[selected.advertiser_id] = (
                    advertiser_counts.get(selected.advertiser_id, 0) + 1
                )
            else:
                break

        return result


class ExplorationReranker(BaseReranker):
    """
    Re-ranker with exploration for new/cold ads.

    Uses epsilon-greedy or Thompson Sampling to balance
    exploitation (showing top ads) and exploration (testing new ads).
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        new_ad_boost: float = 1.5,
        min_impressions_for_stable: int = 1000,
    ):
        """
        Initialize exploration re-ranker.

        Args:
            epsilon: Probability of exploration
            new_ad_boost: Score multiplier for new ads
            min_impressions_for_stable: Min impressions to consider ad stable
        """
        self.epsilon = epsilon
        self.new_ad_boost = new_ad_boost
        self.min_impressions_for_stable = min_impressions_for_stable

    def rerank(
        self,
        candidates: list[AdCandidate],
        user_context: UserContext,
        **kwargs: Any,
    ) -> list[AdCandidate]:
        """
        Re-rank with exploration.

        With probability epsilon, shuffle to explore.
        Otherwise, boost new ads slightly.
        """
        if not candidates:
            return []

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Explore: shuffle candidates
            shuffled = list(candidates)
            random.shuffle(shuffled)
            logger.debug("Exploration: shuffled candidates")
            return shuffled

        # Exploitation with new ad boost
        result = []
        for candidate in candidates:
            impressions = candidate.metadata.get("impressions", 0)
            if impressions < self.min_impressions_for_stable:
                # New ad: apply boost
                candidate.score *= self.new_ad_boost

            result.append(candidate)

        # Re-sort by adjusted score
        result.sort(key=lambda c: c.score, reverse=True)

        return result


class BusinessRulesReranker(BaseReranker):
    """
    Re-ranker applying business rules.

    Applies various business constraints and optimizations.
    """

    def __init__(
        self,
        boost_rules: list[dict[str, Any]] | None = None,
        penalty_rules: list[dict[str, Any]] | None = None,
    ):
        """
        Initialize business rules re-ranker.

        Args:
            boost_rules: Rules for boosting certain ads
            penalty_rules: Rules for penalizing certain ads
        """
        self.boost_rules = boost_rules or []
        self.penalty_rules = penalty_rules or []

    def rerank(
        self,
        candidates: list[AdCandidate],
        user_context: UserContext,
        **kwargs: Any,
    ) -> list[AdCandidate]:
        """Apply business rules to re-rank candidates."""
        if not candidates:
            return []

        for candidate in candidates:
            multiplier = 1.0

            # Apply boost rules
            for rule in self.boost_rules:
                if self._match_rule(rule, candidate, user_context):
                    multiplier *= rule.get("boost", 1.2)

            # Apply penalty rules
            for rule in self.penalty_rules:
                if self._match_rule(rule, candidate, user_context):
                    multiplier *= rule.get("penalty", 0.8)

            candidate.score *= multiplier

        # Re-sort
        candidates.sort(key=lambda c: c.score, reverse=True)

        return candidates

    def _match_rule(
        self,
        rule: dict[str, Any],
        candidate: AdCandidate,
        user_context: UserContext,
    ) -> bool:
        """Check if rule matches candidate."""
        conditions = rule.get("conditions", {})

        # Check advertiser
        if "advertiser_id" in conditions:
            if candidate.advertiser_id != conditions["advertiser_id"]:
                return False

        # Check campaign
        if "campaign_id" in conditions:
            if candidate.campaign_id != conditions["campaign_id"]:
                return False

        # Check creative type
        if "creative_type" in conditions:
            if candidate.creative_type != conditions["creative_type"]:
                return False

        # Check user attributes
        if "user_os" in conditions:
            if user_context.os != conditions["user_os"]:
                return False

        if "user_country" in conditions:
            if user_context.country != conditions["user_country"]:
                return False

        return True


class CompositeReranker(BaseReranker):
    """
    Composite re-ranker chaining multiple re-rankers.
    """

    def __init__(self, rerankers: list[BaseReranker]):
        self.rerankers = rerankers

    def rerank(
        self,
        candidates: list[AdCandidate],
        user_context: UserContext,
        **kwargs: Any,
    ) -> list[AdCandidate]:
        """Apply all re-rankers in sequence."""
        result = candidates

        for reranker in self.rerankers:
            result = reranker.rerank(result, user_context, **kwargs)

        return result
