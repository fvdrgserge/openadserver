"""
Bidding and ranking module.

Implements various bidding strategies and ranking algorithms.
"""

from enum import IntEnum
from typing import Any

from liteads.common.logger import get_logger
from liteads.models import BidType
from liteads.schemas.internal import AdCandidate

logger = get_logger(__name__)


class RankingStrategy(IntEnum):
    """Ranking strategy enum."""

    ECPM = 1  # eCPM ranking (standard)
    REVENUE = 2  # Revenue optimization
    ENGAGEMENT = 3  # Click optimization
    CONVERSION = 4  # Conversion optimization
    HYBRID = 5  # Hybrid strategy


class Bidding:
    """
    Bidding and ranking calculator.

    Calculates eCPM and ranks candidates based on bidding strategy.
    """

    def __init__(
        self,
        strategy: RankingStrategy = RankingStrategy.ECPM,
        min_ecpm: float = 0.01,
    ):
        """
        Initialize bidding module.

        Args:
            strategy: Ranking strategy to use
            min_ecpm: Minimum eCPM threshold
        """
        self.strategy = strategy
        self.min_ecpm = min_ecpm

    def calculate_ecpm(self, candidate: AdCandidate) -> float:
        """
        Calculate eCPM for a candidate.

        eCPM = Effective Cost Per Mille (1000 impressions)

        For different bid types:
        - CPM: eCPM = bid
        - CPC: eCPM = bid × pCTR × 1000
        - CPA: eCPM = bid × pCTR × pCVR × 1000
        - OCPM: eCPM = bid × pCTR × 1000 (optimized)
        """
        bid = candidate.bid
        pctr = max(candidate.pctr, 0.0001)  # Avoid division by zero
        pcvr = max(candidate.pcvr, 0.0001)

        if candidate.bid_type == BidType.CPM:
            ecpm = bid
        elif candidate.bid_type == BidType.CPC:
            ecpm = bid * pctr * 1000
        elif candidate.bid_type == BidType.CPA:
            ecpm = bid * pctr * pcvr * 1000
        elif candidate.bid_type == BidType.OCPM:
            # OCPM: Optimized CPM, uses expected value
            ecpm = bid * pctr * 1000
        else:
            ecpm = bid * pctr * 1000  # Default to CPC-like

        return max(ecpm, self.min_ecpm)

    def calculate_score(self, candidate: AdCandidate) -> float:
        """
        Calculate ranking score based on strategy.

        Different strategies optimize for different objectives:
        - ECPM: Pure revenue optimization
        - REVENUE: Similar to ECPM with adjustments
        - ENGAGEMENT: Prioritize high CTR ads
        - CONVERSION: Prioritize high CVR ads
        - HYBRID: Balance of multiple factors
        """
        ecpm = self.calculate_ecpm(candidate)
        pctr = candidate.pctr
        pcvr = candidate.pcvr

        if self.strategy == RankingStrategy.ECPM:
            score = ecpm

        elif self.strategy == RankingStrategy.REVENUE:
            # Revenue with quality adjustment
            quality_factor = min(pctr / 0.01, 2.0)  # Boost high quality
            score = ecpm * quality_factor

        elif self.strategy == RankingStrategy.ENGAGEMENT:
            # Prioritize CTR
            score = ecpm * (1 + pctr * 10)

        elif self.strategy == RankingStrategy.CONVERSION:
            # Prioritize CVR
            score = ecpm * (1 + pcvr * 100)

        elif self.strategy == RankingStrategy.HYBRID:
            # Balanced scoring
            ctr_factor = 1 + pctr * 5
            cvr_factor = 1 + pcvr * 20
            score = ecpm * ctr_factor * cvr_factor

        else:
            score = ecpm

        return score

    def rank(
        self,
        candidates: list[AdCandidate],
        apply_ecpm: bool = True,
    ) -> list[AdCandidate]:
        """
        Rank candidates by score.

        Args:
            candidates: List of candidates to rank
            apply_ecpm: Whether to calculate and apply eCPM

        Returns:
            Sorted list of candidates (highest score first)
        """
        if not candidates:
            return []

        for candidate in candidates:
            if apply_ecpm:
                candidate.ecpm = self.calculate_ecpm(candidate)
            candidate.score = self.calculate_score(candidate)

        # Sort by score descending
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)

        logger.debug(
            f"Ranked {len(ranked)} candidates",
            top_score=ranked[0].score if ranked else 0,
        )

        return ranked


class SecondPriceAuction:
    """
    Second price auction implementation.

    Winner pays the second highest bid (plus small increment).
    """

    def __init__(self, increment: float = 0.01):
        """
        Initialize second price auction.

        Args:
            increment: Small increment above second price
        """
        self.increment = increment

    def run_auction(
        self,
        candidates: list[AdCandidate],
    ) -> tuple[AdCandidate | None, float]:
        """
        Run second price auction.

        Args:
            candidates: Sorted list of candidates (highest eCPM first)

        Returns:
            Tuple of (winner, price_to_pay)
        """
        if not candidates:
            return None, 0.0

        if len(candidates) == 1:
            # Only one bidder, pay minimum
            winner = candidates[0]
            return winner, self.increment

        # Winner is the highest bidder
        winner = candidates[0]

        # Price is second highest bid + increment
        second_price = candidates[1].ecpm
        price = second_price + self.increment

        return winner, price


class BudgetPacing:
    """
    Budget pacing for smooth ad delivery.

    Ensures budget is spent evenly throughout the day.
    """

    def __init__(
        self,
        daily_budget: float,
        hours_remaining: int = 24,
        smoothing_factor: float = 1.2,
    ):
        """
        Initialize budget pacing.

        Args:
            daily_budget: Total daily budget
            hours_remaining: Hours remaining in the day
            smoothing_factor: Factor to adjust pacing (>1 = aggressive)
        """
        self.daily_budget = daily_budget
        self.hours_remaining = max(hours_remaining, 1)
        self.smoothing_factor = smoothing_factor

    def get_hourly_budget(self, spent_today: float) -> float:
        """
        Get recommended hourly budget.

        Args:
            spent_today: Amount spent today so far

        Returns:
            Recommended hourly budget
        """
        remaining_budget = max(0, self.daily_budget - spent_today)
        ideal_hourly = remaining_budget / self.hours_remaining

        # Apply smoothing factor
        return ideal_hourly * self.smoothing_factor

    def should_serve(
        self,
        candidate: AdCandidate,
        spent_this_hour: float,
        hourly_budget: float,
    ) -> bool:
        """
        Determine if ad should be served based on pacing.

        Args:
            candidate: Ad candidate
            spent_this_hour: Amount spent this hour
            hourly_budget: Budget for this hour

        Returns:
            True if ad should be served
        """
        if spent_this_hour >= hourly_budget:
            return False

        # Use probabilistic pacing
        remaining_ratio = (hourly_budget - spent_this_hour) / hourly_budget
        return remaining_ratio > 0.1  # Serve if >10% budget remaining

    def adjust_bid(
        self,
        bid: float,
        spent_today: float,
        target_spend: float,
    ) -> float:
        """
        Adjust bid based on pacing status.

        Args:
            bid: Original bid
            spent_today: Amount spent today
            target_spend: Target spend by this time

        Returns:
            Adjusted bid
        """
        if target_spend <= 0:
            return bid

        pacing_ratio = spent_today / target_spend

        if pacing_ratio < 0.8:
            # Under-pacing: increase bid
            return bid * 1.2
        elif pacing_ratio > 1.2:
            # Over-pacing: decrease bid
            return bid * 0.8
        else:
            # On track
            return bid
