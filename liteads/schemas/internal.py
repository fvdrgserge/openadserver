"""
Internal data schemas for passing data between modules.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdCandidate:
    """Ad candidate for ranking."""

    campaign_id: int
    creative_id: int
    advertiser_id: int
    bid: float
    bid_type: int

    # Targeting match info
    targeting_score: float = 1.0

    # Predicted scores
    pctr: float = 0.0  # Predicted CTR
    pcvr: float = 0.0  # Predicted CVR

    # Calculated scores
    ecpm: float = 0.0  # Effective CPM
    score: float = 0.0  # Final ranking score

    # Creative info
    title: str | None = None
    description: str | None = None
    image_url: str | None = None
    video_url: str | None = None
    landing_url: str = ""
    creative_type: int = 1
    width: int | None = None
    height: int | None = None

    # Extra info
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserContext:
    """User context for ad serving."""

    user_id: str | None = None
    user_hash: int = 0  # Hash for bucketing

    # Device
    os: str = ""
    os_version: str = ""
    device_model: str = ""
    device_brand: str = ""

    # Geo
    ip: str = ""
    country: str = ""
    region: str = ""
    city: str = ""
    latitude: float | None = None
    longitude: float | None = None

    # Context
    app_id: str = ""
    app_name: str = ""
    network: str = ""
    carrier: str = ""

    # Features (for ML)
    age: int | None = None
    gender: str | None = None
    interests: list[str] = field(default_factory=list)
    app_categories: list[str] = field(default_factory=list)
    custom_features: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureVector:
    """Feature vector for ML prediction."""

    # Sparse features (categorical)
    sparse_features: dict[str, int] = field(default_factory=dict)
    # {feature_name: feature_index}

    # Dense features (numerical)
    dense_features: list[float] = field(default_factory=list)

    # Feature names (for debugging)
    feature_names: list[str] = field(default_factory=list)


@dataclass
class PredictionResult:
    """ML prediction result."""

    campaign_id: int
    creative_id: int
    pctr: float = 0.0
    pcvr: float = 0.0
    model_version: str = ""
    latency_ms: float = 0.0


@dataclass
class FrequencyInfo:
    """Frequency control information."""

    user_id: str
    campaign_id: int
    daily_count: int = 0
    hourly_count: int = 0
    daily_cap: int | None = None
    hourly_cap: int | None = None

    @property
    def is_capped(self) -> bool:
        """Check if frequency cap is reached."""
        if self.daily_cap and self.daily_count >= self.daily_cap:
            return True
        if self.hourly_cap and self.hourly_count >= self.hourly_cap:
            return True
        return False


@dataclass
class BudgetInfo:
    """Budget information."""

    campaign_id: int
    budget_daily: float | None = None
    budget_total: float | None = None
    spent_today: float = 0.0
    spent_total: float = 0.0

    @property
    def remaining_daily(self) -> float | None:
        """Get remaining daily budget."""
        if self.budget_daily is None:
            return None
        return max(0.0, self.budget_daily - self.spent_today)

    @property
    def remaining_total(self) -> float | None:
        """Get remaining total budget."""
        if self.budget_total is None:
            return None
        return max(0.0, self.budget_total - self.spent_total)

    @property
    def has_budget(self) -> bool:
        """Check if campaign has remaining budget."""
        if self.budget_daily and self.spent_today >= self.budget_daily:
            return False
        if self.budget_total and self.spent_total >= self.budget_total:
            return False
        return True
