"""
Custom exceptions for LiteAds.
"""

from typing import Any


class LiteAdsError(Exception):
    """Base exception for LiteAds."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ConfigError(LiteAdsError):
    """Configuration related errors."""

    pass


class DatabaseError(LiteAdsError):
    """Database related errors."""

    pass


class CacheError(LiteAdsError):
    """Cache (Redis) related errors."""

    pass


class ValidationError(LiteAdsError):
    """Request validation errors."""

    pass


class AdNotFoundError(LiteAdsError):
    """No ads found for the request."""

    pass


class AdFilteredError(LiteAdsError):
    """All ads were filtered out."""

    pass


class FrequencyCapError(LiteAdsError):
    """Frequency cap exceeded."""

    pass


class BudgetExhaustedError(LiteAdsError):
    """Campaign budget exhausted."""

    pass


class ModelNotFoundError(LiteAdsError):
    """ML model not found."""

    pass


class ModelPredictionError(LiteAdsError):
    """ML model prediction failed."""

    pass


class TimeoutError(LiteAdsError):
    """Operation timed out."""

    pass


class RateLimitError(LiteAdsError):
    """Rate limit exceeded."""

    pass
