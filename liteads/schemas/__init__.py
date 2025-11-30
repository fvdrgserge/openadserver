"""
Pydantic schemas for API requests and responses.
"""

from liteads.schemas.internal import (
    AdCandidate,
    BudgetInfo,
    FeatureVector,
    FrequencyInfo,
    PredictionResult,
    UserContext,
)
from liteads.schemas.request import (
    AdRequest,
    ContextInfo,
    DeviceInfo,
    EventRequest,
    GeoInfo,
    UserFeatures,
)
from liteads.schemas.response import (
    AdListResponse,
    AdResponse,
    CreativeResponse,
    ErrorResponse,
    EventResponse,
    HealthResponse,
    TrackingUrls,
)

__all__ = [
    # Request schemas
    "AdRequest",
    "EventRequest",
    "DeviceInfo",
    "GeoInfo",
    "ContextInfo",
    "UserFeatures",
    # Response schemas
    "AdResponse",
    "AdListResponse",
    "EventResponse",
    "HealthResponse",
    "ErrorResponse",
    "CreativeResponse",
    "TrackingUrls",
    # Internal schemas
    "AdCandidate",
    "UserContext",
    "FeatureVector",
    "PredictionResult",
    "FrequencyInfo",
    "BudgetInfo",
]
