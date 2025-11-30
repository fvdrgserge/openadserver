"""
Prometheus metrics middleware for monitoring.

Provides:
- Request latency histograms
- Request counters by endpoint
- Active request gauge
- Business metrics (impressions, clicks, etc.)
"""

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from liteads.common.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# Prometheus Metrics Definitions
# =============================================================================

# Application info
APP_INFO = Info("liteads_app", "LiteAds application information")
APP_INFO.info({
    "version": "1.0.0",
    "name": "liteads",
    "description": "Lightweight Ad Server",
})

# HTTP request metrics
HTTP_REQUEST_TOTAL = Counter(
    "liteads_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "liteads_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "liteads_http_requests_in_progress",
    "HTTP requests currently in progress",
    ["method", "endpoint"],
)

# Ad serving metrics
AD_REQUESTS_TOTAL = Counter(
    "liteads_ad_requests_total",
    "Total ad requests",
    ["slot_id", "status"],
)

AD_IMPRESSIONS_TOTAL = Counter(
    "liteads_ad_impressions_total",
    "Total ad impressions",
    ["campaign_id", "creative_id"],
)

AD_CLICKS_TOTAL = Counter(
    "liteads_ad_clicks_total",
    "Total ad clicks",
    ["campaign_id", "creative_id"],
)

AD_CONVERSIONS_TOTAL = Counter(
    "liteads_ad_conversions_total",
    "Total ad conversions",
    ["campaign_id", "creative_id"],
)

# Recommendation engine metrics
RETRIEVAL_LATENCY = Histogram(
    "liteads_retrieval_latency_seconds",
    "Retrieval stage latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
)

FILTER_LATENCY = Histogram(
    "liteads_filter_latency_seconds",
    "Filter stage latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
)

RANKING_LATENCY = Histogram(
    "liteads_ranking_latency_seconds",
    "Ranking stage latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
)

ML_PREDICTION_LATENCY = Histogram(
    "liteads_ml_prediction_latency_seconds",
    "ML prediction latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)

CANDIDATES_COUNT = Histogram(
    "liteads_candidates_count",
    "Number of candidates at each stage",
    ["stage"],
    buckets=(0, 10, 50, 100, 200, 500, 1000),
)

# Cache metrics
CACHE_HIT_TOTAL = Counter(
    "liteads_cache_hit_total",
    "Total cache hits",
    ["cache_type"],
)

CACHE_MISS_TOTAL = Counter(
    "liteads_cache_miss_total",
    "Total cache misses",
    ["cache_type"],
)

# Database metrics
DB_QUERY_LATENCY = Histogram(
    "liteads_db_query_latency_seconds",
    "Database query latency",
    ["query_type"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

# Model metrics
MODEL_VERSION = Gauge(
    "liteads_model_version",
    "Currently loaded model version",
    ["model_name"],
)


# =============================================================================
# Metrics Middleware
# =============================================================================

class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware that collects Prometheus metrics for all HTTP requests.
    """

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process request and record metrics."""
        method = request.method
        endpoint = self._get_endpoint(request)

        # Track in-progress requests
        HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()

        start_time = time.perf_counter()
        status_code = 500  # Default to error

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise
        finally:
            # Calculate duration
            duration = time.perf_counter() - start_time

            # Record metrics
            HTTP_REQUEST_TOTAL.labels(
                method=method,
                endpoint=endpoint,
                status=str(status_code),
            ).inc()

            HTTP_REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)

            HTTP_REQUESTS_IN_PROGRESS.labels(
                method=method,
                endpoint=endpoint,
            ).dec()

    def _get_endpoint(self, request: Request) -> str:
        """Get endpoint path, normalizing path parameters."""
        path = request.url.path

        # Normalize common path patterns
        # e.g., /api/v1/ad/123 -> /api/v1/ad/{id}
        parts = path.split("/")
        normalized = []
        for part in parts:
            if part.isdigit():
                normalized.append("{id}")
            else:
                normalized.append(part)

        return "/".join(normalized)


# =============================================================================
# Metrics Endpoint
# =============================================================================

async def metrics_endpoint() -> StarletteResponse:
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    return StarletteResponse(
        content=generate_latest(),
        media_type="text/plain; charset=utf-8",
    )


# =============================================================================
# Helper Functions for Recording Business Metrics
# =============================================================================

def record_ad_request(slot_id: str, success: bool) -> None:
    """Record an ad request."""
    status = "success" if success else "no_fill"
    AD_REQUESTS_TOTAL.labels(slot_id=slot_id, status=status).inc()


def record_impression(campaign_id: int, creative_id: int) -> None:
    """Record an ad impression."""
    AD_IMPRESSIONS_TOTAL.labels(
        campaign_id=str(campaign_id),
        creative_id=str(creative_id),
    ).inc()


def record_click(campaign_id: int, creative_id: int) -> None:
    """Record an ad click."""
    AD_CLICKS_TOTAL.labels(
        campaign_id=str(campaign_id),
        creative_id=str(creative_id),
    ).inc()


def record_conversion(campaign_id: int, creative_id: int) -> None:
    """Record an ad conversion."""
    AD_CONVERSIONS_TOTAL.labels(
        campaign_id=str(campaign_id),
        creative_id=str(creative_id),
    ).inc()


def record_retrieval_latency(duration: float) -> None:
    """Record retrieval stage latency."""
    RETRIEVAL_LATENCY.observe(duration)


def record_filter_latency(duration: float) -> None:
    """Record filter stage latency."""
    FILTER_LATENCY.observe(duration)


def record_ranking_latency(duration: float) -> None:
    """Record ranking stage latency."""
    RANKING_LATENCY.observe(duration)


def record_ml_prediction_latency(duration: float) -> None:
    """Record ML prediction latency."""
    ML_PREDICTION_LATENCY.observe(duration)


def record_candidates_count(stage: str, count: int) -> None:
    """Record candidate count at a stage."""
    CANDIDATES_COUNT.labels(stage=stage).observe(count)


def record_cache_hit(cache_type: str) -> None:
    """Record a cache hit."""
    CACHE_HIT_TOTAL.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str) -> None:
    """Record a cache miss."""
    CACHE_MISS_TOTAL.labels(cache_type=cache_type).inc()


def record_db_query_latency(query_type: str, duration: float) -> None:
    """Record database query latency."""
    DB_QUERY_LATENCY.labels(query_type=query_type).observe(duration)


def set_model_version(model_name: str, version: float) -> None:
    """Set the current model version."""
    MODEL_VERSION.labels(model_name=model_name).set(version)
