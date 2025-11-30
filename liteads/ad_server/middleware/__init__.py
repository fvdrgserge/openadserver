"""
Middleware for ad server.
"""

from liteads.ad_server.middleware.metrics import (
    MetricsMiddleware,
    metrics_endpoint,
    record_ad_request,
    record_cache_hit,
    record_cache_miss,
    record_candidates_count,
    record_click,
    record_conversion,
    record_db_query_latency,
    record_filter_latency,
    record_impression,
    record_ml_prediction_latency,
    record_ranking_latency,
    record_retrieval_latency,
    set_model_version,
)

__all__ = [
    "MetricsMiddleware",
    "metrics_endpoint",
    "record_ad_request",
    "record_impression",
    "record_click",
    "record_conversion",
    "record_retrieval_latency",
    "record_filter_latency",
    "record_ranking_latency",
    "record_ml_prediction_latency",
    "record_candidates_count",
    "record_cache_hit",
    "record_cache_miss",
    "record_db_query_latency",
    "set_model_version",
]
