"""
Health check endpoints.
"""

from fastapi import APIRouter

from liteads.common.cache import redis_client
from liteads.common.config import get_settings
from liteads.common.database import db
from liteads.schemas.response import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns service status and dependency health.
    """
    settings = get_settings()

    # Check database
    db_healthy = await db.health_check()

    # Check Redis
    redis_healthy = await redis_client.health_check()

    status = "healthy" if (db_healthy and redis_healthy) else "degraded"

    return HealthResponse(
        status=status,
        version=settings.app_version,
        database=db_healthy,
        redis=redis_healthy,
    )


@router.get("/ping")
async def ping() -> dict:
    """Simple ping endpoint."""
    return {"pong": True}


@router.get("/ready")
async def readiness_check() -> dict:
    """Readiness check for Kubernetes."""
    db_healthy = await db.health_check()
    redis_healthy = await redis_client.health_check()

    if not db_healthy or not redis_healthy:
        return {"ready": False, "reason": "Dependencies not ready"}

    return {"ready": True}


@router.get("/live")
async def liveness_check() -> dict:
    """Liveness check for Kubernetes."""
    return {"alive": True}
