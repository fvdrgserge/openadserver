"""
Tests for ad serving endpoints.
"""

from typing import Any

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_ad_request_empty(
    client: AsyncClient,
    sample_ad_request: dict[str, Any],
) -> None:
    """Test ad request with empty database returns empty ads."""
    response = await client.post("/api/v1/ad/request", json=sample_ad_request)

    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert data["ads"] == []
    assert data["count"] == 0


@pytest.mark.asyncio
async def test_ad_request_validation_error(client: AsyncClient) -> None:
    """Test ad request with invalid data returns validation error."""
    # Missing required field 'slot_id'
    invalid_request = {
        "device": {"os": "android"},
    }

    response = await client.post("/api/v1/ad/request", json=invalid_request)
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_ad_request_with_user_features(
    client: AsyncClient,
    sample_ad_request: dict[str, Any],
) -> None:
    """Test ad request with user features."""
    sample_ad_request["user_features"] = {
        "age": 25,
        "gender": "male",
        "interests": ["sports", "technology"],
    }

    response = await client.post("/api/v1/ad/request", json=sample_ad_request)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
