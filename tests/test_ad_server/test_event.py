"""
Tests for event tracking endpoints.
"""

from typing import Any

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_track_event_post(
    client: AsyncClient,
    sample_event_request: dict[str, Any],
) -> None:
    """Test event tracking via POST."""
    response = await client.post("/api/v1/event/track", json=sample_event_request)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


@pytest.mark.asyncio
async def test_track_event_get(client: AsyncClient) -> None:
    """Test event tracking via GET (pixel tracking)."""
    response = await client.get(
        "/api/v1/event/track",
        params={
            "type": "impression",
            "req": "test_request_123",
            "ad": "ad_100_200",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


@pytest.mark.asyncio
async def test_track_click_event(
    client: AsyncClient,
    sample_event_request: dict[str, Any],
) -> None:
    """Test click event tracking."""
    sample_event_request["event_type"] = "click"

    response = await client.post("/api/v1/event/track", json=sample_event_request)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


@pytest.mark.asyncio
async def test_track_conversion_event(
    client: AsyncClient,
    sample_event_request: dict[str, Any],
) -> None:
    """Test conversion event tracking."""
    sample_event_request["event_type"] = "conversion"
    sample_event_request["extra"] = {"value": 100.0}

    response = await client.post("/api/v1/event/track", json=sample_event_request)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
