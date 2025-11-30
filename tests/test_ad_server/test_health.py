"""
Tests for health check endpoints.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_ping(client: AsyncClient) -> None:
    """Test ping endpoint."""
    response = await client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"pong": True}


@pytest.mark.asyncio
async def test_live(client: AsyncClient) -> None:
    """Test liveness endpoint."""
    response = await client.get("/live")
    assert response.status_code == 200
    assert response.json() == {"alive": True}
