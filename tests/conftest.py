"""
Pytest configuration and fixtures.
"""

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from liteads.ad_server.main import app
from liteads.common.database import Base, get_session


# Test database URL (use SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def test_db() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=NullPool,
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(test_db: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create a test client with database override."""

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        yield test_db

    app.dependency_overrides[get_session] = override_get_session

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def sample_ad_request() -> dict[str, Any]:
    """Sample ad request data."""
    return {
        "slot_id": "test_slot",
        "user_id": "test_user_123",
        "device": {
            "os": "android",
            "os_version": "13.0",
            "model": "Pixel 7",
            "brand": "Google",
            "screen_width": 1080,
            "screen_height": 2400,
        },
        "geo": {
            "ip": "1.2.3.4",
            "country": "CN",
            "city": "shanghai",
        },
        "context": {
            "app_id": "com.test.app",
            "app_version": "1.0.0",
            "network": "wifi",
        },
        "num_ads": 1,
    }


@pytest.fixture
def sample_event_request() -> dict[str, Any]:
    """Sample event request data."""
    return {
        "request_id": "test_request_123",
        "ad_id": "ad_100_200",
        "event_type": "impression",
        "user_id": "test_user_123",
        "timestamp": 1700000000,
    }
