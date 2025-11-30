#!/usr/bin/env python3
"""
Database initialization script.

Creates tables and sets up initial data for LiteAds.

Usage:
    python scripts/init_db.py [--drop-existing] [--seed]
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from liteads.common.config import get_settings
from liteads.common.database import engine, get_session
from liteads.common.logger import get_logger
from liteads.models.base import Base

logger = get_logger(__name__)


async def create_tables(drop_existing: bool = False) -> None:
    """Create all database tables."""
    from sqlalchemy import text

    async with engine.begin() as conn:
        if drop_existing:
            logger.warning("Dropping existing tables...")
            await conn.run_sync(Base.metadata.drop_all)

        logger.info("Creating database tables...")
        await conn.run_sync(Base.metadata.create_all)

        # Create indexes for performance
        logger.info("Creating additional indexes...")

        # Index for campaign status and time
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_campaigns_status_time
            ON campaigns (status, start_time, end_time)
        """))

        # Index for targeting rules lookup
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_targeting_rules_campaign
            ON targeting_rules (campaign_id, is_include)
        """))

        # Index for hourly stats
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_hourly_stats_time
            ON hourly_stats (stat_hour DESC, campaign_id)
        """))

    logger.info("Database tables created successfully")


async def seed_data() -> None:
    """Seed initial data for development/testing."""
    from datetime import datetime, timedelta

    from liteads.models.ad import Advertiser, Campaign, Creative, TargetingRule

    logger.info("Seeding initial data...")

    async with get_session() as session:
        # Check if data already exists
        from sqlalchemy import select
        result = await session.execute(select(Advertiser).limit(1))
        if result.scalar():
            logger.info("Data already exists, skipping seed")
            return

        # Create advertisers
        advertisers = [
            Advertiser(
                name="Demo Advertiser 1",
                balance=10000.0,
                daily_budget=1000.0,
                status=1,
            ),
            Advertiser(
                name="Demo Advertiser 2",
                balance=5000.0,
                daily_budget=500.0,
                status=1,
            ),
            Advertiser(
                name="Demo Advertiser 3",
                balance=20000.0,
                daily_budget=2000.0,
                status=1,
            ),
        ]

        for adv in advertisers:
            session.add(adv)
        await session.flush()

        logger.info(f"Created {len(advertisers)} advertisers")

        # Create campaigns
        now = datetime.utcnow()
        campaigns = []

        campaign_configs = [
            {
                "name": "Game Install Campaign",
                "advertiser_id": advertisers[0].id,
                "budget_daily": 500.0,
                "budget_total": 5000.0,
                "bid_type": 2,  # CPC
                "bid_amount": 2.5,
                "category": "game",
            },
            {
                "name": "E-commerce Sale",
                "advertiser_id": advertisers[0].id,
                "budget_daily": 300.0,
                "budget_total": 3000.0,
                "bid_type": 1,  # CPM
                "bid_amount": 10.0,
                "category": "ecom",
            },
            {
                "name": "Finance App",
                "advertiser_id": advertisers[1].id,
                "budget_daily": 200.0,
                "budget_total": 2000.0,
                "bid_type": 3,  # CPA
                "bid_amount": 50.0,
                "category": "finance",
            },
            {
                "name": "Education Course",
                "advertiser_id": advertisers[2].id,
                "budget_daily": 400.0,
                "budget_total": 4000.0,
                "bid_type": 2,  # CPC
                "bid_amount": 1.5,
                "category": "education",
            },
            {
                "name": "Video Streaming",
                "advertiser_id": advertisers[2].id,
                "budget_daily": 600.0,
                "budget_total": 6000.0,
                "bid_type": 1,  # CPM
                "bid_amount": 15.0,
                "category": "entertainment",
            },
        ]

        for config in campaign_configs:
            campaign = Campaign(
                name=config["name"],
                advertiser_id=config["advertiser_id"],
                budget_daily=config["budget_daily"],
                budget_total=config["budget_total"],
                bid_type=config["bid_type"],
                bid_amount=config["bid_amount"],
                start_time=now - timedelta(days=1),
                end_time=now + timedelta(days=30),
                status=1,
            )
            session.add(campaign)
            campaigns.append((campaign, config))

        await session.flush()
        logger.info(f"Created {len(campaigns)} campaigns")

        # Create creatives for each campaign
        creative_count = 0
        for campaign, config in campaigns:
            creatives_data = [
                {
                    "title": f"{config['name']} - Banner Ad",
                    "description": f"Check out {config['name']}!",
                    "image_url": f"https://example.com/images/{campaign.id}_banner.jpg",
                    "landing_url": f"https://example.com/landing/{campaign.id}",
                    "creative_type": 1,  # Banner
                },
                {
                    "title": f"{config['name']} - Native Ad",
                    "description": f"Discover {config['name']} today",
                    "image_url": f"https://example.com/images/{campaign.id}_native.jpg",
                    "landing_url": f"https://example.com/landing/{campaign.id}",
                    "creative_type": 2,  # Native
                },
            ]

            for data in creatives_data:
                creative = Creative(
                    campaign_id=campaign.id,
                    title=data["title"],
                    description=data["description"],
                    image_url=data["image_url"],
                    landing_url=data["landing_url"],
                    creative_type=data["creative_type"],
                    status=1,
                )
                session.add(creative)
                creative_count += 1

        await session.flush()
        logger.info(f"Created {creative_count} creatives")

        # Create targeting rules
        targeting_count = 0
        for campaign, config in campaigns:
            # Age targeting
            if config["category"] == "game":
                age_rule = TargetingRule(
                    campaign_id=campaign.id,
                    rule_type="age",
                    rule_value={"min": 18, "max": 35},
                    is_include=True,
                )
                session.add(age_rule)
                targeting_count += 1

            # Device targeting
            device_rule = TargetingRule(
                campaign_id=campaign.id,
                rule_type="device",
                rule_value={"os": ["android", "ios"]},
                is_include=True,
            )
            session.add(device_rule)
            targeting_count += 1

            # Geo targeting (default to all)
            geo_rule = TargetingRule(
                campaign_id=campaign.id,
                rule_type="geo",
                rule_value={"countries": ["CN"]},
                is_include=True,
            )
            session.add(geo_rule)
            targeting_count += 1

        await session.commit()
        logger.info(f"Created {targeting_count} targeting rules")

    logger.info("Database seeding completed successfully")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Initialize LiteAds database")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing tables before creating",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed initial demo data",
    )

    args = parser.parse_args()

    settings = get_settings()
    logger.info(f"Initializing database: {settings.database.host}:{settings.database.port}")

    await create_tables(drop_existing=args.drop_existing)

    if args.seed:
        await seed_data()

    logger.info("Database initialization complete!")


if __name__ == "__main__":
    asyncio.run(main())
