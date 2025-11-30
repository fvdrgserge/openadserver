#!/usr/bin/env python3
"""
Generate mock data for testing and development.
"""

import asyncio
import random
from decimal import Decimal

from liteads.common.database import db, get_session
from liteads.models import Advertiser, BidType, Campaign, Creative, CreativeType, Status


async def generate_mock_data(
    num_advertisers: int = 10,
    campaigns_per_advertiser: int = 5,
    creatives_per_campaign: int = 3,
) -> None:
    """Generate mock advertisers, campaigns, and creatives."""

    await db.init()

    async with db.session() as session:
        print(f"Generating {num_advertisers} advertisers...")

        for i in range(num_advertisers):
            # Create advertiser
            advertiser = Advertiser(
                name=f"Advertiser {i + 1}",
                company=f"Company {i + 1} Ltd.",
                contact_email=f"contact{i + 1}@example.com",
                balance=Decimal(str(random.randint(1000, 100000))),
                status=Status.ACTIVE,
            )
            session.add(advertiser)
            await session.flush()

            print(f"  Created advertiser: {advertiser.name} (ID: {advertiser.id})")

            # Create campaigns
            for j in range(campaigns_per_advertiser):
                campaign = Campaign(
                    advertiser_id=advertiser.id,
                    name=f"Campaign {i + 1}-{j + 1}",
                    description=f"Description for campaign {j + 1}",
                    budget_daily=Decimal(str(random.randint(50, 500))),
                    budget_total=Decimal(str(random.randint(1000, 10000))),
                    bid_type=random.choice([BidType.CPM, BidType.CPC]),
                    bid_amount=Decimal(str(random.uniform(0.5, 10.0))).quantize(
                        Decimal("0.0001")
                    ),
                    freq_cap_daily=random.choice([3, 5, 10]),
                    freq_cap_hourly=random.choice([1, 2]),
                    status=Status.ACTIVE,
                )
                session.add(campaign)
                await session.flush()

                # Create creatives
                for k in range(creatives_per_campaign):
                    creative_type = random.choice(
                        [CreativeType.BANNER, CreativeType.NATIVE]
                    )
                    width, height = random.choice(
                        [(300, 250), (728, 90), (320, 50), (300, 600)]
                    )

                    creative = Creative(
                        campaign_id=campaign.id,
                        title=f"Ad Title {i + 1}-{j + 1}-{k + 1}",
                        description=f"This is an amazing product! Click to learn more.",
                        image_url=f"https://via.placeholder.com/{width}x{height}",
                        landing_url=f"https://example.com/landing/{campaign.id}/{k + 1}",
                        creative_type=creative_type,
                        width=width,
                        height=height,
                        status=Status.ACTIVE,
                    )
                    session.add(creative)

        await session.commit()

    print(f"\nGenerated:")
    print(f"  - {num_advertisers} advertisers")
    print(f"  - {num_advertisers * campaigns_per_advertiser} campaigns")
    print(
        f"  - {num_advertisers * campaigns_per_advertiser * creatives_per_campaign} creatives"
    )

    await db.close()


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate mock data for LiteAds")
    parser.add_argument(
        "--advertisers", type=int, default=10, help="Number of advertisers"
    )
    parser.add_argument(
        "--campaigns", type=int, default=5, help="Campaigns per advertiser"
    )
    parser.add_argument(
        "--creatives", type=int, default=3, help="Creatives per campaign"
    )

    args = parser.parse_args()

    asyncio.run(
        generate_mock_data(
            num_advertisers=args.advertisers,
            campaigns_per_advertiser=args.campaigns,
            creatives_per_campaign=args.creatives,
        )
    )


if __name__ == "__main__":
    main()
