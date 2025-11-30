#!/usr/bin/env python3
"""
Initialize test data in the database for end-to-end testing.

Creates advertisers, campaigns, creatives, and targeting rules.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def init_database():
    """Initialize database with test data."""
    import asyncpg

    print("Connecting to PostgreSQL...")
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="liteads",
        password="liteads_password",
        database="liteads",
    )

    print("Creating tables...")

    # Create tables
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS advertisers (
            id BIGSERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            company VARCHAR(255),
            contact_email VARCHAR(255),
            contact_phone VARCHAR(50),
            balance DECIMAL(15,2) DEFAULT 0.00 NOT NULL,
            credit_limit DECIMAL(15,2) DEFAULT 0.00 NOT NULL,
            status SMALLINT DEFAULT 1 NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS campaigns (
            id BIGSERIAL PRIMARY KEY,
            advertiser_id BIGINT NOT NULL REFERENCES advertisers(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            budget_daily DECIMAL(15,2),
            budget_total DECIMAL(15,2),
            spent_today DECIMAL(15,2) DEFAULT 0.00 NOT NULL,
            spent_total DECIMAL(15,2) DEFAULT 0.00 NOT NULL,
            bid_type SMALLINT DEFAULT 1 NOT NULL,
            bid_amount DECIMAL(10,4) DEFAULT 1.0000 NOT NULL,
            start_time TIMESTAMP WITH TIME ZONE,
            end_time TIMESTAMP WITH TIME ZONE,
            freq_cap_daily SMALLINT,
            freq_cap_hourly SMALLINT,
            status SMALLINT DEFAULT 1 NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS creatives (
            id BIGSERIAL PRIMARY KEY,
            campaign_id BIGINT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
            title VARCHAR(255),
            description TEXT,
            image_url VARCHAR(500),
            video_url VARCHAR(500),
            landing_url VARCHAR(500) NOT NULL,
            creative_type SMALLINT DEFAULT 1 NOT NULL,
            width SMALLINT,
            height SMALLINT,
            status SMALLINT DEFAULT 1 NOT NULL,
            impressions BIGINT DEFAULT 0 NOT NULL,
            clicks BIGINT DEFAULT 0 NOT NULL,
            conversions BIGINT DEFAULT 0 NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS targeting_rules (
            id BIGSERIAL PRIMARY KEY,
            campaign_id BIGINT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
            rule_type VARCHAR(50) NOT NULL,
            rule_value JSONB NOT NULL,
            is_include BOOLEAN DEFAULT TRUE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS ad_events (
            id BIGSERIAL PRIMARY KEY,
            request_id VARCHAR(64) NOT NULL,
            campaign_id BIGINT,
            creative_id BIGINT,
            event_type SMALLINT NOT NULL,
            event_time TIMESTAMP WITH TIME ZONE NOT NULL,
            user_id VARCHAR(64),
            ip_address VARCHAR(45),
            cost DECIMAL(10,6) DEFAULT 0.000000 NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
        )
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS hourly_stats (
            id BIGSERIAL PRIMARY KEY,
            campaign_id BIGINT NOT NULL,
            creative_id BIGINT,
            stat_hour TIMESTAMP WITH TIME ZONE NOT NULL,
            impressions BIGINT DEFAULT 0 NOT NULL,
            clicks BIGINT DEFAULT 0 NOT NULL,
            conversions BIGINT DEFAULT 0 NOT NULL,
            cost DECIMAL(15,4) DEFAULT 0.0000 NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
        )
    """)

    # Check if data exists
    existing_count = await conn.fetchval("SELECT COUNT(*) FROM advertisers")
    if existing_count > 0:
        print(f"Database already has {existing_count} advertisers, skipping insert...")
        await conn.close()
        return

    print("Inserting test data...")

    # Insert advertisers
    advertisers = [
        ("游戏广告商", "游戏公司A", "game@example.com", 50000.00),
        ("电商广告商", "电商公司B", "shop@example.com", 100000.00),
        ("金融广告商", "金融公司C", "finance@example.com", 80000.00),
        ("教育广告商", "教育公司D", "edu@example.com", 30000.00),
        ("本地生活", "生活服务E", "local@example.com", 20000.00),
    ]

    advertiser_ids = []
    for name, company, email, balance in advertisers:
        row = await conn.fetchrow(
            """
            INSERT INTO advertisers (name, company, contact_email, balance, status)
            VALUES ($1, $2, $3, $4, 1)
            RETURNING id
            """,
            name, company, email, balance
        )
        advertiser_ids.append(row['id'])

    print(f"  Inserted {len(advertisers)} advertisers (IDs: {advertiser_ids})")

    # Insert campaigns
    # Use dynamic advertiser IDs
    a1, a2, a3, a4, a5 = advertiser_ids
    campaigns = [
        # advertiser_id, name, budget_daily, budget_total, bid_type, bid_amount
        (a1, "王者荣耀推广", 1000.00, 30000.00, 2, 2.50),  # CPC
        (a1, "原神暑期活动", 500.00, 15000.00, 1, 15.00),  # CPM
        (a2, "618大促活动", 5000.00, 100000.00, 2, 1.80),  # CPC
        (a2, "双11预热", 3000.00, 50000.00, 2, 2.20),  # CPC
        (a3, "理财产品推广", 2000.00, 60000.00, 2, 5.00),  # CPC
        (a3, "信用卡活动", 1500.00, 45000.00, 1, 25.00),  # CPM
        (a4, "在线课程推广", 800.00, 24000.00, 2, 3.00),  # CPC
        (a5, "外卖优惠", 2000.00, 40000.00, 2, 1.50),  # CPC
        (a5, "打车补贴", 1000.00, 30000.00, 2, 1.20),  # CPC
        (a1, "新游戏预约", 300.00, 9000.00, 1, 10.00),  # CPM
    ]

    campaign_ids = []
    for adv_id, name, budget_d, budget_t, bid_type, bid_amt in campaigns:
        row = await conn.fetchrow(
            """
            INSERT INTO campaigns (advertiser_id, name, budget_daily, budget_total,
                                   bid_type, bid_amount, status)
            VALUES ($1, $2, $3, $4, $5, $6, 1)
            RETURNING id
            """,
            adv_id, name, budget_d, budget_t, bid_type, bid_amt
        )
        campaign_ids.append(row['id'])

    print(f"  Inserted {len(campaigns)} campaigns (IDs: {campaign_ids})")

    # Insert creatives - use campaign IDs
    c = campaign_ids  # shorthand
    creatives = [
        (c[0], "王者荣耀新英雄上线", "限时福利，新英雄免费试玩", "banner", 300, 250),
        (c[0], "王者荣耀视频广告", "精彩操作集锦", "video", 640, 360),
        (c[1], "原神3.0新地图", "探索须弥，领取原石", "banner", 300, 250),
        (c[2], "618超级红包", "满减叠加，低至5折", "native", 200, 200),
        (c[2], "618品牌特卖", "大牌直降，限时抢购", "banner", 300, 100),
        (c[3], "双11提前购", "定金翻倍，预售优惠", "native", 200, 200),
        (c[4], "低息理财产品", "年化收益4.5%起", "banner", 300, 250),
        (c[5], "信用卡申请", "开卡即送积分", "native", 200, 200),
        (c[6], "Python课程", "从入门到精通", "banner", 300, 250),
        (c[7], "外卖新人红包", "首单立减20元", "native", 200, 200),
        (c[8], "打车优惠券", "新用户专享5折", "banner", 300, 100),
        (c[9], "新游预约", "预约送限定皮肤", "banner", 300, 250),
    ]

    for camp_id, title, desc, c_type, width, height in creatives:
        creative_type = 1 if c_type == "banner" else (2 if c_type == "video" else 3)
        await conn.execute(
            """
            INSERT INTO creatives (campaign_id, title, description, image_url, landing_url,
                                   creative_type, width, height, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 1)
            """,
            camp_id, title, desc,
            f"https://via.placeholder.com/{width}x{height}",
            f"https://example.com/landing/{camp_id}",
            creative_type, width, height
        )

    print(f"  Inserted {len(creatives)} creatives")

    # Insert targeting rules - use campaign IDs
    # Note: rule_value must be a dict with specific keys based on rule_type:
    # - platform/os: {"values": [...]}
    # - geo: {"countries": [...], "cities": [...]}
    # - age: {"min": N, "max": N}
    import json
    targeting_rules = [
        (c[0], "os", json.dumps({"values": ["android", "ios"]}), True),
        (c[0], "geo", json.dumps({"countries": ["CN"]}), True),
        (c[1], "age", json.dumps({"min": 18, "max": 34}), True),
        (c[2], "os", json.dumps({"values": ["android"]}), True),
        (c[2], "geo", json.dumps({"countries": ["CN"]}), True),
        (c[4], "age", json.dumps({"min": 25, "max": 99}), True),
        (c[6], "interest", json.dumps({"values": ["programming", "technology"]}), True),
        (c[7], "geo", json.dumps({"countries": ["CN"]}), True),
    ]

    for camp_id, rule_type, rule_value, is_include in targeting_rules:
        await conn.execute(
            """
            INSERT INTO targeting_rules (campaign_id, rule_type, rule_value, is_include)
            VALUES ($1, $2, $3, $4)
            """,
            camp_id, rule_type, rule_value, is_include
        )

    print(f"  Inserted {len(targeting_rules)} targeting rules")

    await conn.close()
    print("\nDatabase initialization complete!")


async def verify_data():
    """Verify data was inserted correctly."""
    import asyncpg

    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="liteads",
        password="liteads_password",
        database="liteads",
    )

    print("\n" + "=" * 50)
    print("Database Summary:")
    print("=" * 50)

    advertisers = await conn.fetchval("SELECT COUNT(*) FROM advertisers")
    campaigns = await conn.fetchval("SELECT COUNT(*) FROM campaigns")
    creatives = await conn.fetchval("SELECT COUNT(*) FROM creatives")
    targeting = await conn.fetchval("SELECT COUNT(*) FROM targeting_rules")

    print(f"  Advertisers: {advertisers}")
    print(f"  Campaigns: {campaigns}")
    print(f"  Creatives: {creatives}")
    print(f"  Targeting Rules: {targeting}")

    print("\nActive Campaigns:")
    rows = await conn.fetch("""
        SELECT c.id, c.name, a.name as advertiser, c.bid_type, c.bid_amount, c.budget_daily
        FROM campaigns c
        JOIN advertisers a ON c.advertiser_id = a.id
        WHERE c.status = 1
        ORDER BY c.id
        LIMIT 10
    """)

    for row in rows:
        bid_type = "CPM" if row["bid_type"] == 1 else "CPC"
        print(f"  [{row['id']}] {row['name']} ({row['advertiser']}) - {bid_type} ${row['bid_amount']}")

    await conn.close()


def main():
    print("=" * 50)
    print("LiteAds Database Initialization")
    print("=" * 50)

    asyncio.run(init_database())
    asyncio.run(verify_data())


if __name__ == "__main__":
    main()
