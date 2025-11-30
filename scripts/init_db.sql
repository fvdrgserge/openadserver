-- LiteAds Database Initialization Script
-- This script is run automatically when PostgreSQL container starts

-- Create extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create advertisers table
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
);

CREATE INDEX IF NOT EXISTS idx_advertiser_status ON advertisers(status);

-- Create campaigns table
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
);

CREATE INDEX IF NOT EXISTS idx_campaign_advertiser ON campaigns(advertiser_id);
CREATE INDEX IF NOT EXISTS idx_campaign_status ON campaigns(status);
CREATE INDEX IF NOT EXISTS idx_campaign_schedule ON campaigns(start_time, end_time);

-- Create creatives table
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
);

CREATE INDEX IF NOT EXISTS idx_creative_campaign ON creatives(campaign_id);
CREATE INDEX IF NOT EXISTS idx_creative_status ON creatives(status);

-- Create targeting_rules table
CREATE TABLE IF NOT EXISTS targeting_rules (
    id BIGSERIAL PRIMARY KEY,
    campaign_id BIGINT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    rule_type VARCHAR(50) NOT NULL,
    rule_value JSONB NOT NULL,
    is_include BOOLEAN DEFAULT TRUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_targeting_campaign ON targeting_rules(campaign_id);
CREATE INDEX IF NOT EXISTS idx_targeting_type ON targeting_rules(rule_type);

-- Create ad_events table
CREATE TABLE IF NOT EXISTS ad_events (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(64) NOT NULL,
    campaign_id BIGINT REFERENCES campaigns(id) ON DELETE SET NULL,
    creative_id BIGINT REFERENCES creatives(id) ON DELETE SET NULL,
    event_type SMALLINT NOT NULL,
    event_time TIMESTAMP WITH TIME ZONE NOT NULL,
    user_id VARCHAR(64),
    ip_address VARCHAR(45),
    cost DECIMAL(10,6) DEFAULT 0.000000 NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_event_request ON ad_events(request_id);
CREATE INDEX IF NOT EXISTS idx_event_campaign ON ad_events(campaign_id);
CREATE INDEX IF NOT EXISTS idx_event_time ON ad_events(event_time);
CREATE INDEX IF NOT EXISTS idx_event_type_time ON ad_events(event_type, event_time);

-- Create hourly_stats table
CREATE TABLE IF NOT EXISTS hourly_stats (
    id BIGSERIAL PRIMARY KEY,
    campaign_id BIGINT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    creative_id BIGINT REFERENCES creatives(id) ON DELETE SET NULL,
    stat_hour TIMESTAMP WITH TIME ZONE NOT NULL,
    impressions BIGINT DEFAULT 0 NOT NULL,
    clicks BIGINT DEFAULT 0 NOT NULL,
    conversions BIGINT DEFAULT 0 NOT NULL,
    cost DECIMAL(15,4) DEFAULT 0.0000 NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_stats_campaign_hour ON hourly_stats(campaign_id, stat_hour);
CREATE INDEX IF NOT EXISTS idx_stats_hour ON hourly_stats(stat_hour);

-- Insert sample data for testing
INSERT INTO advertisers (name, company, balance, status) VALUES
    ('Demo Advertiser', 'Demo Company Inc.', 10000.00, 1);

INSERT INTO campaigns (advertiser_id, name, budget_daily, budget_total, bid_type, bid_amount, status) VALUES
    (1, 'Demo Campaign', 100.00, 1000.00, 1, 5.0000, 1);

INSERT INTO creatives (campaign_id, title, description, image_url, landing_url, creative_type, width, height, status) VALUES
    (1, 'Demo Ad', 'This is a demo advertisement', 'https://via.placeholder.com/300x250', 'https://example.com', 1, 300, 250, 1);

-- Print success message
DO $$
BEGIN
    RAISE NOTICE 'LiteAds database initialized successfully!';
END $$;
