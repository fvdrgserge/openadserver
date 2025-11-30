#!/usr/bin/env python3
"""
Generate training data for ML models.

Creates realistic training data for CTR/CVR prediction models.

Usage:
    python scripts/generate_training_data.py --output data/ --samples 100000
"""

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from liteads.common.logger import get_logger

logger = get_logger(__name__)


def generate_user_features(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Generate user feature data."""
    np.random.seed(seed)

    data = {
        "user_id": [f"user_{i}" for i in np.random.randint(0, n_samples // 10, n_samples)],
        "user_gender": np.random.choice(["male", "female", "unknown"], n_samples, p=[0.45, 0.45, 0.1]),
        "user_age_bucket": np.random.choice(
            ["under_18", "18-24", "25-34", "35-44", "45+"],
            n_samples,
            p=[0.05, 0.25, 0.35, 0.20, 0.15],
        ),
        "user_device_os": np.random.choice(["android", "ios", "other"], n_samples, p=[0.55, 0.40, 0.05]),
        "user_network_type": np.random.choice(["wifi", "4g", "5g", "3g"], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
        "user_click_count_7d": np.random.exponential(5, n_samples).astype(int),
        "user_click_count_30d": np.random.exponential(15, n_samples).astype(int),
        "user_conversion_count_7d": np.random.exponential(0.3, n_samples).astype(int),
        "user_ctr_7d": np.clip(np.random.beta(2, 50, n_samples), 0, 0.3),
        "user_cvr_7d": np.clip(np.random.beta(1, 100, n_samples), 0, 0.1),
        "user_avg_session_duration": np.random.exponential(180, n_samples),
    }

    return pd.DataFrame(data)


def generate_ad_features(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Generate ad feature data."""
    np.random.seed(seed + 1)

    n_campaigns = min(200, n_samples // 100)
    n_creatives = min(1000, n_samples // 20)
    n_advertisers = min(50, n_samples // 500)

    data = {
        "campaign_id": [f"camp_{i % n_campaigns}" for i in range(n_samples)],
        "creative_id": [f"creative_{i % n_creatives}" for i in range(n_samples)],
        "advertiser_id": [f"adv_{i % n_advertisers}" for i in range(n_samples)],
        "ad_category": np.random.choice(
            ["game", "ecom", "finance", "education", "entertainment", "social", "utility"],
            n_samples,
            p=[0.2, 0.25, 0.1, 0.1, 0.15, 0.1, 0.1],
        ),
        "creative_type": np.random.choice(["banner", "native", "video"], n_samples, p=[0.4, 0.4, 0.2]),
        "bid_type": np.random.choice(["cpm", "cpc", "cpa"], n_samples, p=[0.3, 0.5, 0.2]),
        "landing_page_type": np.random.choice(["app", "h5", "deep_link"], n_samples, p=[0.5, 0.35, 0.15]),
        "ad_bid": np.random.uniform(0.5, 50, n_samples),
        "ad_ctr_7d": np.clip(np.random.beta(2, 50, n_samples), 0, 0.2),
        "ad_cvr_7d": np.clip(np.random.beta(1, 100, n_samples), 0, 0.1),
        "ad_impression_count_7d": np.random.exponential(5000, n_samples).astype(int),
        "ad_click_count_7d": np.random.exponential(50, n_samples).astype(int),
        "creative_ctr": np.clip(np.random.beta(2, 50, n_samples), 0, 0.2),
        "advertiser_quality_score": np.random.uniform(0.3, 1.0, n_samples),
    }

    return pd.DataFrame(data)


def generate_context_features(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Generate context feature data."""
    np.random.seed(seed + 2)

    # Generate timestamps over last 30 days
    base_time = datetime.now() - timedelta(days=30)
    timestamps = [base_time + timedelta(seconds=random.randint(0, 30 * 24 * 3600)) for _ in range(n_samples)]

    data = {
        "slot_id": np.random.choice(["slot_1", "slot_2", "slot_3", "slot_4", "slot_5"], n_samples),
        "request_hour": [t.hour for t in timestamps],
        "request_day_of_week": [t.weekday() for t in timestamps],
        "is_weekend": [1 if t.weekday() >= 5 else 0 for t in timestamps],
        "is_peak_hour": [1 if 9 <= t.hour <= 12 or 19 <= t.hour <= 22 else 0 for t in timestamps],
        "geo_country": np.random.choice(["CN", "US", "JP", "KR", "TW"], n_samples, p=[0.7, 0.1, 0.1, 0.05, 0.05]),
        "geo_city": np.random.choice(
            ["shanghai", "beijing", "shenzhen", "guangzhou", "hangzhou", "other"],
            n_samples,
            p=[0.2, 0.2, 0.15, 0.15, 0.1, 0.2],
        ),
        "slot_ctr": np.clip(np.random.beta(2, 50, n_samples), 0, 0.15),
        "hour_ctr": np.clip(np.random.beta(2, 50, n_samples), 0, 0.15),
        "timestamp": [int(t.timestamp()) for t in timestamps],
    }

    return pd.DataFrame(data)


def generate_labels(
    user_df: pd.DataFrame,
    ad_df: pd.DataFrame,
    context_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate click and conversion labels based on features."""
    np.random.seed(seed + 3)

    n_samples = len(user_df)

    # Base click probability
    base_ctr = 0.02

    # Feature effects on CTR
    ctr_boost = np.zeros(n_samples)

    # User effects
    ctr_boost += (user_df["user_ctr_7d"].values - 0.02) * 2
    ctr_boost += np.where(user_df["user_gender"] == "female", 0.005, 0)
    ctr_boost += np.where(user_df["user_age_bucket"] == "25-34", 0.005, 0)

    # Ad effects
    ctr_boost += (ad_df["ad_ctr_7d"].values - 0.02) * 3
    ctr_boost += (ad_df["creative_ctr"].values - 0.02) * 2
    ctr_boost += np.where(ad_df["creative_type"] == "video", 0.01, 0)
    ctr_boost += np.where(ad_df["ad_category"] == "game", 0.005, 0)

    # Context effects
    ctr_boost += np.where(context_df["is_peak_hour"] == 1, 0.005, 0)
    ctr_boost += np.where(context_df["is_weekend"] == 1, 0.002, 0)

    # Calculate final click probability
    click_prob = np.clip(base_ctr + ctr_boost + np.random.normal(0, 0.005, n_samples), 0.001, 0.3)

    # Generate clicks
    clicks = np.random.binomial(1, click_prob)

    # Generate conversions (only for clicks)
    base_cvr = 0.05
    cvr_boost = (ad_df["ad_cvr_7d"].values - 0.01) * 5
    cvr_boost += np.where(ad_df["bid_type"] == "cpa", 0.02, 0)
    conversion_prob = np.clip(base_cvr + cvr_boost, 0.01, 0.3)

    conversions = clicks * np.random.binomial(1, conversion_prob)

    return pd.DataFrame({
        "click": clicks,
        "conversion": conversions,
    })


def generate_training_data(n_samples: int, output_dir: Path, seed: int = 42) -> None:
    """Generate complete training dataset."""
    logger.info(f"Generating {n_samples} training samples...")

    # Generate features
    user_df = generate_user_features(n_samples, seed)
    ad_df = generate_ad_features(n_samples, seed)
    context_df = generate_context_features(n_samples, seed)

    # Generate labels
    labels_df = generate_labels(user_df, ad_df, context_df, seed)

    # Combine all features
    data = pd.concat([user_df, ad_df, context_df, labels_df], axis=1)

    # Split into train/val/test
    n_train = int(n_samples * 0.8)
    n_val = int(n_samples * 0.1)

    indices = np.random.permutation(n_samples)

    train_data = data.iloc[indices[:n_train]]
    val_data = data.iloc[indices[n_train:n_train + n_val]]
    test_data = data.iloc[indices[n_train + n_val:]]

    # Save to files
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)

    logger.info(f"Saved training data to {train_path} ({len(train_data)} samples)")
    logger.info(f"Saved validation data to {val_path} ({len(val_data)} samples)")
    logger.info(f"Saved test data to {test_path} ({len(test_data)} samples)")

    # Print statistics
    logger.info("\nData Statistics:")
    logger.info(f"  Click rate: {labels_df['click'].mean():.4f}")
    logger.info(f"  Conversion rate (of clicks): {labels_df['conversion'].sum() / max(labels_df['click'].sum(), 1):.4f}")

    # Save as parquet for faster loading
    train_data.to_parquet(output_dir / "train.parquet", index=False)
    val_data.to_parquet(output_dir / "val.parquet", index=False)
    test_data.to_parquet(output_dir / "test.parquet", index=False)

    logger.info("\nAlso saved as Parquet format for faster loading")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate training data for ML models")
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)

    # Generate training data
    generate_training_data(args.samples, output_dir, args.seed)

    logger.info("\nData generation complete!")


if __name__ == "__main__":
    main()
