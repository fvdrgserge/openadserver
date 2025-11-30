#!/usr/bin/env python3
"""
Training script for DeepFM CTR prediction model.

Example usage:
    python scripts/train_model.py --data data/train.csv --epochs 10
"""

import argparse
from pathlib import Path

import pandas as pd

from liteads.common.logger import get_logger
from liteads.ml_engine.data import AdDataModule
from liteads.ml_engine.features import FeatureBuilder
from liteads.ml_engine.models import DeepFM
from liteads.ml_engine.training import Trainer, TrainingConfig

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DeepFM model")

    # Data arguments
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/train.csv",
        help="Path to training data (CSV or Parquet)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data (optional)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio if no val-data provided",
    )
    parser.add_argument(
        "--label-cols",
        type=str,
        nargs="+",
        default=["click"],
        help="Label column names",
    )

    # Model arguments
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="Default embedding dimension",
    )
    parser.add_argument(
        "--dnn-hidden-units",
        type=int,
        nargs="+",
        default=[256, 128, 64],
        help="DNN hidden layer sizes",
    )
    parser.add_argument(
        "--dnn-dropout",
        type=float,
        default=0.2,
        help="DNN dropout rate",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Early stopping patience",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepfm_ctr",
        help="Model name for saving",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    return parser.parse_args()


def generate_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate sample training data for demonstration."""
    import numpy as np

    np.random.seed(42)

    data = {
        # User features
        "user_id": [f"user_{i % 1000}" for i in range(n_samples)],
        "user_gender": np.random.choice(["male", "female", "unknown"], n_samples),
        "user_age_bucket": np.random.choice(["18-24", "25-34", "35-44", "45+"], n_samples),
        "user_device_os": np.random.choice(["ios", "android", "other"], n_samples),
        "user_network_type": np.random.choice(["wifi", "4g", "5g"], n_samples),
        "user_click_count_7d": np.random.exponential(5, n_samples).astype(int),
        "user_click_count_30d": np.random.exponential(20, n_samples).astype(int),
        "user_conversion_count_7d": np.random.exponential(0.5, n_samples).astype(int),
        "user_ctr_7d": np.random.beta(2, 100, n_samples),
        "user_cvr_7d": np.random.beta(1, 200, n_samples),
        "user_avg_session_duration": np.random.exponential(300, n_samples),
        # Ad features
        "campaign_id": [f"camp_{i % 100}" for i in range(n_samples)],
        "creative_id": [f"creative_{i % 500}" for i in range(n_samples)],
        "advertiser_id": [f"adv_{i % 50}" for i in range(n_samples)],
        "ad_category": np.random.choice(["game", "ecom", "finance", "education"], n_samples),
        "creative_type": np.random.choice(["banner", "native", "video"], n_samples),
        "bid_type": np.random.choice(["cpm", "cpc", "cpa"], n_samples),
        "landing_page_type": np.random.choice(["app", "h5", "deep_link"], n_samples),
        "ad_bid": np.random.uniform(1, 50, n_samples),
        "ad_ctr_7d": np.random.beta(2, 100, n_samples),
        "ad_cvr_7d": np.random.beta(1, 200, n_samples),
        "ad_impression_count_7d": np.random.exponential(1000, n_samples).astype(int),
        "ad_click_count_7d": np.random.exponential(10, n_samples).astype(int),
        "creative_ctr": np.random.beta(2, 100, n_samples),
        "advertiser_quality_score": np.random.uniform(0.5, 1.0, n_samples),
        # Context features
        "slot_id": np.random.choice(["slot_1", "slot_2", "slot_3"], n_samples),
        "request_hour": np.random.randint(0, 24, n_samples),
        "request_day_of_week": np.random.randint(0, 7, n_samples),
        "is_weekend": np.random.choice([0, 1], n_samples),
        "is_peak_hour": np.random.choice([0, 1], n_samples),
        "geo_country": np.random.choice(["CN", "US", "JP"], n_samples),
        "geo_city": np.random.choice(["shanghai", "beijing", "shenzhen"], n_samples),
        "slot_ctr": np.random.beta(2, 100, n_samples),
        "hour_ctr": np.random.beta(2, 100, n_samples),
    }

    # Generate labels (click) based on features
    click_prob = (
        0.01
        + 0.02 * (data["ad_ctr_7d"] > 0.02)
        + 0.01 * (data["user_ctr_7d"] > 0.02)
        + 0.005 * (data["is_peak_hour"] == 1)
    )
    data["click"] = np.random.binomial(1, click_prob)

    return pd.DataFrame(data)


def main() -> None:
    """Main training function."""
    args = parse_args()

    logger.info("Starting training with configuration:")
    logger.info(f"  Train data: {args.train_data}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Device: {args.device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate data
    train_path = Path(args.train_data)
    if train_path.exists():
        logger.info(f"Loading training data from {train_path}")
        if train_path.suffix == ".csv":
            train_df = pd.read_csv(train_path)
        elif train_path.suffix == ".parquet":
            train_df = pd.read_parquet(train_path)
        else:
            raise ValueError(f"Unsupported file format: {train_path.suffix}")
    else:
        logger.info("Training data not found, generating sample data...")
        train_df = generate_sample_data(n_samples=50000)
        train_df.to_csv(train_path, index=False)
        logger.info(f"Sample data saved to {train_path}")

    # Load validation data if provided
    val_df = None
    if args.val_data:
        val_path = Path(args.val_data)
        if val_path.exists():
            if val_path.suffix == ".csv":
                val_df = pd.read_csv(val_path)
            elif val_path.suffix == ".parquet":
                val_df = pd.read_parquet(val_path)

    # Setup data module
    logger.info("Setting up data module...")
    feature_builder = FeatureBuilder(device=args.device)
    data_module = AdDataModule(
        feature_builder=feature_builder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    data_module.setup_from_dataframe(
        train_df=train_df,
        val_df=val_df,
        label_cols=args.label_cols,
        val_split=args.val_split,
    )

    # Get model configuration from data
    model_config = data_module.get_model_config()

    # Create model
    logger.info("Creating DeepFM model...")
    model = DeepFM(
        sparse_feature_dims=model_config["sparse_feature_dims"],
        sparse_embedding_dims=model_config["sparse_embedding_dims"],
        dense_feature_dim=model_config["dense_feature_dim"],
        sequence_feature_dims=model_config["sequence_feature_dims"],
        sequence_embedding_dims=model_config["sequence_embedding_dims"],
        fm_k=model_config.get("fm_k", 8),
        dnn_hidden_units=args.dnn_hidden_units,
        dnn_dropout=args.dnn_dropout,
        l2_reg_embedding=args.weight_decay,
        l2_reg_dnn=args.weight_decay,
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=str(output_dir / "checkpoints"),
        device=args.device,
    )

    trainer = Trainer(model=model, config=training_config)

    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Train
    logger.info("Starting training...")
    metrics = trainer.fit(train_loader, val_loader)

    # Save final model
    final_model_path = output_dir / f"{args.model_name}.pt"
    trainer._save_checkpoint(str(final_model_path))

    # Save feature builder
    feature_builder_path = output_dir / f"{args.model_name}_features.pkl"
    feature_builder.save(str(feature_builder_path))

    logger.info(f"Training complete!")
    logger.info(f"Best validation loss: {metrics.best_val_loss:.4f}")
    logger.info(f"Best validation AUC: {metrics.best_val_auc:.4f}")
    logger.info(f"Model saved to {final_model_path}")
    logger.info(f"Feature builder saved to {feature_builder_path}")


if __name__ == "__main__":
    main()
