#!/usr/bin/env python3
"""
End-to-end test script for LiteAds with LR model.

This script:
1. Generates synthetic Criteo-like training data
2. Trains an LR model for CTR prediction
3. Tests the complete API flow: request → model prediction → ad response

Usage:
    python scripts/e2e_test_lr.py

For full test with server:
    python scripts/e2e_test_lr.py --with-server
"""

import argparse
import asyncio
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_criteo_like_data(
    num_samples: int = 10000,
    num_sparse_features: int = 26,
    num_dense_features: int = 13,
    sparse_vocab_sizes: list[int] | None = None,
) -> dict:
    """
    Generate synthetic data similar to Criteo CTR dataset.

    Criteo dataset has:
    - 1 label (click/no-click)
    - 13 numerical features (I1-I13)
    - 26 categorical features (C1-C26)

    Args:
        num_samples: Number of samples to generate
        num_sparse_features: Number of categorical features
        num_dense_features: Number of numerical features
        sparse_vocab_sizes: Vocabulary sizes for each sparse feature

    Returns:
        Dictionary with features and labels
    """
    print(f"Generating {num_samples} Criteo-like samples...")

    # Default vocab sizes (inspired by Criteo)
    if sparse_vocab_sizes is None:
        sparse_vocab_sizes = [
            1000, 500, 200, 100, 50,  # C1-C5
            300, 200, 100, 1000, 500,  # C6-C10
            200, 100, 50, 500, 300,  # C11-C15
            200, 100, 50, 1000, 500,  # C16-C20
            200, 100, 50, 300, 200, 100,  # C21-C26
        ][:num_sparse_features]

    # Generate sparse features (categorical)
    sparse_features = np.zeros((num_samples, num_sparse_features), dtype=np.int64)
    for i, vocab_size in enumerate(sparse_vocab_sizes):
        sparse_features[:, i] = np.random.randint(0, vocab_size, num_samples)

    # Generate dense features (numerical) with realistic distributions
    dense_features = np.zeros((num_samples, num_dense_features), dtype=np.float32)
    for i in range(num_dense_features):
        # Mix of normal and log-normal distributions (like real Criteo data)
        if i % 2 == 0:
            dense_features[:, i] = np.random.randn(num_samples) * 2 + 5
        else:
            dense_features[:, i] = np.random.lognormal(0, 1, num_samples)

    # Normalize dense features
    dense_mean = dense_features.mean(axis=0, keepdims=True)
    dense_std = dense_features.std(axis=0, keepdims=True) + 1e-6
    dense_features = (dense_features - dense_mean) / dense_std

    # Generate labels with realistic CTR (2-5%)
    # Use a simple logistic function based on features
    logits = np.zeros(num_samples)

    # Add contribution from dense features
    for i in range(num_dense_features):
        weight = np.random.randn() * 0.1
        logits += weight * dense_features[:, i]

    # Add some noise
    logits += np.random.randn(num_samples) * 0.5

    # Shift logits to get ~3% CTR
    logits -= 3.5

    # Convert to probabilities and sample
    probs = 1 / (1 + np.exp(-logits))
    labels = (np.random.rand(num_samples) < probs).astype(np.float32)

    actual_ctr = labels.mean()
    print(f"Generated data with CTR: {actual_ctr:.2%}")

    return {
        "sparse_features": sparse_features,
        "dense_features": dense_features,
        "labels": labels,
        "sparse_vocab_sizes": sparse_vocab_sizes,
    }


def train_lr_model(
    sparse_features: np.ndarray,
    dense_features: np.ndarray,
    labels: np.ndarray,
    sparse_vocab_sizes: list[int],
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 0.01,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    """
    Train LR model on the generated data.

    Args:
        sparse_features: Sparse feature indices
        dense_features: Dense feature values
        labels: Click labels
        sparse_vocab_sizes: Vocabulary sizes for sparse features
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to use

    Returns:
        Trained model and model config
    """
    from liteads.ml_engine.models.lr import LogisticRegression

    print(f"\nTraining LR model on {len(labels)} samples...")
    print(f"  Sparse features: {sparse_features.shape[1]}")
    print(f"  Dense features: {dense_features.shape[1]}")
    print(f"  Device: {device}")

    # Create model
    model = LogisticRegression(
        sparse_feature_dims=sparse_vocab_sizes,
        dense_feature_dim=dense_features.shape[1],
        l2_reg=0.0001,
    )
    model = model.to(device)

    # Create data loaders
    train_size = int(0.8 * len(labels))

    train_sparse = torch.tensor(sparse_features[:train_size], dtype=torch.long)
    train_dense = torch.tensor(dense_features[:train_size], dtype=torch.float32)
    train_labels = torch.tensor(labels[:train_size], dtype=torch.float32).unsqueeze(1)

    val_sparse = torch.tensor(sparse_features[train_size:], dtype=torch.long)
    val_dense = torch.tensor(dense_features[train_size:], dtype=torch.float32)
    val_labels = torch.tensor(labels[train_size:], dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(train_sparse, train_dense, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_sparse, batch_dense, batch_labels in train_loader:
            batch_sparse = batch_sparse.to(device)
            batch_dense = batch_dense.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            outputs = model(batch_sparse, batch_dense)
            loss = criterion(outputs["ctr"], batch_labels)

            # Add regularization
            reg_loss = model.get_regularization_loss()
            total_loss_batch = loss + reg_loss

            total_loss_batch.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_sparse.to(device), val_dense.to(device))
            val_loss = criterion(val_outputs["ctr"], val_labels.to(device))

            # Calculate AUC
            val_preds = val_outputs["ctr"].cpu().numpy()
            val_true = val_labels.numpy()
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(val_true, val_preds)
            except:
                auc = 0.0

        model.train()

        print(f"  Epoch {epoch + 1}/{epochs}: train_loss={avg_loss:.4f}, val_loss={val_loss.item():.4f}, val_auc={auc:.4f}")

    model.eval()
    print("Training complete!")

    # Model config for saving
    model_config = {
        "sparse_feature_dims": sparse_vocab_sizes,
        "dense_feature_dim": dense_features.shape[1],
        "l2_reg_embedding": 0.0001,
    }

    return model, model_config


def save_model(
    model: nn.Module,
    model_config: dict,
    output_dir: str,
    model_type: str = "lr",
) -> str:
    """Save trained model to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_file = output_path / f"{model_type}_ctr.pt"

    checkpoint = {
        "model_type": model_type,
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "version": f"{model_type}_v1",
    }

    torch.save(checkpoint, model_file)
    print(f"\nModel saved to: {model_file}")

    return str(model_file)


def test_model_prediction(model: nn.Module, sparse_vocab_sizes: list[int], device: str = "cpu") -> None:
    """Test model prediction with sample inputs."""
    print("\n" + "=" * 60)
    print("Testing Model Prediction")
    print("=" * 60)

    model.eval()
    model = model.to(device)

    # Create sample inputs
    batch_size = 5
    num_sparse = len(sparse_vocab_sizes)
    num_dense = model.dense_feature_dim

    sparse_features = torch.zeros(batch_size, num_sparse, dtype=torch.long)
    for i, vocab_size in enumerate(sparse_vocab_sizes):
        sparse_features[:, i] = torch.randint(0, vocab_size, (batch_size,))

    dense_features = torch.randn(batch_size, num_dense)

    sparse_features = sparse_features.to(device)
    dense_features = dense_features.to(device)

    # Make prediction
    with torch.no_grad():
        start_time = time.time()
        outputs = model(sparse_features, dense_features)
        latency_ms = (time.time() - start_time) * 1000

    predictions = outputs["ctr"].cpu().numpy()

    print(f"\nPrediction Results:")
    print(f"  Input shape: sparse={sparse_features.shape}, dense={dense_features.shape}")
    print(f"  Latency: {latency_ms:.2f}ms for {batch_size} samples")
    print(f"  Predictions: {predictions.flatten()}")
    print(f"  Mean pCTR: {predictions.mean():.4f}")


async def test_api_flow(model_path: str, feature_builder_path: str | None = None) -> None:
    """Test the ML predictor flow (without starting server)."""
    print("\n" + "=" * 60)
    print("Testing ML Predictor Flow")
    print("=" * 60)

    from liteads.ml_engine.serving import ModelPredictor

    # Create predictor
    predictor = ModelPredictor(
        model_path=model_path,
        feature_builder_path=feature_builder_path,
        device="cpu",
        warmup_samples=10,
    )

    # Load model
    predictor.load()
    print(f"Loaded model type: {predictor.model_type}")

    # Test with raw features (pre-transformed)
    # Since we don't have a fitted feature builder, we'll pass pre-transformed features
    sparse_dims = predictor.model.sparse_feature_dims
    num_sparse = len(sparse_dims)
    num_dense = predictor.model.dense_feature_dim

    # Generate features within valid vocabulary ranges
    sample_features = {
        "sparse_features": [random.randint(0, vs - 1) for vs in sparse_dims],
        "dense_features": [random.gauss(0, 1) for _ in range(num_dense)],
    }

    # Make prediction
    result = predictor.predict(sample_features)

    print(f"\nPrediction Result:")
    print(f"  pCTR: {result.pctr:.4f}")
    print(f"  Model Version: {result.model_version}")
    print(f"  Latency: {result.latency_ms:.2f}ms")


async def test_full_api_server(base_url: str = "http://localhost:8000") -> None:
    """Test full API server with HTTP requests."""
    import httpx

    print("\n" + "=" * 60)
    print("Testing Full API Server")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Health check
        print("\n1. Health Check...")
        try:
            response = await client.get(f"{base_url}/health")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"   Error: {e}")
            print("   Server may not be running. Start with: python -m liteads.ad_server.main")
            return

        # Ad request
        print("\n2. Ad Request...")
        ad_request = {
            "slot_id": "banner_home",
            "user_id": f"test_user_{random.randint(1, 1000)}",
            "device": {
                "os": "android",
                "os_version": "13.0",
                "model": "Pixel 8",
            },
            "geo": {
                "country": "CN",
                "city": "shanghai",
            },
            "context": {
                "app_id": "app_1",
                "app_version": "1.0.0",
            },
            "num_ads": 1,
        }

        try:
            response = await client.post(
                f"{base_url}/api/v1/ad/request",
                json=ad_request,
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   Request ID: {result.get('request_id')}")
                print(f"   Ads returned: {len(result.get('ads', []))}")
                if result.get("ads"):
                    ad = result["ads"][0]
                    print(f"   First ad:")
                    print(f"     - Campaign ID: {ad.get('campaign_id')}")
                    print(f"     - Creative ID: {ad.get('creative_id')}")
                    print(f"     - pCTR: {ad.get('pctr', 'N/A')}")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")

        # Event tracking
        print("\n3. Event Tracking...")
        try:
            event_data = {
                "request_id": "test_req_123",
                "ad_id": "ad_1_1",
                "event_type": "impression",
            }
            response = await client.post(
                f"{base_url}/api/v1/event/track",
                json=event_data,
            )
            print(f"   Status: {response.status_code}")
        except Exception as e:
            print(f"   Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="End-to-end test for LiteAds with LR model")
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--with-server",
        action="store_true",
        help="Test with running API server",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="API server URL",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for training (cpu/cuda/mps)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LiteAds End-to-End Test with LR Model")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[Step 1] Generating Criteo-like training data...")
    data = generate_criteo_like_data(
        num_samples=args.samples,
        num_sparse_features=26,
        num_dense_features=13,
    )

    # Step 2: Train model
    print("\n[Step 2] Training LR model...")
    model, model_config = train_lr_model(
        sparse_features=data["sparse_features"],
        dense_features=data["dense_features"],
        labels=data["labels"],
        sparse_vocab_sizes=data["sparse_vocab_sizes"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Step 3: Save model
    print("\n[Step 3] Saving trained model...")
    model_path = save_model(model, model_config, args.output_dir, model_type="lr")

    # Step 4: Test prediction
    print("\n[Step 4] Testing model prediction...")
    test_model_prediction(model, data["sparse_vocab_sizes"], args.device)

    # Step 5: Test ML predictor
    print("\n[Step 5] Testing ML predictor flow...")
    asyncio.run(test_api_flow(model_path))

    # Step 6: Test full API server (if requested)
    if args.with_server:
        print("\n[Step 6] Testing full API server...")
        asyncio.run(test_full_api_server(args.server_url))
    else:
        print("\n[Step 6] Skipped full API server test (use --with-server to enable)")

    print("\n" + "=" * 60)
    print("End-to-End Test Complete!")
    print("=" * 60)
    print(f"\nTrained model saved to: {model_path}")
    print("\nTo test with the full API server:")
    print("  1. Start the server: python -m liteads.ad_server.main")
    print("  2. Run: python scripts/e2e_test_lr.py --with-server")


if __name__ == "__main__":
    main()
