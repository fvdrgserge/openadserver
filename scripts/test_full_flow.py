#!/usr/bin/env python3
"""
Standalone test for full ad request flow with LR model.

This script tests the complete flow without needing external services:
1. Ad request parsing
2. Candidate retrieval (mock)
3. CTR prediction using trained LR model
4. Ranking and ad selection
5. Response formatting

Usage:
    python scripts/test_full_flow.py
"""

import asyncio
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class MockAdCandidate:
    """Mock ad candidate for testing."""
    campaign_id: int
    creative_id: int
    advertiser_id: int
    bid: float
    bid_type: int  # 1=CPM, 2=CPC
    pctr: float = 0.01
    pcvr: float = 0.001
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MockUserContext:
    """Mock user context for testing."""
    user_id: str
    gender: str = None
    age: int = None
    os: str = None
    country: str = None
    city: str = None
    interests: list = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def load_lr_model(model_path: str):
    """Load trained LR model."""
    from liteads.ml_engine.serving import ModelPredictor

    predictor = ModelPredictor(
        model_path=model_path,
        device="cpu",
        warmup_samples=5,
    )
    predictor.load()
    return predictor


def generate_mock_candidates(num_candidates: int = 10) -> list[MockAdCandidate]:
    """Generate mock ad candidates."""
    candidates = []
    for i in range(num_candidates):
        candidates.append(MockAdCandidate(
            campaign_id=100 + i,
            creative_id=1000 + i * 10,
            advertiser_id=10 + (i % 5),
            bid=random.uniform(0.5, 5.0),
            bid_type=random.choice([1, 2]),
            metadata={
                "category": random.choice(["game", "shopping", "finance", "social"]),
                "creative_type": random.choice(["banner", "video", "native"]),
                "impressions": random.randint(1000, 100000),
                "clicks": random.randint(10, 1000),
            }
        ))
    return candidates


def build_features_for_prediction(
    user_context: MockUserContext,
    candidate: MockAdCandidate,
    sparse_vocab_sizes: list[int],
    num_dense: int,
) -> dict:
    """Build feature dict for model prediction."""
    # For testing without feature builder, we generate random features
    # In production, this would use the FeatureBuilder
    return {
        "sparse_features": [random.randint(0, vs - 1) for vs in sparse_vocab_sizes],
        "dense_features": [random.gauss(0, 1) for _ in range(num_dense)],
    }


async def run_prediction_pipeline(
    predictor,
    user_context: MockUserContext,
    candidates: list[MockAdCandidate],
) -> list[tuple[MockAdCandidate, float]]:
    """Run CTR prediction for all candidates."""
    # Get model dimensions
    sparse_dims = predictor.model.sparse_feature_dims
    num_dense = predictor.model.dense_feature_dim

    # Build features for all candidates
    features_batch = []
    for candidate in candidates:
        features = build_features_for_prediction(
            user_context, candidate, sparse_dims, num_dense
        )
        features_batch.append(features)

    # Batch prediction
    results = predictor.predict_batch(features_batch)

    # Pair candidates with predictions
    scored_candidates = []
    for candidate, result in zip(candidates, results):
        candidate.pctr = result.pctr
        scored_candidates.append((candidate, result.pctr))

    return scored_candidates


def rank_candidates(
    scored_candidates: list[tuple[MockAdCandidate, float]],
    ranking_strategy: str = "ecpm",
) -> list[MockAdCandidate]:
    """Rank candidates by eCPM (expected CPM)."""
    if ranking_strategy == "ecpm":
        # eCPM = pCTR * bid (for CPC) or bid (for CPM)
        def calc_ecpm(item):
            candidate, pctr = item
            if candidate.bid_type == 1:  # CPM
                return candidate.bid
            else:  # CPC
                return pctr * candidate.bid * 1000  # Convert to CPM scale

        sorted_candidates = sorted(scored_candidates, key=calc_ecpm, reverse=True)
    else:
        # Just sort by pCTR
        sorted_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)

    return [c for c, _ in sorted_candidates]


def format_ad_response(
    request_id: str,
    selected_ads: list[MockAdCandidate],
) -> dict:
    """Format ad response."""
    ads = []
    for ad in selected_ads:
        ads.append({
            "ad_id": f"ad_{ad.campaign_id}_{ad.creative_id}",
            "campaign_id": ad.campaign_id,
            "creative_id": ad.creative_id,
            "advertiser_id": ad.advertiser_id,
            "pctr": round(ad.pctr, 4),
            "bid": round(ad.bid, 2),
            "metadata": ad.metadata,
        })

    return {
        "request_id": request_id,
        "ads": ads,
        "fill_count": len(ads),
    }


async def test_full_flow():
    """Test the complete ad request flow."""
    print("=" * 60)
    print("Testing Full Ad Request Flow with LR Model")
    print("=" * 60)

    # Step 1: Load model
    print("\n[Step 1] Loading trained LR model...")
    model_path = project_root / "models" / "lr_ctr.pt"
    if not model_path.exists():
        print(f"  Model not found at {model_path}")
        print("  Please run e2e_test_lr.py first to train a model")
        return

    predictor = load_lr_model(str(model_path))
    print(f"  Model type: {predictor.model_type}")
    print(f"  Sparse features: {len(predictor.model.sparse_feature_dims)}")
    print(f"  Dense features: {predictor.model.dense_feature_dim}")

    # Step 2: Create user context (simulate incoming request)
    print("\n[Step 2] Simulating ad request...")
    user_context = MockUserContext(
        user_id=f"user_{random.randint(1, 10000)}",
        gender="male",
        age=28,
        os="android",
        country="CN",
        city="shanghai",
        interests=["gaming", "technology"],
        metadata={
            "slot_id": "banner_home",
            "hour": 14,
            "day_of_week": 5,
        }
    )
    print(f"  User ID: {user_context.user_id}")
    print(f"  Device: {user_context.os}")
    print(f"  Location: {user_context.city}, {user_context.country}")

    # Step 3: Retrieve candidates (mock)
    print("\n[Step 3] Retrieving ad candidates...")
    candidates = generate_mock_candidates(num_candidates=20)
    print(f"  Retrieved {len(candidates)} candidates")

    # Step 4: CTR prediction
    print("\n[Step 4] Running CTR prediction with LR model...")
    start_time = time.time()
    scored_candidates = await run_prediction_pipeline(predictor, user_context, candidates)
    prediction_time = (time.time() - start_time) * 1000
    print(f"  Prediction latency: {prediction_time:.2f}ms for {len(candidates)} candidates")

    pctrs = [pctr for _, pctr in scored_candidates]
    print(f"  pCTR range: [{min(pctrs):.4f}, {max(pctrs):.4f}]")
    print(f"  Mean pCTR: {sum(pctrs) / len(pctrs):.4f}")

    # Step 5: Ranking
    print("\n[Step 5] Ranking by eCPM...")
    ranked_candidates = rank_candidates(scored_candidates, "ecpm")
    print("  Top 5 candidates:")
    for i, c in enumerate(ranked_candidates[:5]):
        ecpm = c.pctr * c.bid * 1000 if c.bid_type == 2 else c.bid
        print(f"    {i + 1}. Campaign {c.campaign_id}: pCTR={c.pctr:.4f}, bid={c.bid:.2f}, eCPM={ecpm:.2f}")

    # Step 6: Format response
    print("\n[Step 6] Formatting response...")
    request_id = f"req_{int(time.time() * 1000)}"
    num_ads = 3
    selected_ads = ranked_candidates[:num_ads]
    response = format_ad_response(request_id, selected_ads)

    print(f"\n  Response:")
    print(f"    Request ID: {response['request_id']}")
    print(f"    Fill count: {response['fill_count']}")
    for ad in response["ads"]:
        print(f"    - Ad {ad['ad_id']}: pCTR={ad['pctr']}, bid={ad['bid']}")

    # Summary
    print("\n" + "=" * 60)
    print("Flow Complete!")
    print("=" * 60)
    print(f"\nTotal end-to-end time: {(time.time() - start_time) * 1000:.2f}ms")
    print("\nThis flow demonstrates:")
    print("  1. Ad request parsing")
    print("  2. Candidate retrieval (mocked)")
    print("  3. CTR prediction using trained LR model")
    print("  4. eCPM-based ranking")
    print("  5. Response formatting")


async def benchmark_prediction(iterations: int = 100):
    """Benchmark prediction performance."""
    print("\n" + "=" * 60)
    print(f"Benchmarking Prediction ({iterations} iterations)")
    print("=" * 60)

    model_path = project_root / "models" / "lr_ctr.pt"
    if not model_path.exists():
        print("Model not found")
        return

    predictor = load_lr_model(str(model_path))

    # Prepare features
    sparse_dims = predictor.model.sparse_feature_dims
    num_dense = predictor.model.dense_feature_dim
    batch_sizes = [1, 10, 50, 100]

    for batch_size in batch_sizes:
        features_batch = []
        for _ in range(batch_size):
            features_batch.append({
                "sparse_features": [random.randint(0, vs - 1) for vs in sparse_dims],
                "dense_features": [random.gauss(0, 1) for _ in range(num_dense)],
            })

        # Warmup
        for _ in range(5):
            predictor.predict_batch(features_batch)

        # Benchmark
        latencies = []
        for _ in range(iterations):
            start = time.time()
            predictor.predict_batch(features_batch)
            latencies.append((time.time() - start) * 1000)

        avg = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]

        print(f"\nBatch size {batch_size}:")
        print(f"  Avg latency: {avg:.3f}ms")
        print(f"  P50 latency: {p50:.3f}ms")
        print(f"  P99 latency: {p99:.3f}ms")
        print(f"  Throughput:  {batch_size / (avg / 1000):.0f} predictions/sec")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test full ad request flow")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    if args.benchmark:
        asyncio.run(benchmark_prediction(args.iterations))
    else:
        asyncio.run(test_full_flow())


if __name__ == "__main__":
    main()
