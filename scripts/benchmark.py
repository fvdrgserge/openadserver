#!/usr/bin/env python3
"""
Performance benchmark script for LiteAds.

Tests API throughput and latency under load.

Usage:
    python scripts/benchmark.py --url http://localhost:8000 --concurrency 50 --requests 1000
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any

import httpx

from liteads.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    error_rate: float


def generate_ad_request() -> dict[str, Any]:
    """Generate a random ad request."""
    return {
        "slot_id": f"slot_{random.randint(1, 5)}",
        "user_id": f"bench_user_{random.randint(1, 10000)}",
        "device": {
            "os": random.choice(["android", "ios"]),
            "os_version": f"{random.randint(10, 17)}.{random.randint(0, 5)}",
            "model": random.choice(["iPhone 15", "Pixel 8", "Galaxy S24"]),
            "screen_width": 1080,
            "screen_height": 2340,
        },
        "geo": {
            "country": random.choice(["CN", "US"]),
            "city": random.choice(["shanghai", "beijing", "new_york"]),
        },
        "context": {
            "app_id": f"app_{random.randint(1, 50)}",
            "app_version": "1.0.0",
            "network": random.choice(["wifi", "4g"]),
        },
        "num_ads": 1,
    }


async def make_request(
    client: httpx.AsyncClient,
    url: str,
    request_data: dict[str, Any],
) -> tuple[bool, float]:
    """Make a single request and return (success, latency_ms)."""
    start_time = time.perf_counter()
    try:
        response = await client.post(url, json=request_data)
        latency_ms = (time.perf_counter() - start_time) * 1000
        success = response.status_code == 200
        return success, latency_ms
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return False, latency_ms


async def worker(
    client: httpx.AsyncClient,
    url: str,
    num_requests: int,
    results: list[tuple[bool, float]],
) -> None:
    """Worker coroutine that makes requests."""
    for _ in range(num_requests):
        request_data = generate_ad_request()
        result = await make_request(client, url, request_data)
        results.append(result)


async def run_benchmark(
    base_url: str,
    concurrency: int,
    total_requests: int,
    timeout: float = 30.0,
) -> BenchmarkResult:
    """Run benchmark with specified concurrency."""
    url = f"{base_url.rstrip('/')}/api/v1/ad/request"

    logger.info(f"Starting benchmark: {total_requests} requests with {concurrency} concurrent workers")
    logger.info(f"Target URL: {url}")

    # Calculate requests per worker
    requests_per_worker = total_requests // concurrency
    extra_requests = total_requests % concurrency

    results: list[tuple[bool, float]] = []

    # Create client with connection pool
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        limits=httpx.Limits(
            max_keepalive_connections=concurrency,
            max_connections=concurrency * 2,
        ),
    ) as client:
        start_time = time.perf_counter()

        # Create workers
        workers = []
        for i in range(concurrency):
            num = requests_per_worker + (1 if i < extra_requests else 0)
            workers.append(worker(client, url, num, results))

        # Run workers concurrently
        await asyncio.gather(*workers)

        total_time = time.perf_counter() - start_time

    # Calculate statistics
    successful = sum(1 for success, _ in results if success)
    failed = len(results) - successful
    latencies = [latency for _, latency in results]

    if not latencies:
        return BenchmarkResult(
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            total_time=total_time,
            avg_latency_ms=0,
            min_latency_ms=0,
            max_latency_ms=0,
            p50_latency_ms=0,
            p95_latency_ms=0,
            p99_latency_ms=0,
            requests_per_second=0,
            error_rate=1.0,
        )

    latencies.sort()

    def percentile(p: float) -> float:
        idx = int(len(latencies) * p)
        return latencies[min(idx, len(latencies) - 1)]

    return BenchmarkResult(
        total_requests=len(results),
        successful_requests=successful,
        failed_requests=failed,
        total_time=total_time,
        avg_latency_ms=statistics.mean(latencies),
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        p50_latency_ms=percentile(0.50),
        p95_latency_ms=percentile(0.95),
        p99_latency_ms=percentile(0.99),
        requests_per_second=len(results) / total_time,
        error_rate=failed / len(results) if results else 0,
    )


def print_results(result: BenchmarkResult) -> None:
    """Print benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nRequests:")
    print(f"  Total:      {result.total_requests}")
    print(f"  Successful: {result.successful_requests}")
    print(f"  Failed:     {result.failed_requests}")
    print(f"  Error Rate: {result.error_rate * 100:.2f}%")

    print(f"\nThroughput:")
    print(f"  Total Time: {result.total_time:.2f}s")
    print(f"  RPS:        {result.requests_per_second:.2f}")

    print(f"\nLatency (ms):")
    print(f"  Min:  {result.min_latency_ms:.2f}")
    print(f"  Avg:  {result.avg_latency_ms:.2f}")
    print(f"  Max:  {result.max_latency_ms:.2f}")
    print(f"  P50:  {result.p50_latency_ms:.2f}")
    print(f"  P95:  {result.p95_latency_ms:.2f}")
    print(f"  P99:  {result.p99_latency_ms:.2f}")

    print("\n" + "=" * 60)


async def warmup(base_url: str, num_requests: int = 100) -> None:
    """Warmup the server with initial requests."""
    url = f"{base_url.rstrip('/')}/api/v1/ad/request"
    logger.info(f"Warming up with {num_requests} requests...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        for _ in range(num_requests):
            try:
                await client.post(url, json=generate_ad_request())
            except Exception:
                pass

    logger.info("Warmup complete")


async def main_async(args: argparse.Namespace) -> None:
    """Async main function."""
    # Health check
    health_url = f"{args.url.rstrip('/')}/health"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(health_url)
            if response.status_code != 200:
                logger.error(f"Server health check failed: {response.status_code}")
                return
        except Exception as e:
            logger.error(f"Cannot connect to server: {e}")
            return

    logger.info("Server health check passed")

    # Warmup
    if args.warmup:
        await warmup(args.url, args.warmup)

    # Run benchmark
    result = await run_benchmark(
        base_url=args.url,
        concurrency=args.concurrency,
        total_requests=args.requests,
        timeout=args.timeout,
    )

    # Print results
    print_results(result)

    # Save results if output specified
    if args.output:
        import json
        from dataclasses import asdict

        with open(args.output, "w") as f:
            json.dump(asdict(result), f, indent=2)
        logger.info(f"Results saved to {args.output}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark LiteAds API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the server",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=50,
        help="Number of concurrent workers",
    )
    parser.add_argument(
        "--requests",
        "-n",
        type=int,
        default=1000,
        help="Total number of requests",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Number of warmup requests (0 to skip)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )

    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
