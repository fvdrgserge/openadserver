"""
Model serving for online prediction.

Provides:
- ModelPredictor: High-performance inference
- ModelCache: Model caching and hot-swap
- BatchPredictor: Efficient batch prediction
"""

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from liteads.common.logger import get_logger
from liteads.ml_engine.features.builder import FeatureBuilder
from liteads.ml_engine.models.deepfm import DeepFM
from liteads.ml_engine.models.lr import FactorizationMachineLR, LogisticRegression

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """Result from model prediction."""

    pctr: float
    pcvr: float | None = None
    pctcvr: float | None = None
    model_version: str = ""
    latency_ms: float = 0.0


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    version: str
    path: str
    loaded_at: float
    num_predictions: int = 0
    avg_latency_ms: float = 0.0


class ModelPredictor:
    """
    High-performance model predictor for online serving.

    Features:
    - Lazy model loading
    - Automatic batching
    - Device management (CPU/GPU)
    - Model warm-up
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        feature_builder_path: str | Path | None = None,
        device: str = "auto",
        use_fp16: bool = False,
        warmup_samples: int = 100,
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to model checkpoint
            feature_builder_path: Path to feature builder state
            device: Device for inference (auto, cpu, cuda, mps)
            use_fp16: Use half precision for inference
            warmup_samples: Number of warmup samples
        """
        self.model_path = Path(model_path) if model_path else None
        self.feature_builder_path = Path(feature_builder_path) if feature_builder_path else None
        self.use_fp16 = use_fp16
        self.warmup_samples = warmup_samples

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Model and feature builder
        self.model: nn.Module | None = None
        self.feature_builder: FeatureBuilder | None = None
        self.model_info: ModelInfo | None = None
        self.model_type: str = "deepfm"  # deepfm, lr, fm_lr

        # Lock for thread-safe operations
        self._lock = threading.Lock()
        self._is_loaded = False

    def load(self) -> None:
        """Load model and feature builder."""
        with self._lock:
            if self._is_loaded:
                return

            start_time = time.time()

            # Load feature builder
            if self.feature_builder_path and self.feature_builder_path.exists():
                self.feature_builder = FeatureBuilder(device=str(self.device))
                self.feature_builder.load(str(self.feature_builder_path))
                logger.info(f"Loaded feature builder from {self.feature_builder_path}")
            else:
                self.feature_builder = FeatureBuilder(device=str(self.device))

            # Load model
            if self.model_path and self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Get model config
                if self.feature_builder._is_fitted:
                    model_config = self.feature_builder.get_model_config()
                else:
                    # Use config from checkpoint if available
                    model_config = checkpoint.get("model_config", {})

                # Detect model type from checkpoint
                self.model_type = checkpoint.get("model_type", "deepfm")

                # Create model based on type
                if self.model_type == "lr":
                    self.model = LogisticRegression(
                        sparse_feature_dims=model_config.get("sparse_feature_dims", []),
                        dense_feature_dim=model_config.get("dense_feature_dim", 0),
                        l2_reg=model_config.get("l2_reg_embedding", 0.0001),
                    )
                elif self.model_type == "fm_lr":
                    self.model = FactorizationMachineLR(
                        sparse_feature_dims=model_config.get("sparse_feature_dims", []),
                        dense_feature_dim=model_config.get("dense_feature_dim", 0),
                        embedding_dim=model_config.get("fm_k", 8),
                        l2_reg=model_config.get("l2_reg_embedding", 0.0001),
                    )
                else:  # deepfm
                    self.model = DeepFM(
                        sparse_feature_dims=model_config.get("sparse_feature_dims", []),
                        sparse_embedding_dims=model_config.get("sparse_embedding_dims", 8),
                        dense_feature_dim=model_config.get("dense_feature_dim", 0),
                        sequence_feature_dims=model_config.get("sequence_feature_dims", {}),
                        sequence_embedding_dims=model_config.get("sequence_embedding_dims", {}),
                        fm_k=model_config.get("fm_k", 8),
                        dnn_hidden_units=model_config.get("dnn_hidden_units", [256, 128, 64]),
                        dnn_dropout=0.0,  # No dropout during inference
                    )

                # Load weights
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model = self.model.to(self.device)
                self.model.eval()

                # Half precision
                if self.use_fp16 and self.device.type == "cuda":
                    self.model = self.model.half()

                # Model info
                version = checkpoint.get("version", self.model_path.stem)
                self.model_info = ModelInfo(
                    version=version,
                    path=str(self.model_path),
                    loaded_at=time.time(),
                )

                logger.info(f"Loaded model from {self.model_path}")

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")

            # Warmup
            if self.model and self.warmup_samples > 0:
                self._warmup()

            self._is_loaded = True

    def _warmup(self) -> None:
        """Warm up model with dummy predictions."""
        logger.info(f"Warming up model with {self.warmup_samples} samples ({self.model_type})...")

        # Get dimensions from model
        if hasattr(self.model, "sparse_feature_dims"):
            sparse_dims = self.model.sparse_feature_dims
            num_sparse = len(sparse_dims)
        else:
            num_sparse = len(getattr(self.model, "sparse_weights", []))
            sparse_dims = [100] * num_sparse  # default

        if hasattr(self.model, "dense_feature_dim"):
            num_dense = self.model.dense_feature_dim
        else:
            num_dense = 0

        # Create dummy inputs with valid indices per feature
        sparse_features = torch.zeros(self.warmup_samples, num_sparse, dtype=torch.long, device=self.device)
        for i, vocab_size in enumerate(sparse_dims):
            sparse_features[:, i] = torch.randint(0, vocab_size, (self.warmup_samples,), device=self.device)

        dense_features = torch.randn(
            self.warmup_samples, num_dense, device=self.device
        )

        if self.use_fp16:
            dense_features = dense_features.half()

        # Run warmup predictions
        with torch.no_grad():
            for _ in range(3):  # Multiple passes
                _ = self.model(sparse_features, dense_features)

        logger.info("Model warmup complete")

    def predict(self, features: dict[str, Any]) -> PredictionResult:
        """
        Make prediction for a single sample.

        Args:
            features: Dictionary of feature values

        Returns:
            PredictionResult with predictions
        """
        return self.predict_batch([features])[0]

    def predict_batch(self, features_batch: list[dict[str, Any]]) -> list[PredictionResult]:
        """
        Make predictions for a batch of samples.

        Args:
            features_batch: List of feature dictionaries

        Returns:
            List of PredictionResult
        """
        if not self._is_loaded:
            self.load()

        if self.model is None:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        # Transform features
        if self.feature_builder and self.feature_builder._is_fitted:
            inputs = self.feature_builder.transform(features_batch)
            sparse_features = inputs.sparse_features
            dense_features = inputs.dense_features
        else:
            # Fallback: expect pre-transformed features
            sparse_features = torch.tensor(
                [f.get("sparse_features", []) for f in features_batch],
                dtype=torch.long,
                device=self.device,
            )
            dense_features = torch.tensor(
                [f.get("dense_features", []) for f in features_batch],
                dtype=torch.float32,
                device=self.device,
            )

        # Half precision
        if self.use_fp16:
            dense_features = dense_features.half()

        # Inference
        with torch.no_grad():
            outputs = self.model(sparse_features, dense_features)

        latency_ms = (time.time() - start_time) * 1000

        # Update stats
        if self.model_info:
            self.model_info.num_predictions += len(features_batch)
            # Running average
            n = self.model_info.num_predictions
            self.model_info.avg_latency_ms = (
                (self.model_info.avg_latency_ms * (n - 1) + latency_ms / len(features_batch)) / n
            )

        # Build results
        results = []
        for i in range(len(features_batch)):
            pctr = outputs.get("ctr", outputs.get("task_0", list(outputs.values())[0]))[i].item()
            pcvr = outputs.get("cvr", outputs.get("task_1"))[i].item() if "cvr" in outputs or "task_1" in outputs else None
            pctcvr = outputs.get("ctcvr")[i].item() if "ctcvr" in outputs else None

            results.append(PredictionResult(
                pctr=pctr,
                pcvr=pcvr,
                pctcvr=pctcvr,
                model_version=self.model_info.version if self.model_info else "",
                latency_ms=latency_ms / len(features_batch),
            ))

        return results

    async def predict_async(self, features: dict[str, Any]) -> PredictionResult:
        """Async prediction for non-blocking inference."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, features)

    async def predict_batch_async(self, features_batch: list[dict[str, Any]]) -> list[PredictionResult]:
        """Async batch prediction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict_batch, features_batch)


class BatchingPredictor:
    """
    Predictor with automatic request batching.

    Collects requests over a time window and processes them together
    for improved throughput.
    """

    def __init__(
        self,
        predictor: ModelPredictor,
        batch_size: int = 32,
        max_wait_ms: float = 5.0,
    ):
        """
        Initialize batching predictor.

        Args:
            predictor: Underlying model predictor
            batch_size: Maximum batch size
            max_wait_ms: Maximum wait time before processing batch
        """
        self.predictor = predictor
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms

        self._queue: deque = deque()
        self._results: dict[int, PredictionResult] = {}
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._request_id = 0

        # Start background worker
        self._running = True
        self._worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self._worker_thread.start()

    def _batch_worker(self) -> None:
        """Background worker that processes batches."""
        while self._running:
            self._event.wait(timeout=self.max_wait_ms / 1000)
            self._event.clear()

            # Collect batch
            with self._lock:
                if not self._queue:
                    continue

                batch = []
                request_ids = []
                while self._queue and len(batch) < self.batch_size:
                    req_id, features = self._queue.popleft()
                    batch.append(features)
                    request_ids.append(req_id)

            if not batch:
                continue

            # Process batch
            try:
                results = self.predictor.predict_batch(batch)

                # Store results
                with self._lock:
                    for req_id, result in zip(request_ids, results):
                        self._results[req_id] = result
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                # Store error results
                with self._lock:
                    for req_id in request_ids:
                        self._results[req_id] = PredictionResult(pctr=0.01)

    def predict(self, features: dict[str, Any], timeout_ms: float = 100.0) -> PredictionResult:
        """
        Submit prediction request and wait for result.

        Args:
            features: Feature dictionary
            timeout_ms: Maximum wait time for result

        Returns:
            PredictionResult
        """
        # Generate request ID
        with self._lock:
            req_id = self._request_id
            self._request_id += 1
            self._queue.append((req_id, features))

        # Signal worker
        self._event.set()

        # Wait for result
        start_time = time.time()
        while time.time() - start_time < timeout_ms / 1000:
            with self._lock:
                if req_id in self._results:
                    return self._results.pop(req_id)
            time.sleep(0.001)

        # Timeout - return default
        logger.warning(f"Prediction timeout for request {req_id}")
        return PredictionResult(pctr=0.01)

    def shutdown(self) -> None:
        """Shutdown the batching predictor."""
        self._running = False
        self._event.set()
        self._worker_thread.join(timeout=1.0)


class ModelCache:
    """
    Model cache for hot-swapping models.

    Supports:
    - Multiple model versions
    - Automatic model loading
    - Hot-swap without downtime
    """

    def __init__(
        self,
        model_dir: str | Path,
        max_models: int = 3,
        device: str = "auto",
    ):
        """
        Initialize model cache.

        Args:
            model_dir: Directory containing model checkpoints
            max_models: Maximum models to keep in cache
            device: Device for inference
        """
        self.model_dir = Path(model_dir)
        self.max_models = max_models
        self.device = device

        self._models: dict[str, ModelPredictor] = {}
        self._current_version: str | None = None
        self._lock = threading.Lock()

    def load_model(self, version: str) -> ModelPredictor:
        """
        Load a specific model version.

        Args:
            version: Model version string

        Returns:
            ModelPredictor instance
        """
        with self._lock:
            if version in self._models:
                return self._models[version]

            # Check cache size
            if len(self._models) >= self.max_models:
                # Remove oldest model (not current)
                for v in list(self._models.keys()):
                    if v != self._current_version:
                        del self._models[v]
                        break

            # Load model
            model_path = self.model_dir / f"{version}.pt"
            feature_builder_path = self.model_dir / f"{version}_features.pkl"

            predictor = ModelPredictor(
                model_path=model_path,
                feature_builder_path=feature_builder_path,
                device=self.device,
            )
            predictor.load()

            self._models[version] = predictor
            return predictor

    def set_current(self, version: str) -> None:
        """Set the current active model version."""
        with self._lock:
            if version not in self._models:
                self.load_model(version)
            self._current_version = version

    def get_current(self) -> ModelPredictor | None:
        """Get the current active model."""
        with self._lock:
            if self._current_version:
                return self._models.get(self._current_version)
            return None

    def list_versions(self) -> list[str]:
        """List available model versions."""
        versions = []
        for path in self.model_dir.glob("*.pt"):
            versions.append(path.stem)
        return sorted(versions)
