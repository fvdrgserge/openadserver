"""
Feature builder for transforming raw data to model inputs.

Coordinates feature processors to build model-ready tensors.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from liteads.common.logger import get_logger
from liteads.ml_engine.features.config import (
    FeatureConfig,
    FeaturesConfigLoader,
    FeaturesConfigSchema,
)
from liteads.ml_engine.features.processor import FeaturePipeline, FeatureProcessorFactory

logger = get_logger(__name__)


@dataclass
class FeatureInfo:
    """Information about a processed feature."""

    name: str
    type: str
    input_dim: int  # Vocabulary size or feature dimension
    embedding_dim: int | None = None  # Embedding dimension if applicable
    offset: int = 0  # Offset in concatenated sparse tensor


@dataclass
class ModelInputs:
    """Model input tensors."""

    # Sparse features (ID, discrete, cross) - indices into embedding tables
    sparse_features: torch.Tensor  # (batch_size, num_sparse_features)

    # Dense features (continuous, time extracted)
    dense_features: torch.Tensor  # (batch_size, num_dense_features)

    # Sequence features with variable length
    sequence_features: dict[str, tuple[torch.Tensor, torch.Tensor]]  # name -> (values, offsets)

    # Labels (if available)
    labels: torch.Tensor | None = None  # (batch_size,) or (batch_size, num_tasks)

    # Feature metadata
    feature_info: dict[str, FeatureInfo] = field(default_factory=dict)


class FeatureBuilder:
    """
    Feature builder that transforms raw data to model inputs.

    Handles the full pipeline:
    1. Load feature configuration
    2. Initialize feature processors
    3. Fit processors on training data
    4. Transform data to model-ready tensors
    """

    def __init__(
        self,
        config_path: str | None = None,
        device: str = "cpu",
    ):
        """
        Initialize feature builder.

        Args:
            config_path: Path to features_config.yaml
            device: Device for tensors (cpu/cuda)
        """
        self.config_loader = FeaturesConfigLoader(config_path)
        self.config: FeaturesConfigSchema | None = None
        self.pipeline: FeaturePipeline | None = None
        self.device = device

        # Feature metadata after fitting
        self._sparse_feature_names: list[str] = []
        self._dense_feature_names: list[str] = []
        self._sequence_feature_names: list[str] = []
        self._feature_info: dict[str, FeatureInfo] = {}

        self._is_fitted = False

    def _init_pipeline(self) -> None:
        """Initialize feature processing pipeline."""
        self.config = self.config_loader.load()
        self.pipeline = FeaturePipeline(self.config)

        # Collect feature names by category
        for feature_config in self.config_loader.get_all_features():
            if feature_config.type in ("id", "discrete", "cross"):
                self._sparse_feature_names.append(feature_config.name)
            elif feature_config.type in ("continuous",):
                self._dense_feature_names.append(feature_config.name)
            elif feature_config.type == "time":
                # Time features expand to multiple dense features
                for extract in feature_config.extract:
                    self._dense_feature_names.append(f"{feature_config.name}_{extract}")
            elif feature_config.type == "sequence":
                self._sequence_feature_names.append(feature_config.name)

    def fit(self, data: list[dict[str, Any]]) -> "FeatureBuilder":
        """
        Fit feature processors on training data.

        Args:
            data: List of sample dictionaries

        Returns:
            Self for chaining
        """
        logger.info(f"Fitting feature builder on {len(data)} samples")

        if self.pipeline is None:
            self._init_pipeline()

        assert self.pipeline is not None
        self.pipeline.fit(data)

        # Build feature info after fitting
        self._build_feature_info()

        self._is_fitted = True
        logger.info("Feature builder fitted successfully")

        return self

    def _build_feature_info(self) -> None:
        """Build feature info metadata after fitting."""
        assert self.pipeline is not None
        assert self.config is not None

        default_dim = self.config.model.default_embedding_dim

        for feature_config in self.config_loader.get_all_features():
            name = feature_config.name
            processor = self.pipeline.processors.get(name)

            if processor is None:
                continue

            if feature_config.type == "id":
                vocab_size = len(processor.encoder.classes_) + 1  # +1 for unknown
                self._feature_info[name] = FeatureInfo(
                    name=name,
                    type="id",
                    input_dim=vocab_size,
                    embedding_dim=feature_config.embedding_dim or default_dim,
                )

            elif feature_config.type == "discrete":
                vocab_size = len(processor.encoder.classes_) + 1
                self._feature_info[name] = FeatureInfo(
                    name=name,
                    type="discrete",
                    input_dim=vocab_size,
                    embedding_dim=feature_config.embedding_dim or default_dim,
                )

            elif feature_config.type == "continuous":
                self._feature_info[name] = FeatureInfo(
                    name=name,
                    type="continuous",
                    input_dim=1,
                )

            elif feature_config.type == "time":
                for extract in feature_config.extract:
                    feat_name = f"{name}_{extract}"
                    self._feature_info[feat_name] = FeatureInfo(
                        name=feat_name,
                        type="time",
                        input_dim=1,
                    )

            elif feature_config.type == "sequence":
                vocab_size = len(processor.encoder.classes_) + 1
                self._feature_info[name] = FeatureInfo(
                    name=name,
                    type="sequence",
                    input_dim=vocab_size,
                    embedding_dim=feature_config.embedding_dim or default_dim,
                )

            elif feature_config.type == "cross":
                self._feature_info[name] = FeatureInfo(
                    name=name,
                    type="cross",
                    input_dim=feature_config.hash_buckets,
                    embedding_dim=feature_config.embedding_dim or default_dim,
                )

    def transform(
        self,
        data: list[dict[str, Any]],
        labels: np.ndarray | None = None,
    ) -> ModelInputs:
        """
        Transform data to model inputs.

        Args:
            data: List of sample dictionaries
            labels: Optional labels array

        Returns:
            ModelInputs with tensors ready for model
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureBuilder must be fitted before transform")

        assert self.pipeline is not None

        # Transform through pipeline
        transformed = self.pipeline.transform(data)

        # Build sparse features tensor
        sparse_values = []
        for name in self._sparse_feature_names:
            if name in transformed:
                sparse_values.append(transformed[name].reshape(-1, 1))

        sparse_tensor = (
            torch.tensor(
                np.hstack(sparse_values),
                dtype=torch.long,
                device=self.device,
            )
            if sparse_values
            else torch.empty(len(data), 0, dtype=torch.long, device=self.device)
        )

        # Build dense features tensor
        dense_values = []
        for name in self._dense_feature_names:
            if name in transformed:
                val = transformed[name]
                if val.ndim == 1:
                    val = val.reshape(-1, 1)
                dense_values.append(val)

        dense_tensor = (
            torch.tensor(
                np.hstack(dense_values),
                dtype=torch.float32,
                device=self.device,
            )
            if dense_values
            else torch.empty(len(data), 0, dtype=torch.float32, device=self.device)
        )

        # Build sequence features
        sequence_features = {}
        for name in self._sequence_feature_names:
            if name in transformed:
                values, offsets = transformed[name]
                sequence_features[name] = (
                    torch.tensor(values, dtype=torch.long, device=self.device),
                    torch.tensor(offsets, dtype=torch.long, device=self.device),
                )

        # Build labels tensor
        labels_tensor = None
        if labels is not None:
            labels_tensor = torch.tensor(
                labels,
                dtype=torch.float32,
                device=self.device,
            )

        return ModelInputs(
            sparse_features=sparse_tensor,
            dense_features=dense_tensor,
            sequence_features=sequence_features,
            labels=labels_tensor,
            feature_info=self._feature_info.copy(),
        )

    def fit_transform(
        self,
        data: list[dict[str, Any]],
        labels: np.ndarray | None = None,
    ) -> ModelInputs:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data, labels)

    def get_model_config(self) -> dict[str, Any]:
        """
        Get configuration for model initialization.

        Returns:
            Dictionary with model configuration parameters
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureBuilder must be fitted first")

        assert self.config is not None

        # Collect sparse feature dimensions
        sparse_feature_dims = []
        sparse_embedding_dims = []
        for name in self._sparse_feature_names:
            info = self._feature_info[name]
            sparse_feature_dims.append(info.input_dim)
            sparse_embedding_dims.append(info.embedding_dim or self.config.model.default_embedding_dim)

        # Collect sequence feature dimensions
        sequence_feature_dims = {}
        sequence_embedding_dims = {}
        for name in self._sequence_feature_names:
            info = self._feature_info[name]
            sequence_feature_dims[name] = info.input_dim
            sequence_embedding_dims[name] = info.embedding_dim or self.config.model.default_embedding_dim

        return {
            "sparse_feature_dims": sparse_feature_dims,
            "sparse_embedding_dims": sparse_embedding_dims,
            "dense_feature_dim": len(self._dense_feature_names),
            "sequence_feature_dims": sequence_feature_dims,
            "sequence_embedding_dims": sequence_embedding_dims,
            "sparse_feature_names": self._sparse_feature_names.copy(),
            "dense_feature_names": self._dense_feature_names.copy(),
            "sequence_feature_names": self._sequence_feature_names.copy(),
            "fm_k": self.config.model.fm_k,
            "dnn_hidden_units": self.config.model.dnn_hidden_units,
            "dnn_dropout": self.config.model.dnn_dropout,
            "dnn_activation": self.config.model.dnn_activation,
            "l2_reg_embedding": self.config.model.l2_reg_embedding,
            "l2_reg_dnn": self.config.model.l2_reg_dnn,
        }

    def save(self, path: str) -> None:
        """Save fitted feature builder state."""
        import pickle

        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted FeatureBuilder")

        state = {
            "pipeline_state": self.pipeline.get_state() if self.pipeline else None,
            "sparse_feature_names": self._sparse_feature_names,
            "dense_feature_names": self._dense_feature_names,
            "sequence_feature_names": self._sequence_feature_names,
            "feature_info": self._feature_info,
            "config_path": str(self.config_loader.config_path),
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Feature builder saved to {path}")

    def load(self, path: str) -> "FeatureBuilder":
        """Load fitted feature builder state."""
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        self._init_pipeline()

        if self.pipeline and state["pipeline_state"]:
            self.pipeline.set_state(state["pipeline_state"])

        self._sparse_feature_names = state["sparse_feature_names"]
        self._dense_feature_names = state["dense_feature_names"]
        self._sequence_feature_names = state["sequence_feature_names"]
        self._feature_info = state["feature_info"]
        self._is_fitted = True

        logger.info(f"Feature builder loaded from {path}")

        return self
