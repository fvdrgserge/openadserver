"""
Feature Processor with Factory Pattern.

Provides unified interface for processing different feature types.
Inspired by ml-interview project's elegant design.
"""

import hashlib
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from liteads.common.logger import get_logger
from liteads.ml_engine.features.config import FeatureConfig, get_feature_config

logger = get_logger(__name__)


# ==============================================================================
# Base Feature Processor (Abstract Factory)
# ==============================================================================


class BaseFeatureProcessor(ABC):
    """Abstract base class for feature processors."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.name = config.name
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: pd.Series) -> "BaseFeatureProcessor":
        """Fit the processor on training data."""
        pass

    @abstractmethod
    def transform(self, data: pd.Series) -> np.ndarray | dict[str, np.ndarray]:
        """Transform data using fitted processor."""
        pass

    def fit_transform(self, data: pd.Series) -> np.ndarray | dict[str, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)

    @abstractmethod
    def get_output_dim(self) -> int:
        """Get output dimension for this feature."""
        pass

    def get_vocab_size(self) -> int | None:
        """Get vocabulary size (for embedding features)."""
        return None


# ==============================================================================
# Concrete Feature Processors
# ==============================================================================


class IDFeatureProcessor(BaseFeatureProcessor):
    """
    ID Feature Processor.

    Encodes ID features using LabelEncoder for embedding lookup.
    """

    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.encoder = LabelEncoder()
        self.unknown_idx = 0  # Index for unknown values

    def fit(self, data: pd.Series) -> "IDFeatureProcessor":
        # Fill NA and convert to string
        data_clean = data.fillna("__UNKNOWN__").astype(str)

        # Fit encoder
        self.encoder.fit(data_clean)

        # Add unknown class if not present
        if "__UNKNOWN__" not in self.encoder.classes_:
            self.encoder.classes_ = np.append(
                self.encoder.classes_, "__UNKNOWN__"
            )
        self.unknown_idx = np.where(
            self.encoder.classes_ == "__UNKNOWN__"
        )[0][0]

        self.is_fitted = True
        logger.debug(
            f"IDFeatureProcessor fitted: {self.name}, "
            f"vocab_size={len(self.encoder.classes_)}"
        )
        return self

    def transform(self, data: pd.Series) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"Processor {self.name} not fitted")

        data_clean = data.fillna("__UNKNOWN__").astype(str)

        # Handle unseen values
        result = np.full(len(data_clean), self.unknown_idx, dtype=np.int64)
        known_mask = data_clean.isin(self.encoder.classes_)
        if known_mask.any():
            result[known_mask] = self.encoder.transform(data_clean[known_mask])

        return result

    def get_output_dim(self) -> int:
        return self.config.embedding_dim or 16

    def get_vocab_size(self) -> int:
        return len(self.encoder.classes_)


class DiscreteFeatureProcessor(BaseFeatureProcessor):
    """
    Discrete Feature Processor.

    Encodes discrete features using LabelEncoder.
    """

    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.encoder = LabelEncoder()
        self.unknown_idx = 0

    def fit(self, data: pd.Series) -> "DiscreteFeatureProcessor":
        data_clean = data.fillna("__UNKNOWN__").astype(str)
        self.encoder.fit(data_clean)

        if "__UNKNOWN__" not in self.encoder.classes_:
            self.encoder.classes_ = np.append(
                self.encoder.classes_, "__UNKNOWN__"
            )
        self.unknown_idx = np.where(
            self.encoder.classes_ == "__UNKNOWN__"
        )[0][0]

        self.is_fitted = True
        logger.debug(
            f"DiscreteFeatureProcessor fitted: {self.name}, "
            f"vocab_size={len(self.encoder.classes_)}"
        )
        return self

    def transform(self, data: pd.Series) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"Processor {self.name} not fitted")

        data_clean = data.fillna("__UNKNOWN__").astype(str)

        result = np.full(len(data_clean), self.unknown_idx, dtype=np.int64)
        known_mask = data_clean.isin(self.encoder.classes_)
        if known_mask.any():
            result[known_mask] = self.encoder.transform(data_clean[known_mask])

        return result

    def get_output_dim(self) -> int:
        return 1  # Single encoded value

    def get_vocab_size(self) -> int:
        return len(self.encoder.classes_)


class ContinuousFeatureProcessor(BaseFeatureProcessor):
    """
    Continuous Feature Processor.

    Normalizes continuous features using StandardScaler.
    Supports optional transformations (log1p, sqrt).
    """

    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.transform_type = config.transform  # log1p, sqrt, etc.

    def _apply_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply pre-scaling transformation."""
        if self.transform_type == "log1p":
            return np.log1p(np.maximum(data, 0))
        elif self.transform_type == "sqrt":
            return np.sqrt(np.maximum(data, 0))
        return data

    def fit(self, data: pd.Series) -> "ContinuousFeatureProcessor":
        # Fill NA with median
        data_filled = data.fillna(data.median()).values.astype(np.float32)

        # Apply transformation
        data_transformed = self._apply_transform(data_filled)

        # Fit scaler
        self.scaler.fit(data_transformed.reshape(-1, 1))

        self.is_fitted = True
        logger.debug(f"ContinuousFeatureProcessor fitted: {self.name}")
        return self

    def transform(self, data: pd.Series) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"Processor {self.name} not fitted")

        # Fill NA with 0 (after scaling, this represents mean)
        data_filled = data.fillna(0).values.astype(np.float32)

        # Apply transformation
        data_transformed = self._apply_transform(data_filled)

        # Scale
        scaled = self.scaler.transform(data_transformed.reshape(-1, 1))

        return scaled.flatten().astype(np.float32)

    def get_output_dim(self) -> int:
        return 1


class TimeFeatureProcessor(BaseFeatureProcessor):
    """
    Time Feature Processor.

    Extracts multiple features from timestamp:
    - hour (0-23)
    - day_of_week (0-6)
    - is_weekend (0/1)
    - is_peak_hour (0/1)
    """

    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.extract_features = config.extract or [
            "hour",
            "day_of_week",
            "is_weekend",
            "is_peak_hour",
        ]

    def fit(self, data: pd.Series) -> "TimeFeatureProcessor":
        # Time features don't need fitting
        self.is_fitted = True
        logger.debug(f"TimeFeatureProcessor fitted: {self.name}")
        return self

    def transform(self, data: pd.Series) -> dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError(f"Processor {self.name} not fitted")

        # Convert to datetime
        dt = pd.to_datetime(data, errors="coerce")

        result = {}

        if "hour" in self.extract_features:
            result[f"{self.name}_hour"] = dt.dt.hour.fillna(12).values.astype(
                np.int64
            )

        if "day_of_week" in self.extract_features:
            result[f"{self.name}_day_of_week"] = dt.dt.dayofweek.fillna(
                0
            ).values.astype(np.int64)

        if "is_weekend" in self.extract_features:
            result[f"{self.name}_is_weekend"] = (
                (dt.dt.dayofweek >= 5).fillna(False).astype(np.int64).values
            )

        if "is_peak_hour" in self.extract_features:
            hour = dt.dt.hour.fillna(12)
            is_peak = ((hour >= 9) & (hour <= 12)) | ((hour >= 19) & (hour <= 22))
            result[f"{self.name}_is_peak_hour"] = is_peak.astype(np.int64).values

        return result

    def get_output_dim(self) -> int:
        return len(self.extract_features)


class SequenceFeatureProcessor(BaseFeatureProcessor):
    """
    Sequence Feature Processor.

    Handles variable-length sequence features (e.g., tags, interests).
    Uses EmbeddingBag with offsets for efficient batch processing.
    """

    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.encoder = LabelEncoder()
        self.max_length = config.max_length
        self.pooling = config.pooling
        self.unknown_idx = 0
        self.pad_idx = 1

    def fit(self, data: pd.Series) -> "SequenceFeatureProcessor":
        # Parse sequences
        all_items = []
        for seq in data:
            if pd.isna(seq) or seq == "":
                continue
            if isinstance(seq, str):
                items = [s.strip() for s in seq.split(",") if s.strip()]
            elif isinstance(seq, list):
                items = seq
            else:
                continue
            all_items.extend(items)

        # Build vocabulary
        vocab = ["__UNKNOWN__", "__PAD__"] + list(set(all_items))
        self.encoder.fit(vocab)

        self.unknown_idx = 0
        self.pad_idx = 1

        self.is_fitted = True
        logger.debug(
            f"SequenceFeatureProcessor fitted: {self.name}, "
            f"vocab_size={len(self.encoder.classes_)}"
        )
        return self

    def transform(self, data: pd.Series) -> dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError(f"Processor {self.name} not fitted")

        all_ids = []
        offsets = [0]

        for seq in data:
            if pd.isna(seq) or seq == "":
                items = []
            elif isinstance(seq, str):
                items = [s.strip() for s in seq.split(",") if s.strip()]
            elif isinstance(seq, list):
                items = [str(s) for s in seq]
            else:
                items = []

            # Truncate to max_length
            items = items[: self.max_length]

            # Encode items
            ids = []
            for item in items:
                if item in self.encoder.classes_:
                    ids.append(
                        np.where(self.encoder.classes_ == item)[0][0]
                    )
                else:
                    ids.append(self.unknown_idx)

            # If empty, add a single unknown
            if not ids:
                ids = [self.unknown_idx]

            all_ids.extend(ids)
            offsets.append(len(all_ids))

        return {
            f"{self.name}_ids": np.array(all_ids, dtype=np.int64),
            f"{self.name}_offsets": np.array(offsets[:-1], dtype=np.int64),
        }

    def get_output_dim(self) -> int:
        return self.config.embedding_dim or 16

    def get_vocab_size(self) -> int:
        return len(self.encoder.classes_)


class CrossFeatureProcessor(BaseFeatureProcessor):
    """
    Cross Feature Processor.

    Creates crossed features by hashing combinations of multiple fields.
    """

    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.fields = config.fields
        self.hash_buckets = config.hash_buckets

    def fit(self, data: pd.DataFrame) -> "CrossFeatureProcessor":
        # Cross features don't need fitting
        self.is_fitted = True
        logger.debug(f"CrossFeatureProcessor fitted: {self.name}")
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"Processor {self.name} not fitted")

        # Concatenate field values and hash
        result = np.zeros(len(data), dtype=np.int64)

        for idx in range(len(data)):
            values = []
            for field in self.fields:
                if field in data.columns:
                    values.append(str(data.iloc[idx][field]))
                else:
                    values.append("__MISSING__")

            combined = "_".join(values)
            hash_val = int(
                hashlib.md5(combined.encode()).hexdigest()[:8], 16
            )
            result[idx] = hash_val % self.hash_buckets

        return result

    def get_output_dim(self) -> int:
        return self.config.embedding_dim or 8

    def get_vocab_size(self) -> int:
        return self.hash_buckets


# ==============================================================================
# Feature Processor Factory
# ==============================================================================


class FeatureProcessorFactory:
    """
    Factory for creating feature processors.

    Maps feature types to processor classes.
    """

    _processors: dict[str, type[BaseFeatureProcessor]] = {
        "id": IDFeatureProcessor,
        "discrete": DiscreteFeatureProcessor,
        "continuous": ContinuousFeatureProcessor,
        "time": TimeFeatureProcessor,
        "sequence": SequenceFeatureProcessor,
        "cross": CrossFeatureProcessor,
    }

    @classmethod
    def create(cls, config: FeatureConfig) -> BaseFeatureProcessor:
        """Create a processor for the given feature config."""
        processor_class = cls._processors.get(config.type)
        if processor_class is None:
            raise ValueError(f"Unknown feature type: {config.type}")
        return processor_class(config)

    @classmethod
    def register(
        cls,
        feature_type: str,
        processor_class: type[BaseFeatureProcessor],
    ) -> None:
        """Register a custom processor class."""
        cls._processors[feature_type] = processor_class


# ==============================================================================
# Feature Pipeline
# ==============================================================================


class FeaturePipeline:
    """
    Complete feature processing pipeline.

    Manages multiple feature processors and provides unified interface.
    """

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize pipeline.

        Args:
            config_path: Path to features_config.yaml
        """
        self.config = get_feature_config(config_path)
        self.processors: dict[str, BaseFeatureProcessor] = {}
        self.is_fitted = False

        # Build processors from config
        self._build_processors()

    def _build_processors(self) -> None:
        """Build all processors from config."""
        all_features = []

        # Collect features from all groups
        for group in [
            self.config.user_features,
            self.config.ad_features,
            self.config.context_features,
        ]:
            all_features.extend(group.id_features)
            all_features.extend(group.discrete_features)
            all_features.extend(group.continuous_features)
            all_features.extend(group.time_features)
            all_features.extend(group.sequence_features)

        all_features.extend(self.config.cross_features)

        # Create processors
        for feat_config in all_features:
            self.processors[feat_config.name] = FeatureProcessorFactory.create(
                feat_config
            )

        logger.info(f"Built {len(self.processors)} feature processors")

    def fit(self, data: pd.DataFrame) -> "FeaturePipeline":
        """
        Fit all processors on training data.

        Args:
            data: Training DataFrame with all feature columns
        """
        logger.info("Fitting feature pipeline...")

        for name, processor in self.processors.items():
            if name not in data.columns:
                # Skip if feature not in data
                logger.warning(f"Feature {name} not in data, using default fit")
                if isinstance(processor, CrossFeatureProcessor):
                    processor.fit(data)
                else:
                    # Create dummy data for fitting
                    dummy = pd.Series(["__UNKNOWN__"] * 10)
                    processor.fit(dummy)
            else:
                if isinstance(processor, CrossFeatureProcessor):
                    processor.fit(data)
                else:
                    processor.fit(data[name])

        self.is_fitted = True
        logger.info("Feature pipeline fitted")
        return self

    def transform(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Transform data using fitted processors.

        Args:
            data: DataFrame to transform

        Returns:
            Dictionary of feature name -> transformed values
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted")

        result = {}

        for name, processor in self.processors.items():
            if name not in data.columns and not isinstance(
                processor, CrossFeatureProcessor
            ):
                logger.warning(f"Feature {name} not in data, skipping")
                continue

            if isinstance(processor, CrossFeatureProcessor):
                transformed = processor.transform(data)
                result[name] = transformed
            elif isinstance(processor, TimeFeatureProcessor):
                # Time features return multiple outputs
                time_features = processor.transform(data[name])
                result.update(time_features)
            elif isinstance(processor, SequenceFeatureProcessor):
                # Sequence features return dict with ids and offsets
                seq_features = processor.transform(data[name])
                result.update(seq_features)
            else:
                result[name] = processor.transform(data[name])

        return result

    def fit_transform(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)

    def get_feature_dims(self) -> dict[str, dict[str, int]]:
        """
        Get dimensions for all features.

        Returns dict with feature info for model building.
        """
        dims = {
            "id_features": {},
            "discrete_features": {},
            "continuous_features": {},
            "sequence_features": {},
            "cross_features": {},
        }

        for name, processor in self.processors.items():
            info = {
                "output_dim": processor.get_output_dim(),
                "vocab_size": processor.get_vocab_size(),
            }

            if isinstance(processor, IDFeatureProcessor):
                dims["id_features"][name] = info
            elif isinstance(processor, DiscreteFeatureProcessor):
                dims["discrete_features"][name] = info
            elif isinstance(processor, ContinuousFeatureProcessor):
                dims["continuous_features"][name] = info
            elif isinstance(processor, SequenceFeatureProcessor):
                dims["sequence_features"][name] = info
            elif isinstance(processor, CrossFeatureProcessor):
                dims["cross_features"][name] = info

        return dims

    def save(self, path: str | Path) -> None:
        """Save fitted pipeline to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "processors": self.processors,
                    "is_fitted": self.is_fitted,
                },
                f,
            )

        logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePipeline":
        """Load pipeline from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        pipeline = cls.__new__(cls)
        pipeline.processors = state["processors"]
        pipeline.is_fitted = state["is_fitted"]
        pipeline.config = None

        logger.info(f"Pipeline loaded from {path}")
        return pipeline
