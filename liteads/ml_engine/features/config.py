"""
Feature configuration loader.

Loads and validates feature configuration from YAML files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from liteads.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureConfig:
    """Single feature configuration."""

    name: str
    type: Literal["id", "discrete", "continuous", "time", "sequence", "cross"]
    description: str = ""

    # Embedding settings
    embedding_dim: int | None = None

    # Continuous feature settings
    transform: str | None = None  # log1p, sqrt, etc.

    # Sequence feature settings
    pooling: str = "mean"  # mean, sum, max
    max_length: int = 20

    # Time feature settings
    extract: list[str] = field(default_factory=list)

    # Cross feature settings
    fields: list[str] = field(default_factory=list)
    hash_buckets: int = 10000


@dataclass
class FeatureGroupConfig:
    """Feature group configuration."""

    id_features: list[FeatureConfig] = field(default_factory=list)
    discrete_features: list[FeatureConfig] = field(default_factory=list)
    continuous_features: list[FeatureConfig] = field(default_factory=list)
    time_features: list[FeatureConfig] = field(default_factory=list)
    sequence_features: list[FeatureConfig] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Model configuration."""

    default_embedding_dim: int = 16
    fm_k: int = 8
    dnn_hidden_units: list[int] = field(default_factory=lambda: [256, 128, 64])
    dnn_dropout: float = 0.2
    dnn_activation: str = "relu"
    l2_reg_embedding: float = 0.0001
    l2_reg_dnn: float = 0.0001


@dataclass
class FeaturesConfigSchema:
    """Complete features configuration schema."""

    user_features: FeatureGroupConfig
    ad_features: FeatureGroupConfig
    context_features: FeatureGroupConfig
    cross_features: list[FeatureConfig]
    model: ModelConfig

    # Feature statistics (populated during training)
    feature_stats: dict[str, Any] = field(default_factory=dict)


class FeaturesConfigLoader:
    """
    Feature configuration loader.

    Loads YAML config and provides typed access to feature definitions.
    """

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to features_config.yaml
        """
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent.parent.parent
                / "configs"
                / "features_config.yaml"
            )
        self.config_path = Path(config_path)
        self._config: dict[str, Any] = {}
        self._schema: FeaturesConfigSchema | None = None

    def load(self) -> FeaturesConfigSchema:
        """Load and parse configuration."""
        if self._schema is not None:
            return self._schema

        logger.info(f"Loading feature config from {self.config_path}")

        with open(self.config_path) as f:
            self._config = yaml.safe_load(f)

        self._schema = self._parse_config()
        return self._schema

    def _parse_config(self) -> FeaturesConfigSchema:
        """Parse raw config into typed schema."""
        feature_groups = self._config.get("feature_groups", {})

        # Parse user features
        user_features = self._parse_feature_group(feature_groups.get("user", {}))

        # Parse ad features
        ad_features = self._parse_feature_group(feature_groups.get("ad", {}))

        # Parse context features
        context_features = self._parse_feature_group(
            feature_groups.get("context", {})
        )

        # Parse cross features
        cross_features = [
            self._parse_feature(f) for f in self._config.get("cross", [])
        ]

        # Parse model config
        model_config = self._parse_model_config(self._config.get("model", {}))

        return FeaturesConfigSchema(
            user_features=user_features,
            ad_features=ad_features,
            context_features=context_features,
            cross_features=cross_features,
            model=model_config,
            feature_stats=self._config.get("feature_stats", {}),
        )

    def _parse_feature_group(self, group: dict[str, Any]) -> FeatureGroupConfig:
        """Parse a feature group."""
        return FeatureGroupConfig(
            id_features=[
                self._parse_feature(f) for f in group.get("id_features", [])
            ],
            discrete_features=[
                self._parse_feature(f) for f in group.get("discrete_features", [])
            ],
            continuous_features=[
                self._parse_feature(f) for f in group.get("continuous_features", [])
            ],
            time_features=[
                self._parse_feature(f) for f in group.get("time_features", [])
            ],
            sequence_features=[
                self._parse_feature(f) for f in group.get("sequence_features", [])
            ],
        )

    def _parse_feature(self, feature: dict[str, Any]) -> FeatureConfig:
        """Parse a single feature configuration."""
        return FeatureConfig(
            name=feature["name"],
            type=feature["type"],
            description=feature.get("description", ""),
            embedding_dim=feature.get("embedding_dim"),
            transform=feature.get("transform"),
            pooling=feature.get("pooling", "mean"),
            max_length=feature.get("max_length", 20),
            extract=feature.get("extract", []),
            fields=feature.get("fields", []),
            hash_buckets=feature.get("hash_buckets", 10000),
        )

    def _parse_model_config(self, model: dict[str, Any]) -> ModelConfig:
        """Parse model configuration."""
        return ModelConfig(
            default_embedding_dim=model.get("default_embedding_dim", 16),
            fm_k=model.get("fm_k", 8),
            dnn_hidden_units=model.get("dnn_hidden_units", [256, 128, 64]),
            dnn_dropout=model.get("dnn_dropout", 0.2),
            dnn_activation=model.get("dnn_activation", "relu"),
            l2_reg_embedding=model.get("l2_reg_embedding", 0.0001),
            l2_reg_dnn=model.get("l2_reg_dnn", 0.0001),
        )

    def get_all_features(self) -> list[FeatureConfig]:
        """Get all feature configurations as flat list."""
        schema = self.load()
        features = []

        for group in [
            schema.user_features,
            schema.ad_features,
            schema.context_features,
        ]:
            features.extend(group.id_features)
            features.extend(group.discrete_features)
            features.extend(group.continuous_features)
            features.extend(group.time_features)
            features.extend(group.sequence_features)

        features.extend(schema.cross_features)

        return features

    def get_features_by_type(self, feature_type: str) -> list[FeatureConfig]:
        """Get all features of a specific type."""
        return [f for f in self.get_all_features() if f.type == feature_type]

    def get_feature_names_by_type(self, feature_type: str) -> list[str]:
        """Get feature names of a specific type."""
        return [f.name for f in self.get_features_by_type(feature_type)]


# Global config loader instance
_config_loader: FeaturesConfigLoader | None = None


def get_feature_config(config_path: str | Path | None = None) -> FeaturesConfigSchema:
    """Get feature configuration (singleton)."""
    global _config_loader
    if _config_loader is None:
        _config_loader = FeaturesConfigLoader(config_path)
    return _config_loader.load()
