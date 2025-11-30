"""
Feature engineering with factory pattern.
"""

from liteads.ml_engine.features.builder import FeatureBuilder, FeatureInfo, ModelInputs
from liteads.ml_engine.features.config import (
    FeatureConfig,
    FeatureGroupConfig,
    FeaturesConfigLoader,
    FeaturesConfigSchema,
    get_feature_config,
)
from liteads.ml_engine.features.processor import (
    BaseFeatureProcessor,
    ContinuousFeatureProcessor,
    CrossFeatureProcessor,
    DiscreteFeatureProcessor,
    FeaturePipeline,
    FeatureProcessorFactory,
    IDFeatureProcessor,
    SequenceFeatureProcessor,
    TimeFeatureProcessor,
)

__all__ = [
    # Config
    "FeatureConfig",
    "FeatureGroupConfig",
    "FeaturesConfigSchema",
    "FeaturesConfigLoader",
    "get_feature_config",
    # Processors
    "BaseFeatureProcessor",
    "IDFeatureProcessor",
    "DiscreteFeatureProcessor",
    "ContinuousFeatureProcessor",
    "TimeFeatureProcessor",
    "SequenceFeatureProcessor",
    "CrossFeatureProcessor",
    "FeatureProcessorFactory",
    "FeaturePipeline",
    # Builder
    "FeatureBuilder",
    "FeatureInfo",
    "ModelInputs",
]
