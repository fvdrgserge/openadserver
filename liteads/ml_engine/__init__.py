"""
Machine Learning Engine for Ad Prediction.

This module provides:
- Feature processing with factory pattern
- DeepFM model for CTR/CVR prediction
- Training utilities
- Model serving for online inference
"""

from liteads.ml_engine.data import AdDataModule, AdDataset
from liteads.ml_engine.features import FeatureBuilder, FeatureConfig, FeaturePipeline
from liteads.ml_engine.models import DeepFM
from liteads.ml_engine.serving import ModelPredictor, PredictionResult
from liteads.ml_engine.training import Trainer, TrainingConfig

__all__ = [
    # Features
    "FeatureBuilder",
    "FeatureConfig",
    "FeaturePipeline",
    # Models
    "DeepFM",
    # Data
    "AdDataModule",
    "AdDataset",
    # Training
    "Trainer",
    "TrainingConfig",
    # Serving
    "ModelPredictor",
    "PredictionResult",
]
