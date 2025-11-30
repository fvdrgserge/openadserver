"""
CTR/CVR prediction module.

Provides prediction interfaces for ML models.
"""

from abc import ABC, abstractmethod
from typing import Any

from liteads.common.logger import get_logger
from liteads.common.utils import Timer
from liteads.schemas.internal import AdCandidate, PredictionResult, UserContext

logger = get_logger(__name__)


class BasePredictor(ABC):
    """Abstract base class for CTR/CVR predictors."""

    @abstractmethod
    async def predict(
        self,
        user_context: UserContext,
        candidate: AdCandidate,
    ) -> PredictionResult:
        """Predict CTR/CVR for a single candidate."""
        pass

    @abstractmethod
    async def predict_batch(
        self,
        user_context: UserContext,
        candidates: list[AdCandidate],
    ) -> list[PredictionResult]:
        """Predict CTR/CVR for multiple candidates."""
        pass


class StatisticalPredictor(BasePredictor):
    """
    Statistical predictor using historical data.

    Uses smoothed CTR based on historical impressions and clicks.
    Good as a baseline or fallback.
    """

    def __init__(
        self,
        default_ctr: float = 0.01,
        default_cvr: float = 0.001,
        smoothing_impressions: int = 1000,
        smoothing_clicks: int = 100,
    ):
        """
        Initialize statistical predictor.

        Args:
            default_ctr: Default CTR for new ads
            default_cvr: Default CVR for new ads
            smoothing_impressions: Smoothing factor for impressions
            smoothing_clicks: Smoothing factor for clicks
        """
        self.default_ctr = default_ctr
        self.default_cvr = default_cvr
        self.smoothing_impressions = smoothing_impressions
        self.smoothing_clicks = smoothing_clicks

    async def predict(
        self,
        user_context: UserContext,
        candidate: AdCandidate,
    ) -> PredictionResult:
        """Predict using smoothed historical CTR."""
        # Get historical stats from candidate metadata
        impressions = candidate.metadata.get("impressions", 0)
        clicks = candidate.metadata.get("clicks", 0)
        conversions = candidate.metadata.get("conversions", 0)

        # Smoothed CTR: (clicks + prior_clicks) / (impressions + prior_impressions)
        smoothed_ctr = (clicks + self.smoothing_clicks * self.default_ctr) / (
            impressions + self.smoothing_clicks
        )

        # Smoothed CVR
        if clicks > 0:
            smoothed_cvr = (conversions + self.smoothing_clicks * self.default_cvr) / (
                clicks + self.smoothing_clicks
            )
        else:
            smoothed_cvr = self.default_cvr

        return PredictionResult(
            campaign_id=candidate.campaign_id,
            creative_id=candidate.creative_id,
            pctr=smoothed_ctr,
            pcvr=smoothed_cvr,
            model_version="statistical_v1",
            latency_ms=0.1,
        )

    async def predict_batch(
        self,
        user_context: UserContext,
        candidates: list[AdCandidate],
    ) -> list[PredictionResult]:
        """Predict for multiple candidates."""
        results = []
        for candidate in candidates:
            result = await self.predict(user_context, candidate)
            results.append(result)
        return results


class MLPredictor(BasePredictor):
    """
    ML-based predictor using trained DeepFM models.

    Supports:
    - PyTorch DeepFM models for CTR/CVR prediction
    - Feature engineering pipeline with factory pattern
    - Async batch inference for low latency
    """

    def __init__(
        self,
        model_path: str | None = None,
        feature_builder_path: str | None = None,
        model_version: str = "v1",
        fallback_ctr: float = 0.01,
        fallback_cvr: float = 0.001,
        device: str = "auto",
    ):
        """
        Initialize ML predictor.

        Args:
            model_path: Path to trained model checkpoint
            feature_builder_path: Path to fitted feature builder
            model_version: Model version identifier
            fallback_ctr: Fallback CTR if prediction fails
            fallback_cvr: Fallback CVR if prediction fails
            device: Device for inference (auto, cpu, cuda, mps)
        """
        self.model_path = model_path
        self.feature_builder_path = feature_builder_path
        self.model_version = model_version
        self.fallback_ctr = fallback_ctr
        self.fallback_cvr = fallback_cvr
        self.device = device
        self._model_predictor = None
        self._is_loaded = False

    async def load_model(self) -> None:
        """Load model and feature builder from storage."""
        if self._is_loaded:
            return

        try:
            from liteads.ml_engine.serving import ModelPredictor

            self._model_predictor = ModelPredictor(
                model_path=self.model_path,
                feature_builder_path=self.feature_builder_path,
                device=self.device,
            )
            self._model_predictor.load()
            self._is_loaded = True
            logger.info(f"Loaded ML model version {self.model_version}")
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}. Using fallback predictions.")
            self._model_predictor = None

    async def predict(
        self,
        user_context: UserContext,
        candidate: AdCandidate,
    ) -> PredictionResult:
        """Predict using ML model."""
        results = await self.predict_batch(user_context, [candidate])
        return results[0]

    async def predict_batch(
        self,
        user_context: UserContext,
        candidates: list[AdCandidate],
    ) -> list[PredictionResult]:
        """Predict for multiple candidates using batch inference."""
        with Timer("ml_prediction") as timer:
            try:
                # Ensure model is loaded
                if not self._is_loaded:
                    await self.load_model()

                # Build features
                features_batch = self._build_features(user_context, candidates)

                # Run inference
                if self._model_predictor is not None:
                    ml_results = await self._model_predictor.predict_batch_async(features_batch)

                    # Build results
                    results = []
                    for i, candidate in enumerate(candidates):
                        ml_result = ml_results[i]
                        results.append(
                            PredictionResult(
                                campaign_id=candidate.campaign_id,
                                creative_id=candidate.creative_id,
                                pctr=ml_result.pctr,
                                pcvr=ml_result.pcvr or self.fallback_cvr,
                                model_version=ml_result.model_version or self.model_version,
                                latency_ms=ml_result.latency_ms,
                            )
                        )
                    return results
                else:
                    # Fallback: return default predictions
                    return [
                        PredictionResult(
                            campaign_id=c.campaign_id,
                            creative_id=c.creative_id,
                            pctr=self.fallback_ctr,
                            pcvr=self.fallback_cvr,
                            model_version="fallback",
                            latency_ms=timer.elapsed_ms / len(candidates),
                        )
                        for c in candidates
                    ]

            except Exception as e:
                logger.error(f"ML prediction failed: {e}")
                # Return fallback predictions
                return [
                    PredictionResult(
                        campaign_id=c.campaign_id,
                        creative_id=c.creative_id,
                        pctr=self.fallback_ctr,
                        pcvr=self.fallback_cvr,
                        model_version="fallback",
                        latency_ms=timer.elapsed_ms / len(candidates) if timer.elapsed_ms else 0,
                    )
                    for c in candidates
                ]

    def _build_features(
        self,
        user_context: UserContext,
        candidates: list[AdCandidate],
    ) -> list[dict[str, Any]]:
        """Build feature dictionaries for prediction."""
        features = []

        for candidate in candidates:
            feature_dict = {
                # User features
                "user_id": user_context.user_id or "unknown",
                "user_gender": user_context.gender or "unknown",
                "user_age_bucket": self._get_age_bucket(user_context.age),
                "user_device_os": user_context.os or "unknown",
                "user_network_type": "wifi",  # Default
                "user_click_count_7d": 0,
                "user_click_count_30d": 0,
                "user_conversion_count_7d": 0,
                "user_ctr_7d": self.fallback_ctr,
                "user_cvr_7d": self.fallback_cvr,
                "user_avg_session_duration": 300,
                "user_interest_tags": user_context.interests or [],
                "user_clicked_categories": [],
                # Ad features
                "campaign_id": str(candidate.campaign_id),
                "creative_id": str(candidate.creative_id),
                "advertiser_id": str(candidate.advertiser_id),
                "ad_category": str(candidate.metadata.get("category", "unknown")),
                "creative_type": candidate.metadata.get("creative_type", "banner"),
                "bid_type": "cpm" if candidate.bid_type == 1 else "cpc",
                "landing_page_type": "app",
                "ad_bid": candidate.bid,
                "ad_ctr_7d": candidate.pctr or self.fallback_ctr,
                "ad_cvr_7d": candidate.pcvr or self.fallback_cvr,
                "ad_impression_count_7d": candidate.metadata.get("impressions", 0),
                "ad_click_count_7d": candidate.metadata.get("clicks", 0),
                "creative_ctr": candidate.pctr or self.fallback_ctr,
                "advertiser_quality_score": 0.8,
                "ad_tags": candidate.metadata.get("tags", []),
                # Context features
                "slot_id": user_context.metadata.get("slot_id", "default"),
                "request_hour": user_context.metadata.get("hour", 12),
                "request_day_of_week": user_context.metadata.get("day_of_week", 0),
                "is_weekend": user_context.metadata.get("is_weekend", 0),
                "is_peak_hour": user_context.metadata.get("is_peak_hour", 0),
                "geo_country": user_context.country or "unknown",
                "geo_city": user_context.city or "unknown",
                "slot_ctr": 0.01,
                "hour_ctr": 0.01,
            }
            features.append(feature_dict)

        return features

    def _get_age_bucket(self, age: int | None) -> str:
        """Convert age to age bucket."""
        if age is None:
            return "unknown"
        if age < 18:
            return "under_18"
        elif age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        else:
            return "45+"


class EnsemblePredictor(BasePredictor):
    """
    Ensemble predictor combining multiple predictors.

    Supports weighted averaging of predictions from multiple models.
    """

    def __init__(
        self,
        predictors: list[tuple[BasePredictor, float]],
    ):
        """
        Initialize ensemble predictor.

        Args:
            predictors: List of (predictor, weight) tuples
        """
        self.predictors = predictors
        total_weight = sum(w for _, w in predictors)
        self.weights = [w / total_weight for _, w in predictors]

    async def predict(
        self,
        user_context: UserContext,
        candidate: AdCandidate,
    ) -> PredictionResult:
        """Predict using weighted ensemble."""
        results = await self.predict_batch(user_context, [candidate])
        return results[0]

    async def predict_batch(
        self,
        user_context: UserContext,
        candidates: list[AdCandidate],
    ) -> list[PredictionResult]:
        """Predict using weighted ensemble for batch."""
        # Collect predictions from all predictors
        all_predictions: list[list[PredictionResult]] = []

        for predictor, _ in self.predictors:
            preds = await predictor.predict_batch(user_context, candidates)
            all_predictions.append(preds)

        # Weighted average
        results = []
        for i, candidate in enumerate(candidates):
            weighted_ctr = sum(
                all_predictions[j][i].pctr * self.weights[j]
                for j in range(len(self.predictors))
            )
            weighted_cvr = sum(
                all_predictions[j][i].pcvr * self.weights[j]
                for j in range(len(self.predictors))
            )

            results.append(
                PredictionResult(
                    campaign_id=candidate.campaign_id,
                    creative_id=candidate.creative_id,
                    pctr=weighted_ctr,
                    pcvr=weighted_cvr,
                    model_version="ensemble",
                    latency_ms=max(p[i].latency_ms for p in all_predictions),
                )
            )

        return results
