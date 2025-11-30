"""
Tests for feature processing module.
"""

import numpy as np
import pytest

from liteads.ml_engine.features.config import FeatureConfig, FeaturesConfigLoader
from liteads.ml_engine.features.processor import (
    ContinuousFeatureProcessor,
    CrossFeatureProcessor,
    DiscreteFeatureProcessor,
    FeaturePipeline,
    FeatureProcessorFactory,
    IDFeatureProcessor,
    SequenceFeatureProcessor,
    TimeFeatureProcessor,
)


@pytest.fixture
def sample_data() -> list[dict]:
    """Create sample data for testing."""
    return [
        {
            "user_id": "user_1",
            "age": 25,
            "gender": "male",
            "interests": ["tech", "sports"],
            "timestamp": 1700000000,
        },
        {
            "user_id": "user_2",
            "age": 35,
            "gender": "female",
            "interests": ["fashion", "food"],
            "timestamp": 1700050000,
        },
        {
            "user_id": "user_3",
            "age": 45,
            "gender": "male",
            "interests": ["tech"],
            "timestamp": 1700100000,
        },
    ]


class TestIDFeatureProcessor:
    """Tests for ID feature processor."""

    def test_fit_transform(self) -> None:
        """Test fit and transform on ID features."""
        config = FeatureConfig(name="user_id", type="id", embedding_dim=16)
        processor = IDFeatureProcessor(config)

        data = [{"user_id": "a"}, {"user_id": "b"}, {"user_id": "a"}]
        processor.fit(data)
        result = processor.transform(data)

        assert len(result) == 3
        assert result[0] == result[2]  # Same user
        assert result[0] != result[1]  # Different users

    def test_unknown_value(self) -> None:
        """Test handling of unknown values during transform."""
        config = FeatureConfig(name="user_id", type="id")
        processor = IDFeatureProcessor(config)

        train_data = [{"user_id": "a"}, {"user_id": "b"}]
        processor.fit(train_data)

        test_data = [{"user_id": "c"}]  # Unknown user
        result = processor.transform(test_data)

        # Unknown values should be mapped to index 0
        assert result[0] == 0


class TestDiscreteFeatureProcessor:
    """Tests for discrete feature processor."""

    def test_fit_transform(self) -> None:
        """Test fit and transform on discrete features."""
        config = FeatureConfig(name="gender", type="discrete")
        processor = DiscreteFeatureProcessor(config)

        data = [{"gender": "male"}, {"gender": "female"}, {"gender": "male"}]
        processor.fit(data)
        result = processor.transform(data)

        assert len(result) == 3
        assert result[0] == result[2]
        assert result[0] != result[1]


class TestContinuousFeatureProcessor:
    """Tests for continuous feature processor."""

    def test_fit_transform(self) -> None:
        """Test fit and transform on continuous features."""
        config = FeatureConfig(name="age", type="continuous")
        processor = ContinuousFeatureProcessor(config)

        data = [{"age": 20}, {"age": 30}, {"age": 40}]
        processor.fit(data)
        result = processor.transform(data)

        # Should be standardized
        assert np.abs(result.mean()) < 0.1
        assert np.abs(result.std() - 1.0) < 0.1

    def test_log_transform(self) -> None:
        """Test log1p transform."""
        config = FeatureConfig(name="clicks", type="continuous", transform="log1p")
        processor = ContinuousFeatureProcessor(config)

        data = [{"clicks": 0}, {"clicks": 100}, {"clicks": 1000}]
        processor.fit(data)
        result = processor.transform(data)

        # Log transform should compress the range
        assert result.max() - result.min() < 10


class TestTimeFeatureProcessor:
    """Tests for time feature processor."""

    def test_extract_features(self) -> None:
        """Test time feature extraction."""
        config = FeatureConfig(
            name="timestamp",
            type="time",
            extract=["hour", "day_of_week", "is_weekend"],
        )
        processor = TimeFeatureProcessor(config)

        # Monday 10:00 UTC
        data = [{"timestamp": 1700000000}]  # 2023-11-14 22:13:20 UTC (Tuesday)
        processor.fit(data)
        result = processor.transform(data)

        # Should have 3 extracted features
        assert "timestamp_hour" in result
        assert "timestamp_day_of_week" in result
        assert "timestamp_is_weekend" in result


class TestSequenceFeatureProcessor:
    """Tests for sequence feature processor."""

    def test_fit_transform(self) -> None:
        """Test sequence feature processing."""
        config = FeatureConfig(
            name="interests",
            type="sequence",
            embedding_dim=8,
            pooling="mean",
            max_length=5,
        )
        processor = SequenceFeatureProcessor(config)

        data = [
            {"interests": ["tech", "sports"]},
            {"interests": ["fashion"]},
            {"interests": ["tech", "food", "travel"]},
        ]
        processor.fit(data)
        values, offsets = processor.transform(data)

        # Check offsets are correct
        assert len(offsets) == 3
        assert offsets[0] == 0

    def test_truncation(self) -> None:
        """Test sequence truncation to max_length."""
        config = FeatureConfig(
            name="interests",
            type="sequence",
            max_length=2,
        )
        processor = SequenceFeatureProcessor(config)

        data = [{"interests": ["a", "b", "c", "d", "e"]}]
        processor.fit(data)
        values, offsets = processor.transform(data)

        # Should only have 2 items
        assert len(values) == 2


class TestCrossFeatureProcessor:
    """Tests for cross feature processor."""

    def test_hash_cross(self) -> None:
        """Test cross feature hashing."""
        config = FeatureConfig(
            name="user_campaign",
            type="cross",
            fields=["user_id", "campaign_id"],
            hash_buckets=1000,
        )
        processor = CrossFeatureProcessor(config)

        data = [
            {"user_id": "u1", "campaign_id": "c1"},
            {"user_id": "u1", "campaign_id": "c2"},
            {"user_id": "u2", "campaign_id": "c1"},
        ]
        processor.fit(data)
        result = processor.transform(data)

        # All results should be within hash buckets
        assert all(0 <= v < 1000 for v in result)

        # Same inputs should produce same hash
        data2 = [{"user_id": "u1", "campaign_id": "c1"}]
        result2 = processor.transform(data2)
        assert result[0] == result2[0]


class TestFeatureProcessorFactory:
    """Tests for feature processor factory."""

    def test_create_all_types(self) -> None:
        """Test factory creates all processor types."""
        types = ["id", "discrete", "continuous", "time", "sequence", "cross"]

        for feature_type in types:
            config = FeatureConfig(name="test", type=feature_type)
            processor = FeatureProcessorFactory.create(config)
            assert processor is not None

    def test_unknown_type_raises(self) -> None:
        """Test factory raises for unknown type."""
        config = FeatureConfig(name="test", type="unknown")  # type: ignore

        with pytest.raises(ValueError):
            FeatureProcessorFactory.create(config)


class TestFeaturePipeline:
    """Tests for feature pipeline."""

    def test_fit_transform(self, sample_data: list[dict]) -> None:
        """Test pipeline fit and transform."""
        # Load config
        loader = FeaturesConfigLoader()
        config = loader.load()

        pipeline = FeaturePipeline(config)

        # Fit and transform
        pipeline.fit(sample_data)
        result = pipeline.transform(sample_data)

        # Should have processed features
        assert len(result) > 0

    def test_state_save_load(self, sample_data: list[dict], tmp_path) -> None:
        """Test pipeline state serialization."""
        loader = FeaturesConfigLoader()
        config = loader.load()

        pipeline = FeaturePipeline(config)
        pipeline.fit(sample_data)

        # Save state
        state = pipeline.get_state()
        assert state is not None

        # Create new pipeline and load state
        pipeline2 = FeaturePipeline(config)
        pipeline2.set_state(state)

        # Transform should produce same results
        result1 = pipeline.transform(sample_data)
        result2 = pipeline2.transform(sample_data)

        for key in result1:
            if isinstance(result1[key], np.ndarray):
                np.testing.assert_array_equal(result1[key], result2[key])
