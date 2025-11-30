"""
Tests for ML models.
"""

import pytest
import torch

from liteads.ml_engine.models.deepfm import DeepFM, MultiTaskDeepFM
from liteads.ml_engine.models.layers import DNN, FM, EmbeddingLayer, SequenceEmbeddingLayer


class TestEmbeddingLayer:
    """Tests for embedding layer."""

    def test_forward(self) -> None:
        """Test embedding forward pass."""
        layer = EmbeddingLayer(
            feature_dims=[100, 50, 200],
            embedding_dims=[8, 8, 8],
        )

        batch_size = 32
        x = torch.randint(0, 50, (batch_size, 3))

        output = layer(x)

        assert output.shape == (batch_size, 24)  # 8 * 3

    def test_heterogeneous_dims(self) -> None:
        """Test with different embedding dimensions."""
        layer = EmbeddingLayer(
            feature_dims=[100, 50],
            embedding_dims=[16, 8],
        )

        x = torch.randint(0, 50, (16, 2))
        output = layer(x)

        assert output.shape == (16, 24)  # 16 + 8


class TestSequenceEmbeddingLayer:
    """Tests for sequence embedding layer."""

    def test_forward(self) -> None:
        """Test sequence embedding forward pass."""
        layer = SequenceEmbeddingLayer(
            vocab_size=100,
            embedding_dim=16,
            mode="mean",
        )

        # Batch of 3 sequences with lengths [2, 3, 1]
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets = torch.tensor([0, 2, 5])

        output = layer(values, offsets)

        assert output.shape == (3, 16)


class TestFM:
    """Tests for FM layer."""

    def test_forward(self) -> None:
        """Test FM forward pass."""
        fm = FM(reduce_sum=True)

        batch_size = 32
        num_features = 5
        embedding_dim = 8

        embeddings = [
            torch.randn(batch_size, embedding_dim)
            for _ in range(num_features)
        ]

        output = fm(embeddings)

        assert output.shape == (batch_size, 1)

    def test_no_reduce(self) -> None:
        """Test FM without reduction."""
        fm = FM(reduce_sum=False)

        batch_size = 16
        embedding_dim = 8
        embeddings = [torch.randn(batch_size, embedding_dim) for _ in range(3)]

        output = fm(embeddings)

        assert output.shape == (batch_size, embedding_dim)


class TestDNN:
    """Tests for DNN layer."""

    def test_forward(self) -> None:
        """Test DNN forward pass."""
        dnn = DNN(
            input_dim=64,
            hidden_units=[128, 64, 32],
            activation="relu",
            dropout=0.1,
        )

        batch_size = 32
        x = torch.randn(batch_size, 64)

        output = dnn(x)

        assert output.shape == (batch_size, 32)

    def test_with_output_dim(self) -> None:
        """Test DNN with explicit output dimension."""
        dnn = DNN(
            input_dim=64,
            hidden_units=[128, 64],
            output_dim=1,
        )

        x = torch.randn(16, 64)
        output = dnn(x)

        assert output.shape == (16, 1)

    def test_activations(self) -> None:
        """Test different activation functions."""
        activations = ["relu", "leaky_relu", "gelu", "tanh"]

        for act in activations:
            dnn = DNN(input_dim=32, hidden_units=[64], activation=act)
            x = torch.randn(8, 32)
            output = dnn(x)
            assert output.shape == (8, 64)


class TestDeepFM:
    """Tests for DeepFM model."""

    @pytest.fixture
    def model(self) -> DeepFM:
        """Create DeepFM model for testing."""
        return DeepFM(
            sparse_feature_dims=[100, 50, 200],
            sparse_embedding_dims=8,
            dense_feature_dim=5,
            fm_k=8,
            dnn_hidden_units=[64, 32],
            dnn_dropout=0.1,
        )

    def test_forward(self, model: DeepFM) -> None:
        """Test DeepFM forward pass."""
        batch_size = 32

        sparse_features = torch.randint(0, 50, (batch_size, 3))
        dense_features = torch.randn(batch_size, 5)

        outputs = model(sparse_features, dense_features)

        assert "task_0" in outputs
        assert outputs["task_0"].shape == (batch_size, 1)
        # Output should be probabilities
        assert (outputs["task_0"] >= 0).all()
        assert (outputs["task_0"] <= 1).all()

    def test_without_dense(self) -> None:
        """Test DeepFM without dense features."""
        model = DeepFM(
            sparse_feature_dims=[100, 50],
            sparse_embedding_dims=8,
            dense_feature_dim=0,
            dnn_hidden_units=[32, 16],
        )

        sparse_features = torch.randint(0, 50, (16, 2))

        outputs = model(sparse_features, None)

        assert outputs["task_0"].shape == (16, 1)

    def test_without_fm(self) -> None:
        """Test DeepFM without FM component."""
        model = DeepFM(
            sparse_feature_dims=[100],
            sparse_embedding_dims=8,
            dense_feature_dim=3,
            use_fm=False,
            dnn_hidden_units=[32],
        )

        sparse_features = torch.randint(0, 50, (16, 1))
        dense_features = torch.randn(16, 3)

        outputs = model(sparse_features, dense_features)

        assert outputs["task_0"].shape == (16, 1)

    def test_without_dnn(self) -> None:
        """Test DeepFM without DNN component (FM only)."""
        model = DeepFM(
            sparse_feature_dims=[100, 50],
            sparse_embedding_dims=8,
            dense_feature_dim=2,
            use_dnn=False,
        )

        sparse_features = torch.randint(0, 50, (16, 2))
        dense_features = torch.randn(16, 2)

        outputs = model(sparse_features, dense_features)

        assert outputs["task_0"].shape == (16, 1)

    def test_regularization_loss(self, model: DeepFM) -> None:
        """Test regularization loss calculation."""
        # Model needs to have non-zero reg coefficients
        model.l2_reg_embedding = 0.01
        model.l2_reg_dnn = 0.01

        reg_loss = model.get_regularization_loss()

        assert reg_loss > 0

    def test_gradient_flow(self, model: DeepFM) -> None:
        """Test gradients flow correctly."""
        sparse_features = torch.randint(0, 50, (8, 3))
        dense_features = torch.randn(8, 5)

        outputs = model(sparse_features, dense_features)
        loss = outputs["task_0"].mean()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestMultiTaskDeepFM:
    """Tests for multi-task DeepFM model."""

    def test_forward(self) -> None:
        """Test multi-task forward pass."""
        model = MultiTaskDeepFM(
            sparse_feature_dims=[100, 50],
            sparse_embedding_dims=8,
            dense_feature_dim=3,
            dnn_hidden_units=[32, 16],
        )

        batch_size = 16
        sparse_features = torch.randint(0, 50, (batch_size, 2))
        dense_features = torch.randn(batch_size, 3)

        outputs = model(sparse_features, dense_features)

        # Should have CTR, CVR, and CTCVR outputs
        assert "ctr" in outputs
        assert "cvr" in outputs
        assert "ctcvr" in outputs

        # CTCVR = CTR * CVR
        expected_ctcvr = outputs["ctr"] * outputs["cvr"]
        torch.testing.assert_close(outputs["ctcvr"], expected_ctcvr)

    def test_outputs_are_probabilities(self) -> None:
        """Test all outputs are valid probabilities."""
        model = MultiTaskDeepFM(
            sparse_feature_dims=[100],
            sparse_embedding_dims=8,
            dense_feature_dim=2,
        )

        sparse_features = torch.randint(0, 50, (32, 1))
        dense_features = torch.randn(32, 2)

        outputs = model(sparse_features, dense_features)

        for name, output in outputs.items():
            assert (output >= 0).all(), f"{name} has negative values"
            assert (output <= 1).all(), f"{name} has values > 1"
