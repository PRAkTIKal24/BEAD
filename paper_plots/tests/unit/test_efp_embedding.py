"""
Unit tests for the EFPEmbedding layer.

This module provides comprehensive tests for the EFPEmbedding class, validating:
- Shape transformations and tensor operations
- Masking behavior for variable-length sequences
- Gate sparsification and threshold functionality
- Configuration options and parameter validation
- Gradient flow and backpropagation
- Edge cases and error handling
- Sparsity monitoring and statistics

Test Structure:
    TestEFPEmbeddingBasic: Core functionality tests
    TestEFPEmbeddingMasking: Jet masking and padding tests
    TestEFPEmbeddingGating: Gate sparsification tests
    TestEFPEmbeddingConfig: Configuration validation tests
    TestEFPEmbeddingGradients: Gradient flow tests
    TestEFPEmbeddingEdgeCases: Edge case and error handling tests
    TestEFPEmbeddingSparsity: Sparsity monitoring tests
"""

import pytest
import torch
import torch.nn as nn

# Import the EFPEmbedding class
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "bead", "src"))
from models.layers import EFPEmbedding


class TestEFPEmbeddingBasic:
    """Test basic functionality of the EFPEmbedding layer."""

    def test_initialization_default_params(self):
        """Test EFPEmbedding initialization with default parameters."""
        embedding = EFPEmbedding(n_efp_features=140)

        assert embedding.n_efp_features == 140
        assert embedding.embedding_dim == 64
        assert embedding.gate_type == "sigmoid"
        assert embedding.gate_threshold == 0.05
        assert embedding.dropout_rate == 0.1
        assert embedding.use_layer_norm is True
        assert embedding.monitor_sparsity is True

        # Check layer initialization
        assert isinstance(embedding.projection, nn.Linear)
        assert isinstance(embedding.gate, nn.Linear)
        assert isinstance(embedding.layer_norm, nn.LayerNorm)
        assert isinstance(embedding.dropout, nn.Dropout)

    def test_initialization_custom_params(self):
        """Test EFPEmbedding initialization with custom parameters."""
        embedding = EFPEmbedding(
            n_efp_features=531,
            embedding_dim=128,
            gate_type="relu6",
            gate_threshold=0.1,
            dropout_rate=0.2,
            use_layer_norm=False,
            monitor_sparsity=False,
        )

        assert embedding.n_efp_features == 531
        assert embedding.embedding_dim == 128
        assert embedding.gate_type == "relu6"
        assert embedding.gate_threshold == 0.1
        assert embedding.dropout_rate == 0.2
        assert embedding.use_layer_norm is False
        assert embedding.monitor_sparsity is False

        # Check optional layers
        assert embedding.layer_norm is None
        assert embedding.dropout is not None  # Should exist when dropout_rate > 0

    def test_forward_shape_transformation(self):
        """Test forward pass shape transformation."""
        batch_size, n_jets, n_efp_features = 32, 3, 140
        embedding_dim = 64

        embedding = EFPEmbedding(
            n_efp_features=n_efp_features, embedding_dim=embedding_dim
        )
        efp_features = torch.randn(batch_size, n_jets, n_efp_features)

        output = embedding(efp_features)

        assert output.shape == (batch_size, n_jets, embedding_dim)
        assert output.dtype == torch.float32

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        embedding = EFPEmbedding(n_efp_features=140, embedding_dim=64)

        for batch_size in [1, 8, 32, 128]:
            efp_features = torch.randn(batch_size, 3, 140)
            output = embedding(efp_features)
            assert output.shape == (batch_size, 3, 64)

    def test_forward_different_jet_counts(self):
        """Test forward pass with different numbers of jets."""
        embedding = EFPEmbedding(n_efp_features=140, embedding_dim=64)

        for n_jets in [1, 2, 3, 5, 10]:
            efp_features = torch.randn(16, n_jets, 140)
            output = embedding(efp_features)
            assert output.shape == (16, n_jets, 64)


class TestEFPEmbeddingMasking:
    """Test masking behavior for variable-length sequences."""

    def test_forward_with_jet_mask(self):
        """Test forward pass with jet masking."""
        batch_size, n_jets, n_efp_features = 8, 3, 140
        embedding_dim = 64

        embedding = EFPEmbedding(
            n_efp_features=n_efp_features, embedding_dim=embedding_dim
        )
        efp_features = torch.randn(batch_size, n_jets, n_efp_features)

        # Create mask: first 2 jets valid, last jet padded
        jet_mask = torch.tensor(
            [
                [True, True, False],  # 2 valid jets
                [True, False, False],  # 1 valid jet
                [True, True, True],  # 3 valid jets
                [False, False, False],  # 0 valid jets
                [True, True, False],  # 2 valid jets
                [True, False, False],  # 1 valid jet
                [True, True, True],  # 3 valid jets
                [True, True, False],  # 2 valid jets
            ]
        )

        output = embedding(efp_features, jet_mask=jet_mask)

        assert output.shape == (batch_size, n_jets, embedding_dim)

        # Check that masked jets have zero embeddings
        for i in range(batch_size):
            for j in range(n_jets):
                if not jet_mask[i, j]:
                    assert torch.allclose(
                        output[i, j], torch.zeros(embedding_dim), atol=1e-6
                    )

    def test_masking_preserves_valid_jets(self):
        """Test that masking preserves embeddings for valid jets."""
        embedding = EFPEmbedding(n_efp_features=140, embedding_dim=64)
        embedding.eval()  # Disable dropout for deterministic behavior
        efp_features = torch.randn(4, 3, 140)

        # Compare output with and without masking for valid jets
        jet_mask = torch.tensor(
            [
                [True, True, False],
                [True, False, True],
                [False, True, True],
                [True, True, True],
            ]
        )

        output_masked = embedding(efp_features, jet_mask=jet_mask)
        output_unmasked = embedding(efp_features)

        # Valid jets should have identical embeddings
        for i in range(4):
            for j in range(3):
                if jet_mask[i, j]:
                    assert torch.allclose(
                        output_masked[i, j], output_unmasked[i, j], atol=1e-6
                    )

    def test_mask_shape_validation(self):
        """Test validation of jet mask shape."""
        embedding = EFPEmbedding(n_efp_features=140, embedding_dim=64)
        efp_features = torch.randn(8, 3, 140)

        # Wrong mask shape should raise ValueError
        wrong_mask = torch.ones(8, 4)  # Wrong number of jets
        with pytest.raises(ValueError, match="jet_mask shape"):
            embedding(efp_features, jet_mask=wrong_mask)

        wrong_mask = torch.ones(4, 3)  # Wrong batch size
        with pytest.raises(ValueError, match="jet_mask shape"):
            embedding(efp_features, jet_mask=wrong_mask)


class TestEFPEmbeddingGating:
    """Test gate sparsification and threshold functionality."""

    def test_gate_activation_functions(self):
        """Test different gate activation functions."""
        n_efp_features, embedding_dim = 140, 64
        efp_features = torch.randn(4, 3, n_efp_features)

        for gate_type in ["sigmoid", "relu6", "tanh"]:
            embedding = EFPEmbedding(
                n_efp_features=n_efp_features,
                embedding_dim=embedding_dim,
                gate_type=gate_type,
            )

            output = embedding(efp_features)
            assert output.shape == (4, 3, embedding_dim)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_gate_threshold_sparsification(self):
        """Test that gate threshold creates sparsity."""
        embedding = EFPEmbedding(
            n_efp_features=140,
            embedding_dim=64,
            gate_threshold=0.5,  # High threshold for more sparsity
            monitor_sparsity=True,
        )

        # Use fixed input to get predictable gate behavior
        efp_features = torch.randn(8, 3, 140)

        embedding.train()  # Enable training mode for sparsity monitoring
        embedding(efp_features)

        # Check that some sparsity was achieved
        stats = embedding.get_sparsity_stats()
        assert stats["sparsity_ratio"] > 0.0
        assert stats["activation_ratio"] < 1.0
        assert stats["total_gates_seen"] > 0

    def test_zero_threshold_no_sparsification(self):
        """Test that zero threshold disables sparsification."""
        embedding = EFPEmbedding(
            n_efp_features=140,
            embedding_dim=64,
            gate_threshold=0.0,  # No threshold
            monitor_sparsity=True,
        )

        efp_features = torch.randn(8, 3, 140)

        embedding.train()
        embedding(efp_features)

        # With zero threshold, sparsity should be minimal
        stats = embedding.get_sparsity_stats()
        assert stats["sparsity_ratio"] < 0.1  # Very low sparsity expected


class TestEFPEmbeddingConfig:
    """Test configuration validation and parameter handling."""

    def test_invalid_n_efp_features(self):
        """Test validation of n_efp_features parameter."""
        with pytest.raises(ValueError, match="n_efp_features must be positive"):
            EFPEmbedding(n_efp_features=0)

        with pytest.raises(ValueError, match="n_efp_features must be positive"):
            EFPEmbedding(n_efp_features=-10)

    def test_invalid_embedding_dim(self):
        """Test validation of embedding_dim parameter."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            EFPEmbedding(n_efp_features=140, embedding_dim=0)

        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            EFPEmbedding(n_efp_features=140, embedding_dim=-5)

    def test_invalid_gate_type(self):
        """Test validation of gate_type parameter."""
        with pytest.raises(ValueError, match="gate_type must be one of"):
            EFPEmbedding(n_efp_features=140, gate_type="invalid")

    def test_invalid_dropout_rate(self):
        """Test validation of dropout_rate parameter."""
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            EFPEmbedding(n_efp_features=140, dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be in"):
            EFPEmbedding(n_efp_features=140, dropout_rate=1.5)

    def test_invalid_gate_threshold(self):
        """Test validation of gate_threshold parameter."""
        with pytest.raises(ValueError, match="gate_threshold must be in"):
            EFPEmbedding(n_efp_features=140, gate_threshold=-0.1)

        with pytest.raises(ValueError, match="gate_threshold must be in"):
            EFPEmbedding(n_efp_features=140, gate_threshold=1.5)

    def test_extra_repr(self):
        """Test string representation of the layer."""
        embedding = EFPEmbedding(
            n_efp_features=140,
            embedding_dim=64,
            gate_type="sigmoid",
            gate_threshold=0.05,
            dropout_rate=0.1,
            use_layer_norm=True,
            monitor_sparsity=True,
        )

        repr_str = str(embedding)
        assert "n_efp_features=140" in repr_str
        assert "embedding_dim=64" in repr_str
        assert "gate_type=sigmoid" in repr_str
        assert "gate_threshold=0.05" in repr_str


class TestEFPEmbeddingGradients:
    """Test gradient flow and backpropagation."""

    def test_gradient_flow(self):
        """Test that gradients flow through the embedding layer."""
        embedding = EFPEmbedding(n_efp_features=140, embedding_dim=64)
        efp_features = torch.randn(4, 3, 140, requires_grad=True)

        output = embedding(efp_features)
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert efp_features.grad is not None
        assert not torch.isnan(efp_features.grad).any()

        # Check that layer parameters have gradients
        assert embedding.projection.weight.grad is not None
        assert embedding.gate.weight.grad is not None
        assert not torch.isnan(embedding.projection.weight.grad).any()
        assert not torch.isnan(embedding.gate.weight.grad).any()

    def test_gradient_flow_with_masking(self):
        """Test gradient flow with jet masking."""
        embedding = EFPEmbedding(n_efp_features=140, embedding_dim=64)
        efp_features = torch.randn(4, 3, 140, requires_grad=True)
        jet_mask = torch.tensor(
            [
                [True, True, False],
                [True, False, True],
                [False, True, True],
                [True, True, True],
            ]
        )

        output = embedding(efp_features, jet_mask=jet_mask)
        loss = output.sum()
        loss.backward()

        # Gradients should exist and be finite
        assert efp_features.grad is not None
        assert torch.isfinite(efp_features.grad).all()
        assert torch.isfinite(embedding.projection.weight.grad).all()
        assert torch.isfinite(embedding.gate.weight.grad).all()

    def test_gradient_magnitude_reasonable(self):
        """Test that gradient magnitudes are reasonable (not exploding/vanishing)."""
        embedding = EFPEmbedding(n_efp_features=140, embedding_dim=64)
        efp_features = torch.randn(8, 3, 140, requires_grad=True)

        output = embedding(efp_features)
        loss = output.mean()
        loss.backward()

        # Check gradient magnitudes are reasonable
        input_grad_norm = efp_features.grad.norm().item()
        weight_grad_norm = embedding.projection.weight.grad.norm().item()

        assert 1e-6 < input_grad_norm < 1e3, (
            f"Input gradient norm {input_grad_norm} seems unreasonable"
        )
        assert 1e-6 < weight_grad_norm < 1e3, (
            f"Weight gradient norm {weight_grad_norm} seems unreasonable"
        )


class TestEFPEmbeddingEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_input_shape(self):
        """Test handling of invalid input shapes."""
        embedding = EFPEmbedding(n_efp_features=140, embedding_dim=64)

        # Wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 3D input"):
            embedding(torch.randn(32, 140))  # 2D instead of 3D

        with pytest.raises(ValueError, match="Expected 3D input"):
            embedding(torch.randn(32, 3, 140, 5))  # 4D instead of 3D

    def test_wrong_feature_count(self):
        """Test handling of wrong number of EFP features."""
        embedding = EFPEmbedding(n_efp_features=140, embedding_dim=64)

        with pytest.raises(ValueError, match="Expected 140 EFP features"):
            embedding(torch.randn(32, 3, 531))  # Wrong feature count

    def test_nan_inf_input_detection(self):
        """Test detection of NaN and Inf values in input."""
        embedding = EFPEmbedding(n_efp_features=140, embedding_dim=64)

        # Test NaN detection
        efp_features_nan = torch.randn(4, 3, 140)
        efp_features_nan[0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="Input contains NaN or Inf values"):
            embedding(efp_features_nan)

        # Test Inf detection
        efp_features_inf = torch.randn(4, 3, 140)
        efp_features_inf[0, 0, 0] = float("inf")
        with pytest.raises(ValueError, match="Input contains NaN or Inf values"):
            embedding(efp_features_inf)

    def test_zero_input_handling(self):
        """Test handling of all-zero input (empty jets)."""
        embedding = EFPEmbedding(n_efp_features=140, embedding_dim=64)

        # All-zero input should not crash
        efp_features = torch.zeros(4, 3, 140)
        output = embedding(efp_features)

        assert output.shape == (4, 3, 64)
        assert torch.isfinite(output).all()

    def test_extreme_values_handling(self):
        """Test handling of extreme input values."""
        embedding = EFPEmbedding(n_efp_features=140, embedding_dim=64)

        # Very large values
        efp_features_large = torch.full((4, 3, 140), 1000.0)
        output_large = embedding(efp_features_large)
        assert torch.isfinite(output_large).all()

        # Very small values
        efp_features_small = torch.full((4, 3, 140), 1e-6)
        output_small = embedding(efp_features_small)
        assert torch.isfinite(output_small).all()


class TestEFPEmbeddingSparsity:
    """Test sparsity monitoring and statistics."""

    def test_sparsity_monitoring_enabled(self):
        """Test sparsity monitoring when enabled."""
        embedding = EFPEmbedding(
            n_efp_features=140,
            embedding_dim=64,
            monitor_sparsity=True,
            gate_threshold=0.3,
        )

        efp_features = torch.randn(8, 3, 140)

        # Initially no statistics
        stats = embedding.get_sparsity_stats()
        assert stats["total_gates_seen"] == 0

        # After forward pass in training mode
        embedding.train()
        embedding(efp_features)

        stats = embedding.get_sparsity_stats()
        assert stats["total_gates_seen"] > 0
        assert 0.0 <= stats["sparsity_ratio"] <= 1.0
        assert 0.0 <= stats["activation_ratio"] <= 1.0
        assert abs(stats["sparsity_ratio"] + stats["activation_ratio"] - 1.0) < 1e-6

    def test_sparsity_monitoring_disabled(self):
        """Test sparsity monitoring when disabled."""
        embedding = EFPEmbedding(
            n_efp_features=140, embedding_dim=64, monitor_sparsity=False
        )

        efp_features = torch.randn(8, 3, 140)
        embedding.train()
        embedding(efp_features)

        stats = embedding.get_sparsity_stats()
        assert stats["sparsity_ratio"] is None
        assert stats["activation_ratio"] is None
        assert stats["total_gates_seen"] == 0

    def test_sparsity_stats_with_masking(self):
        """Test sparsity statistics with jet masking."""
        embedding = EFPEmbedding(
            n_efp_features=140, embedding_dim=64, monitor_sparsity=True
        )

        efp_features = torch.randn(4, 3, 140)
        jet_mask = torch.tensor(
            [
                [True, True, False],
                [True, False, False],
                [False, False, False],
                [True, True, True],
            ]
        )

        embedding.train()
        embedding(efp_features, jet_mask=jet_mask)

        stats = embedding.get_sparsity_stats()

        # Should only count gates for valid jets (6 valid jets total: 2+1+0+3)
        expected_gates = 6 * 64  # 6 valid jets * 64 embedding dimensions
        assert stats["total_gates_seen"] == expected_gates

    def test_reset_sparsity_stats(self):
        """Test resetting sparsity statistics."""
        embedding = EFPEmbedding(
            n_efp_features=140, embedding_dim=64, monitor_sparsity=True
        )

        efp_features = torch.randn(4, 3, 140)

        # Accumulate some statistics
        embedding.train()
        embedding(efp_features)

        stats_before = embedding.get_sparsity_stats()
        assert stats_before["total_gates_seen"] > 0

        # Reset statistics
        embedding.reset_sparsity_stats()

        stats_after = embedding.get_sparsity_stats()
        assert stats_after["total_gates_seen"] == 0
        assert stats_after["sparsity_ratio"] == 0.0
        assert stats_after["activation_ratio"] == 0.0

    def test_sparsity_eval_mode(self):
        """Test that sparsity monitoring only works in training mode."""
        embedding = EFPEmbedding(
            n_efp_features=140, embedding_dim=64, monitor_sparsity=True
        )

        efp_features = torch.randn(4, 3, 140)

        # In eval mode, statistics should not update
        embedding.eval()
        embedding(efp_features)

        stats = embedding.get_sparsity_stats()
        assert stats["total_gates_seen"] == 0


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
