"""
Unit tests for helper functions.
"""

import torch
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from bead.src.utils.helper import get_ntxent_outputs
from bead.src.utils.ntxent_utils import generate_augmented_views


class MockConfig:
    """Mock configuration class for testing."""
    def __init__(self, ntxent_sigma=0.1, temperature=0.07, is_ddp_active=False, world_size=1, device=None):
        self.ntxent_sigma = ntxent_sigma  # Using consistent parameter naming
        self.ntxent_temperature = temperature
        self.is_ddp_active = is_ddp_active
        self.world_size = world_size
        self.device = device if device is not None else torch.device('cpu')


class MockModel(torch.nn.Module):
    """Mock model for testing."""
    def __init__(self, output_type="ae"):
        """Initialize mock model."""
        super(MockModel, self).__init__()
        self.output_type = output_type
    
    def forward(self, x):
        """Mock forward pass."""
        batch_size = x.shape[0]
        if self.output_type == "ae":
            recon = x  # Just return the input as reconstruction for simplicity
            zk = torch.randn(batch_size, 10)  # Random latent vector
            return (recon, zk, zk, zk, zk, zk)  # Basic AE format
        elif self.output_type == "vae":
            recon = x
            mu = torch.randn(batch_size, 10)
            logvar = torch.randn(batch_size, 10)
            return (recon, mu, logvar, mu, mu, mu)  # VAE format
        elif self.output_type == "flow":
            recon = x
            mu = torch.randn(batch_size, 10)
            logvar = torch.randn(batch_size, 10)
            ldj = torch.randn(batch_size)
            z0 = torch.randn(batch_size, 10)
            zk = torch.randn(batch_size, 10)
            return (recon, mu, logvar, ldj, z0, zk)  # Flow model format


class TestNTXentHelper:
    def test_get_ntxent_outputs_flow(self):
        """Test that get_ntxent_outputs properly handles flow model outputs."""
        model = MockModel(output_type="flow")
        inputs = torch.randn(5, 32)  # Batch size 5, dim 32
        config = MockConfig()
        
        outputs = get_ntxent_outputs(model, inputs, config)
        
        # Check that we get the expected 7 outputs
        assert len(outputs) == 7
        
        # Check that zk_i and zk_j are different
        zk_i, zk_j = outputs[-2], outputs[-1]
        assert not torch.allclose(zk_i, zk_j)
        
        # Check that the outputs have the expected shapes
        recon, mu, logvar, ldj, z0, zk_i, zk_j = outputs
        assert recon.shape == inputs.shape
        assert mu.shape == (5, 10)
        assert logvar.shape == (5, 10)
        assert ldj.shape == (5,)
        assert z0.shape == (5, 10)
        assert zk_i.shape == (5, 10)
        assert zk_j.shape == (5, 10)
    
    def test_get_ntxent_outputs_vae(self):
        """Test that get_ntxent_outputs properly handles VAE model outputs."""
        model = MockModel(output_type="vae")
        inputs = torch.randn(5, 32)
        config = MockConfig()
        
        outputs = get_ntxent_outputs(model, inputs, config)
        
        # Check that we get the expected 7 outputs
        assert len(outputs) == 7
        
        # Check that the last two outputs (zk_i and zk_j) are different
        assert not torch.allclose(outputs[-2], outputs[-1])
    
    def test_get_ntxent_outputs_ae(self):
        """Test that get_ntxent_outputs properly handles basic AE model outputs."""
        model = MockModel(output_type="ae")
        inputs = torch.randn(5, 32)
        config = MockConfig()
        
        outputs = get_ntxent_outputs(model, inputs, config)
        
        # Check that we get the expected 7 outputs
        assert len(outputs) == 7
        
        # For AE, all the middle outputs are the same as zk
        assert torch.allclose(outputs[1], outputs[5])  # mu = zk
        assert torch.allclose(outputs[2], outputs[5])  # logvar = zk
        
        # But zk_i and zk_j should be different
        assert not torch.allclose(outputs[5], outputs[6])  # zk_i ≠ zk_j


class TestNTXentUtils:
    def test_generate_augmented_views(self):
        """Test that generate_augmented_views produces two different augmented views."""
        # Create a simple test tensor
        test_tensor = torch.ones(10, 5)
        
        # Generate augmented views with default sigma
        view1, view2 = generate_augmented_views(test_tensor)
        
        # Check that both views have the same shape as the input
        assert view1.shape == test_tensor.shape
        assert view2.shape == test_tensor.shape
        
        # Check that the views are different from each other
        assert not torch.allclose(view1, view2)
        
        # Check that the views are different from the original
        assert not torch.allclose(view1, test_tensor)
        assert not torch.allclose(view2, test_tensor)
        
        # Test with a specific sigma value
        sigma = 0.05
        view1, view2 = generate_augmented_views(test_tensor, sigma=sigma)
        
        # Check that the noise level is approximately as expected
        # (mean should be close to 1.0, with small deviation)
        assert 0.95 < view1.mean() < 1.05
        assert 0.95 < view2.mean() < 1.05
        
        # Standard deviation should be close to sigma
        # (not exactly sigma due to sampling variance)
        assert abs((view1 - test_tensor).std() - sigma) < 0.02
        assert abs((view2 - test_tensor).std() - sigma) < 0.02
