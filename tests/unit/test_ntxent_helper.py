"""
Unit tests for NT-Xent helper functions.
"""

import torch
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from bead.src.utils.helper import get_ntxent_outputs


class MockConfig:
    """Mock configuration class for testing."""
    def __init__(self, ntxent_noise_sigma=0.1):
        self.ntxent_noise_sigma = ntxent_noise_sigma


class MockModel(torch.nn.Module):
    """Mock model for testing."""
    def __init__(self, output_type="ae"):
        super(MockModel, self).__init__()
        self.output_type = output_type
    
    def forward(self, x):
        # Return different output formats based on model type
        batch_size, features = x.shape
        latent_dim = 8
        
        # Mock reconstruction (same shape as input)
        recon = x.clone()
        
        # Mock latent variables
        zk = torch.randn(batch_size, latent_dim)
        
        if self.output_type == "ae":
            # Basic AE: (recon, zk)
            return recon, zk
        elif self.output_type == "vae":
            # VAE: (recon, mu, logvar, zk)
            mu = torch.randn(batch_size, latent_dim)
            logvar = torch.randn(batch_size, latent_dim)
            return recon, mu, logvar, zk
        else:  # flow
            # Flow: (recon, mu, logvar, ldj, z0, zk)
            mu = torch.randn(batch_size, latent_dim)
            logvar = torch.randn(batch_size, latent_dim)
            ldj = torch.randn(batch_size, 1)
            z0 = torch.randn(batch_size, latent_dim)
            return recon, mu, logvar, ldj, z0, zk


class TestNTXentHelper:
    def test_get_ntxent_outputs_flow(self):
        """Test that get_ntxent_outputs properly handles flow model outputs."""
        config = MockConfig(ntxent_noise_sigma=0.1)
        model = MockModel(output_type="flow")
        
        # Create sample input
        batch_size, features = 4, 16
        inputs = torch.randn(batch_size, features)
        
        # Call the function
        recon, mu, logvar, ldj, z0, zk, zk_j = get_ntxent_outputs(model, inputs, config)
        
        # Check output types
        assert isinstance(recon, torch.Tensor)
        assert isinstance(mu, torch.Tensor)
        assert isinstance(logvar, torch.Tensor)
        assert isinstance(ldj, torch.Tensor)
        assert isinstance(z0, torch.Tensor)
        assert isinstance(zk, torch.Tensor)
        assert isinstance(zk_j, torch.Tensor)
        
        # Check shapes
        assert recon.shape == inputs.shape
        assert mu.shape == (batch_size, 8)
        assert logvar.shape == (batch_size, 8)
        assert z0.shape == (batch_size, 8)
        assert zk.shape == (batch_size, 8)
        assert zk_j.shape == (batch_size, 8)
        
        # zk and zk_j should be different (from different augmented views)
        assert not torch.allclose(zk, zk_j)
    
    def test_get_ntxent_outputs_vae(self):
        """Test that get_ntxent_outputs properly handles VAE model outputs."""
        config = MockConfig(ntxent_noise_sigma=0.1)
        model = MockModel(output_type="vae")
        
        # Create sample input
        batch_size, features = 4, 16
        inputs = torch.randn(batch_size, features)
        
        # Call the function
        recon, mu, logvar, ldj, z0, zk, zk_j = get_ntxent_outputs(model, inputs, config)
        
        # Check output types
        assert isinstance(recon, torch.Tensor)
        assert isinstance(mu, torch.Tensor)
        assert isinstance(logvar, torch.Tensor)
        assert isinstance(ldj, torch.Tensor)
        assert isinstance(z0, torch.Tensor)
        assert isinstance(zk, torch.Tensor)
        assert isinstance(zk_j, torch.Tensor)
        
        # For VAE, zk should be same as z0 (no flow)
        assert torch.allclose(zk, z0)
        
        # zk and zk_j should be different
        assert not torch.allclose(zk, zk_j)
    
    def test_get_ntxent_outputs_ae(self):
        """Test that get_ntxent_outputs properly handles basic AE model outputs."""
        config = MockConfig(ntxent_noise_sigma=0.1)
        model = MockModel(output_type="ae")
        
        # Create sample input
        batch_size, features = 4, 16
        inputs = torch.randn(batch_size, features)
        
        # Call the function
        recon, mu, logvar, ldj, z0, zk, zk_j = get_ntxent_outputs(model, inputs, config)
        
        # Check output types
        assert isinstance(recon, torch.Tensor)
        assert isinstance(mu, torch.Tensor)
        assert isinstance(logvar, torch.Tensor)
        assert isinstance(ldj, torch.Tensor)
        assert isinstance(z0, torch.Tensor)
        assert isinstance(zk, torch.Tensor)
        assert isinstance(zk_j, torch.Tensor)
        
        # For AE, mu and logvar should be zero tensors
        assert torch.all(mu == 0)
        assert torch.all(logvar == 0)
        
        # zk and zk_j should be different
        assert not torch.allclose(zk, zk_j)
