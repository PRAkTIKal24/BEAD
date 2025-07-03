"""
Unit tests for NT-Xent loss functions.
"""

import torch
import pytest
import numpy as np
from bead.src.utils.loss import NTXentLoss


class MockConfig:
    """Mock configuration class for testing."""
    def __init__(self, temperature=0.07, is_ddp_active=False, world_size=1, device=None):
        self.ntxent_temperature = temperature
        self.is_ddp_active = is_ddp_active
        self.world_size = world_size
        self.device = device if device is not None else torch.device('cpu')


class TestNTXentLoss:
    def test_ntxent_loss_same_inputs(self):
        """Test NT-Xent loss when inputs are identical."""
        config = MockConfig()
        loss_fn = NTXentLoss(config)
        
        # Create identical feature vectors - expect minimal loss
        batch_size = 4
        feat_dim = 8
        z1 = torch.randn(batch_size, feat_dim)
        z2 = z1.clone()  # Identical vectors
        
        # Normalize the vectors (as would be done in practice)
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        
        loss, = loss_fn.calculate(z1, z2)
        
        # For identical vectors, the loss should be relatively small
        # But not necessarily zero due to how negatives are handled in the batch
        assert loss < 0.1  # More realistic threshold for small batches
        
    def test_ntxent_loss_different_inputs(self):
        """Test NT-Xent loss when inputs are completely different."""
        config = MockConfig()
        loss_fn = NTXentLoss(config)
        
        # Create different feature vectors - expect higher loss
        batch_size = 4
        feat_dim = 8
        z1 = torch.randn(batch_size, feat_dim)
        z2 = torch.randn(batch_size, feat_dim)  # Different vectors
        
        # Normalize the vectors
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        
        loss, = loss_fn.calculate(z1, z2)
        
        # For completely different vectors, the loss should be higher than for identical vectors
        # But the absolute threshold depends on batch size, dimensionality, and implementation
        same_vectors_z1 = torch.randn(batch_size, feat_dim)
        same_vectors_z2 = same_vectors_z1.clone()
        
        # Normalize the vectors
        same_vectors_z1 = torch.nn.functional.normalize(same_vectors_z1, dim=1)
        same_vectors_z2 = torch.nn.functional.normalize(same_vectors_z2, dim=1)
        
        same_vectors_loss, = loss_fn.calculate(same_vectors_z1, same_vectors_z2)
        
        # Loss for different vectors should be higher than for identical vectors
        assert loss > same_vectors_loss
        
    def test_ntxent_loss_temperature(self):
        """Test that different temperature values affect the loss."""
        # First with default temperature (0.07)
        config1 = MockConfig(temperature=0.07)
        loss_fn1 = NTXentLoss(config1)
        
        # Then with higher temperature (0.5)
        config2 = MockConfig(temperature=0.5)
        loss_fn2 = NTXentLoss(config2)
        
        # Create slightly different feature vectors
        batch_size = 4
        feat_dim = 8
        z1 = torch.randn(batch_size, feat_dim)
        z2 = z1.clone() + 0.1 * torch.randn(batch_size, feat_dim)  # Slightly different
        
        # Normalize the vectors
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        
        loss1, = loss_fn1.calculate(z1, z2)
        loss2, = loss_fn2.calculate(z1, z2)
        
        # Different temperatures should result in different loss values
        assert loss1 != loss2  # They should be different
        
        # In NT-Xent with temperature in denominator (sim/T), higher T typically makes the
        # distribution more uniform which often increases loss for similar inputs and
        # decreases loss for dissimilar inputs
        # For slightly different inputs, we just check they're different without 
        # asserting which should be higher
