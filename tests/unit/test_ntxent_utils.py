"""
Unit tests for NT-Xent utilities.
"""

import torch
import pytest
import numpy as np
from bead.src.utils.ntxent_utils import generate_augmented_views


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
