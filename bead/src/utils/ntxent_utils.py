"""
NT-Xent (Normalized Temperature-Scaled Cross Entropy Loss) utilities.

This module provides utility functions for implementing NT-Xent loss in BEAD models.
Functions include data augmentation methods for creating contrastive views of inputs.
"""

import torch


def generate_augmented_views(x, sigma=0.1):
    """
    Generate two augmented views of the input by adding Gaussian noise.
    Used for NT-Xent contrastive learning which requires positive pairs.
    
    Args:
        x (torch.Tensor): Input tensor to augment
        sigma (float): Standard deviation of the Gaussian noise
        
    Returns:
        tuple: (x_aug1, x_aug2) Two differently augmented views of the input tensor
    """
    # Generate two different noise tensors with the same shape as the input
    noise1 = torch.randn_like(x) * sigma
    noise2 = torch.randn_like(x) * sigma
    
    # Add noise to create two different augmented views
    x_aug1 = x + noise1
    x_aug2 = x + noise2
    
    return x_aug1, x_aug2
