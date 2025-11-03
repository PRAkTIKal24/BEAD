#!/usr/bin/env python3
"""
Unit tests for the GPU-accelerated plotting functionality.

These tests verify that the plotting module correctly handles:
1. Detection of GPU availability
2. Using GPU-accelerated methods when available
3. Proper fallback to CPU methods when GPU is not available
4. Different dimensionality reduction methods (PCA, t-SNE, UMAP, TriMap)
5. Subsampling functionality
"""

import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from bead.src.utils.plotting import (
    reduce_dim_subsampled,
    CUUMAP_AVAILABLE,
    UMAP_AVAILABLE,
    TRIMAP_AVAILABLE,
)


class TestGPUAcceleratedPlotting(unittest.TestCase):
    """Test GPU-accelerated plotting functionality."""

    def setUp(self):
        """Set up test fixtures for each test method."""
        # Create synthetic data for testing
        np.random.seed(42)  # For reproducibility
        self.data = np.random.normal(size=(100, 50))  # 100 samples, 50 features

    def test_reduce_dim_subsampled_pca(self):
        """Test PCA dimensionality reduction."""
        # Test basic PCA functionality
        reduced_data, method, indices = reduce_dim_subsampled(
            self.data, method="pca", verbose=False
        )

        # Check output shapes and types
        self.assertEqual(reduced_data.shape, (100, 2))  # Default output is 2D
        self.assertEqual(method, "pca")
        self.assertEqual(len(indices), 100)  # Should include all samples
        self.assertIsInstance(reduced_data, np.ndarray)

    def test_reduce_dim_subsampled_tsne(self):
        """Test t-SNE dimensionality reduction."""
        # Test basic t-SNE functionality
        reduced_data, method, indices = reduce_dim_subsampled(
            self.data, method="tsne", verbose=False
        )

        # Check output shapes and types
        self.assertEqual(reduced_data.shape, (100, 2))
        self.assertEqual(method, "t-sne")
        self.assertEqual(len(indices), 100)
        self.assertIsInstance(reduced_data, np.ndarray)

    @unittest.skipIf(not TRIMAP_AVAILABLE, "TriMap is not installed")
    def test_reduce_dim_subsampled_trimap(self):
        """Test TriMap dimensionality reduction."""
        # Test basic TriMap functionality
        reduced_data, method, indices = reduce_dim_subsampled(
            self.data, method="trimap", verbose=False
        )

        # Check output shapes and types
        self.assertEqual(reduced_data.shape, (100, 2))
        self.assertEqual(method, "trimap")
        self.assertEqual(len(indices), 100)
        self.assertIsInstance(reduced_data, np.ndarray)

    @unittest.skipIf(
        not UMAP_AVAILABLE and not CUUMAP_AVAILABLE, "UMAP is not installed"
    )
    def test_reduce_dim_subsampled_umap(self):
        """Test UMAP dimensionality reduction."""
        # Test basic UMAP functionality
        reduced_data, method, indices = reduce_dim_subsampled(
            self.data, method="umap", verbose=False
        )

        # Check output shapes and types
        self.assertEqual(reduced_data.shape, (100, 2))
        self.assertEqual(method, "umap")
        self.assertEqual(len(indices), 100)
        self.assertIsInstance(reduced_data, np.ndarray)

    def test_subsampling(self):
        """Test subsampling functionality."""
        # Test with subsampling
        reduced_data, method, indices = reduce_dim_subsampled(
            self.data, method="pca", n_samples=50, verbose=False
        )

        # Check that only n_samples were used
        self.assertEqual(reduced_data.shape, (50, 2))
        self.assertEqual(len(indices), 50)

        # Check that indices are unique (no duplicates)
        self.assertEqual(len(indices), len(np.unique(indices)))

    def test_invalid_method(self):
        """Test handling of invalid reduction method."""
        # Should gracefully handle invalid method by falling back to t-SNE
        with self.assertWarns(Warning):  # Should issue a warning
            reduced_data, method, indices = reduce_dim_subsampled(
                self.data, method="invalid_method", verbose=False
            )

        # Check that it fell back to t-SNE
        self.assertEqual(reduced_data.shape, (100, 2))
        self.assertEqual(len(indices), 100)


class TestGPUAvailability(unittest.TestCase):
    """Test handling of GPU availability."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.data = np.random.normal(size=(100, 50))

    @patch("bead.src.utils.helper.get_device")
    def test_gpu_usage_when_available(self, mock_get_device):
        """Test that GPU implementations are used when available."""
        # Import the plotting module directly for this test
        from bead.src.utils import plotting

        # Skip test if cuML is not available
        if not plotting.CUML_AVAILABLE:
            self.skipTest("cuML is not available, skipping GPU-specific test")

        # Mock GPU availability
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_get_device.return_value = mock_device

        # Override CUML_AVAILABLE flag to True for testing
        original_flag = plotting.CUML_AVAILABLE
        plotting.CUML_AVAILABLE = True

        try:
            # Use mock_module to create a mock for cuPCA since it might not exist
            module_mock = MagicMock()
            mock_instance = MagicMock()
            mock_instance.fit_transform.return_value = np.zeros((100, 2))
            module_mock.return_value = mock_instance

            # Temporarily add the mock to the plotting module
            original_cuPCA = getattr(plotting, "cuPCA", None)
            plotting.cuPCA = module_mock

            # Call function with PCA method
            reduce_dim_subsampled(self.data, method="pca", verbose=False)

            # Assert our mock was called
            module_mock.assert_called_once()

        finally:
            # Restore original attributes
            plotting.CUML_AVAILABLE = original_flag
            if original_cuPCA is not None:
                plotting.cuPCA = original_cuPCA
            else:
                delattr(plotting, "cuPCA")

    @patch("bead.src.utils.helper.get_device")
    def test_cpu_fallback(self, mock_get_device):
        """Test fallback to CPU when GPU is not available."""
        # Import plotting module directly
        from bead.src.utils import plotting

        # Override CUML_AVAILABLE flag to True for testing
        original_flag = plotting.CUML_AVAILABLE
        plotting.CUML_AVAILABLE = True

        try:
            # Mock CPU-only environment
            mock_device = MagicMock()
            mock_device.type = "cpu"
            mock_get_device.return_value = mock_device

            # Mock CPU PCA
            with patch("bead.src.utils.plotting.PCA") as mock_PCA:
                mock_PCA_instance = MagicMock()
                mock_PCA_instance.fit_transform.return_value = np.zeros((100, 2))
                mock_PCA.return_value = mock_PCA_instance

                # Call function with PCA method
                reduce_dim_subsampled(self.data, method="pca", verbose=False)

                # Assert sklearn PCA was called, not cuPCA
                mock_PCA.assert_called()

        finally:
            # Restore original flag
            plotting.CUML_AVAILABLE = original_flag

    @patch("bead.src.utils.helper.get_device")
    def test_cpu_usage_when_gpu_unavailable(self, mock_get_device):
        """Test that CPU implementations are used when GPU libraries are not installed."""
        # Import plotting module directly
        from bead.src.utils import plotting

        # Save original flag and override CUML_AVAILABLE to False for testing
        original_flag = plotting.CUML_AVAILABLE
        plotting.CUML_AVAILABLE = False

        try:
            # Mock environment with GPU hardware but no cuML
            mock_device = MagicMock()
            mock_device.type = "cuda"
            mock_get_device.return_value = mock_device

            # Mock CPU PCA
            with patch("bead.src.utils.plotting.PCA") as mock_PCA:
                mock_PCA_instance = MagicMock()
                mock_PCA_instance.fit_transform.return_value = np.zeros((100, 2))
                mock_PCA.return_value = mock_PCA_instance

                # Call function with PCA method
                reduce_dim_subsampled(self.data, method="pca", verbose=False)

                # Assert sklearn PCA was called since cuML is not available
                mock_PCA.assert_called()

        finally:
            # Restore original flag
            plotting.CUML_AVAILABLE = original_flag

    def test_tensor_conversion(self):
        """Test handling of torch tensors as input."""
        # Create a torch tensor
        tensor_data = torch.tensor(self.data, dtype=torch.float32)

        # Process the tensor
        reduced_data, method, indices = reduce_dim_subsampled(
            tensor_data, method="pca", verbose=False
        )

        # Check that the tensor was properly converted and processed
        self.assertEqual(reduced_data.shape, (100, 2))
        self.assertEqual(method, "pca")
        self.assertEqual(len(indices), 100)
        self.assertIsInstance(reduced_data, np.ndarray)


if __name__ == "__main__":
    unittest.main()
