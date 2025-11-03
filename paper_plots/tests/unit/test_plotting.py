#!/usr/bin/env python3
"""
Unit tests for the plotting functionality.

These tests verify that the plotting module correctly handles:
1. Loss plots generation
2. Latent space visualization
3. Mean and log-variance visualization
4. ROC curve generation
5. ROC curve overlay functionality
"""

import unittest
import os
import numpy as np
import tempfile
import shutil
from types import SimpleNamespace
from unittest.mock import patch, call

from bead.src.utils.plotting import (
    plot_losses,
    plot_latent_variables,
    plot_mu_logvar,
    plot_roc_curve,
)


class TestPlotLosses(unittest.TestCase):
    """Test loss plotting functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "results")
        self.save_dir = os.path.join(self.temp_dir, "plots", "loss")
        os.makedirs(self.output_dir)
        os.makedirs(self.save_dir)

        # Create dummy config
        self.config = SimpleNamespace()
        self.config.project_name = "test_project"
        self.config.epochs = 10

        # Create dummy loss data files
        self.train_epoch_loss = np.random.rand(10)
        self.val_epoch_loss = np.random.rand(10)
        np.save(
            os.path.join(self.output_dir, "train_epoch_loss_data.npy"),
            self.train_epoch_loss,
        )
        np.save(
            os.path.join(self.output_dir, "val_epoch_loss_data.npy"),
            self.val_epoch_loss,
        )

        # Create dummy loss component files for different categories
        self.categories = ["train", "val", "test"]
        self.loss_components = ["loss", "reco", "kl"]

        for category in self.categories:
            for component in self.loss_components:
                # Create dummy data with 10 epochs, 100 events per epoch
                data = np.random.rand(10 * 100)
                np.save(
                    os.path.join(self.output_dir, f"{component}_{category}.npy"), data
                )

    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.temp_dir)

    @patch("matplotlib.pyplot.savefig")
    def test_plot_losses_generates_files(self, mock_savefig):
        """Test that plot_losses generates the expected files."""
        # Run the function
        plot_losses(self.output_dir, self.save_dir, self.config, verbose=True)

        # Check that savefig was called the expected number of times
        # 1 for train_metrics.pdf + 1 for each category (train, val, test)
        expected_calls = 1 + len(self.categories)
        self.assertEqual(mock_savefig.call_count, expected_calls)

        # Check specific calls
        mock_savefig.assert_any_call(os.path.join(self.save_dir, "train_metrics.pdf"))
        for category in self.categories:
            mock_savefig.assert_any_call(
                os.path.join(self.save_dir, f"loss_components_{category}.pdf")
            )

    def test_plot_losses_missing_files(self):
        """Test that plot_losses raises FileNotFoundError when files are missing."""
        # Remove one of the required files
        os.remove(os.path.join(self.output_dir, "train_epoch_loss_data.npy"))

        # Run the function, expecting an exception
        with self.assertRaises(FileNotFoundError):
            plot_losses(self.output_dir, self.save_dir, self.config)


class TestLatentVariablesPlotting(unittest.TestCase):
    """Test latent variable plotting functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

        # Create directory structure
        self.paths = {
            "output_path": os.path.join(self.temp_dir, "output"),
            "data_path": os.path.join(self.temp_dir, "data"),
        }

        # Create required subdirectories
        os.makedirs(os.path.join(self.paths["output_path"], "results"))
        os.makedirs(os.path.join(self.paths["output_path"], "plots", "latent_space"))

        self.file_type = "h5"
        h5_tensor_path = os.path.join(
            self.paths["data_path"], self.file_type, "tensors", "processed"
        )
        os.makedirs(h5_tensor_path, exist_ok=True)

        # Create dummy config
        self.config = SimpleNamespace()
        self.config.input_level = "constituent"
        self.config.latent_space_size = 10
        self.config.latent_space_plot_style = "pca"  # Use PCA for speed in tests
        self.config.file_type = self.file_type
        self.config.project_name = "test_project"
        self.config.subsample_plot = False
        self.config.subsample_size = 1000

        # Generate test data for latent variables
        self.n_samples = 100
        self.latent_dim = 10

        # Create dummy data for training set
        np.random.seed(42)
        self.train_z0 = np.random.randn(self.n_samples, self.latent_dim)
        self.train_zk = np.random.randn(self.n_samples, self.latent_dim)
        self.train_gen_labels = np.random.randint(0, 3, self.n_samples)

        # Save training set data
        np.save(
            os.path.join(self.paths["output_path"], "results", "train_z0_data.npy"),
            self.train_z0,
        )
        np.save(
            os.path.join(self.paths["output_path"], "results", "train_zk_data.npy"),
            self.train_zk,
        )
        np.save(
            os.path.join(
                self.paths["data_path"],
                self.file_type,
                "tensors",
                "processed",
                f"train_gen_label_{self.config.input_level}.npy",
            ),
            self.train_gen_labels,
        )

        # Create dummy data for test set with signal
        self.n_background = self.n_samples  # Make all samples background for test
        self.n_signal = 0

        self.test_z0 = np.random.randn(self.n_samples, self.latent_dim)
        self.test_zk = np.random.randn(self.n_samples, self.latent_dim)
        self.test_gen_labels = np.random.randint(0, 3, self.n_background)
        self.test_labels = np.zeros(self.n_samples)  # All samples are background

        # Save test set data
        np.save(
            os.path.join(self.paths["output_path"], "results", "test_z0_data.npy"),
            self.test_z0,
        )
        np.save(
            os.path.join(self.paths["output_path"], "results", "test_zk_data.npy"),
            self.test_zk,
        )
        np.save(
            os.path.join(
                self.paths["data_path"],
                self.file_type,
                "tensors",
                "processed",
                f"test_gen_label_{self.config.input_level}.npy",
            ),
            self.test_gen_labels,
        )
        np.save(
            os.path.join(
                self.paths["output_path"],
                "results",
                f"test_{self.config.input_level}_label.npy",
            ),
            self.test_labels,
        )

    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.temp_dir)

    @patch("matplotlib.pyplot.savefig")
    @patch(
        "bead.src.utils.plotting.reduce_dim_subsampled",
        return_value=(np.random.randn(100, 2), "pca", np.arange(100)),
    )
    def test_plot_latent_variables(self, mock_reduce_dim, mock_savefig):
        """Test plotting latent variables."""
        # Run the function
        plot_latent_variables(self.config, self.paths, verbose=True)

        # Check that savefig was called the expected number of times
        # 2 plots (z0, zk) for each prefix (train_, test_)
        expected_calls = 2 * 2
        self.assertEqual(mock_savefig.call_count, expected_calls)

        # Check for specific filenames in the calls
        expected_filenames = [
            os.path.join(
                self.paths["output_path"], "plots", "latent_space", "train_z0.pdf"
            ),
            os.path.join(
                self.paths["output_path"], "plots", "latent_space", "train_zk.pdf"
            ),
            os.path.join(
                self.paths["output_path"], "plots", "latent_space", "test_z0.pdf"
            ),
            os.path.join(
                self.paths["output_path"], "plots", "latent_space", "test_zk.pdf"
            ),
        ]

        for filename in expected_filenames:
            mock_savefig.assert_any_call(filename, format="pdf")


class TestMuLogvarPlotting(unittest.TestCase):
    """Test mu and logvar plotting functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

        # Create directory structure
        self.paths = {
            "output_path": os.path.join(self.temp_dir, "output"),
            "data_path": os.path.join(self.temp_dir, "data"),
        }

        # Create required subdirectories
        os.makedirs(os.path.join(self.paths["output_path"], "results"))
        os.makedirs(os.path.join(self.paths["output_path"], "plots", "latent_space"))

        self.file_type = "h5"
        h5_tensor_path = os.path.join(
            self.paths["data_path"], self.file_type, "tensors", "processed"
        )
        os.makedirs(h5_tensor_path, exist_ok=True)

        # Create dummy config
        self.config = SimpleNamespace()
        self.config.input_level = "constituent"
        self.config.latent_space_size = 10
        self.config.latent_space_plot_style = "pca"  # Use PCA for speed in tests
        self.config.file_type = self.file_type
        self.config.project_name = "test_project"
        self.config.subsample_plot = False
        self.config.subsample_size = 1000

        # Generate test data
        self.n_samples = 100
        self.latent_dim = 10

        # Create dummy data for training set
        np.random.seed(42)
        self.train_mu = np.random.randn(self.n_samples, self.latent_dim)
        self.train_logvar = np.random.randn(self.n_samples, self.latent_dim)
        self.train_gen_labels = np.random.randint(0, 3, self.n_samples)

        # Save training set data
        np.save(
            os.path.join(self.paths["output_path"], "results", "train_mu_data.npy"),
            self.train_mu,
        )
        np.save(
            os.path.join(self.paths["output_path"], "results", "train_logvar_data.npy"),
            self.train_logvar,
        )
        np.save(
            os.path.join(
                self.paths["data_path"],
                self.file_type,
                "tensors",
                "processed",
                f"train_gen_label_{self.config.input_level}.npy",
            ),
            self.train_gen_labels,
        )

        # Create dummy data for test set with signal
        self.n_background = self.n_samples  # Make all samples background for test
        self.n_signal = 0

        self.test_mu = np.random.randn(self.n_samples, self.latent_dim)
        self.test_logvar = np.random.randn(self.n_samples, self.latent_dim)
        self.test_gen_labels = np.random.randint(0, 3, self.n_background)
        self.test_labels = np.zeros(self.n_samples)  # All samples are background

        # Save test set data
        np.save(
            os.path.join(self.paths["output_path"], "results", "test_mu_data.npy"),
            self.test_mu,
        )
        np.save(
            os.path.join(self.paths["output_path"], "results", "test_logvar_data.npy"),
            self.test_logvar,
        )
        np.save(
            os.path.join(
                self.paths["data_path"],
                self.file_type,
                "tensors",
                "processed",
                f"test_gen_label_{self.config.input_level}.npy",
            ),
            self.test_gen_labels,
        )
        np.save(
            os.path.join(
                self.paths["output_path"],
                "results",
                f"test_{self.config.input_level}_label.npy",
            ),
            self.test_labels,
        )

    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.temp_dir)

    @patch("matplotlib.pyplot.savefig")
    @patch(
        "bead.src.utils.plotting.reduce_dim_subsampled",
        return_value=(np.random.randn(100, 2), "pca", np.arange(100)),
    )
    def test_plot_mu_logvar(self, mock_reduce_dim, mock_savefig):
        """Test plotting mu and logvar visualization."""
        # Run the function
        plot_mu_logvar(self.config, self.paths, verbose=True)

        # Check that savefig was called the expected number of times
        # 2 plots (mu embedding, uncertainty) for each prefix (train_, test_)
        expected_calls = 2 * 2
        self.assertEqual(mock_savefig.call_count, expected_calls)

        # Check for specific filenames in the calls
        expected_filenames = [
            os.path.join(
                self.paths["output_path"],
                "plots",
                "latent_space",
                f"{self.config.project_name}_train_mu.pdf",
            ),
            os.path.join(
                self.paths["output_path"],
                "plots",
                "latent_space",
                "train_uncertainty.pdf",
            ),
            os.path.join(
                self.paths["output_path"],
                "plots",
                "latent_space",
                f"{self.config.project_name}_test_mu.pdf",
            ),
            os.path.join(
                self.paths["output_path"],
                "plots",
                "latent_space",
                "test_uncertainty.pdf",
            ),
        ]

        for filename in expected_filenames:
            mock_savefig.assert_any_call(filename)


class TestROCCurvePlotting(unittest.TestCase):
    """Test ROC curve plotting functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

        # Create directory structure
        self.paths = {
            "output_path": os.path.join(self.temp_dir, "output"),
            "data_path": os.path.join(self.temp_dir, "data"),
        }

        # Create required subdirectories
        os.makedirs(os.path.join(self.paths["output_path"], "results"))
        os.makedirs(os.path.join(self.paths["output_path"], "plots", "loss"))

        # Create dummy config
        self.config = SimpleNamespace()
        self.config.input_level = "constituent"
        self.config.project_name = "test_project"
        self.config.skip_to_roc = False  # Skip to ROC curve plotting

        # Create ground truth labels (1D array with binary values)
        self.n_samples = 100
        self.n_background = 70

        # Binary labels (0 for background, 1 for signal)
        self.labels = np.zeros(self.n_samples)
        self.labels[self.n_background :] = 1  # Signal samples

        # Create loss data for different components
        self.loss_components = ["loss", "reco", "kl"]
        for component in self.loss_components:
            # Create loss values that are lower for background, higher for signal
            loss_data = np.random.rand(self.n_samples)
            loss_data[: self.n_background] *= 0.5  # Lower scores for background
            loss_data[self.n_background :] += 0.5  # Higher scores for signal

            np.save(
                os.path.join(
                    self.paths["output_path"], "results", f"{component}_test.npy"
                ),
                loss_data,
            )

        # Save labels
        np.save(
            os.path.join(
                self.paths["output_path"],
                "results",
                f"test_{self.config.input_level}_label.npy",
            ),
            self.labels,
        )

    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.temp_dir)

    @patch("matplotlib.pyplot.savefig")
    def test_plot_roc_curve_basic(self, mock_savefig):
        """Test basic ROC curve plotting."""
        # Run the function
        plot_roc_curve(self.config, self.paths, verbose=True)

        # Check that savefig was called with the expected filename
        mock_savefig.assert_called_once_with(
            os.path.join(self.paths["output_path"], "plots", "loss", "roc.pdf")
        )

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.figure")
    def test_plot_roc_curve_overlay(self, mock_figure, mock_savefig):
        """Test ROC curve overlay functionality."""
        # Setup the overlay config
        self.config.overlay_roc = True
        self.config.skip_to_roc = True  # Skip to ROC curve plotting
        self.config.overlay_roc_projects = []  # Empty list so we don't need to mock external projects
        self.config.overlay_roc_save_location = "overlay_roc"
        self.config.overlay_roc_filename = "combined_roc.pdf"

        # Create directory for overlay ROC
        os.makedirs(
            os.path.join(self.paths["output_path"], "plots", "overlay_roc"),
            exist_ok=True,
        )

        # Run the function
        plot_roc_curve(self.config, self.paths, verbose=True)

        # Check that savefig was called with expected filenames
        expected_calls = [
            # call(os.path.join(self.paths["output_path"], "plots", "loss", "roc.pdf")),
            call(
                os.path.join(
                    self.paths["output_path"],
                    "plots",
                    "overlay_roc",
                    "combined_roc.pdf",
                )
            )
        ]
        mock_savefig.assert_has_calls(expected_calls, any_order=False)

        # Check that figure was called at least twice (once for regular ROC, once for overlay)
        # Note: The actual implementation may create multiple figures for internal purposes
        self.assertGreaterEqual(mock_figure.call_count, 2)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.xscale")
    def test_plot_roc_curve_overlay_log_scale(
        self, mock_xscale, mock_figure, mock_savefig
    ):
        """Test ROC curve overlay uses log scale."""
        # Setup the overlay config
        self.config.overlay_roc = True
        self.config.overlay_roc_projects = []  # Empty list so we don't need to mock external projects
        self.config.overlay_roc_save_location = "overlay_roc"
        self.config.overlay_roc_filename = "combined_roc.pdf"

        # Create directory for overlay ROC
        os.makedirs(
            os.path.join(self.paths["output_path"], "plots", "overlay_roc"),
            exist_ok=True,
        )

        # Run the function
        plot_roc_curve(self.config, self.paths, verbose=True)

        # Verify log scale was set for x-axis
        mock_xscale.assert_called_once_with("log")

    def test_plot_roc_curve_missing_files(self):
        """Test ROC curve plotting with missing files."""
        # Remove the labels file
        os.remove(
            os.path.join(
                self.paths["output_path"],
                "results",
                f"test_{self.config.input_level}_label.npy",
            )
        )

        # Run the function, expecting an exception
        with self.assertRaises(FileNotFoundError):
            plot_roc_curve(self.config, self.paths)


if __name__ == "__main__":
    unittest.main()
