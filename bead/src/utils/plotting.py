"""
Visualization utilities for model results.

This module provides functions for creating visualizations of model training results,
latent space embeddings, and performance metrics. These plots are essential for
understanding model behavior and evaluating anomaly detection performance.

Functions:
    plot_losses: Generate plots for training and validation losses.
    reduce_dim_subsampled: Reduce dimensionality with optional subsampling.
    plot_latent_variables: Visualize latent space embeddings.
    plot_mu_logvar: Plot latent space mean and variance.
    plot_roc_curve: Generate ROC curves from model results.
"""

import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve

# Import GPU-accelerated algorithms if available, otherwise use CPU versions
try:
    from cuml.decomposition import PCA as cuPCA
    from cuml.manifold import TSNE as cuTSNE

    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    warnings.warn(
        "cuML not installed. Using sklearn CPU implementations instead.", stacklevel=2
    )

# Import CPU implementations as fallback
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Try importing UMAP (both GPU and CPU versions)
try:
    if CUML_AVAILABLE:
        from cuml.manifold import UMAP as cuUMAP

        CUUMAP_AVAILABLE = True
    else:
        CUUMAP_AVAILABLE = False
except ImportError:
    CUUMAP_AVAILABLE = False

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Import TriMap as an alternative
try:
    import trimap

    TRIMAP_AVAILABLE = True
except ImportError:
    TRIMAP_AVAILABLE = False
    warnings.warn(
        "TriMap not installed. This dimensionality reduction method will not be available.",
        stacklevel=2,
    )

# Import utility function to check device availability
from . import helper


def plot_losses(output_dir, save_dir, config, verbose: bool = False):
    """
    Generate plots for training and validation losses over epochs.

    This function creates two types of visualizations:
    1. Training and validation total loss curves per epoch
    2. Component-wise loss curves (reconstruction, KL divergence, etc.) for train/val/test sets

    Parameters
    ----------
    output_dir : str
        Directory containing the saved model output files (.npy)
    save_dir : str
        Directory where the generated plots will be saved
    config : object
        Configuration object containing model parameters like project_name and epochs
    verbose : bool, optional
        Whether to print progress messages, default is False

    Raises
    ------
    FileNotFoundError
        If required loss data files are not found in the output directory
    """
    if verbose:
        print("Making Loss Plots...")
    # --------- Plot Train & Validation Epoch Losses ---------
    train_loss_file = os.path.join(output_dir, "train_epoch_loss_data.npy")
    val_loss_file = os.path.join(output_dir, "val_epoch_loss_data.npy")

    if os.path.exists(train_loss_file) and os.path.exists(val_loss_file):
        # Load epoch-wise loss arrays (assumed 1D arrays with one value per epoch)
        train_epoch_loss = np.load(train_loss_file)
        val_epoch_loss = np.load(val_loss_file)

        epochs = np.arange(1, len(train_epoch_loss) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_epoch_loss, label="Train Loss", marker="o")
        plt.plot(epochs, val_epoch_loss, label="Validation Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(config.project_name)
        plt.legend()
        plt.tight_layout()

        # Save plot as PDF
        plt.savefig(os.path.join(save_dir, "train_metrics.pdf"))
        plt.close()
    else:
        raise FileNotFoundError(
            "Epoch loss data files not found. Make sure to run the train mode first."
        )

    # --------- Plot Loss Components ---------
    # Define the prefixes and categories of interest
    prefixes = ["loss", "reco", "kl", "emd", "l1", "l2"]
    categories = ["train", "val", "test"]

    # Iterate over each category
    for cat in categories:
        files_for_cat = []
        # For each prefix, build the file path and check if it exists.
        for prefix in prefixes:
            file_path = os.path.join(output_dir, f"{prefix}_{cat}.npy")
            if os.path.exists(file_path):
                files_for_cat.append(file_path)

        # If any files are found for the current category, process and plot them.
        if files_for_cat:
            plt.figure(figsize=(8, 6))

            for file in files_for_cat:
                data = np.load(file)
                num_epochs = config.epochs
                total_length = len(data)

                # Determine the number of events per epoch.
                # Assumes that total_length divides evenly by config.epochs.
                events_per_epoch = total_length // num_epochs

                avg_losses = []
                for epoch in range(num_epochs):
                    start_idx = epoch * events_per_epoch
                    end_idx = start_idx + events_per_epoch
                    epoch_data = data[start_idx:end_idx]
                    avg_losses.append(np.mean(epoch_data))
                avg_losses = np.array(avg_losses)

                epochs = np.arange(1, num_epochs + 1)
                # Use the file prefix (without the _category.npy) as the label.
                base_name = os.path.basename(file)
                label = os.path.splitext(base_name)[0]

                plt.plot(epochs, avg_losses, label=label, marker="o")

            plt.xlabel("Epoch")
            plt.ylabel("Average Loss")
            plt.title(config.project_name)
            plt.legend()
            plt.tight_layout()

            # Save the plot to a file with category in its name.
            save_filename = os.path.join(save_dir, f"loss_components_{cat}.pdf")
            plt.savefig(save_filename)
            plt.close()

        else:
            print(
                f"No loss component data files found for {cat} set. Make sure to run the appropriate {cat} mode first."
            )

    if verbose:
        print("Loss plots generated successfully and saved to: ", save_dir)


def reduce_dim_subsampled(
    data, method="trimap", n_components=2, n_samples=None, verbose=False
):
    """
    Reduce dimensionality of data with optional subsampling for large datasets.

    This function applies dimensionality reduction techniques (PCA, t-SNE, TriMap, or UMAP)
    to high-dimensional data, with options for subsampling large datasets to improve
    computational efficiency. It will use GPU-accelerated methods when available.

    Parameters
    ----------
    data : numpy.ndarray
        Input data of shape (n_samples, n_features)
    method : str, optional
        Dimensionality reduction method to use: "pca", "tsne", "trimap", or "umap", default is "tsne"
    n_components : int, optional
        Number of dimensions to reduce to, default is 2
    n_samples : int, optional
        Number of samples to use (subsampling), if None uses all data, default is None
    verbose : bool, optional
        Whether to print progress messages, default is False

    Returns
    -------
    numpy.ndarray
        Reduced data with shape (n_samples, n_components)
    str
        The name of the dimensionality reduction method used
    numpy.ndarray
        Indices of the samples used for subsampling

    Raises
    ------
    ValueError
        If an invalid dimensionality reduction method is specified
    """
    # Convert to numpy array if it's a tensor
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Check device availability for GPU acceleration
    device = helper.get_device()
    use_gpu = CUML_AVAILABLE and device.type == "cuda"

    # Subsampling for large datasets
    if n_samples is not None and data.shape[0] > n_samples:
        if verbose:
            print(f"Subsampling {data.shape[0]} points to {n_samples} points...")
        indices = np.random.choice(data.shape[0], n_samples, replace=False)
        data = data[indices]
    else:
        indices = np.arange(data.shape[0])  # Use all data

    # PCA preprocessing for high-dimensional data
    if data.shape[1] > 50:
        if verbose:
            print(f"Applying PCA to reduce from {data.shape[1]} to 50 dimensions...")
        try:
            if use_gpu:
                data = cuPCA(n_components=50).fit_transform(data)
            else:
                data = PCA(n_components=50).fit_transform(data)
        except Exception as e:
            warnings.warn(
                f"Error in PCA preprocessing, using original data: {str(e)}",
                stacklevel=2,
            )

    # Apply the selected dimensionality reduction method
    method = method.lower()
    try:
        if method == "pca":
            if verbose:
                print("Reducing to 2 dimensions using PCA...")
            if use_gpu:
                reducer = cuPCA(n_components=n_components)
            else:
                reducer = PCA(n_components=n_components)
            reduced = reducer.fit_transform(data)
            method_used = "pca"

        elif method == "tsne":
            if verbose:
                print("Reducing to 2 dimensions using t-SNE...")
            if use_gpu:
                # cuML's T-SNE has different parameters
                reducer = cuTSNE(
                    n_components=n_components,
                    random_state=42,
                    learning_rate="auto",
                    init="random",
                )
            else:
                reducer = TSNE(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(data)
            method_used = "t-sne"

        elif method == "trimap" and TRIMAP_AVAILABLE:
            if verbose:
                print("Reducing to 2 dimensions using TriMap...")
            reducer = trimap.TRIMAP(n_dims=n_components)
            reduced = reducer.fit_transform(data)
            method_used = "trimap"

        elif method == "umap":
            if verbose:
                print("Reducing to 2 dimensions using UMAP...")
            if use_gpu and CUUMAP_AVAILABLE:
                reducer = cuUMAP(n_components=n_components, random_state=42)
            elif UMAP_AVAILABLE:
                reducer = umap.UMAP(n_components=n_components, random_state=42)
            else:
                raise ImportError(
                    "UMAP is not available. Install 'umap-learn' package."
                )
            reduced = reducer.fit_transform(data)
            method_used = "umap"

        else:
            # Fall back to PCA if the specified method is not available
            if method != "pca" and method != "tsne":
                warnings.warn(
                    f"Method '{method}' is not available. Falling back to t-SNE.",
                    stacklevel=2,
                )

            if use_gpu:
                reducer = cuTSNE(n_components=n_components, random_state=42)
            else:
                reducer = TSNE(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(data)
            method_used = "t-sne"

    except Exception as e:
        warnings.warn(
            f"Error using {method}, falling back to PCA: {str(e)}", stacklevel=2
        )
        if use_gpu:
            reducer = cuPCA(n_components=n_components)
        else:
            reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(data)
        method_used = "pca"

    # Convert to numpy array if it's a GPU array
    if hasattr(reduced, "to_numpy"):
        reduced = reduced.to_numpy()

    return reduced, method_used, indices


def plot_latent_variables(config, paths, verbose=False):
    """
    Visualize latent space embeddings from the model.

    This function creates 2D projections of the latent space using dimensionality
    reduction techniques (PCA, t-SNE, UMap or TriMap) for both initial (z0) and final (zk)
    latent variables, color-coded by class.

    Parameters
    ----------
    config : object
        Configuration object containing parameters like latent_space_size,
        latent_space_plot_style, input_level, etc.
    paths : dict
        Dictionary of paths including output_path and data_path
    verbose : bool, optional
        Whether to print progress and debugging information, default is False

    Notes
    -----
    The function handles both training and test data, with different color
    schemes for each. For test data, signal samples are shown in red, while
    background samples are colored according to their generator type.
    """
    prefixes = ["train_", "test_"]

    def reduce_dim(data):
        # Use our improved dimensionality reduction function
        reduced, method, _ = reduce_dim_subsampled(
            data=data,
            method=config.latent_space_plot_style,
            n_components=2,
            n_samples=None,
            verbose=verbose,
        )
        return reduced, method

    for prefix in prefixes:
        # Construct file paths
        label_path = os.path.join(
            paths["output_path"], "results", f"{prefix}{config.input_level}_label.npy"
        )
        gen_label_path = os.path.join(
            paths["data_path"],
            config.file_type,
            "tensors",
            "processed",
            f"{prefix}gen_label_{config.input_level}.npy",
        )
        z0_path = os.path.join(paths["output_path"], "results", f"{prefix}z0_data.npy")
        zk_path = os.path.join(paths["output_path"], "results", f"{prefix}zk_data.npy")

        # Check if required files exist
        required_files = [gen_label_path, z0_path, zk_path]
        if prefix == "test_":
            required_files.append(label_path)  # Only test prefix requires label file

        if not all(os.path.exists(f) for f in required_files):
            if verbose:
                print(f"Skipping {prefix} due to missing files")
            continue

        try:
            gen_labels = np.load(gen_label_path)
            z0 = np.load(z0_path)
            zk = np.load(zk_path)
            labels = np.load(label_path) if prefix == "test_" else None

            # Determine minimum length among relevant arrays
            if prefix == "train_":
                min_length = min(len(gen_labels), len(z0), len(zk))
                gen_labels = gen_labels[:min_length]
                z0 = z0[:min_length]
                zk = zk[:min_length]
            else:
                min_length = min(len(z0), len(zk), len(labels))
                z0 = z0[:min_length]
                zk = zk[:min_length]
                labels = labels[:min_length] if labels is not None else None

            if prefix == "test_":
                n_background = np.sum(labels == 0)
                if len(gen_labels) != n_background:
                    print(f"Skipping {prefix}: gen_label/label mismatch after clipping")
                    continue

        except Exception as e:
            if verbose:
                print(f"Error loading {prefix} files: {e}")
            continue

        # Define colors based on prefix
        if prefix == "train_":
            # Only Herwig, Pythia, and Sherpa classes for train prefix
            colors = []
            for i in range(len(gen_labels)):
                if gen_labels[i] == 0:
                    colors.append("green")
                elif gen_labels[i] == 1:
                    colors.append("blue")
                elif gen_labels[i] == 2:
                    colors.append("yellow")
                else:
                    colors.append("black")  # Fallback for unexpected labels
        else:
            # For test prefix, include signal (red) and background classes
            colors = []
            for i in range(n_background):
                if gen_labels[i] == 0:
                    colors.append("green")
                elif gen_labels[i] == 1:
                    colors.append("blue")
                elif gen_labels[i] == 2:
                    colors.append("yellow")
                else:
                    colors.append("black")
            colors.extend(["red"] * (len(labels) - n_background))

        # Plot latent variables
        for data, latent_suffix in [(z0, "_z0"), (zk, "_zk")]:
            if config.subsample_plot:
                colors_z = []
                reduced, method, indices = reduce_dim_subsampled(
                    data,
                    method=config.latent_space_plot_style,
                    n_samples=config.subsample_size,
                    verbose=verbose,
                )
                colors_z = [colors[i] for i in indices]
            else:
                reduced, method = reduce_dim(data)
                colors_z = colors
            plt.figure(figsize=(8, 6))
            plt.scatter(
                reduced[:, 0],
                reduced[:, 1],
                c=colors_z,
                alpha=0.7,
                edgecolors="w",
                s=60,
            )
            plt.title(
                f"{latent_suffix[1:].upper()} {method.upper()} Embedding ({prefix[:-1]})"
            )
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")

            # Create legend based on prefix
            if prefix == "train_":
                legend = [
                    mpatches.Patch(color="green", label="Herwig"),
                    mpatches.Patch(color="blue", label="Pythia"),
                    mpatches.Patch(color="yellow", label="Sherpa"),
                ]
            else:
                legend = [
                    mpatches.Patch(color="green", label="Herwig"),
                    mpatches.Patch(color="blue", label="Pythia"),
                    mpatches.Patch(color="yellow", label="Sherpa"),
                    mpatches.Patch(color="red", label="Signal"),
                ]
            plt.legend(handles=legend)

            save_path = os.path.join(
                paths["output_path"],
                "plots",
                "latent_space",
                f"{prefix[:-1]}{latent_suffix}.pdf",
            )
            plt.savefig(save_path, format="pdf")
            plt.close()


def plot_mu_logvar(config, paths, verbose=False):
    """
    Visualize the mean (mu) and log-variance (logvar) of the latent space distribution.

    This function creates two types of visualizations:
    1. 2D projection of the mean vectors in latent space, color-coded by class
    2. Histogram of uncertainties derived from the log-variance

    Parameters
    ----------
    config : object
        Configuration object containing parameters like latent_space_size,
        latent_space_plot_style, input_level, etc.
    paths : dict
        Dictionary of paths including output_path and data_path
    verbose : bool, optional
        Whether to print progress and debugging information, default is False

    Notes
    -----
    The uncertainty is calculated as the mean of the standard deviation (sigma)
    across all dimensions of the latent space, where sigma = exp(0.5 * logvar).
    """
    prefixes = ["train_", "test_"]

    def reduce_dim(data):
        # Use our improved dimensionality reduction function
        reduced, method, _ = reduce_dim_subsampled(
            data=data,
            method=config.latent_space_plot_style,
            n_components=2,
            n_samples=None,
            verbose=verbose,
        )
        return reduced, method

    for prefix in prefixes:
        mu_path = os.path.join(paths["output_path"], "results", f"{prefix}mu_data.npy")
        logvar_path = os.path.join(
            paths["output_path"], "results", f"{prefix}logvar_data.npy"
        )
        label_path = os.path.join(
            paths["output_path"], "results", f"{prefix}{config.input_level}_label.npy"
        )
        gen_label_path = os.path.join(
            paths["data_path"],
            config.file_type,
            "tensors",
            "processed",
            f"{prefix}gen_label_{config.input_level}.npy",
        )

        # Check if required files exist
        required_files = [mu_path, logvar_path, gen_label_path]
        if prefix == "test_":
            required_files.append(label_path)  # Only test prefix requires label file

        if not all(os.path.exists(f) for f in required_files):
            if verbose:
                print(f"Skipping {prefix} due to missing files")
            continue

        try:
            mu = np.load(mu_path)
            logvar = np.load(logvar_path)
            gen_labels = np.load(gen_label_path)
            labels = np.load(label_path) if prefix == "test_" else None

            # Determine minimum length among relevant arrays
            if prefix == "train_":
                min_length = min(len(gen_labels), len(mu), len(logvar))
                gen_labels = gen_labels[:min_length]
                mu = mu[:min_length]
                logvar = logvar[:min_length]
            else:
                min_length = min(len(mu), len(logvar), len(labels))
                mu = mu[:min_length]
                logvar = logvar[:min_length]
                labels = labels[:min_length] if labels is not None else None

            if prefix == "test_":
                n_background = np.sum(labels == 0)
                if len(gen_labels) != n_background:
                    print(f"Skipping {prefix}: gen_label/label mismatch after clipping")
                    continue

        except Exception as e:
            if verbose:
                print(f"Error loading {prefix} files: {e}")
            continue

        # Define colors based on prefix
        if prefix == "train_":
            # Only Herwig, Pythia, and Sherpa classes for train prefix
            colors = []
            for i in range(len(gen_labels)):
                if gen_labels[i] == 0:
                    colors.append("green")
                elif gen_labels[i] == 1:
                    colors.append("blue")
                elif gen_labels[i] == 2:
                    colors.append("yellow")
                else:
                    colors.append("black")
        else:
            # For test prefix, include signal (red) and background classes
            colors = []
            for i in range(n_background):
                if gen_labels[i] == 0:
                    colors.append("green")
                elif gen_labels[i] == 1:
                    colors.append("blue")
                elif gen_labels[i] == 2:
                    colors.append("yellow")
                else:
                    colors.append("black")
            colors.extend(["red"] * (len(labels) - n_background))

        # Plot latent means (mu)
        if config.subsample_plot:
            colors_z = []
            reduced_mu, method, indices = reduce_dim_subsampled(
                mu,
                method=config.latent_space_plot_style,
                n_samples=config.subsample_size,
                verbose=verbose,
            )
            colors_z = [colors[i] for i in indices]
        else:
            reduced_mu, method = reduce_dim(mu)
            colors_z = colors
        plt.figure(figsize=(8, 6))
        plt.scatter(
            reduced_mu[:, 0],
            reduced_mu[:, 1],
            c=colors_z,
            alpha=0.7,
            edgecolors="w",
            s=60,
        )
        plt.title(f"Mu {method.upper()} Embedding ({prefix[:-1]})")
        # Create legend based on prefix
        if prefix == "train_":
            legend = [
                mpatches.Patch(color="green", label="Herwig"),
                mpatches.Patch(color="blue", label="Pythia"),
                mpatches.Patch(color="yellow", label="Sherpa"),
            ]
        else:
            legend = [
                mpatches.Patch(color="green", label="Herwig"),
                mpatches.Patch(color="blue", label="Pythia"),
                mpatches.Patch(color="yellow", label="Sherpa"),
                mpatches.Patch(color="red", label="Signal"),
            ]
        plt.legend(handles=legend)
        plt.savefig(
            os.path.join(
                paths["output_path"],
                "plots",
                "latent_space",
                f"{config.project_name}_{prefix[:-1]}_mu.pdf",
            )
        )
        plt.close()

        # Plot uncertainty
        sigma = np.exp(0.5 * logvar)
        uncertainty = np.mean(sigma, axis=1)
        if prefix == "test_":
            background_uncertainty = uncertainty[:n_background]
            signal_uncertainty = uncertainty[n_background:]
        else:
            background_uncertainty = uncertainty
            signal_uncertainty = []
        plt.figure(figsize=(8, 6))
        for color, values in zip(
            ["green", "blue", "yellow", "red"],
            [
                background_uncertainty[gen_labels == 0],
                background_uncertainty[gen_labels == 1],
                background_uncertainty[gen_labels == 2],
                signal_uncertainty,
            ],
            strict=False,
        ):
            if len(values) > 0:
                plt.hist(values, bins=30, alpha=0.6, color=color)
        plt.title(f"Uncertainty Distribution ({prefix[:-1]})")
        plt.savefig(
            os.path.join(
                paths["output_path"],
                "plots",
                "latent_space",
                f"{prefix[:-1]}_uncertainty.pdf",
            )
        )
        plt.close()


def plot_roc_curve(config, paths, verbose: bool = False):
    """
    Generate and save ROC curves for available loss component files.

    This function computes and plots Receiver Operating Characteristic (ROC) curves
    for different loss components, evaluating their effectiveness as anomaly scores.

    If config.overlay_roc is True, it also generates an overlay plot comparing ROC curves
    across different projects specified in config.overlay_roc_projects.

    Parameters
    ----------
    config : object
        Configuration object containing parameters like input_level and project_name
    paths : dict
        Dictionary containing paths, particularly output_path
    verbose : bool, optional
        Whether to print additional debug information, default is False

    Raises
    ------
    FileNotFoundError
        If the required label file is not found
    ValueError
        If ground truth labels are not a 1D array or if there's a length mismatch
        between loss scores and ground truth labels

    Notes
    -----
    The function generates a single plot containing ROC curves for all available
    loss components (total loss, reconstruction loss, KL divergence, etc.),
    with the area under the curve (AUC) displayed in the legend.

    If overlay_roc is enabled, it also creates a combined plot showing ROC curves
    from multiple projects for comparison.
    """
    # Store ROC data for potential overlay with current project
    current_project_roc = {}

    if not config.overlay_roc:
        # Load ground truth binary labels from 'label.npy'
        label_path = os.path.join(
            paths["output_path"], "results", "test_" + config.input_level + "_label.npy"
        )
        output_dir = os.path.join(paths["output_path"], "results")
        # check if the label file exists
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        else:
            ground_truth = np.load(label_path)

        # Ensure ground_truth is a 1D array
        if ground_truth.ndim != 1:
            raise ValueError("Ground truth labels must be a 1D array.")

        # Define the loss component prefixes to search for.
        loss_components = ["loss", "reco", "kl", "emd", "l1", "l2"]

        # Iterate over each loss component and generate ROC curve.
        plt.figure(figsize=(8, 6))

        for component in loss_components:
            file_path = os.path.join(output_dir, f"{component}_test.npy")
            if not os.path.exists(file_path):
                continue  # Skip if the file does not exist

            # Load loss scores
            data = np.load(file_path)

            # Ensure that data is a 1D array (flatten if necessary).
            if data.ndim > 1:
                data = data.flatten()

            # Check if the length of data matches the length of ground_truth
            if len(data) != len(ground_truth):
                raise ValueError(
                    f"Length mismatch: {file_path} has {len(data)} entries; "
                    f"ground truth has {len(ground_truth)} entries."
                )

            # Compute ROC curve and AUC.
            fpr, tpr, thresholds = roc_curve(ground_truth, data)
            roc_auc = auc(fpr, tpr)

            # Store loss component ROC data for overlay
            if component == "loss":
                current_project_roc = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": roc_auc,
                    "name": config.project_name,
                }

            # Plot the ROC curve.
            plt.plot(
                fpr, tpr, label=f"{component.capitalize()} AUC = {roc_auc:.2f}", lw=2
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {config.project_name}")
        plt.legend(loc="best")
        plt.tight_layout()

        # Save the plot as a PDF file.
        save_filename = os.path.join(paths["output_path"], "plots", "loss", "roc.pdf")
        plt.savefig(save_filename)
        plt.close()

        # Generate per-signal ROC plots if enabled
        if hasattr(config, "plot_roc_per_signal") and config.plot_roc_per_signal:
            if verbose:
                print("Generating per-signal ROC plots...")

            # Get signal file information from CSV files
            try:
                from . import helper

                csv_path = os.path.join(paths["data_path"], "csv")
                bkg_count = helper.get_bkg_test_count_from_csv(csv_path)

                # Check actual data length first
                output_dir = os.path.join(paths["output_path"], "results")
                loss_file_path = os.path.join(output_dir, "loss_test.npy")

                if os.path.exists(loss_file_path):
                    actual_data_length = len(np.load(loss_file_path))

                    # Use corrected signal file info that accounts for actual data length
                    signal_file_info = helper.get_signal_file_info_corrected(
                        csv_path, actual_data_length, bkg_count, "sig_test"
                    )

                    if verbose:
                        print(f"Actual data length: {actual_data_length}")
                        print(f"Background events: {bkg_count}")
                        print(
                            f"Available signal events: {actual_data_length - bkg_count}"
                        )
                else:
                    # Fallback to original method if no data file exists
                    signal_file_info = helper.get_signal_file_info_from_csv(
                        csv_path, "sig_test"
                    )

                if verbose:
                    print(
                        f"Found {len(signal_file_info)} signal files for per-signal ROC plotting"
                    )
                    print(f"Background test event count: {bkg_count}")
                    for info in signal_file_info:
                        print(
                            f"  {info['sig_filename']}: {info['events_count']} events (indices {bkg_count + info['start_idx']}:{bkg_count + info['end_idx']})"
                        )

                # Define the loss component prefixes to search for
                loss_components = ["loss", "reco", "kl", "emd", "l1", "l2"]

                # Generate ROC plot for each signal file
                for signal_info in signal_file_info:
                    sig_filename = signal_info["sig_filename"]
                    sig_start_idx = bkg_count + signal_info["start_idx"]
                    sig_end_idx = bkg_count + signal_info["end_idx"]

                    if verbose:
                        print(
                            f"Processing signal file: {sig_filename} (indices {sig_start_idx}:{sig_end_idx})"
                        )

                    # Create ground truth labels for this specific signal vs all backgrounds
                    # Background events are labeled as 0, this specific signal events as 1
                    per_signal_ground_truth = np.zeros(
                        bkg_count + signal_info["events_count"]
                    )
                    per_signal_ground_truth[bkg_count:] = 1

                    plt.figure(figsize=(8, 6))

                    for component in loss_components:
                        file_path = os.path.join(output_dir, f"{component}_test.npy")
                        if not os.path.exists(file_path):
                            continue

                        # Load loss scores and extract data for this signal vs backgrounds
                        full_data = np.load(file_path)
                        if full_data.ndim > 1:
                            full_data = full_data.flatten()

                        # Validate array bounds before extraction
                        if sig_end_idx > len(full_data):
                            if verbose:
                                print(
                                    f"Warning: Signal indices {sig_start_idx}:{sig_end_idx} exceed data length {len(full_data)} for {component} and {sig_filename}, skipping"
                                )
                            continue

                        # Extract background losses and this specific signal's losses
                        bkg_losses = full_data[:bkg_count]
                        signal_losses = full_data[sig_start_idx:sig_end_idx]
                        per_signal_data = np.concatenate([bkg_losses, signal_losses])

                        # Check length consistency
                        if len(per_signal_data) != len(per_signal_ground_truth):
                            if verbose:
                                print(
                                    f"Warning: Length mismatch for {component} and {sig_filename}, skipping"
                                )
                                print(
                                    f"  Data length: {len(per_signal_data)}, Ground truth length: {len(per_signal_ground_truth)}"
                                )
                                print(
                                    f"  Full data length: {len(full_data)}, Background count: {bkg_count}"
                                )
                                print(
                                    f"  Signal indices: {sig_start_idx}:{sig_end_idx}, Signal events: {sig_end_idx - sig_start_idx}"
                                )
                            continue

                        # Compute ROC curve and AUC
                        fpr, tpr, thresholds = roc_curve(
                            per_signal_ground_truth, per_signal_data
                        )
                        roc_auc = auc(fpr, tpr)

                        # Output AUC and TPR at specific FPR thresholds for verbose mode
                        if (
                            verbose and component == "loss"
                        ):  # Only output for main loss component to avoid spam
                            print(
                                f"  {sig_filename} - {component.upper()} AUC: {roc_auc:.4f}"
                            )
                            target_fpr = [1e-4, 1e-3, 1e-2]
                            for threshold in target_fpr:
                                if threshold <= fpr.max() and threshold >= fpr.min():
                                    # Interpolate to find TPR at the desired FPR
                                    tpr_at_fpr = np.interp(threshold, fpr, tpr)
                                    print(
                                        f"    TPR at FPR {threshold:.1e}: {tpr_at_fpr:.4f}"
                                    )
                                else:
                                    print(
                                        f"    TPR at FPR {threshold:.1e}: N/A (FPR out of range)"
                                    )

                        # Plot the ROC curve
                        plt.plot(
                            fpr,
                            tpr,
                            label=f"{component.capitalize()} AUC = {roc_auc:.2f}",
                            lw=2,
                        )

                    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Guess")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title(f"ROC Curve - {config.project_name} - {sig_filename}")
                    plt.legend(loc="best")
                    plt.tight_layout()

                    # Save the per-signal ROC plot
                    per_signal_filename = os.path.join(
                        paths["output_path"], "plots", "loss", f"roc_{sig_filename}.pdf"
                    )
                    plt.savefig(per_signal_filename)
                    plt.close()

                    if verbose:
                        print(f"Saved per-signal ROC plot: {per_signal_filename}")

            except Exception as e:
                print(f"Error generating per-signal ROC plots: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()

    # Create ROC overlay if enabled
    if hasattr(config, "overlay_roc") and config.overlay_roc:
        if verbose:
            print("Generating ROC overlay plot...")

        # Create a new figure for the overlay plot with log scale
        plt.figure(figsize=(10, 8))

        # Plot current project's ROC curve first
        if current_project_roc:
            plt.plot(
                current_project_roc["fpr"],
                current_project_roc["tpr"],
                label=f"{current_project_roc['name']} (AUC = {current_project_roc['auc']:.3f})",
                lw=2,
            )

        # Process each project in the overlay list
        for project_path in config.overlay_roc_projects:
            try:
                # Parse workspace_name/project_name format
                workspace_name, project_name = project_path.split("/")

                # Construct paths to the other project's files
                other_project_results = os.path.join(
                    "bead/workspaces", workspace_name, project_name, "output", "results"
                )

                # Load label file from the other project
                other_label_path = os.path.join(
                    other_project_results, f"test_{config.input_level}_label.npy"
                )

                # Skip if the label file doesn't exist
                if not os.path.exists(other_label_path):
                    if verbose:
                        print(f"Skipping {project_path}: label file not found")
                    continue

                other_ground_truth = np.load(other_label_path)

                # Load loss component file (using only the main "loss" component for overlay)
                other_loss_path = os.path.join(other_project_results, "loss_test.npy")
                if not os.path.exists(other_loss_path):
                    if verbose:
                        print(f"Skipping {project_path}: loss file not found")
                    continue

                other_loss_data = np.load(other_loss_path)

                # Ensure data is 1D
                if other_loss_data.ndim > 1:
                    other_loss_data = other_loss_data.flatten()

                # Check if lengths match
                if len(other_loss_data) != len(other_ground_truth):
                    if verbose:
                        print(f"Skipping {project_path}: length mismatch")
                    continue

                # Calculate ROC curve and AUC
                fpr, tpr, _ = roc_curve(other_ground_truth, other_loss_data)
                roc_auc = auc(fpr, tpr)

                # Print out TPR at specific FPR thresholds using numpy interpolation
                target_fpr = [1e-4, 1e-3, 1e-2, 1e-1]
                for threshold in target_fpr:
                    if threshold in fpr:
                        tpr_at_fpr = tpr[np.where(fpr == threshold)[0][0]]
                    else:
                        # Interpolate to find TPR at the desired FPR
                        tpr_at_fpr = np.interp(threshold, fpr, tpr)
                    if verbose:
                        print(
                            f"TPR for {project_name} at FPR {threshold:.1e}: {tpr_at_fpr:.3f}"
                        )

                # Plot the ROC curve
                plt.plot(fpr, tpr, label=f"{project_name} (AUC = {roc_auc:.3f})", lw=2)

                if verbose:
                    print(f"Added {project_path} to overlay plot (AUC = {roc_auc:.3f})")

            except Exception as e:
                if verbose:
                    print(f"Error processing {project_path}: {str(e)}")

        # Add the random guess line
        plt.plot([0, 0], [1, 1], "k--", lw=2, label="Random Guess")

        # Set x-axis to log scale and zoom to specified region
        plt.xscale("log")
        plt.xlim(1e-4, 1e-1)  # Set x-axis range to show 1E-4, 1E-2 and 1E-1
        plt.ylim(0, 0.42)

        # Add labels and title
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.tight_layout()

        # Create directory for the overlay plot if it doesn't exist
        overlay_dir = os.path.join(
            paths["output_path"], "plots", config.overlay_roc_save_location
        )
        os.makedirs(overlay_dir, exist_ok=True)

        # Save the overlay plot
        overlay_filename = os.path.join(overlay_dir, config.overlay_roc_filename)
        plt.savefig(overlay_filename)

        if verbose:
            print(f"ROC overlay plot saved to {overlay_filename}")

        plt.close()


def plot_test_loss_histogram(config, paths, loss_component="loss_test", verbose=False):
    """
    Generate histogram plots for test loss components with generator and signal labels.

    This function creates TWO histogram visualizations of test loss values:
    1. Standard histogram plot with signal histogram pushed to the background
    2. Line plot version with background histograms and signal as a red line on top

    Both plots are color-coded by generator type (Herwig, Pythia, Sherpa) and
    signal vs background classification, using the same color scheme as the latent space plots.

    Parameters
    ----------
    config : object
        Configuration object containing parameters like input_level and project_name
    paths : dict
        Dictionary containing paths, particularly output_path and data_path
    loss_component : str, optional
        Name of the loss component file to plot (without .npy extension), default is "loss_test"
    verbose : bool, optional
        Whether to print progress and debugging information, default is False

    Notes
    -----
    The function uses the same color scheme as latent space plots:
    - Green: Herwig background
    - Blue: Pythia background
    - Yellow: Sherpa background
    - Red: Signal samples

    Generates two output files:
    - {loss_component}_histogram.pdf: Standard overlapping histograms (signal in back)
    - {loss_component}_histogram_line.pdf: Background histograms + signal as line plot
    """
    if verbose:
        print(f"Creating test loss histogram for {loss_component}...")

    try:
        # Load test loss data
        loss_file_path = os.path.join(
            paths["output_path"], "results", f"{loss_component}.npy"
        )
        if not os.path.exists(loss_file_path):
            if verbose:
                print(f"Loss file {loss_file_path} not found, skipping histogram plot")
            return

        loss_data = np.load(loss_file_path)

        # Load test labels (signal vs background)
        label_path = os.path.join(
            paths["output_path"], "results", f"test_{config.input_level}_label.npy"
        )
        if not os.path.exists(label_path):
            if verbose:
                print(f"Label file {label_path} not found, skipping histogram plot")
            return

        labels = np.load(label_path)

        # Load generator labels for background samples
        gen_label_path = os.path.join(
            paths["data_path"],
            config.file_type,
            "tensors",
            "processed",
            f"test_gen_label_{config.input_level}.npy",
        )
        if not os.path.exists(gen_label_path):
            if verbose:
                print(
                    f"Generator label file {gen_label_path} not found, skipping histogram plot"
                )
            return

        gen_labels = np.load(gen_label_path)

        # Ensure data consistency
        n_background = np.sum(labels == 0)
        _n_signal = np.sum(labels == 1)

        if len(gen_labels) != n_background:
            if verbose:
                print(
                    f"Generator labels length ({len(gen_labels)}) doesn't match background samples ({n_background})"
                )
            return

        if len(loss_data) != len(labels):
            if verbose:
                print(
                    f"Loss data length ({len(loss_data)}) doesn't match label length ({len(labels)})"
                )
            return

        # Flatten loss data if necessary
        if loss_data.ndim > 1:
            loss_data = loss_data.flatten()

        # Separate background and signal loss values
        background_mask = labels == 0
        signal_mask = labels == 1

        background_loss = loss_data[background_mask]
        signal_loss = loss_data[signal_mask]

        # Separate background by generator type
        herwig_mask = gen_labels == 0
        pythia_mask = gen_labels == 1
        sherpa_mask = gen_labels == 2

        herwig_loss = background_loss[herwig_mask]
        pythia_loss = background_loss[pythia_mask]
        sherpa_loss = background_loss[sherpa_mask]

        # Create histogram plot
        plt.figure(figsize=(10, 6))

        # Determine histogram range and bins
        all_loss_values = np.concatenate(
            [herwig_loss, pythia_loss, sherpa_loss, signal_loss]
        )
        hist_range = (
            np.percentile(all_loss_values, 1),
            np.percentile(all_loss_values, 99),
        )
        bins = 50

        # === PLOT 1: Standard histogram with signal pushed to back ===
        plt.figure(figsize=(10, 6))

        # Plot histograms with same colors as latent space plots
        alpha = 0.7

        # Plot signal FIRST so it appears in the back
        if len(signal_loss) > 0:
            plt.hist(
                signal_loss,
                bins=bins,
                range=hist_range,
                alpha=alpha,
                color="red",
                label=f"Signal (n={len(signal_loss)})",
                density=True,
            )

        # Then plot background histograms on top
        if len(herwig_loss) > 0:
            plt.hist(
                herwig_loss,
                bins=bins,
                range=hist_range,
                alpha=alpha,
                color="green",
                label=f"Herwig (n={len(herwig_loss)})",
                density=True,
            )

        if len(pythia_loss) > 0:
            plt.hist(
                pythia_loss,
                bins=bins,
                range=hist_range,
                alpha=alpha,
                color="blue",
                label=f"Pythia (n={len(pythia_loss)})",
                density=True,
            )

        if len(sherpa_loss) > 0:
            plt.hist(
                sherpa_loss,
                bins=bins,
                range=hist_range,
                alpha=alpha,
                color="yellow",
                label=f"Sherpa (n={len(sherpa_loss)})",
                density=True,
            )

        plt.xlabel(f"{loss_component.replace('_test', '').title()} Loss")
        plt.ylabel("Density")
        plt.title(
            f"{config.project_name} - {loss_component.replace('_test', '').title()} Loss Distribution"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the standard plot
        save_path = os.path.join(
            paths["output_path"], "plots", "loss", f"{loss_component}_histogram.pdf"
        )
        plt.savefig(save_path, format="pdf")
        plt.close()

        if verbose:
            print(f"Test loss histogram saved to: {save_path}")

        # === PLOT 2: Background histograms + signal as line plot ===
        plt.figure(figsize=(10, 6))

        # Plot background histograms
        if len(herwig_loss) > 0:
            plt.hist(
                herwig_loss,
                bins=bins,
                range=hist_range,
                alpha=alpha,
                color="green",
                label=f"Herwig (n={len(herwig_loss)})",
                density=True,
            )

        if len(pythia_loss) > 0:
            plt.hist(
                pythia_loss,
                bins=bins,
                range=hist_range,
                alpha=alpha,
                color="blue",
                label=f"Pythia (n={len(pythia_loss)})",
                density=True,
            )

        if len(sherpa_loss) > 0:
            plt.hist(
                sherpa_loss,
                bins=bins,
                range=hist_range,
                alpha=alpha,
                color="yellow",
                label=f"Sherpa (n={len(sherpa_loss)})",
                density=True,
            )

        # Plot signal as a line on top
        if len(signal_loss) > 0:
            # Calculate histogram for signal to get bin centers and heights
            signal_counts, bin_edges = np.histogram(
                signal_loss, bins=bins, range=hist_range, density=True
            )
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Plot as a line
            plt.plot(
                bin_centers,
                signal_counts,
                color="red",
                linewidth=2.5,
                label=f"Signal (n={len(signal_loss)})",
                marker="o",
                markersize=3,
                alpha=0.9,
            )

        plt.xlabel(f"{loss_component.replace('_test', '').title()} Loss")
        plt.ylabel("Density")
        plt.title(
            f"{config.project_name} - {loss_component.replace('_test', '').title()} Loss Distribution (Line)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the line plot version
        save_path_line = os.path.join(
            paths["output_path"],
            "plots",
            "loss",
            f"{loss_component}_histogram_line.pdf",
        )
        plt.savefig(save_path_line, format="pdf")
        plt.close()

        if verbose:
            print(f"Test loss histogram (line version) saved to: {save_path_line}")

    except Exception as e:
        if verbose:
            print(f"Error creating test loss histogram: {e}")
