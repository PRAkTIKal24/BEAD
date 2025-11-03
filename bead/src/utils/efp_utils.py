"""
Energy-Flow Polynomial (EFP) utilities for BEAD.

This module provides helper functions for EFP feature generation, including
configuration validation, preprocessing, and computation utilities.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

try:
    import energyflow as ef

    ENERGYFLOW_AVAILABLE = True
except ImportError:
    ENERGYFLOW_AVAILABLE = False
    ef = None

logger = logging.getLogger(__name__)


def validate_efp_config(config) -> Dict[str, Union[int, str, bool]]:
    """
    Validate and adjust EFP configuration parameters based on investigation results.

    Args:
        config: Configuration object with EFP parameters

    Returns:
        Dict with validated EFP parameters and metadata

    Raises:
        ImportError: If energyflow is not available
        ValueError: If configuration parameters are invalid
    """
    if not ENERGYFLOW_AVAILABLE:
        raise ImportError(
            "energyflow package is required for EFP computation. "
            "Install with: pip install energyflow"
        )

    # Get base parameters
    nmax = getattr(config, "efp_nmax", 5)
    dmax = getattr(config, "efp_dmax", 6)
    extended_mode = getattr(config, "efp_extended_mode", False)

    # Apply extended mode if requested
    if extended_mode:
        nmax = 6
        dmax = 7
        logger.info("EFP extended mode enabled: using n≤6, d≤7 for ~531 EFPs")
    else:
        logger.info("EFP standard mode: using n≤5, d≤6 for ~140 EFPs")

    # Validate parameters
    if nmax < 2 or nmax > 8:
        raise ValueError(f"efp_nmax must be between 2 and 8, got {nmax}")
    if dmax < 2 or dmax > 10:
        raise ValueError(f"efp_dmax must be between 2 and 10, got {dmax}")

    # Validate other parameters
    beta = getattr(config, "efp_beta", 1.0)
    if beta <= 0:
        raise ValueError(f"efp_beta must be positive, got {beta}")

    measure = getattr(config, "efp_measure", "hadr")
    if measure not in ["hadr", "ee", "gen"]:
        raise ValueError(f"efp_measure must be 'hadr', 'ee', or 'gen', got {measure}")

    n_jobs = getattr(config, "efp_n_jobs", 4)
    if n_jobs < 1:
        n_jobs = 1
    elif n_jobs > 16:
        logger.warning(
            f"efp_n_jobs={n_jobs} is very high, consider reducing for memory efficiency"
        )

    # Create EFPSet to get actual feature count
    try:
        efpset = ef.EFPSet(
            f"n<={nmax}",
            f"d<={dmax}",
            measure=measure,
            beta=beta,
            normed=getattr(config, "efp_normed", True),
            coords="ptyphim",
            check_input=False,
        )
        # Use compute output length as feature count (not len(efps))
        n_graphs = len(efpset.graphs())
        n_efps = n_graphs  # compute() returns one value per graph
        logger.info(
            f"EFPSet created successfully with {n_graphs} graphs → {n_efps} EFP features"
        )

    except Exception as e:
        raise ValueError(f"Failed to create EFPSet with given parameters: {e}") from e

    # Return validated configuration
    validated_config = {
        "nmax": nmax,
        "dmax": dmax,
        "n_efps": n_efps,
        "beta": beta,
        "measure": measure,
        "normed": getattr(config, "efp_normed", True),
        "n_jobs": n_jobs,
        "extended_mode": extended_mode,
        "cache_dir": getattr(config, "efp_cache_dir", None),
        "standardize": getattr(config, "efp_standardize_meanvar", True),
        "feature_prefix": getattr(config, "efp_feature_prefix", "EFP_"),
        "eps": getattr(config, "efp_eps", 1e-12),
    }

    return validated_config


def create_efpset(efp_config: Dict) -> "ef.EFPSet":
    """
    Create an EFPSet object from validated configuration.

    Args:
        efp_config: Validated EFP configuration dictionary

    Returns:
        EnergyFlow EFPSet object
    """
    if not ENERGYFLOW_AVAILABLE:
        raise ImportError("energyflow package is required")

    efpset = ef.EFPSet(
        f"n<={efp_config['nmax']}",
        f"d<={efp_config['dmax']}",
        measure=efp_config["measure"],
        beta=efp_config["beta"],
        normed=efp_config["normed"],
        coords="ptyphim",  # Explicitly use [pT, y, phi, m] coordinates
        check_input=True,  # Enable input validation to catch format errors
    )

    logger.debug(f"Created EFPSet with {len(efpset.efps)} EFPs")
    return efpset


def preprocess_jet_constituents(
    constituents: np.ndarray, mask_threshold: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess jet constituents for EFP computation with proper [pT, y, phi, m] format.

    Filters out zero-padded constituents and handles edge cases.

    Args:
        constituents: Array of shape (n_particles, 4) with [pT, eta, phi, mass]
        mask_threshold: Minimum pT threshold for valid particles

    Returns:
        Tuple of (filtered_constituents, mask) where mask indicates valid particles
    """
    import energyflow as ef  # Import here to avoid dependency issues

    # Extract features
    pt = constituents[:, 0]  # pT
    eta = constituents[:, 1]  # eta
    phi = constituents[:, 2]  # phi
    mass = constituents[:, 3]  # mass (from PID column)

    # Create mask for valid particles (pT > threshold)
    mask = pt > mask_threshold

    # Calculate proper rapidity using EnergyFlow utility
    rapidity = ef.ys_from_pts_etas_ms(pt, eta, mass)

    # Create properly formatted constituents: [pT, y, phi, m]
    proper_constituents = np.stack([pt, rapidity, phi, mass], axis=-1)

    # Filter constituents
    filtered_constituents = proper_constituents[mask]

    # Handle empty jets
    if len(filtered_constituents) == 0:
        logger.debug("Empty jet detected (no constituents with pT > threshold)")
        # Return single dummy particle to avoid EFP computation errors
        filtered_constituents = np.array([[mask_threshold, 0.0, 0.0, 0.0]])
        mask = np.array([False])  # Mark as invalid

    return filtered_constituents, mask


def compute_efps_for_jet(
    jet_constituents: np.ndarray, efpset: "ef.EFPSet", handle_empty: bool = True
) -> np.ndarray:
    """
    Compute EFPs for a single jet.

    Args:
        jet_constituents: Array of shape (n_particles, 3) with (pT, eta, phi)
        efpset: EnergyFlow EFPSet object
        handle_empty: Whether to handle empty jets gracefully

    Returns:
        Array of EFP values for the jet
    """
    try:
        # Preprocess constituents
        filtered_constituents, mask = preprocess_jet_constituents(jet_constituents)

        # Compute EFPs
        if handle_empty and not np.any(mask):
            # Empty jet - return zeros (use graph count for feature size)
            efp_values = np.zeros(len(efpset.graphs()))
        else:
            efp_values = efpset.compute(filtered_constituents)

        return efp_values.astype(np.float32)

    except Exception as e:
        logger.warning(f"EFP computation failed for jet: {e}")
        # Return zeros on failure (use graph count for feature size)
        return np.zeros(len(efpset.graphs()), dtype=np.float32)


def compute_efps_batch(
    jets_constituents: np.ndarray, efpset: "ef.EFPSet", n_jobs: int = 4
) -> np.ndarray:
    """
    Compute EFPs for a batch of jets using multiprocessing.

    Args:
        jets_constituents: Array of shape (n_jets, n_particles, 3)
        efpset: EnergyFlow EFPSet object
        n_jobs: Number of parallel workers

    Returns:
        Array of shape (n_jets, n_efps) with EFP values
    """
    n_jets = jets_constituents.shape[0]
    n_efps = len(efpset.graphs())  # Use graph count for feature size

    # Prepare jets list for batch_compute
    jets_to_process = []
    valid_jet_indices = []

    for i, jet in enumerate(jets_constituents):
        filtered_constituents, mask = preprocess_jet_constituents(jet)
        if np.any(mask):  # Has valid particles (to avoid computing EFPs for empty jets)
            jets_to_process.append(filtered_constituents)
            valid_jet_indices.append(i)

    # Initialize output array
    efp_results = np.zeros((n_jets, n_efps), dtype=np.float32)

    if jets_to_process:
        try:
            # Compute EFPs for valid jets
            start_time = time.time()
            batch_results = efpset.batch_compute(jets_to_process, n_jobs=n_jobs)
            computation_time = time.time() - start_time

            logger.debug(
                f"Computed EFPs for {len(jets_to_process)}/{n_jets} jets in {computation_time:.3f}s"
            )

            # Fill results for valid jets
            for idx, jet_idx in enumerate(valid_jet_indices):
                efp_results[jet_idx] = batch_results[idx].astype(np.float32)

        except Exception as e:
            logger.error(f"Batch EFP computation failed: {e}")
            # Fall back to individual computation
            for i, jet in enumerate(jets_constituents):
                efp_results[i] = compute_efps_for_jet(jet, efpset, handle_empty=True)

    return efp_results


def standardize_efps(
    efp_data: np.ndarray,
    compute_stats: bool = True,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, Dict]:
    """
    Standardize EFP features to zero mean and unit variance.

    Args:
        efp_data: Array of shape (n_samples, n_efps)
        compute_stats: Whether to compute mean/std from data
        mean: Pre-computed mean (if not computing from data)
        std: Pre-computed std (if not computing from data)
        eps: Small value to avoid division by zero

    Returns:
        Tuple of (standardized_data, stats_dict)
    """
    if compute_stats:
        mean = np.mean(efp_data, axis=0)
        std = np.std(efp_data, axis=0)

    # Avoid division by zero
    std_safe = np.where(std < eps, eps, std)

    # Standardize
    standardized_data = (efp_data - mean) / std_safe

    stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "eps": eps}

    return standardized_data.astype(np.float32), stats


def get_efp_cache_path(
    cache_dir: Union[str, Path], efp_config: Dict, prefix: str = ""
) -> Path:
    """
    Generate cache file path for EFP features.

    Args:
        cache_dir: Base cache directory
        efp_config: EFP configuration dictionary
        prefix: Optional prefix for filename

    Returns:
        Path object for cache file
    """
    if cache_dir is None:
        raise ValueError("cache_dir cannot be None")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with configuration parameters
    filename_parts = []
    if prefix:
        filename_parts.append(prefix)

    filename_parts.extend(
        [
            f"n{efp_config['nmax']}",
            f"d{efp_config['dmax']}",
            f"b{efp_config['beta']:.1f}",
            f"{efp_config['measure']}",
        ]
    )

    if efp_config.get("extended_mode", False):
        filename_parts.append("ext")

    filename = "_".join(filename_parts) + "_efp.pt"
    return cache_dir / filename
