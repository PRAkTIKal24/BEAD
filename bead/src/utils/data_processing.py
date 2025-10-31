"""
Data processing utilities for HDF5/NumPy arrays.

This module provides functions for loading, processing, and preparing data for training
and inference. It includes utilities for selecting the top jets and constituents, normalizing
data, and converting between different data formats.

Functions:
    load_data: Load data from HDF5 files.
    select_top_jets_and_constituents: Select top N jets and M constituents.
    process_and_save_tensors: Process input file and save as PyTorch tensors.
    preproc_inputs: Preprocess inputs for training or inference.
"""

import os
import pickle
import sys

import h5py
import numpy as np
import torch

from . import helper, normalization
from .efp_utils import (
    compute_efps_batch,
    create_efpset,
    standardize_efps,
    validate_efp_config,
)


def load_data(file_path, file_type="h5", verbose: bool = False):
    """
    Load data from either an HDF5 file or .npy files.
    """

    if verbose:
        print(f"Loading data from {file_path}...")
    if file_type == "h5":
        with h5py.File(file_path, "r") as h5file:
            events = h5file["events"][:]
            jets = h5file["jets"][:]
            constituents = h5file["constituents"][:]
    elif file_type == "npy":
        raise NotImplementedError(
            "Loading .npy files is not yet supported. Please retry with --mode = convert_csv and --options = h5 first."
        )
    else:
        raise ValueError(
            "Unsupported file type. First convert to 'h5' or 'npy' using --mode = convert_csv and --options = [chosen file_type]."
        )
    return events, jets, constituents


def select_top_jets_and_constituents(
    jets, constituents, n_jets=3, n_constits=15, verbose=False
):
    """
    Select top n_jets per event and, for each selected jet, top n_constits constituents.

    Returns:
        jets_out: (num_events, n_jets, jets.shape[1])
        constits_out: (num_events, n_jets * n_constits, constituents.shape[1])

    """
    # --- Pre-sort jets ---
    # Sort by event id (ascending) then by descending pT (column 4).
    sort_idx_j = np.lexsort((-jets[:, 4], jets[:, 0]))
    jets_sorted = jets[sort_idx_j]

    # --- Pre-sort constituents ---
    # Sort by event id, then by jet id, then by descending pT.
    sort_idx_c = np.lexsort(
        (-constituents[:, 4], constituents[:, 1], constituents[:, 0])
    )
    constits_sorted = constituents[sort_idx_c]

    # --- Group jets by event ---
    evt_ids, evt_start, evt_counts = np.unique(
        jets_sorted[:, 0], return_index=True, return_counts=True
    )
    num_events = len(evt_ids)

    # Pre-allocate output arrays:
    jets_out = np.zeros((num_events, n_jets, jets.shape[1]), dtype=jets.dtype)
    constits_out = np.zeros(
        (num_events, n_jets * n_constits, constituents.shape[1]),
        dtype=constituents.dtype,
    )

    # Process each event:
    for i, evt_id in enumerate(evt_ids):
        # --- Select jets for this event ---
        start_j = evt_start[i]
        count_j = evt_counts[i]
        evt_jets = jets_sorted[start_j : start_j + count_j]
        n_used_jets = min(n_jets, evt_jets.shape[0])
        jets_out[i, :n_used_jets, :] = evt_jets[:n_used_jets]

        # --- Extract constituents for this event ---
        left_c = np.searchsorted(constits_sorted[:, 0], evt_id, side="left")
        right_c = np.searchsorted(constits_sorted[:, 0], evt_id, side="right")
        evt_constits = constits_sorted[left_c:right_c]

        # Fill constituent slots sequentially
        constit_idx = 0
        for j in range(n_used_jets):
            jet = jets_out[i, j, :]
            jet_id = jet[1]
            jet_btag = jet[3]
            # Filter constituents that belong to this jet.
            mask = evt_constits[:, 1] == jet_id
            jet_constits = evt_constits[mask]
            n_used_constits = min(n_constits, jet_constits.shape[0])
            # Fill the flat constituent array
            constits_out[i, constit_idx : constit_idx + n_used_constits, :] = (
                jet_constits[:n_used_constits]
            )
            # Ensure correct jet id and btag
            constits_out[i, constit_idx : constit_idx + n_used_constits, 1] = jet_id
            constits_out[i, constit_idx : constit_idx + n_used_constits, 3] = jet_btag
            constit_idx += n_constits  # Move to next jet slot

    if verbose:
        print(f"Jets shape after selection: {jets_out.shape}")
        print(f"Constitutents shape after selection: {constits_out.shape}")

    return jets_out, constits_out


def process_and_save_tensors(
    in_path, out_path, output_prefix, config, verbose: bool = False
):
    """
    Process the input file, parallelize selections, and save the results as PyTorch tensors.
    """

    file_type = config.file_type
    n_jets = config.num_jets
    n_constits = config.num_constits
    # n_workers = config.parallel_workers
    norm = config.normalizations

    if verbose:
        print(
            f"Processing {n_jets} jets and {n_constits} constituents from {in_path}..."
        )

    # Load the data
    events, jets, constituents = load_data(in_path, file_type, verbose)
    if verbose:
        print(
            f"Loaded {len(events)} events, {len(jets)} jets, and {len(constituents)} constituents from {in_path}"
        )
        print(
            f"Events shape: {events.shape}\nJets shape: {jets.shape}\nConstituents shape: {constituents.shape}"
        )

    # Apply normalizations
    if norm:
        if verbose:
            print(f"Normalizing data using {norm}...")
        if norm == "pj_custom":
            jets_norm, jet_comp_scaler = normalization.normalize_jet_pj_custom(jets)
            constituents_norm, constit_comp_scaler = (
                normalization.normalize_constit_pj_custom(constituents)
            )
        else:
            jets_norm, jet_comp_scaler = helper.normalize_data(jets, norm)
            constituents_norm, constit_comp_scaler = helper.normalize_data(
                constituents, norm
            )
        if verbose:
            print("Normalization complete.")
            print(
                f"Jets shape after normalization: {jets_norm.shape}\nConstituents shape after normalization: {constituents_norm.shape}"
            )

    # Parallel processing for top jets and constituents
    # if verbose:
    #     print(
    #         f"Selecting top {n_jets} jets and their top {n_constits} constituents in parallel..."
    #     )
    jet_selection, constits_selection = select_top_jets_and_constituents(
        jets_norm, constituents_norm, n_jets, n_constits, verbose
    )
    if verbose:
        print(
            f"Jets shape after selection: {jet_selection.shape}\nConstituents shape after selection: {constits_selection.shape}"
        )

    # Compute EFP features if enabled
    efp_tensor = None
    if hasattr(config, "should_compute_efp") and config.should_compute_efp():
        if verbose:
            print("Computing EFP features...")
        efp_tensor = compute_efp_features(
            constits_selection, config, n_jets, n_constits, verbose
        )
        if verbose:
            print(f"EFP tensor shape: {efp_tensor.shape}")

    # Convert to PyTorch tensors
    if verbose:
        print("Converting to PyTorch tensors...")
    evt_tensor, jet_tensor, constits_tensor = [
        helper.convert_to_tensor(data)
        for data in [events, jet_selection, constits_selection]
    ]

    # Save tensors
    if verbose:
        efp_msg = f" and {output_prefix}_efp.pt" if efp_tensor is not None else ""
        print(
            f"Saving tensors to {output_prefix}_events.pt, {output_prefix}_jets.pt, {output_prefix}_constituents.pt{efp_msg}..."
        )
    torch.save(evt_tensor, out_path + f"/{output_prefix}_events.pt")
    torch.save(jet_tensor, out_path + f"/{output_prefix}_jets.pt")
    torch.save(constits_tensor, out_path + f"/{output_prefix}_constituents.pt")

    # Save EFP tensor if computed
    if efp_tensor is not None:
        torch.save(efp_tensor, out_path + f"/{output_prefix}_efp.pt")
        if verbose:
            print(f"EFP tensor saved: {efp_tensor.shape}")

    # Save normalization scalers as pickle files
    if norm:
        if verbose:
            print(
                f"Saving normalization scalers to {output_prefix}_jet_scaler.pkl and {output_prefix}_constituent_scaler.pkl..."
            )
        jet_scaler_path = out_path + "/" + f"{output_prefix}_jet_scaler.pkl"
        constit_scaler_path = out_path + "/" + f"{output_prefix}_constituent_scaler.pkl"
        with open(jet_scaler_path, "wb") as f:
            pickle.dump(jet_comp_scaler, f)
        with open(constit_scaler_path, "wb") as f:
            pickle.dump(constit_comp_scaler, f)

    if verbose:
        print(
            f"Tensors saved to {output_prefix}_events.pt, {output_prefix}_jets.pt and {output_prefix}_constituents.pt"
        )
        if norm:
            print(
                f"Normalization scalers saved to {output_prefix}_jet_scaler.pkl and {output_prefix}_constituent_scaler.pkl"
            )


def preproc_inputs(paths, config, keyword, verbose: bool = False):
    # Load data from files whose names start with 'bkg'
    input_path = os.path.join(
        paths["data_path"], config.file_type, "tensors", "processed"
    )

    # Default: no EFP computation or usage
    include_efp = False
    efp_tensor = None

    try:
        # Validate EFP mode
        if config.efp_mode:
            include_efp = True
            if verbose:
                print("EFP integration: EFPs will be computed and used for training")

            events_tensor, jets_tensor, constituents_tensor, efp_tensor = (
                helper.load_tensors(input_path, keyword, include_efp=include_efp)
            )
            if verbose:
                print(
                    f"Loaded tensors from {input_path}/{keyword}_events.pt, {keyword}_jets.pt, {keyword}_constituents.pt, and {keyword}_efp.pt"
                )
                print(
                    f"Events tensor shape: {events_tensor.shape}\nJets tensor shape: {jets_tensor.shape}\nConstituents tensor shape: {constituents_tensor.shape}"
                )
                if efp_tensor is not None:
                    print(f"EFP tensor shape: {efp_tensor.shape}")
                else:
                    print("EFP tensor: None (files not found)")

        else:
            if verbose:
                print("EFPs disabled: no EFP computation or usage")

            events_tensor, jets_tensor, constituents_tensor = helper.load_tensors(
                input_path, keyword
            )
            if verbose:
                print(
                    f"Loaded tensors from {input_path}/{keyword}_events.pt, {keyword}_jets.pt and {keyword}_constituents.pt"
                )
                print(
                    f"Events tensor shape: {events_tensor.shape}\nJets tensor shape: {jets_tensor.shape}\nConstituents tensor shape: {constituents_tensor.shape}"
                )
    except ValueError as e:
        print(e)
        sys.exit(1)

    # Reshape the data as per configs.input_features
    try:
        jets_tensor, constituents_tensor = helper.select_features(
            jets_tensor, constituents_tensor, config.input_features
        )

        # Prepare data tuple with or without EFP tensor
        if include_efp and efp_tensor is not None:
            data = (events_tensor, jets_tensor, constituents_tensor, efp_tensor)
        else:
            data = (events_tensor, jets_tensor, constituents_tensor)

        if verbose:
            print("Data reshaped successfully")
            print("Events tensor shape:", events_tensor.shape)
            print("Jets tensor shape:", jets_tensor.shape)
            print("Constituents tensor shape:", constituents_tensor.shape)
            if include_efp and efp_tensor is not None:
                print("EFP tensor shape:", efp_tensor.shape)
    except ValueError as e:
        print(e)

    if keyword == "bkg_train":
        # Split the data into training and validation sets
        if verbose:
            print("Splitting data into training and validation sets...")
            print(
                f"Train:Val split ratio: {config.train_size * 100}:{(1 - config.train_size) * 100}"
            )
        try:
            # Apply the function to each tensor, producing a list of tuples.
            if include_efp and efp_tensor is not None:
                splits = [
                    helper.train_val_split(t, config.train_size)
                    for t in (
                        events_tensor,
                        jets_tensor,
                        constituents_tensor,
                        efp_tensor,
                    )
                ]
            else:
                splits = [
                    helper.train_val_split(t, config.train_size)
                    for t in (events_tensor, jets_tensor, constituents_tensor)
                ]
        except ValueError as e:
            print(e)
        # Unpack the list of tuples into two transposed tuples.
        trains, vals = zip(*splits, strict=False)
        # Repack into a single tuple
        data = trains + vals

    return data


def compute_efp_features(
    constituents_selection, config, n_jets, n_constits, verbose=False
):
    """
    Compute EFP features for the selected constituents.

    Args:
        constituents_selection: Array of shape (num_events, n_jets * n_constits, features)
        config: Configuration object with EFP parameters
        n_jets: Number of jets per event
        n_constits: Number of constituents per jet
        verbose: Whether to print progress information

    Returns:
        torch.Tensor: EFP features of shape (num_events, n_jets, n_efps)
    """
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    logger = logging.getLogger(__name__)

    # Validate EFP configuration
    try:
        efp_config = validate_efp_config(config)
        if verbose:
            logger.info(f"EFP config validated: {efp_config['n_efps']} features")
    except Exception as e:
        logger.error(f"EFP configuration validation failed: {e}")
        raise

    # Create EFPSet
    try:
        efpset = create_efpset(efp_config)
        if verbose:
            logger.info("EFPSet created successfully")
    except Exception as e:
        logger.error(f"EFPSet creation failed: {e}")
        raise

    # Reshape constituents from (events, jets*constits, features) to (events, jets, constits, features)
    num_events = constituents_selection.shape[0]
    total_features = constituents_selection.shape[1]  # noqa: F841

    # Extract features from constituent array
    # Based on BEAD data structure: [evt_id, jet_id, constit_id, b_tagged, constit_pt, constit_eta, constit_phi_sin, constit_phi_cos, generator_id]
    # Where generator_id is now mass
    pt_values = constituents_selection[:, :, 4]  # pT
    eta_values = constituents_selection[:, :, 5]  # eta
    phi_sin = constituents_selection[:, :, 6]  # phi_sin
    phi_cos = constituents_selection[:, :, 7]  # phi_cos
    mass_values = constituents_selection[:, :, 8]  # mass (from generator_id/PID)

    # Reconstruct phi from sin/cos
    phi_values = np.arctan2(phi_sin, phi_cos)

    # Reshape to (events, jets, constits, 4) for [pT, eta, phi, mass]
    # Note: We pass eta here, but preprocess_jet_constituents will convert it to rapidity
    constituents_reshaped = np.stack(
        [pt_values, eta_values, phi_values, mass_values], axis=-1
    )
    constituents_reshaped = constituents_reshaped.reshape(
        num_events, n_jets, n_constits, 4
    )

    if verbose:
        logger.info(f"Reshaped constituents: {constituents_reshaped.shape}")

    # Compute EFPs for each event
    n_efps = efp_config["n_efps"]
    efp_results = np.zeros((num_events, n_jets, n_efps), dtype=np.float32)

    for event_idx in range(num_events):
        # Process all jets for this event
        jets_for_event = constituents_reshaped[
            event_idx
        ]  # Shape: (n_jets, n_constits, 4)

        try:
            # Compute EFPs for all jets in this event
            event_efps = compute_efps_batch(
                jets_for_event, efpset, n_jobs=getattr(config, "efp_n_jobs", 4)
            )
            efp_results[event_idx] = event_efps

        except Exception as e:
            logger.warning(f"EFP computation failed for event {event_idx}: {e}")
            # Fill with zeros on failure
            efp_results[event_idx] = np.zeros((n_jets, n_efps), dtype=np.float32)

    if verbose:
        logger.info(f"EFP computation completed: {efp_results.shape}")

    # Apply standardization if requested
    if getattr(config, "efp_standardize_meanvar", True):
        if verbose:
            logger.info("Applying EFP standardization...")

        # Flatten for standardization: (events * jets, n_efps)
        efp_flat = efp_results.reshape(-1, n_efps)
        efp_standardized, efp_stats = standardize_efps(efp_flat)
        efp_results = efp_standardized.reshape(num_events, n_jets, n_efps)

        if verbose:
            logger.info("EFP standardization completed")

    # Convert to PyTorch tensor
    efp_tensor = torch.from_numpy(efp_results).float()

    if verbose:
        logger.info(
            f"EFP tensor created: {efp_tensor.shape}, dtype: {efp_tensor.dtype}"
        )

    return efp_tensor
