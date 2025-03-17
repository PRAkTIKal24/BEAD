# Custom normalization functions for analysis.

import numpy as np
from . import helper


def normalize_jet_pj_custom(data):
    """
    Normalizes jet data for HEP analysis using a chained normalization approach.

    Input data is expected as a NumPy array of shape (N, 7) with columns in the order:
    0: event_id         (unchanged)
    1: jet_id           (unchanged)
    2: num_constituents (to be normalized via "robust")
    3: b_tagged         (already integer; left unchanged)
    4: jet_pt           (to be normalized via "log+standard")
    5: jet_eta          (to be normalized via "standard")
    6: jet_phi          (to be normalized via "sin_cos" transformation)

    The output array will have 8 columns:
    [event_id, jet_id, num_constituents_norm, b_tagged, jet_pt_norm, jet_eta_norm, jet_phi_sin, jet_phi_cos]

    Args:
        data (np.ndarray): Input array of shape (N, 7).

    Returns:
        normalized_data (np.ndarray): Output array of shape (N, 8).
        scalers (dict): Dictionary containing the fitted scalers for each feature.
    """

    # Initialize the dictionary to store the fitted scalers.
    scalers = {}

    # 1. event_id, jet_id and b_tagged (columns 0, 1 and 3), unchanged.
    event_id = data[:, 0].reshape(-1, 1).astype(float)
    jet_id = data[:, 1].reshape(-1, 1).astype(float)
    b_tagged = data[:, 3].reshape(-1, 1).astype(float)

    # 2. num_constituents: column 2, use "robust"
    num_constituents = data[:, 2].reshape(-1, 1).astype(float)
    norm_num_const, scalers["num_constituents"] = helper.normalize_data(
        num_constituents, "robust"
    )

    # 4. jet_pt: column 4, use chain "log+standard"
    jet_pt = data[:, 4].reshape(-1, 1).astype(float)
    norm_jet_pt, scalers["jet_pt"] = helper.normalize_data(jet_pt, "log+standard")

    # 5. jet_eta: column 5, use "standard"
    jet_eta = data[:, 5].reshape(-1, 1).astype(float)
    norm_jet_eta, scalers["jet_eta"] = helper.normalize_data(jet_eta, "standard")

    # 6. jet_phi: column 6, use "sin_cos"
    jet_phi = data[:, 6].reshape(-1, 1).astype(float)
    norm_jet_phi, scalers["jet_phi"] = helper.normalize_data(jet_phi, "sincos")
    # norm_jet_phi will have 2 columns: sin and cos.

    # Concatenate the processed features:
    normalized_data = np.hstack(
        [
            event_id,  # unchanged
            jet_id,  # unchanged
            norm_num_const,  # normalized num_constituents
            b_tagged,  # unchanged
            norm_jet_pt,  # normalized jet_pt
            norm_jet_eta,  # normalized jet_eta
            norm_jet_phi,  # two columns: jet_phi_sin and jet_phi_cos
        ]
    )

    return normalized_data, scalers


def normalize_constit_pj_custom(data):
    """
    Normalizes jet data for HEP analysis using a chained normalization approach.

    Input data is expected as a NumPy array of shape (N, 7) with columns in the order:
        0: event_id         (unchanged)
        1: jet_id           (unchanged)
        2: constit_id       (unchanged)
        3: b_tagged         (unchanged)
        4: constit_pt           (to be normalized via "log+standard")
        5: constit_eta          (to be normalized via "standard")
        6: constit_phi          (to be normalized via "sin_cos" transformation)

    The output array will have 8 columns:
        [event_id, jet_id, constit_id, b_tagged, constit_pt_norm, constit_eta_norm, constit_phi_sin, constit_phi_cos]

    Args:
        data (np.ndarray): Input array of shape (N, 7).

    Returns:
        normalized_data (np.ndarray): Output array of shape (N, 8).
        scalers (dict): Dictionary containing the fitted scalers for each feature.
    """

    # Initialize the dictionary to store the fitted scalers.
    scalers = {}

    # 1. event_id, jet_id and constit_id (columns 0 - 3), unchanged.
    event_id = data[:, 0].reshape(-1, 1).astype(float)
    jet_id = data[:, 1].reshape(-1, 1).astype(float)
    constit_id = data[:, 2].reshape(-1, 1).astype(float)
    b_tagged = data[:, 3].reshape(-1, 1).astype(float)

    # 4. constit_pt: column 4, use chain "log+standard"
    constit_pt = data[:, 4].reshape(-1, 1).astype(float)
    norm_constit_pt, scalers["constit_pt"] = helper.normalize_data(
        constit_pt, "log+standard"
    )

    # 5. constit_eta: column 5, use "standard"
    constit_eta = data[:, 5].reshape(-1, 1).astype(float)
    norm_constit_eta, scalers["constit_eta"] = helper.normalize_data(
        constit_eta, "standard"
    )

    # 6. constit_phi: column 6, use "sin_cos"
    constit_phi = data[:, 6].reshape(-1, 1).astype(float)
    norm_constit_phi, scalers["constit_phi"] = helper.normalize_data(
        constit_phi, "sincos"
    )
    # norm_constit_phi will have 2 columns: sin and cos.

    # Concatenate the processed features:
    normalized_data = np.hstack(
        [
            event_id,  # unchanged
            jet_id,  # unchanged
            constit_id,  # unchanged
            b_tagged,  # unchanged
            norm_constit_pt,  # normalized constit_pt
            norm_constit_eta,  # normalized constit_eta
            norm_constit_phi,  # two columns: constit_phi_sin and constit_phi_cos
        ]
    )

    return normalized_data, scalers


def invert_normalize_jet_pj_custom(normalized_data, scalers):
    """
    Inverts the normalization applied by normalize_jet_data_np_chained.

    The input normalized_data is assumed to be a NumPy array of shape (N, 8) with columns:
        0: event_id              (unchanged)
        1: jet_id                (unchanged)
        2: num_constituents_norm (normalized via "robust")
        3: b_tagged              (unchanged)
        4: jet_pt_norm           (normalized via "log+standard")
        5: jet_eta_norm          (normalized via "standard")
        6-7: jet_phi_sin, jet_phi_cos (normalized via "sin_cos")

    Returns:
        original_data: NumPy array of shape (N, 7) with columns:
          [event_id, jet_id, num_constituents, b_tagged, jet_pt, jet_eta, jet_phi]

    Note:
      - The scaler for jet_pt (chain "log+standard") is expected to invert first the StandardScaler then the Log1pScaler,
        so that the original jet_pt is recovered.
      - The scaler for jet_phi (chain "sin_cos") converts the 2 columns back to the original angle using arctan2.
    """
    # 1. The unchanged columns: event_id, jet_id, b_tagged.
    event_id = normalized_data[:, 0].reshape(-1, 1)
    jet_id = normalized_data[:, 1].reshape(-1, 1)
    b_tagged = normalized_data[:, 3].reshape(-1, 1)

    # 2. Invert num_constituents (chain: "robust")
    norm_num_const = normalized_data[:, 2].reshape(-1, 1)
    original_num_const = helper.invert_normalize_data(
        norm_num_const, scalers["num_constituents"]
    )

    # 3. Invert jet_pt (chain: "log+standard")
    norm_jet_pt = normalized_data[:, 4].reshape(-1, 1)
    original_jet_pt = helper.invert_normalize_data(norm_jet_pt, scalers["jet_pt"])

    # 4. Invert jet_eta (chain: "standard")
    norm_jet_eta = normalized_data[:, 5].reshape(-1, 1)
    original_jet_eta = helper.invert_normalize_data(norm_jet_eta, scalers["jet_eta"])

    # 5. Invert jet_phi (chain: "sin_cos")
    # The chain "sin_cos" returns 2 columns; we pass these into its inverse_transform.
    norm_jet_phi = normalized_data[:, 6:8]
    original_jet_phi = helper.invert_normalize_data(norm_jet_phi, scalers["jet_phi"])

    # Concatenate the recovered columns in order:
    original_data = np.hstack(
        [
            event_id,
            jet_id,
            original_num_const,
            b_tagged,
            original_jet_pt,
            original_jet_eta,
            original_jet_phi,
        ]
    )

    return original_data


def invert_normalize_constit_pj_custom(normalized_data, scalers):
    """
    Inverts the normalization applied by normalize_jet_data_np_chained.

    The input normalized_data is assumed to be a NumPy array of shape (N, 8) with columns:
        0: event_id              (unchanged)
        1: jet_id                (unchanged)
        2: num_constituents_norm (normalized via "robust")
        3: b_tagged              (unchanged)
        4: jet_pt_norm           (normalized via "log+standard")
        5: jet_eta_norm          (normalized via "standard")
        6-7: jet_phi_sin, jet_phi_cos (normalized via "sin_cos")

    Returns:
        original_data: NumPy array of shape (N, 7) with columns:
          [event_id, jet_id, num_constituents, b_tagged, jet_pt, jet_eta, jet_phi]

    Note:
      - The scaler for jet_pt (chain "log+standard") is expected to invert first the StandardScaler then the Log1pScaler,
        so that the original jet_pt is recovered.
      - The scaler for jet_phi (chain "sin_cos") converts the 2 columns back to the original angle using arctan2.
    """
    # 1. The unchanged columns: event_id, jet_id, constit_id, b_tagged.
    event_id = normalized_data[:, 0].reshape(-1, 1)
    jet_id = normalized_data[:, 1].reshape(-1, 1)
    constit_id = normalized_data[:, 2].reshape(-1, 1)
    b_tagged = normalized_data[:, 3].reshape(-1, 1)

    # 3. Invert constit_pt (chain: "log+standard")
    norm_constit_pt = normalized_data[:, 4].reshape(-1, 1)
    original_constit_pt = helper.invert_normalize_data(
        norm_constit_pt, scalers["constit_pt"]
    )

    # 4. Invert constit_eta (chain: "standard")
    norm_constit_eta = normalized_data[:, 5].reshape(-1, 1)
    original_constit_eta = helper.invert_normalize_data(
        norm_constit_eta, scalers["constit_eta"]
    )

    # 5. Invert constit_phi (chain: "sin_cos")
    # The chain "sin_cos" returns 2 columns; we pass these into its inverse_transform.
    norm_constit_phi = normalized_data[:, 6:8]
    original_constit_phi = helper.invert_normalize_data(
        norm_constit_phi, scalers["constit_phi"]
    )

    # Concatenate the recovered columns in order:
    original_data = np.hstack(
        [
            event_id,
            jet_id,
            original_num_const,
            b_tagged,
            original_constit_pt,
            original_constit_eta,
            original_constit_phi,
        ]
    )

    return original_data
