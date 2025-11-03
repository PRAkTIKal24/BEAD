#!/usr/bin/env python3
"""
Test script to validate zprime CSV and tensor file ordering.
Run this on your local setup where you have zprime data.
"""

import os
import sys

# Add the bead package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))


def test_zprime_ordering():
    """Test the corrected ordering and indexing for zprime files."""
    from bead.src.utils.helper import (
        get_signal_file_info_corrected,
        get_bkg_test_count_from_csv,
    )
    import numpy as np

    workspace = "zprime"
    csv_path = f"bead/workspaces/{workspace}/data/csv"

    print("Testing zprime workspace corrected indexing...")
    print(f"CSV path: {csv_path}")
    print("=" * 60)

    # Check if paths exist
    if not os.path.exists(csv_path):
        print(f"Error: CSV path does not exist: {csv_path}")
        return False

    # Get background count
    bkg_count = get_bkg_test_count_from_csv(csv_path)
    print(f"Background test event count: {bkg_count}")

    # Check actual data length from results
    possible_projects = [
        "convvae_house_sc_anneal",
        "hp_convvae",
        "2classzp/hp_convvae",
        "convvae_h",
        "convvae_p",
        "convvae",
    ]

    actual_data_length = None

    for project in possible_projects:
        test_results_path = f"bead/workspaces/{workspace}/{project}/output/results"
        if os.path.exists(test_results_path):
            loss_file = os.path.join(test_results_path, "loss_test.npy")
            if os.path.exists(loss_file):
                try:
                    data = np.load(loss_file)
                    actual_data_length = len(data)
                    print(f"Found data in: {test_results_path}")
                    print(f"Actual data length: {actual_data_length}")
                    break
                except Exception as e:
                    print(f"Error loading {loss_file}: {e}")

    if actual_data_length is None:
        print("Error: Could not find actual data files to determine length")
        return False

    # Get corrected signal file info
    print("\nGetting corrected signal file info...")
    signal_file_info = get_signal_file_info_corrected(
        csv_path, actual_data_length, bkg_count, "sig_test"
    )

    print(f"Available signal events: {actual_data_length - bkg_count}")
    total_corrected_events = 0
    for info in signal_file_info:
        print(
            f"  {info['sig_filename']}: {info['events_count']} events (indices {bkg_count + info['start_idx']}:{bkg_count + info['end_idx']})"
        )
        total_corrected_events += info["events_count"]

    print(f"Total corrected signal events: {total_corrected_events}")
    print(f"Expected total events: {bkg_count + total_corrected_events}")

    # Validate that indices don't exceed data length
    max_index = max(bkg_count + info["end_idx"] for info in signal_file_info)
    indices_valid = max_index <= actual_data_length

    print(f"Max index: {max_index}")
    print(f"Indices within bounds: {indices_valid}")

    return indices_valid


def check_existing_numpy_files():
    """Check existing numpy files to see their actual lengths."""
    workspace = "zprime"

    # Look for existing inference results
    possible_projects = [
        "convvae_house_sc_anneal",
        "hp_convvae",
        "2classzp/hp_convvae",
        "convvae_h",
        "convvae_p",
    ]

    for project in possible_projects:
        results_path = f"bead/workspaces/{workspace}/{project}/output/results"
        if os.path.exists(results_path):
            print(f"Found results in: {results_path}")

            # Check numpy files
            numpy_files = [
                f for f in os.listdir(results_path) if f.endswith("_test.npy")
            ]
            for npy_file in numpy_files:
                file_path = os.path.join(results_path, npy_file)
                try:
                    import numpy as np

                    data = np.load(file_path)
                    print(f"  {npy_file}: shape {data.shape}, length {len(data)}")
                except Exception as e:
                    print(f"  {npy_file}: Error loading - {e}")
            print()


if __name__ == "__main__":
    print("Zprime Ordering Validation Test")
    print("=" * 40)

    try:
        indices_valid = test_zprime_ordering()
        print()
        check_existing_numpy_files()

        if indices_valid:
            print("✅ SUCCESS: Corrected indices are within data bounds!")
        else:
            print("❌ ISSUE: Corrected indices still exceed data bounds!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()
