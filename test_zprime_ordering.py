#!/usr/bin/env python3
"""
Test script to validate zprime CSV and tensor file ordering.
Run this on your local setup where you have zprime data.
"""

import os
import sys

# Add the bead package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_zprime_ordering():
    """Test the ordering validation for zprime files."""
    from bead.src.utils.helper import validate_csv_tensor_ordering, get_signal_file_info_from_csv, get_bkg_test_count_from_csv
    
    workspace = "zprime"
    csv_path = f"bead/workspaces/{workspace}/data/csv"
    tensor_path = f"bead/workspaces/{workspace}/data/h5/tensors/processed"
    
    print(f"Testing zprime workspace ordering...")
    print(f"CSV path: {csv_path}")
    print(f"Tensor path: {tensor_path}")
    print("=" * 60)
    
    # Check if paths exist
    if not os.path.exists(csv_path):
        print(f"Error: CSV path does not exist: {csv_path}")
        return False
    
    if not os.path.exists(tensor_path):
        print(f"Error: Tensor path does not exist: {tensor_path}")
        return False
    
    # List CSV files
    csv_files = [f for f in os.listdir(csv_path) if 'sig_test' in f and f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  {f}")
    print()
    
    # List tensor files
    tensor_files = [f for f in os.listdir(tensor_path) if 'sig_test' in f and f.endswith('.pt')]
    print(f"Found {len(tensor_files)} tensor files:")
    for f in tensor_files:
        print(f"  {f}")
    print()
    
    # Validate ordering
    print("Validating CSV vs Tensor ordering...")
    csv_order, tensor_order, matches = validate_csv_tensor_ordering(csv_path, tensor_path)
    
    print(f"CSV signal order ({len(csv_order)}): {csv_order}")
    print(f"Tensor signal order ({len(tensor_order)}): {tensor_order}")
    print(f"Orders match: {matches}")
    print()
    
    # Get signal file info
    print("Getting signal file info from CSV...")
    signal_file_info = get_signal_file_info_from_csv(csv_path, "sig_test")
    bkg_count = get_bkg_test_count_from_csv(csv_path)
    
    print(f"Background test event count: {bkg_count}")
    total_signal_events = 0
    for info in signal_file_info:
        print(f"  {info['sig_filename']}: {info['events_count']} events (indices {bkg_count + info['start_idx']}:{bkg_count + info['end_idx']})")
        total_signal_events += info['events_count']
    
    print(f"Total signal events: {total_signal_events}")
    print(f"Expected total events: {bkg_count + total_signal_events}")
    print()
    
    return matches

def check_existing_numpy_files():
    """Check existing numpy files to see their actual lengths."""
    workspace = "zprime"
    
    # Look for existing inference results
    possible_projects = [
        "convvae_house_sc_anneal",
        "hp_convvae", 
        "2classzp/hp_convvae",
        "convvae_h",
        "convvae_p"
    ]
    
    for project in possible_projects:
        results_path = f"bead/workspaces/{workspace}/{project}/output/results"
        if os.path.exists(results_path):
            print(f"Found results in: {results_path}")
            
            # Check numpy files
            numpy_files = [f for f in os.listdir(results_path) if f.endswith('_test.npy')]
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
        matches = test_zprime_ordering()
        print()
        check_existing_numpy_files()
        
        if matches:
            print("✅ SUCCESS: CSV and tensor ordering matches!")
        else:
            print("❌ ISSUE: CSV and tensor ordering mismatch detected!")
            print("   This explains the ROC plotting problems.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()