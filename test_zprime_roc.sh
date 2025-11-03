#!/bin/bash

# Test script for zprime ROC plotting with enhanced debugging
# Run this on your local setup where you have zprime data

echo "Testing zprime ROC plotting with enhanced debugging..."
echo "================================================================"

# Test the ordering validation first
echo "Step 1: Validating CSV vs Tensor file ordering..."
uv run python test_zprime_ordering.py

echo ""
echo "Step 2: Running ROC per-signal plotting with verbose output..."
echo "================================================================"

# Run ROC plotting with verbose output to see the new debugging info
# Replace 'your_project_name' with the actual project name you want to test
uv run bead -m plot -p zprime your_project_name -o roc_per_signal -v

echo ""
echo "Test completed. Please copy the above output and share it."