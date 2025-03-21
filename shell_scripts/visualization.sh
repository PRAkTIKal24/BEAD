#!/bin/bash

WORKSPACE_NAME="Kinsley_BEAD_Workspace"
PROJECT_NAME="anomaly_detection"
PLOT_TYPE=$1  # Optional argument: train_metrics, test_metrics, or all

# Navigate to BEAD directory
if ! cd BEAD/bead; then
    echo "Error: Directory BEAD/bead not found!"
    exit 1
fi

# Generate plots based on the specified type
if [ "$PLOT_TYPE" = "train_metrics" ]; then
    echo "Generating training metrics plots..."
    poetry run bead -m plot -p $WORKSPACE_NAME $PROJECT_NAME -o train_metrics
elif [ "$PLOT_TYPE" = "test_metrics" ]; then
    echo "Generating test metrics plots..."
    poetry run bead -m plot -p $WORKSPACE_NAME $PROJECT_NAME -o test_metrics
else
    echo "Generating all plots..."
    poetry run bead -m plot -p $WORKSPACE_NAME $PROJECT_NAME
fi

echo "Plot generation completed."
