#!/bin/bash

WORKSPACE_NAME="Kinsley_BEAD_Workspace"
PROJECT_NAME="anomaly_detection"

# Navigate to BEAD directory
if ! cd BEAD/bead; then
    echo "Error: Directory BEAD/bead not found!"
    exit 1
fi

# Run anomaly detection
echo "Running anomaly detection..."
poetry run bead -m detect -p $WORKSPACE_NAME $PROJECT_NAME

echo "Anomaly detection completed. Results saved in workspaces/$WORKSPACE_NAME/$PROJECT_NAME/output/results/"
