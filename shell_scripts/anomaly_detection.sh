#!/bin/bash

WORKSPACE_NAME="Kinsley_BEAD_Workspace"
PROJECT_NAME="anomaly_detection"

# Navigate to BEAD directory
cd BEAD/bead

# Run anomaly detection
echo "Running anomaly detection..."
poetry run bead -m detect -p $WORKSPACE_NAME $PROJECT_NAME

echo "Anomaly detection completed. Results saved in workspaces/$WORKSPACE_NAME/$PROJECT_NAME/output/results/"
