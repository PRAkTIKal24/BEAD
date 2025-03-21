#!/bin/bash

WORKSPACE_NAME="Kinsley_BEAD_Workspace"
PROJECT_NAME="anomaly_detection"

# Navigate to BEAD directory
if ! cd BEAD/bead; then
    echo "Error: Directory BEAD/bead not found!"
    exit 1
fi

# Train the model
echo "Starting model training..."
poetry run bead -m train -p $WORKSPACE_NAME $PROJECT_NAME

echo "Model training completed. Model saved in workspaces/$WORKSPACE_NAME/$PROJECT_NAME/output/models/"
