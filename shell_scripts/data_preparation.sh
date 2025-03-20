#!/bin/bash

WORKSPACE_NAME="Kinsley_BEAD_Workspace"
PROJECT_NAME="anomaly_detection"

# Navigate to BEAD directory
cd BEAD/bead

# Check if input data exists
if [ ! -d "workspaces/$WORKSPACE_NAME/data/csv" ]; then
    mkdir -p "workspaces/$WORKSPACE_NAME/data/csv"
    echo "Created CSV directory. Please copy your *_input_data.csv files to workspaces/$WORKSPACE_NAME/data/csv/"
    exit 1
fi

# Convert CSV to H5 format
echo "Converting CSV data to H5 format..."
poetry run bead -m convert_csv -p $WORKSPACE_NAME $PROJECT_NAME

# Prepare inputs for model training
echo "Preparing inputs for model training..."
poetry run bead -m prepare_inputs -p $WORKSPACE_NAME $PROJECT_NAME

echo "Data preparation completed successfully."
