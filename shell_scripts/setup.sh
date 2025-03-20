#!/bin/bash

"""
Since I have already installed poetry and trimap, I won't be performing that step again here
Install trimap separately due to compatibility issues
pip install trimap

Install BEAD package
poetry install

"""
# Navigate to BEAD directory
cd BEAD/bead

# Create a new workspace and project
WORKSPACE_NAME="Kinsley_BEAD_Workspace"
PROJECT_NAME="anomaly_detection"

echo "Creating new workspace: $WORKSPACE_NAME and project: $PROJECT_NAME"
poetry run bead -m new_project -p $WORKSPACE_NAME $PROJECT_NAME

echo "Project setup complete. Please copy your *_input_data.csv files to workspaces/$WORKSPACE_NAME/data/csv/"