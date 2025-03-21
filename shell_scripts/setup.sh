#!/bin/bash

# Install poetry if not already installed
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"  # Ensure Poetry is in PATH
fi

# Ensure trimap is installed
echo "Installing trimap..."
pip install trimap || { echo "Failed to install trimap!"; exit 1; }

# Navigate to BEAD directory
if ! cd BEAD/bead; then
    echo "Error: Directory BEAD/bead not found!"
    exit 1
fi

# Install BEAD package using Poetry
echo "Installing BEAD dependencies..."
poetry install || { echo "Poetry install failed!"; exit 1; }

# Define workspace and project names
WORKSPACE_NAME="Kinsley_BEAD_Workspace"
PROJECT_NAME="anomaly_detection"

# Create a new workspace and project
echo "Creating new workspace: $WORKSPACE_NAME and project: $PROJECT_NAME"
poetry run bead -m new_project -p "$WORKSPACE_NAME" "$PROJECT_NAME"

# Final instruction for the user
echo "Project setup complete. Please copy your *_input_data.csv files to workspaces/$WORKSPACE_NAME/data/csv/"