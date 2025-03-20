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

# Run the full pipeline using the chain mode
echo "Running full BEAD pipeline..."
poetry run bead -m chain -p $WORKSPACE_NAME $PROJECT_NAME -o convertcsv_prepareinputs_train_detect_plot

echo "Full pipeline completed successfully."
