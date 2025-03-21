#!/bin/bash

WORKSPACE_NAME="Kinsley_BEAD_Workspace"
PROJECT_NAME="anomaly_detection"
MODEL_NAME=${1:-"TransformerAE"}  # Default to TransformerAE if not specified
EPOCHS=${2:-500}                  # Default to 500 epochs if not specified
LEARNING_RATE=${3:-0.0001}        # Default to 0.0001 if not specified

# Navigate to BEAD directory
if ! cd BEAD/bead; then
    echo "Error: Directory BEAD/bead not found!"
    exit 1
fi

CONFIG_FILE="workspaces/$WORKSPACE_NAME/$PROJECT_NAME/config/${PROJECT_NAME}_config.py"

# Update model configuration
echo "Updating model configuration..."
sed -i "s/c.model_name\s*=.*/c.model_name = \"$MODEL_NAME\"/" "$CONFIG_FILE"
sed -i "s/c.epochs\s*=.*/c.epochs = $EPOCHS/" "$CONFIG_FILE"
sed -i "s/c.learning_rate\s*=.*/c.learning_rate = $LEARNING_RATE/" "$CONFIG_FILE"

echo "Configuration updated successfully."
echo "Model: $MODEL_NAME"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
