#!/bin/bash
# This utility script updates BEAD configuration files
# Usage: ./update_config.sh WORKSPACE_NAME PROJECT_NAME MODEL_NAME EPOCHS SAVE_EVERY

WORKSPACE_NAME=$1
PROJECT_NAME=$2
MODEL_NAME=${3:-"Planar_ConvVAE"}  # Default to Planar_ConvVAE if not specified
EPOCHS=${4:-500}                  # Default to 500 epochs if not specified
SAVE_EVERY=${5:-100}              # Default to saving every 100 epochs

# Navigate to BEAD directory
BEAD_DIR="$HOME/BEAD/bead"
cd $BEAD_DIR

CONFIG_FILE="$BEAD_DIR/workspaces/$WORKSPACE_NAME/$PROJECT_NAME/config/${PROJECT_NAME}_config.py"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found at $CONFIG_FILE"
    exit 1
fi

# Update model configuration
echo "Updating model configuration..."
sed -i "s/c.model_name\s*=.*/c.model_name                   = \"$MODEL_NAME\"/" "$CONFIG_FILE"
sed -i "s/c.epochs\s*=.*/c.epochs                      = $EPOCHS/" "$CONFIG_FILE"
sed -i "s/c.save_model_every\s*=.*/c.save_model_every            = $SAVE_EVERY/" "$CONFIG_FILE"

echo "Configuration updated successfully."
echo "Model: $MODEL_NAME"
echo "Epochs: $EPOCHS"
echo "Save Checkpoints Every: $SAVE_EVERY epochs"
