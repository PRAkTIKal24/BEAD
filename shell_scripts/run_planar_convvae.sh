#!/bin/bash
#$ -N bead_planar_convvae
#$ -pe smp 4
#$ -l mem=16G
#$ -l h_rt=48:00:00
#$ -j y
#$ -o bead_planar_convvae.log

# Load necessary modules
module load anaconda3/2020.07
module load python/3.8

# Set up working directories
WORKSPACE_NAME="monotop_200_A"
PROJECT_NAME="Planar_ConvVAE_500epochs"
BEAD_DIR="$HOME/BEAD/bead"

# Navigate to BEAD directory
cd $BEAD_DIR

# Create new project
echo "Creating new project: $PROJECT_NAME in workspace: $WORKSPACE_NAME"
poetry run bead -m new_project -p $WORKSPACE_NAME $PROJECT_NAME

# Check if input data exists
if [ ! -d "workspaces/$WORKSPACE_NAME/data/csv" ]; then
    echo "ERROR: Input data directory not found at workspaces/$WORKSPACE_NAME/data/csv"
    echo "Please ensure monotop_200_A input data CSV files are in the correct location"
    exit 1
fi

# Modify config file to set epochs and checkpoint frequency
CONFIG_FILE="$BEAD_DIR/workspaces/$WORKSPACE_NAME/$PROJECT_NAME/config/${PROJECT_NAME}_config.py"

# Update configuration for 500 epochs and saving every 100 epochs
echo "Updating model configuration..."
sed -i 's/c.epochs                      = 100/c.epochs                      = 500/' $CONFIG_FILE
sed -i 's/c.save_model_every            = 0/c.save_model_every            = 100/' $CONFIG_FILE
sed -i 's/c.model_name                   = "ConvVAE"/c.model_name                   = "Planar_ConvVAE"/' $CONFIG_FILE
echo "Configuration updated successfully."

# Run the full workflow using chain mode for efficiency
echo "Starting full workflow execution..."
poetry run bead -m chain -p $WORKSPACE_NAME $PROJECT_NAME -o convertcsv_prepareinputs_train_detect_plot

echo "Workflow completed successfully!"
echo "Results available in: workspaces/$WORKSPACE_NAME/$PROJECT_NAME/output/results/"
echo "Trained models saved in: workspaces/$WORKSPACE_NAME/$PROJECT_NAME/output/models/"