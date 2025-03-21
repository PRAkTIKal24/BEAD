#!/bin/bash
#$ -N bead_all_normflow
#$ -pe smp 4
#$ -l mem=16G
#$ -l h_rt=96:00:00
#$ -j y
#$ -o bead_all_normflow.log

# Load necessary modules
module load anaconda3/2020.07
module load python/3.8

# Set up working directories
WORKSPACE_NAME="monotop_200_A"
BEAD_DIR="$HOME/BEAD/bead"

# Navigate to BEAD directory
cd $BEAD_DIR

# Check if input data exists
if [ ! -d "workspaces/$WORKSPACE_NAME/data/csv" ]; then
    echo "ERROR: Input data directory not found at workspaces/$WORKSPACE_NAME/data/csv"
    echo "Please ensure monotop_200_A input data CSV files are in the correct location"
    exit 1
fi

# Define array of all NormFlow+ConvVAE model combinations
# These models are identified from the models.py file
MODELS=(
    "Planar_ConvVAE"
    "OrthogonalSylvester_ConvVAE"
    "HouseholderSylvester_ConvVAE"
    "TriangularSylvester_ConvVAE"
    "IAF_ConvVAE"
    "ConvFlow_ConvVAE"
    "NSFAR_ConvVAE"
)

# Create results directory for comparison
RESULTS_DIR="$BEAD_DIR/model_comparison_results"
mkdir -p $RESULTS_DIR
echo "Model comparison results will be saved in $RESULTS_DIR"

# Process the first model with full data preparation
FIRST_MODEL="${MODELS[0]}"
PROJECT_NAME="${FIRST_MODEL}_500epochs"

echo "Processing first model: $FIRST_MODEL with full data preparation"
echo "Creating new project: $PROJECT_NAME in workspace: $WORKSPACE_NAME"

# Create new project
poetry run bead -m new_project -p $WORKSPACE_NAME $PROJECT_NAME

# Modify config file
CONFIG_FILE="$BEAD_DIR/workspaces/$WORKSPACE_NAME/$PROJECT_NAME/config/${PROJECT_NAME}_config.py"

# Update configuration
echo "Updating model configuration for $FIRST_MODEL..."
sed -i "s/c.epochs                      = 100/c.epochs                      = 500/" $CONFIG_FILE
sed -i "s/c.save_model_every            = 0/c.save_model_every            = 100/" $CONFIG_FILE
sed -i "s/c.model_name                   = \"ConvVAE\"/c.model_name                   = \"$FIRST_MODEL\"/" $CONFIG_FILE
echo "Configuration updated successfully."

# Run the full workflow with chaining for the first model
echo "Running full workflow for $FIRST_MODEL..."
poetry run bead -m chain -p $WORKSPACE_NAME $PROJECT_NAME -o convertcsv_prepareinputs_train_detect_plot

# Copy results to comparison directory
mkdir -p "$RESULTS_DIR/$PROJECT_NAME"
cp -r "$BEAD_DIR/workspaces/$WORKSPACE_NAME/$PROJECT_NAME/output/results/"* "$RESULTS_DIR/$PROJECT_NAME/"
echo "Results for $FIRST_MODEL copied to comparison directory"

# Process remaining models (skipping data preparation)
for ((i=1; i<${#MODELS[@]}; i++)); do
    MODEL="${MODELS[$i]}"
    PROJECT_NAME="${MODEL}_500epochs"
    
    echo "Processing model: $MODEL (skipping data preparation)"
    echo "Creating new project: $PROJECT_NAME in workspace: $WORKSPACE_NAME"
    
    # Create new project
    poetry run bead -m new_project -p $WORKSPACE_NAME $PROJECT_NAME
    
    # Modify config file
    CONFIG_FILE="$BEAD_DIR/workspaces/$WORKSPACE_NAME/$PROJECT_NAME/config/${PROJECT_NAME}_config.py"
    
    # Update configuration
    echo "Updating model configuration for $MODEL..."
    sed -i "s/c.epochs                      = 100/c.epochs                      = 500/" $CONFIG_FILE
    sed -i "s/c.save_model_every            = 0/c.save_model_every            = 100/" $CONFIG_FILE
    sed -i "s/c.model_name                   = \"ConvVAE\"/c.model_name                   = \"$MODEL\"/" $CONFIG_FILE
    echo "Configuration updated successfully."
    
    # Run the workflow with chaining, skipping data preparation steps
    echo "Running workflow for $MODEL (train, detect, plot)..."
    poetry run bead -m chain -p $WORKSPACE_NAME $PROJECT_NAME -o train_detect_plot
    
    # Copy results to comparison directory
    mkdir -p "$RESULTS_DIR/$PROJECT_NAME"
    cp -r "$BEAD_DIR/workspaces/$WORKSPACE_NAME/$PROJECT_NAME/output/results/"* "$RESULTS_DIR/$PROJECT_NAME/"
    echo "Results for $MODEL copied to comparison directory"
done

echo "All model combinations completed successfully!"
echo "Comparison results available in: $RESULTS_DIR"
