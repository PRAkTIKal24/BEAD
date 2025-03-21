#!/bin/bash
#############################################################
# Job submission script for BEAD monotop analysis using 
# Planar_ConvVAE model on CSF cluster
#############################################################

#$ -cwd                     # Run job from current directory
#$ -j y                     # Merge stdout and stderr
#$ -o monotop_planar.log    # Output file name
#$ -N BEAD_monotop         # Job name
#$ -pe smp.pe 4            # Request 4 cores (adjust as needed)
#$ -l h_rt=24:00:00        # Request 24 hours runtime
#$ -l nvidia_v100=1        # Request 1 NVIDIA V100 GPU
#$ -m bea                  # Email at beginning, end, and abort
#$ -M njangid22@example.com  # Email address (replace with yours)

# Define workspace and project names
WORKSPACE_NAME="monotop_analysis"
PROJECT_NAME="planar_convvae_500ep"
DATA_NAME="monotop_200_A"

# Print job info
echo "=========================================="
echo "Starting BEAD job for ${DATA_NAME} analysis"
echo "Using model: Planar_ConvVAE"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Host: $(hostname)"
echo "=========================================="

# Load required modules for CSF cluster
module load apps/anaconda3
module load libs/cuda/11.7.0

# Assume environment is already set up
# Activate conda environment if needed
# conda activate bead_env

# Go to the BEAD directory
cd $HOME/BEAD/bead

# Step 1: Create new project workspace
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating new project workspace..."
poetry run bead -m new_project -p $WORKSPACE_NAME $PROJECT_NAME

# Step 2: Update configuration for our specific requirements
# Get the path to the config file
CONFIG_FILE="workspaces/$WORKSPACE_NAME/$PROJECT_NAME/config/${PROJECT_NAME}_config.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Updating configuration in $CONFIG_FILE..."

# Make a backup of the original config
cp $CONFIG_FILE ${CONFIG_FILE}.bak

# Update the configuration settings
# Set model to Planar_ConvVAE, epochs to 500, save frequency to 100, and signal to monotop_200_A
sed -i "s/c.model_name\s*=\s*\"[^\"]*\"/c.model_name = \"Planar_ConvVAE\"/g" $CONFIG_FILE
sed -i "s/c.n_epochs\s*=\s*[0-9]*/c.n_epochs = 500/g" $CONFIG_FILE
sed -i "s/c.save_frequency\s*=\s*[0-9]*/c.save_frequency = 100/g" $CONFIG_FILE
sed -i "s/c.signal_names\s*=\s*\[[^]]*\]/c.signal_names = [\"monotop_200_A\"]/g" $CONFIG_FILE

# Step 3: Ensure input data is in the correct location
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking input data location..."
DATA_DIR="workspaces/$WORKSPACE_NAME/data/csv"
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory: $DATA_DIR"
    mkdir -p $DATA_DIR
fi

# Verify that input data exists
if [ ! -f "$DATA_DIR/${DATA_NAME}_input_data.csv" ]; then
    echo "WARNING: Input data file ${DATA_NAME}_input_data.csv not found in $DATA_DIR"
    echo "Please ensure the input data is available before running the next steps"
fi

# Step 4: Run the full BEAD workflow chain
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting BEAD workflow chain..."
poetry run bead -m chain -p $WORKSPACE_NAME $PROJECT_NAME -o convertcsv_prepareinputs_train_detect

# Step 5: Generate plots
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generating plots..."
poetry run bead -m plot -p $WORKSPACE_NAME $PROJECT_NAME

echo "[$(date '+%Y-%m-%d %H:%M:%S')] BEAD workflow completed"
echo "Results are available in: workspaces/$WORKSPACE_NAME/$PROJECT_NAME/output/"
echo "=========================================="