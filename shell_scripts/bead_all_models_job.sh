#!/bin/bash
#############################################################
# Job submission script for BEAD monotop analysis using 
# all NormFlow+ConvVAE model combinations on CSF cluster
#############################################################

#$ -cwd                      # Run job from current directory
#$ -j y                      # Merge stdout and stderr
#$ -o monotop_all_models.log # Output file name
#$ -N BEAD_all_models       # Job name
#$ -pe smp.pe 4             # Request 4 cores (adjust as needed)
#$ -l h_rt=72:00:00         # Request 72 hours runtime (3 days)
#$ -l nvidia_v100=1         # Request 1 NVIDIA V100 GPU
#$ -m bea                   # Email at beginning, end, and abort
#$ -M njangid22@example.com # Email address (replace with yours)

# Define workspace and data names
WORKSPACE_NAME="monotop_analysis"
DATA_NAME="monotop_200_A"

# Print job info
echo "=========================================="
echo "Starting BEAD job for ${DATA_NAME} analysis with all NormFlow+ConvVAE models"
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

# Define all NormFlow+ConvVAE model combinations to test
# Based on available models in models.py
MODELS=(
    "Planar_ConvVAE"
    "Radial_ConvVAE"
    "RealNVP_ConvVAE"
    "MAF_ConvVAE"
    "IAF_ConvVAE"
)

# Store start time
START_TIME=$(date +%s)

# Flag to track if data preparation has been done
DATA_PREPARED=false

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing ${#MODELS[@]} model combinations..."

# Loop through all models
for MODEL in "${MODELS[@]}"; do
    # Create a descriptive project name
    PROJECT_NAME="${MODEL,,}_500ep"  # lowercase model name
    
    echo "=========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing model: $MODEL"
    
    # Step 1: Create new project
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating new project: $PROJECT_NAME"
    poetry run bead -m new_project -p $WORKSPACE_NAME $PROJECT_NAME
    
    # Step 2: Update configuration
    CONFIG_FILE="workspaces/$WORKSPACE_NAME/$PROJECT_NAME/config/${PROJECT_NAME}_config.py"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Updating configuration in $CONFIG_FILE"
    
    # Make a backup of the original config
    cp $CONFIG_FILE ${CONFIG_FILE}.bak
    
    # Update the configuration settings
    sed -i "s/c.model_name\s*=\s*\"[^\"]*\"/c.model_name = \"$MODEL\"/g" $CONFIG_FILE
    sed -i "s/c.n_epochs\s*=\s*[0-9]*/c.n_epochs = 500/g" $CONFIG_FILE
    sed -i "s/c.save_frequency\s*=\s*[0-9]*/c.save_frequency = 100/g" $CONFIG_FILE
    sed -i "s/c.signal_names\s*=\s*\[[^]]*\]/c.signal_names = [\"monotop_200_A\"]/g" $CONFIG_FILE
    
    # Step 3: Run the appropriate workflow
    if [ "$DATA_PREPARED" = false ]; then
        # First time - need to convert CSV and prepare inputs
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running full workflow for $MODEL (including data preparation)..."
        poetry run bead -m chain -p $WORKSPACE_NAME $PROJECT_NAME -o convertcsv_prepareinputs_train_detect
        DATA_PREPARED=true
    else
        # Data already prepared in first run - just train and detect
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running train and detect for $MODEL (using prepared data)..."
        poetry run bead -m chain -p $WORKSPACE_NAME $PROJECT_NAME -o train_detect
    fi
    
    # Step 4: Generate plots
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generating plots for $MODEL..."
    poetry run bead -m plot -p $WORKSPACE_NAME $PROJECT_NAME
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed processing for $MODEL"
    echo "Results available in: workspaces/$WORKSPACE_NAME/$PROJECT_NAME/output/"
    echo "=========================================="
done

# Calculate and display total runtime
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(( (TOTAL_TIME % 3600) / 60 ))
SECONDS=$((TOTAL_TIME % 60))

echo "=========================================="
echo "All models processing completed!"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Date: $(date)"
echo "=========================================="