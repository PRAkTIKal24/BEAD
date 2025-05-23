#!/bin/bash --login
#SBATCH -J bead_run_a100
#SBATCH -p gpuA
#SBATCH -G 2
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 2-0
#SBATCH -o ../jobs_bead/bead_a100_%j.out
#SBATCH -e ../jobs_bead/bead_a100_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pratik.jawahar@postgrad.manchester.ac.uk

# --- User Defined Variables ---
WORKSPACE_NAME=$1
PROJECT_NAME=$2
OPTIONS=$3
# Path to your BEAD repository's 'bead' subdirectory
BEAD_DIR="~/scratch/x10719pj/BEAD"

# --- Environment Setup ---
echo "Loading modules..."
#module purge # Clean environment
#module load anaconda3/2023.09-0 # Or your preferred Anaconda version
module load cuda/12.6.0       # Ensure compatibility with PyTorch version

#echo "Activating Conda environment..."
#conda activate bead_env # Activate your pre-configured Conda environment

# Check if WORKSPACE_NAME and PROJECT_NAME are provided
if [ -z "$WORKSPACE_NAME" ] || [ -z "$PROJECT_NAME" ]; then
    echo "Error: WORKSPACE_NAME and PROJECT_NAME must be provided as arguments."
    echo "Usage: sbatch $0 <WORKSPACE_NAME> <PROJECT_NAME>"
    exit 1
fi

# --- Application Execution ---
echo "Changing to BEAD directory: $BEAD_DIR"
cd $BEAD_DIR

# Create job output/error directory if it doesn't exist
mkdir -p ../jobs_bead

echo "Starting BEAD run..."
echo "Chain Options: $OPTIONS"
echo "Workspace: $WORKSPACE_NAME"
echo "Project: $PROJECT_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs per GPU: $SLURM_CPUS_PER_GPU"

# Run BEAD training command
uv run bead -m chain -p $WORKSPACE_NAME $PROJECT_NAME -o $OPTIONS -v

echo "Training finished."
echo "Job completed at: $(date)"
