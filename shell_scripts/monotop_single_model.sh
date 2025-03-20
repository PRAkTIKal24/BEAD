#!/bin/bash
#SBATCH --job-name=monotop_200_A_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load required modules
module load anaconda
source activate my_env1  

# Create directories if they don’t exist
mkdir -p logs models plots

# Train the model
python run.py --process monotop_200_A \
              --model Planar_ConvVAE \
              --epochs 500 \
              --save_frequency 100

# Generate plots
python run.py --mode plot --process monotop_200_A

echo "Job completed successfully."
