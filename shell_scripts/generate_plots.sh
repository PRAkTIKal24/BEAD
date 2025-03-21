#!/bin/bash
#$ -N bead_plots
#$ -pe smp 2
#$ -l mem=8G
#$ -l h_rt=6:00:00
#$ -j y
#$ -o bead_plots.log

# Load necessary modules
module load anaconda3/2020.07
module load python/3.8

# Set up working directories
WORKSPACE_NAME="monotop_200_A"
BEAD_DIR="$HOME/BEAD/bead"

# Navigate to BEAD directory
cd $BEAD_DIR

# Define array of all NormFlow+ConvVAE model combinations
MODELS=(
    "Planar_ConvVAE"
    "OrthogonalSylvester_ConvVAE"
    "HouseholderSylvester_ConvVAE"
    "TriangularSylvester_ConvVAE"
    "IAF_ConvVAE"
    "ConvFlow_ConvVAE"
    "NSFAR_ConvVAE"
)

# Generate plots for each model
for MODEL in "${MODELS[@]}"; do
    PROJECT_NAME="${MODEL}_500epochs"
    
    echo "Generating plots for model: $MODEL"
    
    # Generate training metrics plots
    echo "Generating training metrics plots..."
    poetry run bead -m plot -p $WORKSPACE_NAME $PROJECT_NAME -o train_metrics
    
    # Generate test metrics plots
    echo "Generating test metrics plots..."
    poetry run bead -m plot -p $WORKSPACE_NAME $PROJECT_NAME -o test_metrics
    
    echo "Plot generation completed for $MODEL"
done

echo "All plots generated successfully!"
