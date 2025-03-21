#!/bin/bash
#$ -N bead_model_eval
#$ -pe smp 2
#$ -l mem=8G
#$ -l h_rt=12:00:00
#$ -j y
#$ -o bead_model_eval.log

# Load necessary modules
module load anaconda3/2020.07
module load python/3.8

# Set up working directories
WORKSPACE_NAME="monotop_200_A"
BEAD_DIR="$HOME/BEAD/bead"
RESULTS_DIR="$BEAD_DIR/model_comparison_results"

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

# Create summary file
SUMMARY_FILE="$RESULTS_DIR/model_comparison_summary.txt"
echo "Model Evaluation Summary" > $SUMMARY_FILE
echo "======================" >> $SUMMARY_FILE
echo "Date: $(date)" >> $SUMMARY_FILE
echo "Workspace: $WORKSPACE_NAME" >> $SUMMARY_FILE
echo "======================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Evaluate each model
for MODEL in "${MODELS[@]}"; do
    PROJECT_NAME="${MODEL}_500epochs"
    RESULTS_PATH="$BEAD_DIR/workspaces/$WORKSPACE_NAME/$PROJECT_NAME/output/results"
    
    echo "Evaluating model: $MODEL"
    echo "" >> $SUMMARY_FILE
    echo "Model: $MODEL" >> $SUMMARY_FILE
    echo "----------------" >> $SUMMARY_FILE
    
    # Check if results exist
    if [ -d "$RESULTS_PATH" ]; then
        # Extract and summarize key metrics
        # This is a placeholder - in a real scenario, you would parse the results files
        # to extract metrics like loss values, anomaly detection performance, etc.
        echo "Results available at: $RESULTS_PATH" >> $SUMMARY_FILE
        
        # Example of extracting information (modify as needed based on actual output format)
        if [ -f "$RESULTS_PATH/test_metrics.txt" ]; then
            echo "Test Metrics:" >> $SUMMARY_FILE
            grep "AUC" "$RESULTS_PATH/test_metrics.txt" >> $SUMMARY_FILE
        fi
        
        if [ -f "$RESULTS_PATH/train_metrics.txt" ]; then
            echo "Final Training Loss:" >> $SUMMARY_FILE
            tail -n 1 "$RESULTS_PATH/train_metrics.txt" >> $SUMMARY_FILE
        fi
    else
        echo "No results found for $MODEL" >> $SUMMARY_FILE
    fi
done

echo "Model evaluation completed. Summary available at: $SUMMARY_FILE"
