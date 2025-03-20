#!/bin/bash
#SBATCH --job-name=monotop_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load necessary modules
module load anaconda
source activate my_env1 

# Define model combinations
NORMFLOW_MODELS=("Planar" "Radial" "Sylvester")
CONV_VAE_MODELS=("ConvVAE" "DeepConvVAE")

# Ensure necessary directories exist
mkdir -p logs models plots

# Loop through all NormFlow + ConvVAE combinations
for normflow in "${NORMFLOW_MODELS[@]}"; do
    for convvae in "${CONV_VAE_MODELS[@]}"; do
        MODEL_NAME="${normflow}_${convvae}"
        echo "Submitting job for model: $MODEL_NAME"

        # Submit a new SLURM job for each combination
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${MODEL_NAME}
#SBATCH --output=logs/${MODEL_NAME}_%j.out
#SBATCH --error=logs/${MODEL_NAME}_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load anaconda
source activate my_env  # Replace with your actual environment

# Run training
python run.py --process monotop_200_A \
              --model $MODEL_NAME \
              --epochs 500 \
              --save_frequency 100

# Run plotting
python run.py --mode plot --process monotop_200_A

EOF
    done
done

echo "All jobs submitted successfully."
