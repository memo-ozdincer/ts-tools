#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=start_geom_stats
#SBATCH --output=logs/start_geom_stats_%j.out

set -e
PROJECT_DIR="/project/memo/code/ts-tools" # Adjust if needed
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_VERSION="3.11"

cd ${PROJECT_DIR}
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load python/${PYTHON_VERSION}
source ${VENV_DIR}/bin/activate

# Set up W&B API key for logging


echo ">>> Running Starting Geometry Statistics script..."

# Test different noise levels to find distribution of higher-order saddle points
python -m src.starting_geometry_stats \
    --max-samples 30 \
    --noise-levels 0.5 1.0 2.0 5.0 10.0 \
    --n-samples-per-noise 5 \
    --wandb \
    --wandb-project "gad-ts-search"

# Alternative: Test more noise levels with fewer samples per level
# python -m src.starting_geometry_stats \
#     --max-samples 30 \
#     --noise-levels 0.1 0.25 0.5 1.0 1.5 2.0 3.0 5.0 7.5 10.0 15.0 \
#     --n-samples-per-noise 3

# Alternative: Focus on a narrow range around typical distances
# python -m src.starting_geometry_stats \
#     --max-samples 50 \
#     --noise-levels 0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0 \
#     --n-samples-per-noise 10

echo "✅ Job completed."
