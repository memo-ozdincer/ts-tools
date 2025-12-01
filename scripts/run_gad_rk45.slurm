#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=gad_rk45
#SBATCH --output=logs/gad_rk45_%j.out
#SBATCH --error=logs/gad_rk45_%j.err

# --- Configuration ---
set -e
PROJECT_DIR="/project/memo/code/ts-tools"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_VERSION="3.11"

# --- Script Start ---
echo ">>> Starting job on node $(hostname)"
cd ${PROJECT_DIR}

echo ">>> Loading Python ${PYTHON_VERSION} module..."
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load python/${PYTHON_VERSION}

echo ">>> Activating virtual environment..."
source ${VENV_DIR}/bin/activate

# Set up W&B API key for logging
export WANDB_API_KEY="ed8d3e45635321a86859f77337bdb4dfb5ceb113"

echo ">>> Running GAD-RK45 Search script..."

# --- Main Experiment: Start from midpoint and search for TS ---
# This version uses the standard GAD dynamics with the RK45 solver.
# Now includes multi-stage early stopping for robust TS detection.
python -m src.gad_rk45_search \
    --max-samples 30 \
    --start-from midpoint_rt \
    --stop-at-ts \
    --t-end 2.0 \
    --max-steps 150 \
    --min-eig-product-threshold -0.0001 \
    --confirmation-steps 10 \
    --wandb \
    --wandb-project "gad-ts-search"

# --- Example with kick mechanism enabled ---
# Uncomment to enable kick mechanism to escape local minima:
# python -m src.gad_rk45_search \
#     --max-samples 30 \
#     --start-from reactant \
#     --stop-at-ts \
#     --t-end 2.0 \
#     --max-steps 10000 \
#     --enable-kick \
#     --kick-force-threshold 0.015 \
#     --kick-magnitude 0.1

# --- Example with eigenvector-following refinement ---
# Uncomment for high-accuracy TS refinement:
# python -m src.gad_rk45_search \
#     --max-samples 30 \
#     --start-from reactant \
#     --stop-at-ts \
#     --t-end 2.0 \
#     --enable-kick \
#     --eigenvector-following \
#     --eigvec-follow-delay 5

echo "✅ Job completed successfully."