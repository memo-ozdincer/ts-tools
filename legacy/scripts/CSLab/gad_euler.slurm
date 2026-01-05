#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=gad_euler
#SBATCH --output=logs/gad_euler_%j.out
#SBATCH --error=logs/gad_euler_%j.err

# --- Configuration ---
set -e
PROJECT_DIR="/project/memo/code/ts-tools"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_VERSION="3.11"

# --- Script Start ---
echo ">>> Starting job on node $(hostname)"
echo ">>> Project Directory: ${PROJECT_DIR}"

cd ${PROJECT_DIR}

echo ">>> Loading Python ${PYTHON_VERSION} module..."
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load python/${PYTHON_VERSION}

echo ">>> Activating virtual environment..."
source ${VENV_DIR}/bin/activate

# Set up W&B API key for logging


echo ">>> Running Euler RMSD script..."
# ARGS ARE HERE NOW
python -m src.gad_gad_euler_rmsd \
    --max-samples 20 \
    --n-steps 150 \
    --start-from midpoint_rt \
    --stop-at-ts \
    --dt 0.001 \
    --wandb \
    --wandb-project "gad-ts-search" \
    --wandb-entity "memo-ozdincer-university-of-toronto"

# --- Example with kick mechanism enabled ---
# Uncomment to enable kick mechanism to escape local minima:
# python -m src.gad_gad_euler_rmsd \
#     --max-samples 300 \
#     --n-steps 1000 \
#     --start-from reactant \
#     --stop-at-ts \
#     --dt 0.001 \
#     --enable-kick \
#     --kick-force-threshold 0.015 \
#     --kick-magnitude 0.1

# python -m src.gad_gad_euler_rmsd \
#     --max-samples 300 \
#     --start-from three_quarter_rt

echo "✅ Job completed successfully."