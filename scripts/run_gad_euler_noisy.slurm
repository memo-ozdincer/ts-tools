#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=gad_euler_noisy
#SBATCH --output=logs/gad_euler_noisy_%j.out
#SBATCH --error=logs/gad_euler_noisy_%j.err

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

echo ">>> Running GAD-Euler Search with noisy starting geometries..."

# === GAD + EIGPROD MINIMIZATION (1000 steps) ===
# GAD dynamics with eigproduct minimization fallback when stalled.
# Gradient descent with larger step size (0.5 Å) to escape weird geometries.

echo ">>> Running GAD + Eigprod (gradient descent, 1000 steps)"
python -m src.gad_gad_euler_rmsd \
    --max-samples 30 \
    --start-from reactant_noise2A \
    --enable-minimization-fallback \
    --eig-descent-lr 0.01 \
    --eig-descent-max-step 0.5 \
    --stop-at-ts \
    --n-steps 1000 \
    --dt 0.005 \
    --wandb \
    --wandb-project "gad-noise-experiments"
    

# --- Noise Level: 1Å ---
# python -m src.gad_gad_euler_rmsd \
#     --max-samples 30 \
#     --start-from reactant_noise1A \
#     --stop-at-ts \
#     --n-steps 300 \
#     --dt 0.01 \
#     --wandb \
#     --wandb-project "gad-noise-experiments"

# --- Noise Level: 0.5Å (easier) ---
# python -m src.gad_gad_euler_rmsd \
#     --max-samples 30 \
#     --start-from reactant_noise0.5A \
#     --stop-at-ts \
#     --n-steps 200 \
#     --dt 0.01 \
#     --wandb \
#     --wandb-project "gad-noise-experiments"

# --- Baseline: No noise (from reactant) ---
# python -m src.gad_gad_euler_rmsd \
#     --max-samples 30 \
#     --start-from reactant \
#     --stop-at-ts \
#     --n-steps 200 \
#     --dt 0.01 \
#     --wandb \
#     --wandb-project "gad-noise-experiments"

# --- With kick mechanism for stuck cases ---
# python -m src.gad_gad_euler_rmsd \
#     --max-samples 30 \
#     --start-from reactant_noise2A \
#     --stop-at-ts \
#     --n-steps 500 \
#     --dt 0.01 \
#     --enable-kick \
#     --kick-magnitude 0.15 \
#     --wandb \
#     --wandb-project "gad-noise-experiments"

echo "✅ Job completed successfully."

