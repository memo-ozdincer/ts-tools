#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=gad_rk45_noisy
#SBATCH --output=logs/gad_rk45_noisy_%j.out
#SBATCH --error=logs/gad_rk45_noisy_%j.err

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


echo ">>> Running GAD-RK45 Search with HYBRID MODE for noisy geometries..."

# === HYBRID FORCE-GAD MODE ===
# This mode is designed for noisy starting geometries (noise levels: 0.5, 1, 2 Å).
# 
# Key innovation: Saddle-order-aware dynamics switching:
#   - At higher-order saddles (2+ negative eigenvalues): Use FORCE DESCENT
#     to escape unstable high-energy regions where noisy geometries often land
#   - At order-1 (TS candidate): Use GAD for precise convergence
#   - At order-0 (minimum): Use standard GAD to climb toward TS
#
# This solves the "plateau problem" where pure eigenvalue optimization gets stuck
# because it fights against steep energy gradients at higher-order saddles.
#
# Parameters:
#   --hybrid-mode: Enable hybrid Force-GAD dynamics (CRITICAL for noisy starts)
#   --adaptive-step-sizing: Larger steps at higher-order saddles, smaller near TS
#   --higher-order-multiplier: Scale factor for higher-order saddles (default: 5.0)
#   --ts-multiplier: Scale factor near TS for refinement (default: 0.5)
#   --force-descent-threshold: Saddle order threshold for force descent (default: 2)

# --- Noise Level: 2Å (most challenging) ---
python -m src.gad_rk45_search \
    --max-samples 30 \
    --start-from reactant_noise2A \
    --stop-at-ts \
    --t-end 5.0 \
    --max-steps 500 \
    --hybrid-mode \
    --adaptive-step-sizing \
    --higher-order-multiplier 10.0 \
    --ts-multiplier 0.5 \
    --force-descent-threshold 2 \
    --wandb \
    --wandb-project "gad-noise-experiments"

# --- Noise Level: 1Å ---
# python -m src.gad_rk45_search \
#     --max-samples 30 \
#     --start-from reactant_noise1A \
#     --stop-at-ts \
#     --t-end 3.0 \
#     --max-steps 300 \
#     --hybrid-mode \
#     --adaptive-step-sizing \
#     --higher-order-multiplier 5.0 \
#     --ts-multiplier 0.5 \
#     --wandb \
#     --wandb-project "gad-ts-search"

# --- Noise Level: 0.5Å (easier) ---
# python -m src.gad_rk45_search \
#     --max-samples 30 \
#     --start-from reactant_noise0.5A \
#     --stop-at-ts \
#     --t-end 2.0 \
#     --max-steps 200 \
#     --hybrid-mode \
#     --adaptive-step-sizing \
#     --higher-order-multiplier 3.0 \
#     --ts-multiplier 0.5 \
#     --wandb \
#     --wandb-project "gad-ts-search"

# --- Baseline comparison (no hybrid mode) ---
# Run this to compare with hybrid mode:
# python -m src.gad_rk45_search \
#     --max-samples 30 \
#     --start-from reactant_noise2A \
#     --stop-at-ts \
#     --t-end 5.0 \
#     --max-steps 500 \
#     --wandb \
#     --wandb-project "gad-ts-search"

# --- Combined: Hybrid + Kick for very noisy geometries ---
# For very noisy geometries that get completely stuck:
# python -m src.gad_rk45_search \
#     --max-samples 30 \
#     --start-from reactant_noise5A \
#     --stop-at-ts \
#     --t-end 10.0 \
#     --max-steps 1000 \
#     --hybrid-mode \
#     --adaptive-step-sizing \
#     --higher-order-multiplier 15.0 \
#     --enable-kick \
#     --kick-magnitude 0.2 \
#     --wandb \
#     --wandb-project "gad-ts-search"

echo "✅ Job completed successfully."