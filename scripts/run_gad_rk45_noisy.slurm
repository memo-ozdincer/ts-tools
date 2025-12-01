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
export WANDB_API_KEY="ed8d3e45635321a86859f77337bdb4dfb5ceb113"

echo ">>> Running GAD-RK45 Search with ADAPTIVE STEP SIZING for noisy geometries..."

# === ADAPTIVE STEP SIZING FOR NOISY GEOMETRIES ===
# Always uses GAD dynamics, but scales step size based on saddle order:
#   - Higher-order saddles (2+ negative eigenvalues): LARGER steps to escape faster
#   - Order-1 (TS candidate): SMALLER steps for precise convergence  
#   - Order-0 (minimum): Normal steps
#
# Parameters:
#   --adaptive-step-sizing: Enable saddle-order-aware step scaling
#   --higher-order-multiplier: Scale factor for higher-order saddles (default: 5.0)
#                             Order-2 uses 5×, Order-3 uses 10×, etc.
#   --ts-multiplier: Scale factor near TS for refinement (default: 0.5)

# --- Noise Level: 2Å (most challenging) ---
python -m src.gad_rk45_search \
    --max-samples 30 \
    --start-from reactant_noise2A \
    --stop-at-ts \
    --t-end 5.0 \
    --max-steps 500 \
    --adaptive-step-sizing \
    --higher-order-multiplier 10.0 \
    --ts-multiplier 0.5 \
    --wandb \
    --wandb-project "gad-noise-experiments"

# --- Noise Level: 1Å ---
# python -m src.gad_rk45_search \
#     --max-samples 30 \
#     --start-from reactant_noise1A \
#     --stop-at-ts \
#     --t-end 3.0 \
#     --max-steps 300 \
#     --adaptive-step-sizing \
#     --higher-order-multiplier 5.0 \
#     --ts-multiplier 0.5 \
#     --wandb \
#     --wandb-project "gad-noise-experiments"

# --- Noise Level: 0.5Å (easier) ---
# python -m src.gad_rk45_search \
#     --max-samples 30 \
#     --start-from reactant_noise0.5A \
#     --stop-at-ts \
#     --t-end 2.0 \
#     --max-steps 200 \
#     --adaptive-step-sizing \
#     --higher-order-multiplier 3.0 \
#     --ts-multiplier 0.5 \
#     --wandb \
#     --wandb-project "gad-noise-experiments"

# --- Baseline comparison (no adaptive step sizing) ---
# Run this to compare with adaptive step sizing:
# python -m src.gad_rk45_search \
#     --max-samples 30 \
#     --start-from reactant_noise2A \
#     --stop-at-ts \
#     --t-end 5.0 \
#     --max-steps 500 \
#     --wandb \
#     --wandb-project "gad-noise-experiments"

# --- Combined: Adaptive + Kick for very noisy geometries ---
# For very noisy geometries that get completely stuck:
# python -m src.gad_rk45_search \
#     --max-samples 30 \
#     --start-from reactant_noise5A \
#     --stop-at-ts \
#     --t-end 10.0 \
#     --max-steps 1000 \
#     --adaptive-step-sizing \
#     --higher-order-multiplier 15.0 \
#     --enable-kick \
#     --kick-magnitude 0.2 \
#     --wandb \
#     --wandb-project "gad-noise-experiments"

echo "✅ Job completed successfully."

