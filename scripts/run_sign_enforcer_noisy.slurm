#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sign_enforcer_noisy
#SBATCH --output=logs/sign_enforcer_noisy_%j.out
#SBATCH --error=logs/sign_enforcer_noisy_%j.err

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

echo ">>> Running Eigenvalue Descent with sign_enforcer on noisy starting geometries..."

# === SIGN ENFORCER EIGENVALUE DESCENT ===
# Uses gradient descent to enforce exactly one negative eigenvalue:
# - If 0 negative eigs: push λ₀ below neg_target
# - If 1 negative eig: done (loss = 0)
# - If >1 negative eigs: push all except λ₀ to be positive
#
# Key parameters:
#   --sign-neg-target: Target for λ₀ when no negatives exist (default: -0.005)
#   --sign-pos-floor: Floor for pushing extra negatives positive (default: 0.001)
#   --use-line-search: Enables line search for optimal step size
#   --adaptive-max-displacement: Reduces step size as optimization progresses

# --- Noise Level: 2Å (most challenging) ---
echo ">>> Running with 2Å noise..."
python -m src.gad_eigenvalue_descent \
    --max-samples 30 \
    --start-from reactant_noise2A \
    --loss-type sign_enforcer \
    --n-steps-opt 300 \
    --lr 0.02 \
    --sign-neg-target -0.01 \
    --sign-pos-floor 0.005 \
    --use-line-search \
    --adaptive-max-displacement \
    --initial-max-displacement 1.0 \
    --final-max-displacement 0.05 \
    --early-stop-eig-product 1e-3 \
    --wandb \
    --wandb-project "gad-noise-experiments"

# --- Noise Level: 1Å ---
echo ">>> Running with 1Å noise..."
python -m src.gad_eigenvalue_descent \
    --max-samples 30 \
    --start-from reactant_noise1A \
    --loss-type sign_enforcer \
    --n-steps-opt 200 \
    --lr 0.02 \
    --sign-neg-target -0.01 \
    --sign-pos-floor 0.005 \
    --use-line-search \
    --adaptive-max-displacement \
    --initial-max-displacement 1.0 \
    --final-max-displacement 0.05 \
    --early-stop-eig-product 1e-3 \
    --wandb \
    --wandb-project "gad-noise-experiments"

# --- Noise Level: 0.5Å (easier) ---
# echo ">>> Running with 0.5Å noise..."
# python -m src.gad_eigenvalue_descent \
#     --max-samples 30 \
#     --start-from reactant_noise0.5A \
#     --loss-type sign_enforcer \
#     --n-steps-opt 150 \
#     --lr 0.02 \
#     --sign-neg-target -0.01 \
#     --sign-pos-floor 0.005 \
#     --use-line-search \
#     --adaptive-max-displacement \
#     --early-stop-eig-product 1e-3 \
#     --wandb \
#     --wandb-project "gad-noise-experiments"

# --- Baseline: No noise (from reactant) ---
# echo ">>> Running baseline (no noise)..."
# python -m src.gad_eigenvalue_descent \
#     --max-samples 30 \
#     --start-from reactant \
#     --loss-type sign_enforcer \
#     --n-steps-opt 150 \
#     --lr 0.02 \
#     --sign-neg-target -0.01 \
#     --sign-pos-floor 0.005 \
#     --use-line-search \
#     --adaptive-max-displacement \
#     --early-stop-eig-product 1e-3 \
#     --wandb \
#     --wandb-project "gad-noise-experiments"

# --- With larger neg_target for stronger push ---
# This may help when starting far from TS
# echo ">>> Running with larger neg_target..."
# python -m src.gad_eigenvalue_descent \
#     --max-samples 30 \
#     --start-from reactant_noise2A \
#     --loss-type sign_enforcer \
#     --n-steps-opt 300 \
#     --lr 0.02 \
#     --sign-neg-target -0.05 \
#     --sign-pos-floor 0.01 \
#     --use-line-search \
#     --adaptive-max-displacement \
#     --initial-max-displacement 2.0 \
#     --final-max-displacement 0.1 \
#     --early-stop-eig-product 1e-3 \
#     --wandb \
#     --wandb-project "gad-noise-experiments"

echo "✅ Job completed successfully."
