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

# === EIGPRODUCT MINIMIZATION EXPERIMENTS ===
# Testing two approaches to escape weird geometries when GAD stalls:
#   1. Larger step sizes (gradient descent with bigger max_step)
#   2. L-BFGS-B optimization (more robust, can take larger steps)

# --- EXPERIMENT 1: Larger step size (gradient descent) ---
# Increases max_step from 0.1 to 0.5 Å to escape weird geometries
# echo ">>> Running Experiment 1: Larger step size (gradient descent)"
# python -m src.gad_gad_euler_rmsd \
#     --max-samples 30 \
#     --start-from reactant_noise2A \
#     --enable-minimization-fallback \
#     --eig-descent-lr 0.01 \
#     --eig-descent-max-step 0.5 \
#     --stop-at-ts \
#     --n-steps 1000 \
#     --dt 0.005 \
#     --wandb \
#     --wandb-project "gad-noise-experiments"

# --- EXPERIMENT 2: L-BFGS-B optimization for Eigprod fallback ---
# Uses scipy L-BFGS-B for more robust eigproduct minimization when GAD stalls
# echo ">>> Running Experiment 2: Eigprod BFGS (fallback mode)"
# python -m src.gad_gad_euler_rmsd \
#     --max-samples 30 \
#     --start-from reactant_noise2A \
#     --enable-minimization-fallback \
#     --use-bfgs \
#     --bfgs-maxiter 50 \
#     --bfgs-gtol 1e-5 \
#     --bfgs-max-step 0.5 \
#     --stop-at-ts \
#     --n-steps 1000 \
#     --dt 0.005 \
#     --wandb \
#     --wandb-project "gad-noise-experiments"

# --- EXPERIMENT 3: GAD BFGS (standalone saddle point optimization) ---
# Uses L-BFGS-B with GAD vector as gradient to directly find saddle points
echo ">>> Running Experiment 3: GAD BFGS (standalone)"
python -m src.gad_gad_euler_rmsd \
    --max-samples 30 \
    --start-from reactant_noise2A \
    --use-gad-bfgs \
    --gad-bfgs-maxiter 100 \
    --gad-bfgs-gtol 1e-5 \
    --gad-bfgs-max-step 1.0 \
    --wandb \
    --wandb-project "gad-noise-experiments"

# --- BASELINE: Original settings (for comparison) ---
# python -m src.gad_gad_euler_rmsd \
#     --max-samples 30 \
#     --start-from reactant_noise2A \
#     --enable-minimization-fallback \
#     --eig-descent-lr 0.01 \
#     --eig-descent-max-step 0.1 \
#     --stop-at-ts \
#     --n-steps 500 \
#     --dt 0.005 \
#     --wandb \
#     --wandb-project "gad-noise-experiments"
    

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
