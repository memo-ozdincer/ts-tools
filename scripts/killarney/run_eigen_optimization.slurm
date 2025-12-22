#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32GB
#SBATCH --job-name=eigen_opt
#SBATCH --output=logs/eig_desc_%j.out
#SBATCH --error=logs/eig_desc_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Set up W&B API key for logging
export WANDB_API_KEY="ed8d3e45635321a86859f77337bdb4dfb5ceb113"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running Eigenvalue Optimization Experiments..."

# Killarney paths
H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CKPT_PATH="/project/6101772/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

# === EXPERIMENT 1: Linear Eigproduct Descent ===
# Tests gradient descent optimization of the eigenvalue product λ₀*λ₁

echo ">>> Running Linear Eigproduct Descent"
python -m src.gad_eigenvalue_descent \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 30 \
    --start-from reactant_noise2A \
    --loss-type eig_product \
    --n-steps-opt 200 \
    --lr 0.01 \
    --use-line-search \
    --adaptive-max-displacement \
    --initial-max-displacement 2.0 \
    --final-max-displacement 0.1 \
    --wandb \
    --wandb-project "gad-noise-experiments"

# === EXPERIMENT 2: Sign Enforcer ===
# echo ">>> Running Sign Enforcer"
# python -m src.gad_eigenvalue_descent \
#     --h5-path "$H5_PATH" \
#     --checkpoint-path "$CKPT_PATH" \
#     --out-dir "$OUT_DIR" \
#     --max-samples 30 \
#     --start-from reactant_noise2A \
#     --loss-type sign_enforcer \
#     --sign-neg-target -5e-3 \
#     --sign-pos-floor 1e-3 \
#     --n-steps-opt 200 \
#     --lr 0.01 \
#     --use-line-search \
#     --wandb \
#     --wandb-project "gad-noise-experiments"

echo "✅ Job completed successfully."
