#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/aip-aspuru/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32GB
#SBATCH --job-name=eigen_opt
#SBATCH --output=logs/eig_desc_%j.out
#SBATCH --error=logs/eig_desc_%j.err

# Activate virtual environment
source .venv/bin/activate

# Set up W&B API key for logging
export WANDB_API_KEY="ed8d3e45635321a86859f77337bdb4dfb5ceb113"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running Eigenvalue Optimization Experiments..."

# Killarney paths
H5_PATH="$HOME/projects/aip-aspuru/memoozd/data/transition1x.h5"
CKPT_PATH="$HOME/projects/aip-aspuru/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

# === EXPERIMENT: EIGPRODUCT BFGS (Pure Optimization) ===
# Tests pure L-BFGS-B optimization of the eigenvalue product λ₀*λ₁
# This helps isolate whether the energy explosion is due to the optimizer or the landscape

echo ">>> Running Eigprod BFGS (Pure Optimization)"
python -m src.gad_eigenvalue_descent \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 30 \
    --start-from reactant_noise2A \
    --loss-type eig_product \
    --use-bfgs \
    --bfgs-maxiter 100 \
    --bfgs-gtol 1e-5 \
    --bfgs-max-step 1.0 \
    --wandb \
    --wandb-project "gad-noise-experiments"

echo "✅ Job completed successfully."
