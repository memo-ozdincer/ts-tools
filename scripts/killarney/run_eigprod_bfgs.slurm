#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/aip-aspuru/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32GB
#SBATCH --job-name=eigprod_bfgs
#SBATCH --output=/scratch/memoozd/ts-tools-output/slurm-%j.txt
#SBATCH --error=/scratch/memoozd/ts-tools-output/slurm-%j.txt

# Activate virtual environment
source .venv/bin/activate

# Set up W&B API key for logging
export WANDB_API_KEY="ed8d3e45635321a86859f77337bdb4dfb5ceb113"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running Eigprod BFGS Optimization..."

# Killarney paths
H5_PATH="$HOME/projects/aip-aspuru/memoozd/data/transition1x.h5"
CKPT_PATH="$HOME/projects/aip-aspuru/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

# === EIGPROD BFGS OPTIMIZATION ===
# Standalone eigenvalue product minimization using L-BFGS-B.
# Directly minimizes λ₀*λ₁ to find transition states (where product < 0).

echo ">>> Running Eigprod BFGS (minimize λ₀*λ₁)"
python -m src.gad_eigenvalue_descent \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 30 \
    --start-from reactant_noise2A \
    --use-bfgs \
    --bfgs-maxiter 100 \
    --bfgs-gtol 1e-5 \
    --bfgs-max-step 1.0 \
    --loss-type eig_product \
    --wandb \
    --wandb-project "gad-noise-experiments"

echo "✅ Job completed successfully."
