#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32GB
#SBATCH --job-name=gad_euler_noisy
#SBATCH --output=logs/gad_euler_noisy_%j.out
#SBATCH --error=logs/gad_euler_noisy_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Set up W&B API key for logging
export WANDB_API_KEY="ed8d3e45635321a86859f77337bdb4dfb5ceb113"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running GAD-Euler (core) with noisy starting geometries..."

# Killarney paths
H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CKPT_PATH="/project/6101772/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

# === CLEAN Euler-GAD (no hybrid, no BFGS/minimization fallback) ===
echo ">>> Running Euler-GAD from reactant_noise2A"
python -m src.runners.gad_euler_core \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 30 \
    --start-from reactant_noise2A \
    --stop-at-ts \
    --n-steps 1000 \
    --dt 0.005 \
    --wandb \
    --wandb-project "gad-noise-experiments"

echo "✅ Job completed successfully."
