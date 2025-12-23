#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32GB
#SBATCH --job-name=gad_rk45
#SBATCH --output=logs/gad_rk45_%j.out
#SBATCH --error=logs/gad_rk45_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# W&B setup (online)
export WANDB_MODE=online
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set (required for WANDB_MODE=online)." >&2
    exit 2
fi

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running GAD-RK45 (core; no hybrid mode)..."

# Killarney paths
H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CKPT_PATH="/project/6101772/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

# --- Main Experiment: Start from midpoint and search for TS ---
# This version uses the standard GAD dynamics with the RK45 solver.
# Now includes multi-stage early stopping for robust TS detection.
python -m src.runners.gad_rk45_core \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 30 \
    --start-from midpoint_rt \
    --stop-at-ts \
    --t-end 2.0 \
    --max-steps 150 \
    --min-eig-product-threshold -0.0001 \
    --wandb \
    --wandb-project "gad-ts-search"

echo "✅ Job completed successfully."
