#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32GB
#SBATCH --job-name=gad_rk45_noisy
#SBATCH --output=logs/gad_rk45_noisy_%j.out
#SBATCH --error=logs/gad_rk45_noisy_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Set up W&B API key for logging
export WANDB_API_KEY="ed8d3e45635321a86859f77337bdb4dfb5ceb113"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running GAD-RK45 Search with HYBRID MODE for noisy geometries..."

# Killarney paths
H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CKPT_PATH="/project/6101772/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

# === HYBRID FORCE-GAD MODE ===
# This mode is designed for noisy starting geometries (noise levels: 0.5, 1, 2 Å).
#
# Key innovation: Saddle-order-aware dynamics switching:
#   - At higher-order saddles (2+ negative eigenvalues): Use FORCE DESCENT
#     to escape unstable high-energy regions where noisy geometries often land
#   - At order-1 (TS candidate): Use GAD for precise convergence
#   - At order-0 (minimum): Use standard GAD to climb toward TS

# --- Noise Level: 2Å (most challenging) ---
python -m src.gad_rk45_search \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
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

echo "✅ Job completed successfully."
