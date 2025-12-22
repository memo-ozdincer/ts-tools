#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32GB
#SBATCH --job-name=start_geom_stats
#SBATCH --output=logs/start_geom_stats_%j.out
#SBATCH --error=logs/start_geom_stats_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Set up W&B API key for logging
export WANDB_API_KEY="ed8d3e45635321a86859f77337bdb4dfb5ceb113"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running Starting Geometry Statistics script..."

# Killarney paths
H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CKPT_PATH="/project/6101772/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

# Test different noise levels to find distribution of higher-order saddle points
python -m src.starting_geometry_stats \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 30 \
    --noise-levels 0.5 1.0 2.0 5.0 10.0 \
    --n-samples-per-noise 5 \
    --wandb \
    --wandb-project "gad-ts-search"

echo "✅ Job completed."
