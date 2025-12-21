#!/bin/bash
#SBATCH -A 6101772
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32GB
#SBATCH --job-name=freq_analysis
#SBATCH --output=logs/freq_analysis_%j.out
#SBATCH --error=logs/freq_analysis_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running frequency analysis script..."

# Killarney paths
H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CKPT_PATH="/project/6101772/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

python -m src.gad_frequency_analysis \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 500 \
    --split test

echo "✅ Job completed successfully."
