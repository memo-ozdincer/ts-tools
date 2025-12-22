#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32GB
#SBATCH --job-name=hip_gad_plateau_midpoint
#SBATCH --output=logs/hip_gad_plateau_midpoint_%j.out
#SBATCH --error=logs/hip_gad_plateau_midpoint_%j.err

mkdir -p logs
source .venv/bin/activate

export WANDB_API_KEY="ed8d3e45635321a86859f77337bdb4dfb5ceb113"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running HIP midpoint_rt plateau experiments (baseline vs plateau-fix)..."

H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CKPT_PATH="/project/6101772/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

START_FROM="midpoint_rt"

echo ">>> [BASELINE] HIP Euler core (fixed dt), start-from=$START_FROM"
python -m src.runners.gad_euler_core \
    --calculator hip \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 200 \
    --n-steps 150 \
    --start-from "$START_FROM" \
    --stop-at-ts \
    --dt 0.001 \
    --wandb \
    --wandb-project "gad-ts-search"

echo ">>> [EXPERIMENT] HIP Euler + neg-eig plateau dt control, start-from=$START_FROM"
python -m src.experiments.scine_gad_plateau \
    --method euler \
    --dt-control neg_eig_plateau \
    --plateau-patience 10 \
    --plateau-boost 1.5 \
    --plateau-shrink 0.5 \
    --dt-min 1e-6 \
    --dt-max 0.05 \
    --max-atom-disp 0.25 \
    --calculator hip \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 200 \
    --n-steps 150 \
    --start-from "$START_FROM" \
    --stop-at-ts \
    --dt 0.001 \
    --wandb \
    --wandb-project "gad-ts-search"

echo "✅ Job completed successfully."
