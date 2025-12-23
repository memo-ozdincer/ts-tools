#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32GB
#SBATCH --job-name=hip_plateau_noisy_1A_30
#SBATCH --output=logs/hip_plateau_noisy_1A_30_%j.out
#SBATCH --error=logs/hip_plateau_noisy_1A_30_%j.err

mkdir -p logs
source .venv/bin/activate

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running HIP noisy plateau experiment (plateau+floor only)..."

H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CKPT_PATH="/project/6101772/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42

echo ">>> HIP Euler + neg-eig plateau dt control (+ floor), start-from=$START_FROM, noise-seed=$NOISE_SEED"
python -m src.experiments.hip_gad_plateau \
    --method euler \
    --dt-control neg_eig_plateau \
    --min-mean-disp 0.0005 \
    --plateau-patience 10 \
    --plateau-boost 1.5 \
    --plateau-shrink 0.5 \
    --dt-min 1e-6 \
    --dt-max 0.05 \
    --max-atom-disp 0.25 \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 30 \
    --n-steps 150 \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --stop-at-ts \
    --dt 0.001 \
    --wandb \
    --wandb-project "gad-ts-search"

echo "✅ Job completed successfully."
