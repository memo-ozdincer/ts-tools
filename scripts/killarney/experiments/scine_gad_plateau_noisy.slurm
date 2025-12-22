#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=scine_gad_plateau_noisy
#SBATCH --output=logs/scine_gad_plateau_noisy_%j.out
#SBATCH --error=logs/scine_gad_plateau_noisy_%j.err

# NOTE: No GPU needed for SCINE - it runs on CPU only

mkdir -p logs
source .venv/bin/activate

export WANDB_API_KEY="ed8d3e45635321a86859f77337bdb4dfb5ceb113"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running SCINE noisy GAD plateau experiments (baseline vs plateau-fix)..."

H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
OUT_DIR="$HOME/scratch/ts-tools-output"

# Pick a noise level here. Supported: reactant_noise2.0A, midpoint_rt_noise2.0A, etc.
START_FROM="midpoint_rt_noise2.0A"
NOISE_SEED=42

echo ">>> [BASELINE] SCINE Euler core (fixed dt), start-from=$START_FROM, noise-seed=$NOISE_SEED"
python -m src.runners.gad_euler_core \
    --calculator scine \
    --scine-functional DFTB0 \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 200 \
    --n-steps 150 \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --stop-at-ts \
    --dt 0.001 \
    --wandb \
    --wandb-project "gad-ts-search-scine"

echo ">>> [EXPERIMENT] SCINE Euler + neg-eig plateau dt control, start-from=$START_FROM, noise-seed=$NOISE_SEED"
python -m src.experiments.scine_gad_plateau \
    --method euler \
    --dt-control neg_eig_plateau \
    --plateau-patience 10 \
    --plateau-boost 1.5 \
    --plateau-shrink 0.5 \
    --dt-min 1e-6 \
    --dt-max 0.05 \
    --max-atom-disp 0.25 \
    --calculator scine \
    --scine-functional DFTB0 \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 200 \
    --n-steps 150 \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --stop-at-ts \
    --dt 0.001 \
    --wandb \
    --wandb-project "gad-ts-search-scine"

echo "✅ Job completed successfully."
