#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=scine_multi_mode_1A
#SBATCH --output=logs/scine_multi_mode_1A_%j.out
#SBATCH --error=logs/scine_multi_mode_1A_%j.err

# NOTE: No GPU needed for SCINE - it runs on CPU only

mkdir -p logs
source .venv/bin/activate

H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
OUT_DIR="$HOME/scratch/ts-tools-output"

# W&B setup
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"
export WANDB_MODE=online
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set (required for WANDB_MODE=online)." >&2
    exit 2
fi

export WANDB_RUN_GROUP="Experiment-multi-mode-hip-scine"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running SCINE multi-mode escape experiment..."

START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42

# Experiment parameters
N_STEPS=1500
MAX_SAMPLES=10

# Multi-mode escape parameters (displacement-based detection)
# escape_disp_threshold: Mean displacement threshold (A) for plateau detection
# escape_window: Number of recent steps to average for plateau detection
# escape_neg_vib_std: Max std(neg_vib) over window (stable saddle index)
# escape_delta: Base perturbation magnitude (Angstrom) along v2
# adaptive_delta: Scale delta by 1/sqrt(|lambda2|) for stronger curvature
# max_escape_cycles: Max attempts to escape high-index saddles
ESCAPE_DISP_THRESHOLD=${ESCAPE_DISP_THRESHOLD:-5e-4}
ESCAPE_WINDOW=${ESCAPE_WINDOW:-20}
ESCAPE_NEG_VIB_STD=${ESCAPE_NEG_VIB_STD:-0.5}
ESCAPE_DELTA=${ESCAPE_DELTA:-0.2}
MAX_ESCAPE_CYCLES=${MAX_ESCAPE_CYCLES:-10}

WANDB_NAME="experiment | multi-mode | SCINE | ${START_FROM} | seed${NOISE_SEED} | job${SLURM_JOB_ID}"

echo ">>> SCINE Euler + multi-mode escape, start-from=$START_FROM, noise-seed=$NOISE_SEED"
python -m src.experiments.scine_multi_mode \
    --method euler \
    --dt-control neg_eig_plateau \
    --plateau-patience 5 \
    --plateau-boost 2.0 \
    --plateau-shrink 0.5 \
    --dt-min 1e-6 \
    --dt-max 0.05 \
    --max-atom-disp 0.25 \
    --escape-disp-threshold "$ESCAPE_DISP_THRESHOLD" \
    --escape-window "$ESCAPE_WINDOW" \
    --escape-neg-vib-std "$ESCAPE_NEG_VIB_STD" \
    --escape-delta "$ESCAPE_DELTA" \
    --adaptive-delta \
    --max-escape-cycles "$MAX_ESCAPE_CYCLES" \
    --scine-functional DFTB0 \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --n-steps "$N_STEPS" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --stop-at-ts \
    --dt 0.001 \
    --wandb \
    --wandb-project "experiment-multi-mode" \
    --wandb-name "$WANDB_NAME"

echo "Job completed successfully."
