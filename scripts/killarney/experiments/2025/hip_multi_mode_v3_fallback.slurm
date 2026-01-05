#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=hip_v3_fallback
#SBATCH --output=logs/hip_v3_fallback_%j.out
#SBATCH --error=logs/hip_v3_fallback_%j.err

# HIP Multi-Mode Escape with V3+ Fallback Kicks
# When v2 kicks become ineffective, automatically escalate to v3, v4, etc.

mkdir -p logs
source .venv/bin/activate

# Ensure CPU-side ops can use the allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CKPT_PATH="/project/6101772/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

# W&B setup
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"
export WANDB_MODE=online
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set (required for WANDB_MODE=online)." >&2
    exit 2
fi

export WANDB_RUN_GROUP="Experiment-v3-fallback-hip"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running HIP multi-mode escape with V3+ fallback kicks..."

START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42

# Experiment parameters
N_STEPS=15000
MAX_SAMPLES=100

# Multi-mode escape parameters (displacement-based detection)
ESCAPE_DISP_THRESHOLD=${ESCAPE_DISP_THRESHOLD:-5e-4}
ESCAPE_WINDOW=${ESCAPE_WINDOW:-20}
ESCAPE_NEG_VIB_STD=${ESCAPE_NEG_VIB_STD:-0.5}
ESCAPE_DELTA=${ESCAPE_DELTA:-0.1}
MIN_INTERATOMIC_DIST=${MIN_INTERATOMIC_DIST:-0.5}
MAX_ESCAPE_CYCLES=${MAX_ESCAPE_CYCLES:-1000}

# NEW: Maximum escape mode (v3+ fallback kicks)
MAX_ESCAPE_MODE=${MAX_ESCAPE_MODE:-6}

WANDB_NAME="experiment | v3-fallback | HIP | ${START_FROM} | seed${NOISE_SEED} | max-mode${MAX_ESCAPE_MODE} | job${SLURM_JOB_ID}"

echo ">>> HIP Euler + multi-mode escape with V3+ fallback, start-from=$START_FROM, noise-seed=$NOISE_SEED, max-escape-mode=$MAX_ESCAPE_MODE"
python -m src.experiments.hip_multi_mode_eckartmw \
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
    --hip-vib-mode proj_tol \
    --hip-rigid-tol 1e-6 \
    --hip-eigh-device cpu \
    --escape-neg-vib-std "$ESCAPE_NEG_VIB_STD" \
    --escape-delta "$ESCAPE_DELTA" \
    --adaptive-delta \
    --min-interatomic-dist "$MIN_INTERATOMIC_DIST" \
    --max-escape-cycles "$MAX_ESCAPE_CYCLES" \
    --max-escape-mode "$MAX_ESCAPE_MODE" \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --n-steps "$N_STEPS" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --stop-at-ts \
    --dt 0.001 \
    --wandb \
    --wandb-project "experiment-v3-fallback" \
    --wandb-name "$WANDB_NAME"

echo "Job completed successfully."
