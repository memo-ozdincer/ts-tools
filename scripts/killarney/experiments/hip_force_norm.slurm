#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=hip_force_norm_1A
#SBATCH --output=logs/hip_force_norm_1A_%j.out
#SBATCH --error=logs/hip_force_norm_1A_%j.err

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

export WANDB_RUN_GROUP="Experiment-force-norm-hip"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running HIP force normalization experiment..."

START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42

# Experiment parameters
N_STEPS=15000
MAX_SAMPLES=100

# Force normalization parameters
# force_norm_threshold: Force magnitude threshold (eV/A). Normalize when |f| < this AND λ₁*λ₂ > 0
# escape_disp_threshold: Mean displacement threshold (A) for plateau detection (logging only)
# escape_window: Number of recent steps for plateau detection
# escape_neg_vib_std: Max std(neg_vib) over window (stable saddle index)
FORCE_NORM_THRESHOLD=${FORCE_NORM_THRESHOLD:-1.0}
ESCAPE_DISP_THRESHOLD=${ESCAPE_DISP_THRESHOLD:-5e-4}
ESCAPE_WINDOW=${ESCAPE_WINDOW:-20}
ESCAPE_NEG_VIB_STD=${ESCAPE_NEG_VIB_STD:-0.5}

WANDB_NAME="experiment | force-norm | HIP | ${START_FROM} | fnorm${FORCE_NORM_THRESHOLD} | seed${NOISE_SEED} | job${SLURM_JOB_ID}"

echo ">>> HIP Euler + force normalization, start-from=$START_FROM, noise-seed=$NOISE_SEED"
python -m src.experiments.multi_mode_force_norm \
    --calculator hip \
    --method euler \
    --dt-control neg_eig_plateau \
    --plateau-patience 5 \
    --plateau-boost 2.0 \
    --plateau-shrink 0.5 \
    --dt-min 1e-6 \
    --dt-max 0.05 \
    --max-atom-disp 0.25 \
    --force-norm-threshold "$FORCE_NORM_THRESHOLD" \
    --escape-disp-threshold "$ESCAPE_DISP_THRESHOLD" \
    --escape-window "$ESCAPE_WINDOW" \
    --escape-neg-vib-std "$ESCAPE_NEG_VIB_STD" \
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
    --wandb-project "experiment-force-norm" \
    --wandb-name "$WANDB_NAME"

echo "Job completed successfully."
