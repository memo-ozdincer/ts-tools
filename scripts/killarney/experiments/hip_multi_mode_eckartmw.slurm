#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=hip_multi_mode_eckartmw_1A
#SBATCH --output=logs/hip_multi_mode_eckartmw_1A_%j.out
#SBATCH --error=logs/hip_multi_mode_eckartmw_1A_%j.err

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

export WANDB_RUN_GROUP="Experiment-multi-mode-eckartmw-hip-scine"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running HIP multi-mode escape (Eckart-MW) experiment..."

START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42

# Experiment parameters
N_STEPS=15000
MAX_SAMPLES=15

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
ESCAPE_DELTA=${ESCAPE_DELTA:-0.1}
MIN_INTERATOMIC_DIST=${MIN_INTERATOMIC_DIST:-0.5}
MAX_ESCAPE_CYCLES=${MAX_ESCAPE_CYCLES:-1000}

WANDB_NAME="experiment | multi-mode-eckartmw | HIP | ${START_FROM} | seed${NOISE_SEED} | job${SLURM_JOB_ID}"

echo ">>> HIP Euler + multi-mode escape (Eckart-MW), start-from=$START_FROM, noise-seed=$NOISE_SEED"
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
    --wandb-project "experiment-multi-mode-eckartmw" \
    --wandb-name "$WANDB_NAME"

echo "Job completed successfully."
