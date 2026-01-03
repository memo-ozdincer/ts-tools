#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=scine_dynamic_gad
#SBATCH --output=logs/scine_dynamic_gad_%j.out
#SBATCH --error=logs/scine_dynamic_gad_%j.err

# SCINE Dynamic GAD Mode Switching
# Instead of kicks, dynamically switch which eigenvector GAD follows:
# - Escalate (v1 -> v2 -> v3) when stuck (tiny displacement)
# - De-escalate (v3 -> v2 -> v1) when progress resumes
# Note: SCINE runs on CPU only

mkdir -p logs
source .venv/bin/activate

# Ensure CPU-side ops can use the allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

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

export WANDB_RUN_GROUP="Experiment-dynamic-gad-mode-scine"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running SCINE Dynamic GAD Mode Switching experiment..."

START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42

# Experiment parameters
N_STEPS=15000
MAX_SAMPLES=100

# Mode switching parameters
MODE_SWITCH_WINDOW=${MODE_SWITCH_WINDOW:-20}
MODE_SWITCH_DISP_THRESHOLD=${MODE_SWITCH_DISP_THRESHOLD:-5e-4}
MODE_SWITCH_NEG_VIB_STD=${MODE_SWITCH_NEG_VIB_STD:-0.5}
MAX_GAD_MODE=${MAX_GAD_MODE:-6}
DEESCALATE_FACTOR=${DEESCALATE_FACTOR:-2.0}
MIN_STEPS_PER_MODE=${MIN_STEPS_PER_MODE:-50}

WANDB_NAME="experiment | dynamic-gad-mode | SCINE | ${START_FROM} | seed${NOISE_SEED} | max-mode${MAX_GAD_MODE} | job${SLURM_JOB_ID}"

echo ">>> SCINE Dynamic GAD Mode Switching, start-from=$START_FROM, noise-seed=$NOISE_SEED, max-gad-mode=$MAX_GAD_MODE"
python -m src.experiments.scine_dynamic_gad_mode \
    --method euler \
    --dt-control neg_eig_plateau \
    --plateau-patience 5 \
    --plateau-boost 2.0 \
    --plateau-shrink 0.5 \
    --dt-min 1e-6 \
    --dt-max 0.05 \
    --max-atom-disp 0.25 \
    --mode-switch-window "$MODE_SWITCH_WINDOW" \
    --mode-switch-disp-threshold "$MODE_SWITCH_DISP_THRESHOLD" \
    --mode-switch-neg-vib-std "$MODE_SWITCH_NEG_VIB_STD" \
    --max-gad-mode "$MAX_GAD_MODE" \
    --deescalate-factor "$DEESCALATE_FACTOR" \
    --min-steps-per-mode "$MIN_STEPS_PER_MODE" \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --n-steps "$N_STEPS" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --stop-at-ts \
    --dt 0.001 \
    --wandb \
    --wandb-project "experiment-dynamic-gad-mode" \
    --wandb-name "$WANDB_NAME"

echo "Job completed successfully."
