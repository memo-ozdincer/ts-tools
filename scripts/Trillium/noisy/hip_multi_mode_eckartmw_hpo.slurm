#!/bin/bash
#SBATCH --job-name=hip_multimode_hpo
#SBATCH --nodes=1#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/hip_multimode_hpo_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/hip_multimode_hpo_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# HIP Multi-Mode Eckart-MW HPO - Trillium
# =============================================================================
# 
# Uses Optuna with TPE sampler to optimize multi-mode GAD hyperparameters
# for HIP. Uses 1x H100 GPU (GAD is sequential per-sample).
#
# TRILLIUM NOTES:
#   - No internet access on compute nodes - W&B runs OFFLINE
#   - Job output MUST go to $SCRATCH
#   - Home/project are READ-ONLY on compute nodes
#   - Results saved to unified SQLite DB with user attributes
#
# CRASH RECOVERY: Progress saved to SQLite after each trial.
#   To resume: RESUME=1 STUDY_NAME=<previous> sbatch ...
#
# Hyperparameters being optimized:
#   - dt: Base time step [0.0005, 0.005] (log scale)
#   - dt_max: Maximum time step [0.01, 0.1] (log scale)
#   - max_atom_disp: Safety cap [0.1, 0.5]
#   - plateau_patience: Steps before boost [3, 20]
#   - plateau_boost: Boost factor [1.2, 3.0]
#   - plateau_shrink: Shrink factor [0.3, 0.7]
#   - escape_disp_threshold: Escape detection [1e-4, 1e-3] (log)
#   - escape_window: Window size [10, 50]
#   - escape_neg_vib_std: Std threshold [0.2, 1.0]
#   - escape_delta: Perturbation size [0.05, 0.3]
#   - adaptive_delta: [True, False]
#   - min_interatomic_dist: Safety [0.3, 0.7]
#
# Objective: success_rate (reaching index-1 saddle) + speed bonus
# =============================================================================

# Exit on error
set -e

# Create logs directory in scratch
mkdir -p /scratch/memoozd/ts-tools-scratch/logs

# Project paths
PROJECT_DIR="/project/rrg-aspuru/memoozd/ts-tools"
SCRATCH_DIR="/scratch/memoozd/ts-tools-scratch"

# Change to scratch for job execution (required by Trillium)
cd "$SCRATCH_DIR"

# Load modules
module purge
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6

# Activate virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

# =============================================================================
# Cache directories - use scratch space (home is read-only on compute nodes)
# =============================================================================
export TMPDIR="${SCRATCH_DIR}/tmp"
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export MPLCONFIGDIR="${SCRATCH_DIR}/.config/matplotlib"
export FONTCONFIG_PATH=/etc/fonts
export FONTCONFIG_FILE=/etc/fonts/fonts.conf

mkdir -p "$TMPDIR"
mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$MPLCONFIGDIR"

# Ensure CPU-side ops can use allocated cores (1 GPU = 24 cores)
export OMP_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export MKL_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24
export VECLIB_MAXIMUM_THREADS=24
export MPLBACKEND=Agg

# Data paths
H5_PATH="/project/rrg-aspuru/memoozd/data/transition1x.h5"
CKPT_PATH="/project/rrg-aspuru/memoozd/models/hip_v2.ckpt"
OUT_DIR="${SCRATCH_DIR}/hpo/hip_multimode_${SLURM_JOB_ID}"
DB_DIR="${SCRATCH_DIR}/dbs"

mkdir -p "$OUT_DIR"
mkdir -p "$DB_DIR"

# =============================================================================
# W&B OFFLINE MODE (Trillium has no internet on compute nodes)
# =============================================================================
export WANDB_MODE=offline
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"

export WANDB_RUN_GROUP="HPO-multi-mode-eckartmw-trillium"

# =============================================================================
# Run Configuration
# =============================================================================
N_STEPS="${N_STEPS:-4000}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
START_FROM="${START_FROM:-midpoint_rt_noise1.0A}"
NOISE_SEED="${NOISE_SEED:-42}"

# HPO configuration
N_TRIALS="${N_TRIALS:-500}"
N_STARTUP_TRIALS="${N_STARTUP_TRIALS:-5}"
STUDY_NAME="${STUDY_NAME:-hip_multi_mode_hpo_${SLURM_JOB_ID}}"

# Resume flag
RESUME_FLAG="${RESUME:-0}"

# Skip verification flag
SKIP_VERIFY="${SKIP_VERIFY:-0}"

echo "=============================================="
echo "HIP Multi-Mode Eckart-MW HPO - Trillium (H100)"
echo "=============================================="
echo "Date: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=============================================="
echo "Configuration:"
echo "  N_STEPS: $N_STEPS"
echo "  MAX_SAMPLES: $MAX_SAMPLES"
echo "  START_FROM: $START_FROM"
echo "  NOISE_SEED: $NOISE_SEED"
echo "=============================================="
echo "HPO Settings:"
echo "  N_TRIALS: $N_TRIALS"
echo "  N_STARTUP_TRIALS: $N_STARTUP_TRIALS"
echo "  STUDY_NAME: $STUDY_NAME"
echo "  RESUME: $RESUME_FLAG"
echo "  SKIP_VERIFY: $SKIP_VERIFY"
echo "  OUT_DIR: $OUT_DIR"
echo "=============================================="
echo "Hyperparameter ranges:"
echo "  dt: [0.0005, 0.005] (log)"
echo "  dt_max: [0.01, 0.1] (log)"
echo "  max_atom_disp: [0.1, 0.5]"
echo "  plateau_patience: [3, 20]"
echo "  plateau_boost: [1.2, 3.0]"
echo "  plateau_shrink: [0.3, 0.7]"
echo "  escape_disp_threshold: [1e-4, 1e-3] (log)"
echo "  escape_window: [10, 50]"
echo "  escape_neg_vib_std: [0.2, 1.0]"
echo "  escape_delta: [0.05, 0.3]"
echo "  adaptive_delta: [True, False]"
echo "  min_interatomic_dist: [0.3, 0.7]"
echo "=============================================="
echo "W&B Mode: OFFLINE (sync later with wandb sync)"
echo "=============================================="
echo ""
echo "Progress saved to SQLite after each trial."
echo "To resume: RESUME=1 STUDY_NAME=$STUDY_NAME sbatch ..."
echo "=============================================="

# Check GPU
nvidia-smi

# Build optional arguments
RESUME_ARG=""
if [[ "$RESUME_FLAG" == "1" ]]; then
    RESUME_ARG="--resume"
    echo "RESUME MODE ENABLED"
fi

SKIP_VERIFY_ARG=""
if [[ "$SKIP_VERIFY" == "1" ]]; then
    SKIP_VERIFY_ARG="--skip-verification"
    echo "SKIPPING VERIFICATION"
fi

# Run HPO
python -m src.noisy.hip_multi_mode_eckartmw_hpo \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$DB_DIR" \
    --n-steps "$N_STEPS" \
    --max-samples "$MAX_SAMPLES" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --n-trials "$N_TRIALS" \
    --n-startup-trials "$N_STARTUP_TRIALS" \
    --study-name "$STUDY_NAME" \
    $RESUME_ARG \
    $SKIP_VERIFY_ARG \
    --wandb \
    --wandb-project "hip-multi-mode-hpo"

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "HIP Multi-Mode HPO completed successfully at $(date)"
else
    echo "HIP Multi-Mode HPO exited with code $EXIT_CODE at $(date)"
    echo "Check logs and resume with: RESUME=1 STUDY_NAME=$STUDY_NAME sbatch ..."
fi
echo "=============================================="
echo "Results saved to: $OUT_DIR"
echo ""
echo "SQLite database: $DB_DIR/${STUDY_NAME}.db"
echo "To analyze: python scripts/analyze_hpo_results.py"
echo ""
echo "W&B offline run saved to: $WANDB_DIR"
echo "To sync later (from login node with internet):"
echo "  wandb sync $WANDB_DIR/offline-run-*"
echo "=============================================="

exit $EXIT_CODE
