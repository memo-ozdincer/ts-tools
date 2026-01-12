#!/bin/bash
#SBATCH --job-name=hip_sella_hpo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/hip_sella_hpo_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/hip_sella_hpo_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# HIP Sella Bayesian Hyperparameter Optimization - Trillium
# =============================================================================
# 
# Uses Optuna with TPE sampler to optimize Sella hyperparameters for HIP.
# Uses 1x H100 GPU (GAD is sequential per-sample, more GPUs don't help).
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
#   - delta0: Initial trust radius [0.15, 0.8] (log scale)
#   - rho_dec: Trust radius decrease threshold [15, 80]
#   - rho_inc: Trust radius increase threshold [1.01, 1.1]
#   - sigma_dec: Trust radius decrease factor [0.75, 0.95]
#   - sigma_inc: Trust radius increase factor [1.1, 1.8]
#   - fmax: Force convergence threshold [1e-4, 1e-2] (log scale)
#   - apply_eckart: Whether to Eckart project Hessians [True, False]
#
# Objective (priority order):
#   1. eigenvalue_ts_rate (exactly 1 neg eigenvalue) - weight 1.0
#   2. speed (fewer steps) - weight 0.01
#   3. sella_convergence_rate - weight 0.001
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
OUT_DIR="${SCRATCH_DIR}/hpo/hip_sella_${SLURM_JOB_ID}"
DB_DIR="${SCRATCH_DIR}/dbs"

mkdir -p "$OUT_DIR"
mkdir -p "$DB_DIR"

# =============================================================================
# W&B OFFLINE MODE (Trillium has no internet on compute nodes)
# =============================================================================
export WANDB_MODE=offline
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"

export WANDB_RUN_GROUP="hip-sella-hpo-trillium"

# =============================================================================
# HPO Configuration
# =============================================================================
N_TRIALS="${N_TRIALS:-50}"
MAX_STEPS="${MAX_STEPS:-100}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
START_FROM="${START_FROM:-midpoint_rt_noise1.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
OPTUNA_SEED="${OPTUNA_SEED:-42}"

# Pre-screening configuration
PRESCREEN="${PRESCREEN:-1}"
PRESCREEN_SAMPLES="${PRESCREEN_SAMPLES:-100}"
HARD_SAMPLES_FILE="${HARD_SAMPLES_FILE:-}"

# Study name for resume capability
STUDY_NAME="${STUDY_NAME:-hip_sella_hpo_job${SLURM_JOB_ID}}"
RESUME_FLAG="${RESUME:-0}"

WANDB_NAME="hip-sella-hpo_${N_TRIALS}trials_${START_FROM}_job${SLURM_JOB_ID}"

echo "=============================================="
echo "HIP Sella Bayesian HPO - Trillium (H100)"
echo "=============================================="
echo "Date: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=============================================="
echo "Configuration:"
echo "  N_TRIALS: $N_TRIALS"
echo "  MAX_STEPS: $MAX_STEPS"
echo "  MAX_SAMPLES: $MAX_SAMPLES"
echo "  START_FROM: $START_FROM"
echo "  NOISE_SEED: $NOISE_SEED"
echo "  OPTUNA_SEED: $OPTUNA_SEED"
echo "  STUDY_NAME: $STUDY_NAME"
echo "  OUT_DIR: $OUT_DIR"
echo "=============================================="
echo "Pre-screening:"
echo "  PRESCREEN: $PRESCREEN"
echo "  PRESCREEN_SAMPLES: $PRESCREEN_SAMPLES"
echo "  HARD_SAMPLES_FILE: ${HARD_SAMPLES_FILE:-<none>}"
echo "=============================================="
echo "Hyperparameter ranges:"
echo "  delta0: [0.15, 0.8] (log)"
echo "  rho_dec: [15, 80]"
echo "  rho_inc: [1.01, 1.1]"
echo "  sigma_dec: [0.75, 0.95]"
echo "  sigma_inc: [1.1, 1.8]"
echo "  fmax: [1e-4, 1e-2] (log)"
echo "  apply_eckart: [True, False]"
echo "=============================================="
echo "Fixed parameters:"
echo "  gamma: 0.0"
echo "  internal: True (always)"
echo "  use_exact_hessian: True (HIP direct)"
echo "  diag_every_n: 1"
echo "  prune_after_n: 10"
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

PRESCREEN_ARG=""
if [[ "$PRESCREEN" == "1" ]]; then
    PRESCREEN_ARG="--prescreen --prescreen-samples $PRESCREEN_SAMPLES"
    echo "PRESCREEN MODE ENABLED"
fi

HARD_SAMPLES_ARG=""
if [[ -n "$HARD_SAMPLES_FILE" ]]; then
    HARD_SAMPLES_ARG="--hard-samples-file $HARD_SAMPLES_FILE"
    PRESCREEN_ARG=""
    echo "LOADING HARD SAMPLES FROM: $HARD_SAMPLES_FILE"
fi

# Run HPO
python -m src.experiments.Sella.hip_sella_hpo \
    --n-trials "$N_TRIALS" \
    --max-steps "$MAX_STEPS" \
    --max-samples "$MAX_SAMPLES" \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$DB_DIR" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --optuna-seed "$OPTUNA_SEED" \
    --study-name "$STUDY_NAME" \
    $RESUME_ARG \
    $PRESCREEN_ARG \
    $HARD_SAMPLES_ARG \
    --verbose \
    --wandb \
    --wandb-project "sella-hpo" \
    --wandb-entity "memo-ozdincer-university-of-toronto" \
    --wandb-name "$WANDB_NAME"

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "HIP Sella HPO completed successfully at $(date)"
else
    echo "HIP Sella HPO exited with code $EXIT_CODE at $(date)"
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
