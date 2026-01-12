#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --job-name=hip_sella_hpo
#SBATCH --output=logs/hip_sella_hpo_%j.out
#SBATCH --error=logs/hip_sella_hpo_%j.err

# =============================================================================
# HIP Sella Bayesian Hyperparameter Optimization
# =============================================================================
# 
# Uses Optuna with TPE sampler to optimize Sella hyperparameters for HIP.
# Uses 1x H100 GPU (GAD is sequential per-sample, more GPUs don't help).
#
# KEY FEATURES:
#   - PRE-SCREENING: Runs default params first to find "hard" samples that
#     don't converge. HPO then focuses only on these hard samples, making
#     optimization more efficient (easy samples always converge anyway).
#   - GOOD PRIORS: Seeds TPE with known-good configs (delta0=0.5, rho_dec=50, sigma_dec=0.9)
#   - CRASH RECOVERY: Progress saved to SQLite after each trial.
#
# CRASH RECOVERY / RESUME:
#   To resume: RESUME=--resume STUDY_NAME=<previous_study_name> sbatch ...
#
# SKIP PRESCREENING (use saved hard samples):
#   HARD_SAMPLES_FILE=/path/to/hard_sample_indices.json PRESCREEN=0 sbatch ...
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
# Fixed parameters:
#   - gamma: 0.0
#   - max_steps: 100
#   - samples per trial: 30 (or fewer if hard samples < 30)
#   - internal: True (ALWAYS use Sella's internal coordinates)
#   - use_exact_hessian: True (HIP direct Hessian, no autograd)
#   - diag_every_n: 1
#   - prune_after_n: 10 (prune bad trials after 10 samples)
#
# Objective (priority order):
#   1. eigenvalue_ts_rate (exactly 1 neg eigenvalue) - weight 1.0
#   2. speed (fewer steps) - weight 0.01
#   3. sella_convergence_rate - weight 0.001
#
# Time estimate: Prescreening ~30min, then ~50 trials × 30 samples × ~15 sec
# =============================================================================

mkdir -p logs
source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MPLBACKEND=Agg

H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CKPT_PATH="/project/6101772/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output/hip_hpo_${SLURM_JOB_ID}"

export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"
export WANDB_MODE=online
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set." >&2
    exit 2
fi

export WANDB_RUN_GROUP="hip-sella-hpo-bayesian"

# HPO configuration
N_TRIALS="${N_TRIALS:-50}"
MAX_STEPS="${MAX_STEPS:-100}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
START_FROM="${START_FROM:-midpoint_rt_noise1.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
OPTUNA_SEED="${OPTUNA_SEED:-42}"

# Pre-screening: Find hard samples first, then focus HPO on them
# Set PRESCREEN=1 to enable, or provide HARD_SAMPLES_FILE to skip prescreening
PRESCREEN="${PRESCREEN:-1}"  # Enabled by default
PRESCREEN_SAMPLES="${PRESCREEN_SAMPLES:-100}"  # Prescreen 100 samples, HPO uses the hard ones
HARD_SAMPLES_FILE="${HARD_SAMPLES_FILE:-}"  # Optional: load hard samples from file

# Study name for resume capability
STUDY_NAME="${STUDY_NAME:-hip_hpo_job${SLURM_JOB_ID}}"
RESUME_FLAG="${RESUME:-}"

WANDB_NAME="hip-sella-hpo_${N_TRIALS}trials_${START_FROM}_job${SLURM_JOB_ID}"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo "=============================================="
echo "HIP Sella Bayesian HPO (1x H100)"
echo "=============================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "N_TRIALS: $N_TRIALS"
echo "MAX_STEPS: $MAX_STEPS"
echo "MAX_SAMPLES: $MAX_SAMPLES (per trial, from hard samples)"
echo "START_FROM: $START_FROM"
echo "NOISE_SEED: $NOISE_SEED"
echo "OPTUNA_SEED: $OPTUNA_SEED"
echo "STUDY_NAME: $STUDY_NAME"
echo "OUT_DIR: $OUT_DIR"
echo "=============================================="
echo "Pre-screening config:"
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
echo "Objective weights:"
echo "  eigenvalue_ts_rate: 1.0 (PRIMARY)"
echo "  speed_bonus: 0.01 (SECONDARY)"
echo "  sella_convergence: 0.001 (TERTIARY)"
echo "=============================================="
echo ""
echo "Progress saved to SQLite after each trial."
echo "To resume: set RESUME=--resume STUDY_NAME=$STUDY_NAME"
echo "=============================================="

# Build optional arguments
RESUME_ARG=""
if [[ -n "$RESUME_FLAG" ]]; then
    RESUME_ARG="--resume"
    echo "RESUME MODE ENABLED"
fi

PRESCREEN_ARG=""
if [[ "$PRESCREEN" == "1" ]]; then
    PRESCREEN_ARG="--prescreen --prescreen-samples $PRESCREEN_SAMPLES"
    echo "PRESCREEN MODE ENABLED (will find hard samples first)"
fi

HARD_SAMPLES_ARG=""
if [[ -n "$HARD_SAMPLES_FILE" ]]; then
    HARD_SAMPLES_ARG="--hard-samples-file $HARD_SAMPLES_FILE"
    PRESCREEN_ARG=""  # Don't prescreen if loading from file
    echo "LOADING HARD SAMPLES FROM: $HARD_SAMPLES_FILE"
fi

python -m src.experiments.Sella.hip_sella_hpo \
    --n-trials "$N_TRIALS" \
    --max-steps "$MAX_STEPS" \
    --max-samples "$MAX_SAMPLES" \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
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
    echo "HIP Sella HPO completed successfully at `date`"
else
    echo "HIP Sella HPO exited with code $EXIT_CODE at `date`"
    echo "Check logs and resume with: RESUME=--resume STUDY_NAME=$STUDY_NAME"
fi
echo "Results saved to: $OUT_DIR"
echo "=============================================="

exit $EXIT_CODE
