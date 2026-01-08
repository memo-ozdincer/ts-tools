#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96GB
#SBATCH --job-name=hip_sella_hpo
#SBATCH --output=logs/hip_sella_hpo_%j.out
#SBATCH --error=logs/hip_sella_hpo_%j.err

# =============================================================================
# HIP Sella Bayesian Hyperparameter Optimization
# =============================================================================
# 
# Uses Optuna with TPE sampler to optimize Sella hyperparameters for HIP.
# Uses 4x L40s GPUs - each trial uses 1 GPU.
#
# CRASH RECOVERY: Progress is saved to SQLite after each trial.
#   To resume: add --resume and same --study-name
#
# Hyperparameters being optimized:
#   - delta0: Initial trust radius [0.15, 0.8] (log scale)
#   - rho_dec: Trust radius decrease threshold [15, 80]
#   - rho_inc: Trust radius increase threshold [1.01, 1.1]
#   - sigma_dec: Trust radius decrease factor [0.75, 0.95]
#   - sigma_inc: Trust radius increase factor [1.1, 1.8]
#   - fmax: Force convergence threshold [1e-4, 1e-2] (log scale)
#   - apply_eckart: Whether to Eckart project Hessians [True, False]
#     (Eckart removes trans/rot from Cartesian Hessian before Sella internal conversion)
#
# Fixed parameters:
#   - gamma: 0.0
#   - max_steps: 100
#   - samples per trial: 30
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
# Time estimate: ~50 trials × 30 samples × ~15 sec = ~6 hours (with pruning ~3-4 hrs)
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

# Study name for resume capability
STUDY_NAME="${STUDY_NAME:-hip_hpo_job${SLURM_JOB_ID}}"
RESUME_FLAG="${RESUME:-}"

WANDB_NAME="hip-sella-hpo_${N_TRIALS}trials_${START_FROM}_job${SLURM_JOB_ID}"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo "=============================================="
echo "HIP Sella Bayesian HPO (4x L40s)"
echo "=============================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "N_TRIALS: $N_TRIALS"
echo "MAX_STEPS: $MAX_STEPS"
echo "MAX_SAMPLES: $MAX_SAMPLES"
echo "START_FROM: $START_FROM"
echo "NOISE_SEED: $NOISE_SEED"
echo "OPTUNA_SEED: $OPTUNA_SEED"
echo "STUDY_NAME: $STUDY_NAME"
echo "OUT_DIR: $OUT_DIR"
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

# Build resume argument
RESUME_ARG=""
if [[ -n "$RESUME_FLAG" ]]; then
    RESUME_ARG="--resume"
    echo "RESUME MODE ENABLED"
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
