#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=hip_gad_hpo
#SBATCH --output=logs/hip_multi_mode_eckartmw_hpo_%j.out
#SBATCH --error=logs/hip_multi_mode_eckartmw_hpo_%j.err

# =============================================================================
# HIP Multi-Mode Eckart-MW GAD Bayesian HPO
# =============================================================================
#
# Uses Optuna with TPE sampler to optimize GAD hyperparameters for HIP.
#
# CRASH RECOVERY: Progress saved to SQLite after each trial.
#   To resume: RESUME=--resume STUDY_NAME=<previous_name> sbatch ...
#
# Hyperparameters being optimized:
#   - dt_min, dt_max: Time step bounds
#   - plateau_patience: Steps before dt adjustment  
#   - plateau_boost, plateau_shrink: dt adjustment factors
#   - escape_disp_threshold: Displacement threshold for plateau detection
#   - escape_window: Window size for plateau detection
#   - escape_neg_vib_std: Stability threshold for saddle index
#   - escape_delta: Base perturbation magnitude
#   - adaptive_delta_scale: Scale factor for adaptive perturbations
#   - trust_radius_max: Maximum allowed displacement per step
#
# Objective: Maximize convergence rate (# samples reaching 1 neg eigenvalue)
# =============================================================================

mkdir -p logs
source .venv/bin/activate

# Ensure CPU-side ops can use the allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MPLBACKEND=Agg

H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CKPT_PATH="/project/6101772/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output/gad_hpo_hip_${SLURM_JOB_ID}"

# W&B configuration
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"
export WANDB_MODE=online
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WARNING: WANDB_API_KEY not set. W&B logging disabled." >&2
    WANDB_FLAG=""
else
    WANDB_FLAG="--wandb"
fi

export WANDB_RUN_GROUP="hip-gad-hpo"

# HPO parameters
N_TRIALS=${N_TRIALS:-100}
N_STEPS_PER_SAMPLE=${N_STEPS_PER_SAMPLE:-800}
N_SAMPLES=${N_SAMPLES:-15}
MAX_SAMPLES=${MAX_SAMPLES:-100}  # For initial difficult sample detection
DIFFICULTY_THRESHOLD=${DIFFICULTY_THRESHOLD:-0.5}

START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42

# Study name for resume capability
STUDY_NAME="${STUDY_NAME:-hip-gad-hpo-${SLURM_JOB_ID}}"
RESUME_FLAG="${RESUME:-}"

WANDB_NAME="hip-gad-hpo_${N_TRIALS}trials_job${SLURM_JOB_ID}"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo "=============================================="
echo "HIP Multi-Mode Eckart-MW GAD HPO"
echo "=============================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "N_TRIALS: $N_TRIALS"
echo "N_STEPS_PER_SAMPLE: $N_STEPS_PER_SAMPLE"
echo "N_SAMPLES: $N_SAMPLES"
echo "MAX_SAMPLES: $MAX_SAMPLES"
echo "DIFFICULTY_THRESHOLD: $DIFFICULTY_THRESHOLD"
echo "STUDY_NAME: $STUDY_NAME"
echo "OUT_DIR: $OUT_DIR"
echo "=============================================="
echo "Progress saved to SQLite after each trial."
echo "To resume: RESUME=--resume STUDY_NAME=$STUDY_NAME sbatch ..."
echo "=============================================="

# Build optional arguments
RESUME_ARG=""
if [[ -n "$RESUME_FLAG" ]]; then
    RESUME_ARG="--resume"
    echo "RESUME MODE ENABLED"
fi

python -m src.experiments.2025.hip_multi_mode_eckartmw_hpo \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --n-trials "$N_TRIALS" \
    --n-steps-per-sample "$N_STEPS_PER_SAMPLE" \
    --n-samples "$N_SAMPLES" \
    --difficulty-threshold "$DIFFICULTY_THRESHOLD" \
    --study-name "$STUDY_NAME" \
    $RESUME_ARG \
    $WANDB_FLAG \
    --wandb-project "gad-hpo" \
    --wandb-entity "memo-ozdincer-university-of-toronto" \
    --wandb-name "$WANDB_NAME"

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "HIP GAD HPO completed successfully at `date`"
else
    echo "HIP GAD HPO exited with code $EXIT_CODE at `date`"
    echo "To resume: RESUME=--resume STUDY_NAME=$STUDY_NAME sbatch ..."
fi
echo "Results saved to: $OUT_DIR/hpo_results.json"
echo "Optuna database: $OUT_DIR/${STUDY_NAME}.db"
echo "=============================================="

exit $EXIT_CODE
