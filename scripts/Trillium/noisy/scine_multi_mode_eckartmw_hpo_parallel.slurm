#!/bin/bash
#SBATCH --job-name=scine_multimode_hpo_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/scine_multimode_hpo_parallel_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/scine_multimode_hpo_parallel_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# SCINE Multi-Mode Eckart-MW Grid Search (Parallel) - Trillium (CPU)
# =============================================================================

set -e

mkdir -p /scratch/memoozd/ts-tools-scratch/logs

PROJECT_DIR="/project/rrg-aspuru/memoozd/ts-tools"
SCRATCH_DIR="/scratch/memoozd/ts-tools-scratch"

cd "$SCRATCH_DIR"

module purge
module load StdEnv/2023
module load python/3.11.5

source "$PROJECT_DIR/.venv/bin/activate"

export TMPDIR="${SCRATCH_DIR}/tmp"
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export MPLCONFIGDIR="${SCRATCH_DIR}/.config/matplotlib"
export FONTCONFIG_PATH=/etc/fonts
export FONTCONFIG_FILE=/etc/fonts/fonts.conf

mkdir -p "$TMPDIR"
mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$MPLCONFIGDIR"

TOTAL_CPUS="${SLURM_CPUS_ON_NODE:-192}"
N_WORKERS="${N_WORKERS:-16}"
THREADS_PER_WORKER=$((TOTAL_CPUS / N_WORKERS))
if [ "$THREADS_PER_WORKER" -lt 1 ]; then
    THREADS_PER_WORKER=1
fi

export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export OPENBLAS_NUM_THREADS="$THREADS_PER_WORKER"
export MKL_NUM_THREADS="$THREADS_PER_WORKER"
export NUMEXPR_NUM_THREADS="$THREADS_PER_WORKER"
export VECLIB_MAXIMUM_THREADS="$THREADS_PER_WORKER"
export MPLBACKEND=Agg

H5_PATH="/project/rrg-aspuru/memoozd/data/transition1x.h5"
OUT_DIR="${SCRATCH_DIR}/hpo/scine_multimode_parallel_${SLURM_JOB_ID}"
DB_DIR="${SCRATCH_DIR}/dbs"

mkdir -p "$OUT_DIR"
mkdir -p "$DB_DIR"

export WANDB_MODE=offline
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"

export WANDB_RUN_GROUP="GRID-multi-mode-eckartmw-trillium"

N_STEPS="${N_STEPS:-10000}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
START_FROM="${START_FROM:-midpoint_rt_noise2.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
SCINE_FUNCTIONAL="${SCINE_FUNCTIONAL:-DFTB0}"

# Submit with: PROJECT_GRADIENT_AND_V=true sbatch scine_multi_mode_eckartmw_hpo_parallel.slurm
PROJECT_GRADIENT_AND_V="${PROJECT_GRADIENT_AND_V:-true}"

N_TRIALS="${N_TRIALS:-18}"
N_STARTUP_TRIALS="${N_STARTUP_TRIALS:-5}"
SEARCH_METHOD="${SEARCH_METHOD:-grid}"
STUDY_NAME="${STUDY_NAME:-scine_multi_mode_grid_parallel_${SLURM_JOB_ID}}"

RESUME_FLAG="${RESUME:-0}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"

echo "=============================================="
echo "SCINE Multi-Mode Eckart-MW GRID - Trillium (Parallel CPU)"
echo "=============================================="
echo "Date: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs available: ${SLURM_CPUS_ON_NODE:-192}"
echo "=============================================="
echo "Configuration:"
echo "  N_STEPS: $N_STEPS"
echo "  MAX_SAMPLES: $MAX_SAMPLES"
echo "  START_FROM: $START_FROM"
echo "  NOISE_SEED: $NOISE_SEED"
echo "  SCINE_FUNCTIONAL: $SCINE_FUNCTIONAL"
echo "  SEARCH_METHOD: $SEARCH_METHOD"
echo "  N_TRIALS: $N_TRIALS"
echo "  N_WORKERS: $N_WORKERS"
echo "  THREADS_PER_WORKER: $THREADS_PER_WORKER"
echo "=============================================="

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

python -m src.noisy.scine_multi_mode_eckartmw_hpo_parallel \
    --h5-path "$H5_PATH" \
    --out-dir "$DB_DIR" \
    --scine-functional "$SCINE_FUNCTIONAL" \
    --n-steps "$N_STEPS" \
    --max-samples "$MAX_SAMPLES" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --n-trials "$N_TRIALS" \
    --n-startup-trials "$N_STARTUP_TRIALS" \
    --study-name "$STUDY_NAME" \
    --search-method "$SEARCH_METHOD" \
    --n-workers "$N_WORKERS" \
    --threads-per-worker "$THREADS_PER_WORKER" \
    $RESUME_ARG \
    $SKIP_VERIFY_ARG \
    --wandb \
    --wandb-project "scine-multi-mode-hpo" \
    --util-log-every 10 \
    $([ "$PROJECT_GRADIENT_AND_V" = "true" ] && echo "--project-gradient-and-v")

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SCINE Multi-Mode HPO completed successfully at $(date)"
else
    echo "SCINE Multi-Mode HPO exited with code $EXIT_CODE at $(date)"
    echo "Check logs and resume with: RESUME=1 STUDY_NAME=$STUDY_NAME sbatch ..."
fi
echo "=============================================="
echo "Results saved to: $OUT_DIR"
echo ""
echo "SQLite database: $DB_DIR/${STUDY_NAME}.db"
echo "=============================================="

exit $EXIT_CODE
