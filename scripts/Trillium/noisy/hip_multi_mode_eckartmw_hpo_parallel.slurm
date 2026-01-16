#!/bin/bash
#SBATCH --job-name=hip_multimode_hpo_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/hip_multimode_hpo_parallel_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/hip_multimode_hpo_parallel_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# HIP Multi-Mode Eckart-MW HPO (Parallel) - Trillium
# =============================================================================
# Uses multiprocessing to parallelize per-sample evaluations on a single GPU.
# =============================================================================

set -e

mkdir -p /scratch/memoozd/ts-tools-scratch/logs

PROJECT_DIR="/project/rrg-aspuru/memoozd/ts-tools"
SCRATCH_DIR="/scratch/memoozd/ts-tools-scratch"

cd "$SCRATCH_DIR"

echo "Clearing Python bytecode cache..."
find "$PROJECT_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$PROJECT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

module purge
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6

source "$PROJECT_DIR/.venv/bin/activate"

export TMPDIR="${SCRATCH_DIR}/tmp"
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export MPLCONFIGDIR="${SCRATCH_DIR}/.config/matplotlib"
export FONTCONFIG_PATH=/etc/fonts
export FONTCONFIG_FILE=/etc/fonts/fonts.conf

mkdir -p "$TMPDIR"
mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$MPLCONFIGDIR"

TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-24}"
N_WORKERS="${N_WORKERS:-4}"
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
CKPT_PATH="/project/rrg-aspuru/memoozd/models/hip_v2.ckpt"
OUT_DIR="${SCRATCH_DIR}/hpo/hip_multimode_parallel_${SLURM_JOB_ID}"
DB_DIR="${SCRATCH_DIR}/dbs"

mkdir -p "$OUT_DIR"
mkdir -p "$DB_DIR"

export WANDB_MODE=offline
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"

export WANDB_RUN_GROUP="HPO-multi-mode-eckartmw-trillium"

N_STEPS="${N_STEPS:-20}"
MAX_SAMPLES="${MAX_SAMPLES:-2}"
START_FROM="${START_FROM:-midpoint_rt_noise1.0A}"
NOISE_SEED="${NOISE_SEED:-42}"

N_TRIALS="${N_TRIALS:-500}"
N_STARTUP_TRIALS="${N_STARTUP_TRIALS:-5}"
STUDY_NAME="${STUDY_NAME:-hip_multi_mode_hpo_parallel_${SLURM_JOB_ID}}"

RESUME_FLAG="${RESUME:-0}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"

echo "=============================================="
echo "HIP Multi-Mode Eckart-MW HPO - Trillium (Parallel)"
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
echo "  N_WORKERS: $N_WORKERS"
echo "  THREADS_PER_WORKER: $THREADS_PER_WORKER"
echo "=============================================="

nvidia-smi

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

python -m src.noisy.hip_multi_mode_eckartmw_hpo_parallel \
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
    --n-workers "$N_WORKERS" \
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
echo "=============================================="

exit $EXIT_CODE
