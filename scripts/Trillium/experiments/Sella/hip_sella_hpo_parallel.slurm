#!/bin/bash
#SBATCH --job-name=hip_sella_hpo_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=4
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/hip_sella_hpo_parallel_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/hip_sella_hpo_parallel_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# HIP Sella Bayesian HPO (Parallel) - Trillium
# =============================================================================

set -e

mkdir -p /scratch/memoozd/ts-tools-scratch/logs

PROJECT_DIR="/project/rrg-aspuru/memoozd/ts-tools"
SCRATCH_DIR="/scratch/memoozd/ts-tools-scratch"

cd "$SCRATCH_DIR"

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

TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-96}"
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
OUT_DIR="${SCRATCH_DIR}/hpo/hip_sella_parallel_${SLURM_JOB_ID}"
DB_DIR="${SCRATCH_DIR}/dbs"

mkdir -p "$OUT_DIR"
mkdir -p "$DB_DIR"

export WANDB_MODE=offline
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"

export WANDB_RUN_GROUP="hip-sella-hpo-trillium"

N_TRIALS="${N_TRIALS:-5000}"
MAX_STEPS="${MAX_STEPS:-300}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
START_FROM="${START_FROM:-midpoint_rt_noise1.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
OPTUNA_SEED="${OPTUNA_SEED:-42}"

PRESCREEN="${PRESCREEN:-1}"
PRESCREEN_SAMPLES="${PRESCREEN_SAMPLES:-100}"
HARD_SAMPLES_FILE="${HARD_SAMPLES_FILE:-}"

STUDY_NAME="${STUDY_NAME:-hip_sella_hpo_parallel_job${SLURM_JOB_ID}}"
RESUME_FLAG="${RESUME:-0}"

WANDB_NAME="hip-sella-hpo_parallel_${N_TRIALS}trials_${START_FROM}_job${SLURM_JOB_ID}"

echo "=============================================="
echo "HIP Sella Bayesian HPO - Trillium (Parallel)"
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
echo "  N_WORKERS: $N_WORKERS"
echo "  THREADS_PER_WORKER: $THREADS_PER_WORKER"
echo "  STUDY_NAME: $STUDY_NAME"
echo "  OUT_DIR: $OUT_DIR"
echo "=============================================="

nvidia-smi

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

python -m src.experiments.Sella.hip_sella_hpo_parallel \
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
    --n-workers "$N_WORKERS" \
    --device-ids "0,1,2,3" \
    $RESUME_ARG \
    $PRESCREEN_ARG \
    $HARD_SAMPLES_ARG \
    --verbose \
    --wandb \
    --wandb-project "sella-hpo" \
    --wandb-entity "memo-ozdincer-university-of-toronto" \
    --wandb-name "$WANDB_NAME" \
    --util-log-every 30

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
echo "=============================================="

exit $EXIT_CODE
