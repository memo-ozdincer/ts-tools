#!/bin/bash
#SBATCH --job-name=scine_multimode_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/scine_multimode_parallel_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/scine_multimode_parallel_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# SCINE Multi-Mode Eckart-MW (Parallel) - Trillium (CPU)
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

TMP_BASE="${SLURM_TMPDIR:-${SCRATCH_DIR}/tmp}"
export TMPDIR="${TMP_BASE}/${SLURM_JOB_ID}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
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
OUT_DIR="${SCRATCH_DIR}/runs/scine_multimode_parallel_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

export WANDB_MODE=offline
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"

export WANDB_RUN_GROUP="Noisy-multi-mode-eckartmw-trillium"

N_STEPS="${N_STEPS:-2000}"
MAX_SAMPLES="${MAX_SAMPLES:-15}"
START_FROM="${START_FROM:-midpoint_rt_noise1.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
SCINE_FUNCTIONAL="${SCINE_FUNCTIONAL:-DFTB0}"

DT="${DT:-0.004998616}"
DT_MAX="${DT_MAX:-0.0787679}"
MAX_ATOM_DISP="${MAX_ATOM_DISP:-0.3508912104455338}"
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-20}"
PLATEAU_BOOST="${PLATEAU_BOOST:-1.8482911248689458}"
PLATEAU_SHRINK="${PLATEAU_SHRINK:-0.6341627930334096}"
ESCAPE_DISP_THRESHOLD="${ESCAPE_DISP_THRESHOLD:-7.24457e-4}"
ESCAPE_WINDOW="${ESCAPE_WINDOW:-10}"
ESCAPE_NEG_VIB_STD="${ESCAPE_NEG_VIB_STD:-0.884664427}"
ESCAPE_DELTA="${ESCAPE_DELTA:-0.2990654358143095}"
MIN_INTERATOMIC_DIST="${MIN_INTERATOMIC_DIST:-0.5247819497457936}"
MAX_ESCAPE_CYCLES="${MAX_ESCAPE_CYCLES:-1000}"

echo "=============================================="
echo "SCINE Multi-Mode Eckart-MW - Trillium (Parallel CPU)"
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
echo "  N_WORKERS: $N_WORKERS"
echo "  THREADS_PER_WORKER: $THREADS_PER_WORKER"
echo "=============================================="
echo "Fixed hyperparameters:"
echo "  dt: $DT"
echo "  dt_max: $DT_MAX"
echo "  max_atom_disp: $MAX_ATOM_DISP"
echo "  plateau_patience: $PLATEAU_PATIENCE"
echo "  plateau_boost: $PLATEAU_BOOST"
echo "  plateau_shrink: $PLATEAU_SHRINK"
echo "  escape_disp_threshold: $ESCAPE_DISP_THRESHOLD"
echo "  escape_window: $ESCAPE_WINDOW"
echo "  escape_neg_vib_std: $ESCAPE_NEG_VIB_STD"
echo "  escape_delta: $ESCAPE_DELTA"
echo "  adaptive_delta: False"
echo "  min_interatomic_dist: $MIN_INTERATOMIC_DIST"
echo "=============================================="

python -m src.noisy.scine_multi_mode_eckartmw_parallel \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --scine-functional "$SCINE_FUNCTIONAL" \
    --n-steps "$N_STEPS" \
    --max-samples "$MAX_SAMPLES" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --n-workers "$N_WORKERS" \
    --threads-per-worker "$THREADS_PER_WORKER" \
    --dt "$DT" \
    --dt-max "$DT_MAX" \
    --max-atom-disp "$MAX_ATOM_DISP" \
    --plateau-patience "$PLATEAU_PATIENCE" \
    --plateau-boost "$PLATEAU_BOOST" \
    --plateau-shrink "$PLATEAU_SHRINK" \
    --escape-disp-threshold "$ESCAPE_DISP_THRESHOLD" \
    --escape-window "$ESCAPE_WINDOW" \
    --escape-neg-vib-std "$ESCAPE_NEG_VIB_STD" \
    --escape-delta "$ESCAPE_DELTA" \
    --no-adaptive-delta \
    --min-interatomic-dist "$MIN_INTERATOMIC_DIST" \
    --max-escape-cycles "$MAX_ESCAPE_CYCLES" \
    --wandb \
    --wandb-project "noisy-multi-mode-eckartmw" \
    --util-log-every 10

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SCINE Multi-Mode (Parallel) completed successfully at $(date)"
else
    echo "SCINE Multi-Mode (Parallel) exited with code $EXIT_CODE at $(date)"
fi
echo "=============================================="
echo "Results saved to: $OUT_DIR"
echo "=============================================="

exit $EXIT_CODE
