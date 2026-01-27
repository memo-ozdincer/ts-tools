#!/bin/bash
#SBATCH --job-name=adaptive_k_hisd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/adaptive_k_hisd_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/adaptive_k_hisd_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Adaptive k-HiSD Runner - Theoretically-Justified High-Index Saddle Dynamics
# =============================================================================
#
# This implements adaptive k-HiSD from the iHiSD paper (Yin et al.):
#   ẋ = -R∇E    where R = I - 2∑ᵢ₌₁ᵏ vᵢvᵢᵀ
#
# Key difference from v₂ kicking:
# - v₂ kick: discrete perturbations that "unstick" GAD but don't systematically
#   reduce the Morse index (works through brute-force exploration)
#
# - Adaptive k-HiSD: continuous k-reflection where k = Morse index
#   Theorem 3.2 guarantees index-k saddles are unstable under k-HiSD
#
# Comparison questions:
# 1. Does adaptive k-HiSD converge faster (fewer steps)?
# 2. Is the index trajectory cleaner (monotonic decrease)?
# 3. Does it avoid the dt collapse issue entirely?
#
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

# Environment setup
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

# Thread configuration
TOTAL_CPUS="${SLURM_CPUS_ON_NODE:-32}"
export OMP_NUM_THREADS="$TOTAL_CPUS"
export OPENBLAS_NUM_THREADS="$TOTAL_CPUS"
export MKL_NUM_THREADS="$TOTAL_CPUS"
export NUMEXPR_NUM_THREADS="$TOTAL_CPUS"
export VECLIB_MAXIMUM_THREADS="$TOTAL_CPUS"
export MPLBACKEND=Agg

# Data paths
H5_PATH="/project/rrg-aspuru/memoozd/data/transition1x.h5"
OUT_DIR="${SCRATCH_DIR}/runs/adaptive_k_hisd_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Configuration (adjust as needed)
# =============================================================================
N_STEPS="${N_STEPS:-5000}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
START_FROM="${START_FROM:-midpoint_rt_noise1.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
SCINE_FUNCTIONAL="${SCINE_FUNCTIONAL:-DFTB0}"

# Adaptive k-HiSD hyperparameters
DT="${DT:-0.005}"
DT_CONTROL="${DT_CONTROL:-adaptive}"  # "fixed" or "adaptive"
DT_MIN="${DT_MIN:-1e-6}"
DT_MAX="${DT_MAX:-0.08}"
MAX_ATOM_DISP="${MAX_ATOM_DISP:-0.35}"
DT_GROW_FACTOR="${DT_GROW_FACTOR:-1.1}"
DT_SHRINK_FACTOR="${DT_SHRINK_FACTOR:-0.5}"
MIN_K="${MIN_K:-1}"      # At index-1, behaves like standard GAD
MAX_K="${MAX_K:-}"       # Empty = no upper limit
TR_THRESHOLD="${TR_THRESHOLD:-1e-6}"

echo "=============================================="
echo "Adaptive k-HiSD Runner - Trillium"
echo "=============================================="
echo "Date: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs available: $TOTAL_CPUS"
echo "=============================================="
echo "Configuration:"
echo "  N_STEPS: $N_STEPS"
echo "  MAX_SAMPLES: $MAX_SAMPLES"
echo "  START_FROM: $START_FROM"
echo "  NOISE_SEED: $NOISE_SEED"
echo "  SCINE_FUNCTIONAL: $SCINE_FUNCTIONAL"
echo "=============================================="
echo "Adaptive k-HiSD Parameters:"
echo "  dt: $DT (control: $DT_CONTROL)"
echo "  dt_min: $DT_MIN, dt_max: $DT_MAX"
echo "  max_atom_disp: $MAX_ATOM_DISP"
echo "  dt_grow_factor: $DT_GROW_FACTOR"
echo "  dt_shrink_factor: $DT_SHRINK_FACTOR"
echo "  k range: [$MIN_K, ${MAX_K:-unlimited}]"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Calculate workers based on CPUs (default: 4 threads per worker)
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "Parallel Execution Config:"
echo "  Total CPUs: $TOTAL_CPUS"
echo "  Workers: $N_WORKERS"
echo "  Threads per worker: $THREADS_PER_WORKER"

# Build command with optional max-k
CMD="python $PROJECT_DIR/src/noisy/scine_adaptive_k_hisd_parallel.py \
    --h5-path $H5_PATH \
    --out-dir $OUT_DIR \
    --scine-functional $SCINE_FUNCTIONAL \
    --n-steps $N_STEPS \
    --max-samples $MAX_SAMPLES \
    --start-from $START_FROM \
    --noise-seed $NOISE_SEED \
    --n-workers $N_WORKERS \
    --threads-per-worker $THREADS_PER_WORKER \
    --dt $DT \
    --dt-control $DT_CONTROL \
    --dt-min $DT_MIN \
    --dt-max $DT_MAX \
    --max-atom-disp $MAX_ATOM_DISP \
    --dt-grow-factor $DT_GROW_FACTOR \
    --dt-shrink-factor $DT_SHRINK_FACTOR \
    --min-k $MIN_K \
    --tr-threshold $TR_THRESHOLD \
    --stop-at-ts"

# Add max-k only if specified
if [ -n "$MAX_K" ]; then
    CMD="$CMD --max-k $MAX_K"
fi

eval $CMD

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Adaptive k-HiSD completed successfully at $(date)"
else
    echo "Adaptive k-HiSD exited with code $EXIT_CODE at $(date)"
fi
echo "=============================================="
echo "Results saved to: $OUT_DIR"
echo "Diagnostic files: $OUT_DIR/diagnostics/"
echo "=============================================="

exit $EXIT_CODE
