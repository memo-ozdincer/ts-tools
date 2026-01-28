#!/bin/bash
#SBATCH --job-name=ihisd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/ihisd_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/ihisd_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# iHiSD Runner - Improved High-index Saddle Dynamics with Crossover
# =============================================================================
#
# This implements iHiSD (Algorithm 1) from the paper:
#   - Uses a crossover parameter theta that smoothly transitions from
#     gradient flow (theta ≈ 0) to full k-HiSD (theta → 1)
#
# The crossover direction is:
#   d = (1 - s*theta) * grad + 2*theta * sum_i(v_i * <grad, v_i>)
#
# Where s = ±1 determines search direction:
#   s = +1: Upward search (gradient ascent → HiSD), good for minima → saddle
#   s = -1: Downward search (gradient descent → HiSD), good for high-index → TS
#
# Key advantages over standard k-HiSD:
# - Nonlocal convergence: Can start outside the region of attraction
# - Smooth transition: Avoids discontinuous jumps in dynamics
# - Guaranteed convergence: Theorem proves convergence to index-k saddles
#
# Theta schedule options:
# - sigmoid: theta = 2/(1 + exp(-lambda*m)) - 1
# - linear: theta = theta_0 + rate * m
# - exponential: theta = 1 - (1 - theta_0) * exp(-rate * m)
#
# Step size (when theta > 0.5):
#   tau = 2 / (L * (2*theta - 1))
# where L is the Lipschitz constant estimate
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
OUT_DIR="${SCRATCH_DIR}/runs/ihisd_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Configuration (adjust as needed)
# =============================================================================
N_STEPS="${N_STEPS:-5000}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
START_FROM="${START_FROM:-midpoint_rt_noise1.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
SCINE_FUNCTIONAL="${SCINE_FUNCTIONAL:-DFTB0}"

# iHiSD hyperparameters
DT="${DT:-0.005}"
DT_MIN="${DT_MIN:-1e-6}"
DT_MAX="${DT_MAX:-0.08}"
MAX_ATOM_DISP="${MAX_ATOM_DISP:-0.35}"

# Crossover parameters
THETA_0="${THETA_0:-1e-11}"
THETA_SCHEDULE="${THETA_SCHEDULE:-sigmoid}"  # "sigmoid", "linear", "exponential"
THETA_RATE="${THETA_RATE:-0.01}"
SEARCH_DIRECTION="${SEARCH_DIRECTION:-1}"    # +1 upward, -1 downward
TARGET_K="${TARGET_K:-1}"
USE_ADAPTIVE_DT="${USE_ADAPTIVE_DT:-true}"   # "true" or "false"
LIPSCHITZ_ESTIMATE="${LIPSCHITZ_ESTIMATE:-1.0}"

GRAD_THRESHOLD="${GRAD_THRESHOLD:-1e-5}"
TR_THRESHOLD="${TR_THRESHOLD:-1e-6}"

echo "=============================================="
echo "iHiSD Runner - Trillium"
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
echo "iHiSD Parameters:"
echo "  dt: $DT (min: $DT_MIN, max: $DT_MAX)"
echo "  max_atom_disp: $MAX_ATOM_DISP"
echo "  theta_0: $THETA_0"
echo "  theta_schedule: $THETA_SCHEDULE"
echo "  theta_rate: $THETA_RATE"
echo "  search_direction: $SEARCH_DIRECTION"
echo "  target_k: $TARGET_K"
echo "  use_adaptive_dt: $USE_ADAPTIVE_DT"
echo "  lipschitz_estimate: $LIPSCHITZ_ESTIMATE"
echo "  grad_threshold: $GRAD_THRESHOLD"
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

# Build command
CMD="python $PROJECT_DIR/src/noisy/scine_ihisd_parallel.py \
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
    --dt-min $DT_MIN \
    --dt-max $DT_MAX \
    --max-atom-disp $MAX_ATOM_DISP \
    --theta-0 $THETA_0 \
    --theta-schedule $THETA_SCHEDULE \
    --theta-rate $THETA_RATE \
    --search-direction $SEARCH_DIRECTION \
    --target-k $TARGET_K \
    --lipschitz-estimate $LIPSCHITZ_ESTIMATE \
    --grad-threshold $GRAD_THRESHOLD \
    --tr-threshold $TR_THRESHOLD"

# Add adaptive dt flag
if [ "$USE_ADAPTIVE_DT" = "true" ]; then
    CMD="$CMD --use-adaptive-dt"
else
    CMD="$CMD --no-adaptive-dt"
fi

eval $CMD

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "iHiSD completed successfully at $(date)"
else
    echo "iHiSD exited with code $EXIT_CODE at $(date)"
fi
echo "=============================================="
echo "Results saved to: $OUT_DIR"
echo "Diagnostic files: $OUT_DIR/diagnostics/"
echo "=============================================="

exit $EXIT_CODE
