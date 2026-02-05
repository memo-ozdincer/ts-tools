#!/bin/bash
#SBATCH --job-name=recursive_hisd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/recursive_hisd_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/recursive_hisd_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Recursive HiSD Runner - Systematic Saddle Index Descent
# =============================================================================
#
# This implements Recursive HiSD from the paper:
#   - Start from a (potentially high-index) saddle point
#   - Recursively descend: index-n → index-(n-1) → ... → index-1 (TS)
#   - At each level, perturb to escape current saddle, then run (k)-HiSD
#
# Key differences from adaptive k-HiSD:
# - Adaptive k-HiSD: starts at k=1, increases k when stuck at high-index
# - Recursive HiSD: explicitly targets each index level sequentially
#
# Algorithm:
# 1. Detect current Morse index n
# 2. If n == 1, done (found TS)
# 3. Perturb along unstable direction to escape
# 4. Run (n-1)-HiSD to find index-(n-1) saddle
# 5. Repeat
#
# Theory: Theorem 5.1 guarantees connectivity between saddles of adjacent
# indices via saddle-saddle connections.
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
OUT_DIR="${SCRATCH_DIR}/runs/recursive_hisd_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Configuration (adjust as needed)
# =============================================================================
N_STEPS="${N_STEPS:-15000}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
START_FROM="${START_FROM:-midpoint_rt_noise2.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
SCINE_FUNCTIONAL="${SCINE_FUNCTIONAL:-DFTB0}"

# Recursive HiSD hyperparameters (suggested single-run defaults)
DT="${DT:-0.003}"
DT_MIN="${DT_MIN:-1e-6}"
DT_MAX="${DT_MAX:-0.06}"
MAX_ATOM_DISP="${MAX_ATOM_DISP:-0.30}"
PERTURB_MAGNITUDE="${PERTURB_MAGNITUDE:-0.02}"
PERTURB_STRATEGY="${PERTURB_STRATEGY:-mixed}"  # "unstable", "random", "mixed"
GRAD_THRESHOLD="${GRAD_THRESHOLD:-5e-6}"
MAX_STEPS_PER_LEVEL="${MAX_STEPS_PER_LEVEL:-600}"
TR_THRESHOLD="${TR_THRESHOLD:-5e-6}"

# Optional grid search (set GRID=1 to enable)
GRID="${GRID:-1}"
DT_GRID=("1e-3" "3e-3" "5e-3")
PERTURB_MAG_GRID=("0.01" "0.02" "0.04")
PERTURB_STRATEGY_GRID=("unstable" "mixed")
GRAD_THRESH_GRID=("1e-5" "5e-6" "1e-6")
MAX_STEPS_PER_LEVEL_GRID=("400" "600" "800")
TR_THRESHOLD_GRID=("1e-5" "5e-6" "1e-6")

echo "=============================================="
echo "Recursive HiSD Runner - Trillium"
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
echo "Recursive HiSD Parameters:"
echo "  dt: $DT (min: $DT_MIN, max: $DT_MAX)"
echo "  max_atom_disp: $MAX_ATOM_DISP"
echo "  perturb_magnitude: $PERTURB_MAGNITUDE"
echo "  perturb_strategy: $PERTURB_STRATEGY"
echo "  grad_threshold: $GRAD_THRESHOLD"
echo "  max_steps_per_level: $MAX_STEPS_PER_LEVEL"
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

if [[ "$GRID" == "1" ]]; then
    echo "Running Recursive HiSD grid search..."
    for DT in "${DT_GRID[@]}"; do
        for PERTURB_MAGNITUDE in "${PERTURB_MAG_GRID[@]}"; do
            for PERTURB_STRATEGY in "${PERTURB_STRATEGY_GRID[@]}"; do
                for GRAD_THRESHOLD in "${GRAD_THRESH_GRID[@]}"; do
                    for MAX_STEPS_PER_LEVEL in "${MAX_STEPS_PER_LEVEL_GRID[@]}"; do
                        for TR_THRESHOLD in "${TR_THRESHOLD_GRID[@]}"; do
                            RUN_TAG="dt${DT}_pm${PERTURB_MAGNITUDE}_ps${PERTURB_STRATEGY}_gt${GRAD_THRESHOLD}_ms${MAX_STEPS_PER_LEVEL}_tr${TR_THRESHOLD}"
                            RUN_OUT_DIR="${OUT_DIR}/${RUN_TAG}"
                            mkdir -p "$RUN_OUT_DIR"
                            python $PROJECT_DIR/src/noisy/scine_recursive_hisd_parallel.py \
                                --h5-path $H5_PATH \
                                --out-dir $RUN_OUT_DIR \
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
                                --perturb-magnitude $PERTURB_MAGNITUDE \
                                --perturb-strategy $PERTURB_STRATEGY \
                                --grad-threshold $GRAD_THRESHOLD \
                                --max-steps-per-level $MAX_STEPS_PER_LEVEL \
                                --tr-threshold $TR_THRESHOLD
                        done
                    done
                done
            done
        done
    done
else
    python $PROJECT_DIR/src/noisy/scine_recursive_hisd_parallel.py \
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
        --perturb-magnitude $PERTURB_MAGNITUDE \
        --perturb-strategy $PERTURB_STRATEGY \
        --grad-threshold $GRAD_THRESHOLD \
        --max-steps-per-level $MAX_STEPS_PER_LEVEL \
        --tr-threshold $TR_THRESHOLD
fi

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Recursive HiSD completed successfully at $(date)"
else
    echo "Recursive HiSD exited with code $EXIT_CODE at $(date)"
fi
echo "=============================================="
echo "Results saved to: $OUT_DIR"
echo "Diagnostic files: $OUT_DIR/diagnostics/"
echo "=============================================="

exit $EXIT_CODE
