#!/bin/bash
#SBATCH --job-name=diag_minima
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/diag_minima_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/diag_minima_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Diagnostic from Labelled Minima
# =============================================================================
#
# Uses ground-truth reactant/product geometries from Transition1x to
# diagnose WHY the optimizer fails at high noise levels:
#
#   Q1. Does DFTB0 predict n_neg=0 at the DFT-computed reactant minimum?
#       If not, the optimizer is trying to find something that doesn't exist.
#
#   Q2. At the noisy starting point, does the gradient (SD direction) or
#       Newton step point closer to the true minimum?
#       This tells us whether the step DIRECTION is the problem.
#
#   Q3. Along the interpolation path noisy start -> true minimum, where
#       does n_neg first reach 0? Where does the quadratic model become
#       valid?  This reveals the "basin of attraction" boundary.
#
# Additional diagnostics at every point:
#   - Condition number (magnitude and log10)
#   - Eigenvalue band populations (8 bands from strongly neg to positive)
#   - Bottom k eigenvalues
#   - Force norms
#
# This is ANALYSIS ONLY — uses labelled data the optimizer cannot access.
# Using this information in the optimizer would be data leakage.
#
# Config:
#   - 300 samples, single seed from SLURM_JOB_ID
#   - Noise levels: 0.5, 1.0, 1.5, 2.0 A (all in one run)
#   - 20 interpolation points per path
#   - Point evaluations only (no optimization), ~6h walltime
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

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# Thread configuration
TOTAL_CPUS="${SLURM_CPUS_ON_NODE:-192}"
export OMP_NUM_THREADS="$TOTAL_CPUS"
export OPENBLAS_NUM_THREADS="$TOTAL_CPUS"
export MKL_NUM_THREADS="$TOTAL_CPUS"
export NUMEXPR_NUM_THREADS="$TOTAL_CPUS"
export VECLIB_MAXIMUM_THREADS="$TOTAL_CPUS"
export MPLBACKEND=Agg

# Data paths
H5_PATH="/project/rrg-aspuru/memoozd/data/transition1x.h5"
OUT_DIR="${SCRATCH_DIR}/runs/diag_minima_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# Configuration
MAX_SAMPLES="${MAX_SAMPLES:-300}"
NOISE_SEED="${SLURM_JOB_ID:-$(date +%s)}"
SCINE_FUNCTIONAL="${SCINE_FUNCTIONAL:-DFTB0}"
NOISE_LEVELS="${NOISE_LEVELS:-0.5,1.0,1.5,2.0}"
N_INTERP_POINTS="${N_INTERP_POINTS:-20}"
SHIFT_EPSILON="${SHIFT_EPSILON:-1e-3}"
LOG_SPECTRUM_K="${LOG_SPECTRUM_K:-10}"

# Parallel worker config
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "=============================================="
echo "Diagnostic from Labelled Minima"
echo "=============================================="
echo "Date:         $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "CPUs:         $TOTAL_CPUS"
echo "Workers:      $N_WORKERS  (${THREADS_PER_WORKER} threads each)"
echo "=============================================="
echo "Configuration:"
echo "  MAX_SAMPLES:         $MAX_SAMPLES"
echo "  NOISE_SEED:          $NOISE_SEED"
echo "  NOISE_LEVELS:        $NOISE_LEVELS"
echo "  N_INTERP_POINTS:     $N_INTERP_POINTS"
echo "  SHIFT_EPSILON:       $SHIFT_EPSILON"
echo "  LOG_SPECTRUM_K:      $LOG_SPECTRUM_K"
echo "  SCINE_FUNCTIONAL:    $SCINE_FUNCTIONAL"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

python "$PROJECT_DIR/src/noisy/v2_tests/scripts/diagnostic_from_minima.py" \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --scine-functional "$SCINE_FUNCTIONAL" \
    --max-samples "$MAX_SAMPLES" \
    --noise-levels "$NOISE_LEVELS" \
    --noise-seed "$NOISE_SEED" \
    --n-workers "$N_WORKERS" \
    --threads-per-worker "$THREADS_PER_WORKER" \
    --n-interpolation-points "$N_INTERP_POINTS" \
    --shift-epsilon "$SHIFT_EPSILON" \
    --log-spectrum-k "$LOG_SPECTRUM_K" \
    > "${OUT_DIR}/diagnostic_report.txt" 2>&1

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Diagnostic completed successfully at $(date)"
else
    echo "Diagnostic exited with code $EXIT_CODE at $(date)"
fi
echo "=============================================="
echo "Results saved to: $OUT_DIR"
echo "  Report:   ${OUT_DIR}/diagnostic_report.txt"
echo "  JSON:     ${OUT_DIR}/diagnostic_from_minima_results.json"
echo "  CSVs:     ${OUT_DIR}/diagnostic_noise*.csv"
echo "=============================================="

exit $EXIT_CODE
