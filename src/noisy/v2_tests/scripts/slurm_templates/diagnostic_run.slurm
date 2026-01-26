#!/bin/bash
#SBATCH --job-name=gad_diagnostics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/gad_diagnostics_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/gad_diagnostics_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# GAD Diagnostic Runner - Comprehensive Logging for Failure Analysis
# =============================================================================
#
# This script runs the enhanced GAD with comprehensive diagnostic logging to
# understand WHY GAD gets stuck and when v₂ kicking helps.
#
# Output includes:
# - Per-step eigenvalue spectrum (first 6 modes)
# - Eigenvalue gaps (critical for singularity detection)
# - Morse index tracking
# - Detailed escape event logs
# - Failure mode analysis
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
OUT_DIR="${SCRATCH_DIR}/runs/gad_diagnostics_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Configuration (adjust as needed)
# =============================================================================
N_STEPS="${N_STEPS:-5000}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
START_FROM="${START_FROM:-midpoint_rt_noise1.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
SCINE_FUNCTIONAL="${SCINE_FUNCTIONAL:-DFTB0}"

# GAD hyperparameters (use defaults from multi_mode_eckartmw)
DT="${DT:-0.001}"
DT_MAX="${DT_MAX:-0.05}"
MAX_ATOM_DISP="${MAX_ATOM_DISP:-0.25}"
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-10}"
PLATEAU_BOOST="${PLATEAU_BOOST:-1.5}"
PLATEAU_SHRINK="${PLATEAU_SHRINK:-0.5}"
ESCAPE_DISP_THRESHOLD="${ESCAPE_DISP_THRESHOLD:-5e-4}"
ESCAPE_WINDOW="${ESCAPE_WINDOW:-20}"
ESCAPE_NEG_VIB_STD="${ESCAPE_NEG_VIB_STD:-0.5}"
ESCAPE_DELTA="${ESCAPE_DELTA:-0.1}"
MIN_INTERATOMIC_DIST="${MIN_INTERATOMIC_DIST:-0.5}"
MAX_ESCAPE_CYCLES="${MAX_ESCAPE_CYCLES:-100}"

echo "=============================================="
echo "GAD Diagnostic Runner - Trillium"
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
echo "GAD Parameters:"
echo "  dt: $DT"
echo "  dt_max: $DT_MAX"
echo "  max_atom_disp: $MAX_ATOM_DISP"
echo "  plateau_patience: $PLATEAU_PATIENCE"
echo "  escape_disp_threshold: $ESCAPE_DISP_THRESHOLD"
echo "  escape_delta: $ESCAPE_DELTA"
echo "=============================================="

python -m src.noisy.v2_tests.runners.run_with_diagnostics \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --calculator scine \
    --scine-functional "$SCINE_FUNCTIONAL" \
    --n-steps "$N_STEPS" \
    --max-samples "$MAX_SAMPLES" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
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
    --min-interatomic-dist "$MIN_INTERATOMIC_DIST" \
    --max-escape-cycles "$MAX_ESCAPE_CYCLES" \
    --stop-at-ts

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "GAD Diagnostics completed successfully at $(date)"
else
    echo "GAD Diagnostics exited with code $EXIT_CODE at $(date)"
fi
echo "=============================================="
echo "Results saved to: $OUT_DIR"
echo "Diagnostic files: $OUT_DIR/diagnostics/"
echo "=============================================="

exit $EXIT_CODE
