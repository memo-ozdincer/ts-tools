#!/bin/bash
#SBATCH --job-name=min_nr_v7h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=09:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v7h_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v7h_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Newton-Raphson Minimization — v7 Line Search Fix (Hard Noise)
# =============================================================================
#
# Same v7 line search fix as the main script, but at harder noise levels
# (1.5 A and 2.0 A) with a single seed to check generalization.
#
# CONFIGS (6 = 3 step-control variants x 2 noise levels):
#   1. n1.5 trust_region          — baseline
#   2. n1.5 line_search           — line search fix
#   3. n1.5 line_search + mw=200  — line search + weight cap
#   4. n2.0 trust_region          — baseline at hardest noise
#   5. n2.0 line_search           — line search at hardest noise
#   6. n2.0 line_search + mw=200  — weight cap at hardest noise
#
# GRID SIZE: 6 configs x 300 samples = 1800 trajectories
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
OUT_DIR="${SCRATCH_DIR}/runs/min_nr_v7h_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Fixed configuration
# =============================================================================
N_STEPS=10000
MAX_SAMPLES=300
SCINE_FUNCTIONAL="DFTB0"
FORCE_CONVERGED="1e-4"
MIN_INTERATOMIC_DIST="0.5"
LOG_SPECTRUM_K=10
TRUST_RADIUS_FLOOR="0.01"
SHIFT_EPSILON="1e-3"

# Single seed for hard noise exploration
NOISE_SEED="${SLURM_JOB_ID:-$(date +%s)}"

# Parallel worker config
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "=============================================="
echo "Newton-Raphson Minimization — v7 Line Search (Hard Noise)"
echo "=============================================="
echo "Date:         $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "CPUs:         $TOTAL_CPUS"
echo "Workers:      $N_WORKERS  (${THREADS_PER_WORKER} threads each)"
echo "=============================================="
echo "Fixed config:"
echo "  SHIFT_EPSILON:      $SHIFT_EPSILON"
echo "  NOISE_SEED:         $NOISE_SEED"
echo "  N_STEPS:            $N_STEPS"
echo "  MAX_SAMPLES:        $MAX_SAMPLES"
echo "  FORCE_CONVERGED:    $FORCE_CONVERGED"
echo "  TRUST_RADIUS_FLOOR: $TRUST_RADIUS_FLOOR"
echo "  LOG_SPECTRUM_K:     $LOG_SPECTRUM_K"
echo "Configs:  6 (3 step-control x 2 noise levels: 1.5A, 2.0A)"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

COMBO=0
FAILED=0

# =============================================================================
# Helper: run one combination
# =============================================================================
run_combo() {
    local noise_level="$1"
    local tag="$2"
    shift 2

    local start_from="midpoint_rt_noise${noise_level}A"
    local run_out="${OUT_DIR}/${tag}"
    mkdir -p "$run_out"

    COMBO=$((COMBO + 1))
    echo "[combo ${COMBO}]  ${tag}  (noise=${noise_level}A, seed=${NOISE_SEED})"

    if python "$PROJECT_DIR/src/noisy/v2_tests/runners/run_minimization_parallel.py" \
            --h5-path "$H5_PATH" \
            --out-dir "$run_out" \
            --scine-functional "$SCINE_FUNCTIONAL" \
            --n-steps "$N_STEPS" \
            --max-samples "$MAX_SAMPLES" \
            --start-from "$start_from" \
            --noise-seed "$NOISE_SEED" \
            --n-workers "$N_WORKERS" \
            --threads-per-worker "$THREADS_PER_WORKER" \
            --method newton_raphson \
            --max-atom-disp 1.3 \
            --force-converged "$FORCE_CONVERGED" \
            --min-interatomic-dist "$MIN_INTERATOMIC_DIST" \
            --project-gradient-and-v \
            --log-spectrum-k "$LOG_SPECTRUM_K" \
            --trust-radius-floor "$TRUST_RADIUS_FLOOR" \
            --shift-epsilon "$SHIFT_EPSILON" \
            "$@"; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# =============================================================================
# Grid: 3 step-control variants x 2 noise levels = 6 configs
# =============================================================================

# --- 1.5 A noise ---

# (1) Trust region baseline
run_combo "1.5" "n1.5_mad1.3_se1e-3_sctr_pgtrue_phfalse" \
    --step-control "trust_region"

# (2) Line search
run_combo "1.5" "n1.5_mad1.3_se1e-3_scls_pgtrue_phfalse" \
    --step-control "line_search"

# (3) Line search + weight cap
run_combo "1.5" "n1.5_mad1.3_se1e-3_scls_mw200_pgtrue_phfalse" \
    --step-control "line_search" \
    --max-nr-weight 200

# --- 2.0 A noise ---

# (4) Trust region baseline
run_combo "2.0" "n2.0_mad1.3_se1e-3_sctr_pgtrue_phfalse" \
    --step-control "trust_region"

# (5) Line search
run_combo "2.0" "n2.0_mad1.3_se1e-3_scls_pgtrue_phfalse" \
    --step-control "line_search"

# (6) Line search + weight cap
run_combo "2.0" "n2.0_mad1.3_se1e-3_scls_mw200_pgtrue_phfalse" \
    --step-control "line_search" \
    --max-nr-weight 200

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "Grid completed at $(date)"
echo "  Total combinations: $COMBO"
echo "  Successful:         $((COMBO - FAILED))"
echo "  Failed:             $FAILED"
echo "  Noise seed:         $NOISE_SEED"
echo "  Results:            $OUT_DIR"
echo "=============================================="

# =============================================================================
# Analysis
# =============================================================================
ANALYSIS_DIR="${OUT_DIR}/analysis"
mkdir -p "$ANALYSIS_DIR"

echo "Running grid analysis ..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_minimization_nr_grid.py" \
    --grid-dir "$OUT_DIR" \
    --result-glob "*/minimization_newton_raphson_*_results.json" \
    --output-dir "$ANALYSIS_DIR" \
    --top-k 10 > "${ANALYSIS_DIR}/report.txt" 2>&1 || true

echo "Running failure autopsy ..."
AUTOPSY_DIR="${OUT_DIR}/autopsy"
mkdir -p "$AUTOPSY_DIR"
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_nr_failure_autopsy.py" \
    --grid-dir "$OUT_DIR" \
    --output-dir "$AUTOPSY_DIR" > "${AUTOPSY_DIR}/autopsy_report.txt" 2>&1 || true

echo "=============================================="
echo "All done.  Results in: $OUT_DIR"
echo "  Grid report:    ${ANALYSIS_DIR}/report.txt"
echo "  Autopsy report: ${AUTOPSY_DIR}/autopsy_report.txt"
echo "=============================================="
