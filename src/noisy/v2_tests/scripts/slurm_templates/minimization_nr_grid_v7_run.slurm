#!/bin/bash
#SBATCH --job-name=min_nr_v7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v7_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v7_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Newton-Raphson Minimization — v7 Line Search Fix
# =============================================================================
#
# DIAGNOSIS (from v6 300-sample analysis):
#   The trust region is the bottleneck. 100% of v6 failures have trust_radius
#   pinned at the floor (0.01 A). Shifted Newton amplifies near-zero modes by
#   1000x, producing oversized steps that collapse the trust radius in ~4 steps.
#   Once at the floor, negative-mode displacement is ~0.0005 A/step — too slow
#   to make progress. Recovery needs ~13 consecutive good steps but any
#   meaningful step triggers collapse again.
#
# FIX:
#   Replace the persistent trust region with a stateless Armijo backtracking
#   line search. The line search starts alpha=1.0 fresh every step (no state
#   to collapse), guarantees energy decrease, and was already proven sound in
#   the SPDN path (SPDN failed due to spectral step builder + GDIIS, not the
#   line search).
#
# SETUP:
#   - Noise: 0.5 A and 1.0 A (test robustness across regimes)
#   - Seed: SLURM_JOB_ID (different starting geometries every run)
#   - Samples: 300
#   - Steps: 10000
#   - All configs: SN epsilon=1e-3 (proven best from v3/v6)
#
# CONFIGS (6 = 3 step-control variants x 2 noise levels):
#   1. n0.5 trust_region          — v6 baseline (should reproduce ~76%)
#   2. n0.5 line_search           — does removing trust region fix collapse?
#   3. n0.5 line_search + mw=200  — does capping weight help further?
#   4. n1.0 trust_region          — baseline at harder noise
#   5. n1.0 line_search           — line search at harder noise
#   6. n1.0 line_search + mw=200  — weight cap at harder noise
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
OUT_DIR="${SCRATCH_DIR}/runs/min_nr_v7_${SLURM_JOB_ID}"

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

# Unique seed per job — different starting geometries every run
NOISE_SEED="${SLURM_JOB_ID:-$(date +%s)}"

# Parallel worker config
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "=============================================="
echo "Newton-Raphson Minimization — v7 Line Search Fix"
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
echo "Configs:  6 (3 step-control x 2 noise levels)"
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
# All use SN epsilon=1e-3 (proven best from v3/v6)
# Tag format: n<noise>_mad1.3_se1e-3_sc<tr|ls>[_mw<v>]_pgtrue_phfalse
# =============================================================================

# --- 0.5 A noise (easy regime) ---

# (1) Trust region baseline — should reproduce v6 ~76%
run_combo "0.5" "n0.5_mad1.3_se1e-3_sctr_pgtrue_phfalse" \
    --step-control "trust_region"

# (2) Line search — test if removing trust region fixes collapse
run_combo "0.5" "n0.5_mad1.3_se1e-3_scls_pgtrue_phfalse" \
    --step-control "line_search"

# (3) Line search + weight cap — does capping at 200x help further?
run_combo "0.5" "n0.5_mad1.3_se1e-3_scls_mw200_pgtrue_phfalse" \
    --step-control "line_search" \
    --max-nr-weight 200

# --- 1.0 A noise (hard regime) ---

# (4) Trust region baseline at harder noise
run_combo "1.0" "n1.0_mad1.3_se1e-3_sctr_pgtrue_phfalse" \
    --step-control "trust_region"

# (5) Line search at harder noise
run_combo "1.0" "n1.0_mad1.3_se1e-3_scls_pgtrue_phfalse" \
    --step-control "line_search"

# (6) Line search + weight cap at harder noise
run_combo "1.0" "n1.0_mad1.3_se1e-3_scls_mw200_pgtrue_phfalse" \
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
