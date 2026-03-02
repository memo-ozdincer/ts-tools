#!/bin/bash
#SBATCH --job-name=min_nr_v6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v6_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v6_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Newton-Raphson Minimization — v6 Back-to-Basics
# =============================================================================
#
# PHILOSOPHY:
#   Strip everything back to core NR + trust radius. No SPDN, no LM damping,
#   no stagnation escape, no blind correction, no mode following, no two-phase
#   annealing. Just the algorithm with different step computation modes.
#
# SETUP:
#   - Noise: 0.5 Å only (easy regime — baseline algorithm quality)
#   - Seed: derived from SLURM_JOB_ID → different starting geometries every run
#   - Samples: 50
#   - Steps: 10000 (matching v3 — some samples converge at step 7000+)
#
# CONFIGS (4 clean, meaningfully different setups):
#   1. Hard Filter (nr_thresh=8e-3)      — v1 baseline: skip near-zero modes
#   2. Shifted Newton ε=2e-4             — conservative shift, tied-best from v4
#   3. Shifted Newton ε=5e-4             — moderate shift, tied-best from v4
#   4. Shifted Newton ε=1e-3             — aggressive shift, v3's best at 1.0Å
#
# GRID SIZE: 4 configs × 50 samples = 200 trajectories
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
OUT_DIR="${SCRATCH_DIR}/runs/min_nr_v6_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Fixed configuration
# =============================================================================
N_STEPS=10000
MAX_SAMPLES=50
NOISE_LEVEL="0.5"
SCINE_FUNCTIONAL="DFTB0"
FORCE_CONVERGED="1e-4"
MIN_INTERATOMIC_DIST="0.5"
LOG_SPECTRUM_K=10
TRUST_RADIUS_FLOOR="0.01"

# Unique seed per job — different starting geometries every run
NOISE_SEED="${SLURM_JOB_ID:-$(date +%s)}"

# Parallel worker config
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "=============================================="
echo "Newton-Raphson Minimization — v6 Back-to-Basics"
echo "=============================================="
echo "Date:         $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "CPUs:         $TOTAL_CPUS"
echo "Workers:      $N_WORKERS  (${THREADS_PER_WORKER} threads each)"
echo "=============================================="
echo "Fixed config:"
echo "  NOISE_LEVEL:        ${NOISE_LEVEL} A"
echo "  NOISE_SEED:         $NOISE_SEED"
echo "  N_STEPS:            $N_STEPS"
echo "  MAX_SAMPLES:        $MAX_SAMPLES"
echo "  FORCE_CONVERGED:    $FORCE_CONVERGED"
echo "  TRUST_RADIUS_FLOOR: $TRUST_RADIUS_FLOOR"
echo "  LOG_SPECTRUM_K:     $LOG_SPECTRUM_K"
echo "Configs:  nrt8e-3, SN_2e-4, SN_5e-4, SN_1e-3"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

COMBO=0
FAILED=0

# =============================================================================
# Helper: run one combination
# =============================================================================
run_combo() {
    local tag="$1"
    shift

    local start_from="midpoint_rt_noise${NOISE_LEVEL}A"
    local run_out="${OUT_DIR}/${tag}"
    mkdir -p "$run_out"

    COMBO=$((COMBO + 1))
    echo "[combo ${COMBO}]  ${tag}  (seed=${NOISE_SEED})"

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
            "$@"; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# =============================================================================
# Grid: 4 clean configs
# =============================================================================

# (1) Hard Filter — v1 baseline: skip modes with |λ| < 8e-3
run_combo "mad1.3_nrt8e-3_pgtrue_phfalse" \
    --nr-threshold "8e-3"

# (2) Shifted Newton ε=2e-4 — conservative shift, tied-best from v4
run_combo "mad1.3_se2e-4_pgtrue_phfalse" \
    --shift-epsilon "2e-4"

# (3) Shifted Newton ε=5e-4 — moderate shift, tied-best from v4
run_combo "mad1.3_se5e-4_pgtrue_phfalse" \
    --shift-epsilon "5e-4"

# (4) Shifted Newton ε=1e-3 — aggressive shift, v3's best at 1.0Å
run_combo "mad1.3_se1e-3_pgtrue_phfalse" \
    --shift-epsilon "1e-3"

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
