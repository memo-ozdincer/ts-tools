#!/bin/bash
#SBATCH --job-name=min_nr_v9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=09:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v9_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v9_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Newton-Raphson Minimization — v9 Relaxed Convergence
# =============================================================================
#
# Motivation — v8 cascade data shows the "false-rejection gap":
#
#   | Noise | Strict (n_neg=0) | eval >= -0.01 | Gap   |
#   |-------|-----------------|---------------|-------|
#   | 0.5 A | 79.3%           | 93.0%         | 13.7  |
#   | 1.0 A | 51.3%           | 90.0%         | 38.7  |
#   | 1.5 A | 31.0%           | 87.7%         | 56.7  |
#   | 2.0 A | 19.7%           | 92.7%         | 73.0  |
#
#   The optimizer already reaches near-minimum geometries (eval >= -0.01)
#   for >87% of molecules at ALL noise levels. The drop from 79% to 20%
#   is almost entirely the strict n_neg=0 criterion rejecting geometries
#   with small residual negative eigenvalues.
#
#   Further, only ~10% of DFT reactant minima have n_neg=0 on the DFTB0
#   surface. The strict criterion is fundamentally inappropriate for cross-
#   surface evaluation.
#
# v9 Approach — "Silver Bullet":
#
#   Use the best algorithmic config from v7/v8 experiments:
#     - Shifted Newton (epsilon=1e-3) — consistently best step builder
#     - Trust region (not line search) — v7 proved TR > LS at all noise
#
#   Test relaxed convergence at multiple thresholds:
#     - STRICT: n_neg == 0 (baseline, same as v7/v8 best)
#     - RELAXED at threshold t: min_vib_eval >= -t AND force < converged
#
#   This directly tests the cascade-data hypothesis with zero algorithmic
#   risk — the optimizer runs identically; only the acceptance criterion
#   changes.
#
# Grid: 4 thresholds x 4 noise levels = 16 combos x 300 samples each
#
# | # | relaxed_threshold | accept_relaxed | Description                  |
# |---|------------------|----------------|------------------------------|
# | 1 | 0.0              | false          | Strict baseline (n_neg==0)   |
# | 2 | 0.001            | true           | Tight relaxed                |
# | 3 | 0.005            | true           | Moderate relaxed             |
# | 4 | 0.01             | true           | Standard relaxed             |
#
# Noise levels: 0.5, 1.0, 1.5, 2.0 A
# Total: 16 combos x 300 samples = 4800 trajectories
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
OUT_DIR="${SCRATCH_DIR}/runs/min_nr_v9_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Fixed configuration — best from v7/v8
# =============================================================================
N_STEPS=10000
MAX_SAMPLES=300
SCINE_FUNCTIONAL="DFTB0"
FORCE_CONVERGED="1e-4"
MIN_INTERATOMIC_DIST="0.5"
LOG_SPECTRUM_K=10
SHIFT_EPSILON="1e-3"
MAD="1.3"
STEP_CONTROL="trust_region"

# Single seed from SLURM_JOB_ID
NOISE_SEED="${SLURM_JOB_ID:-$(date +%s)}"

# Parallel worker config
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "=============================================="
echo "Newton-Raphson Minimization — v9 Relaxed Conv"
echo "=============================================="
echo "Date:         $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "CPUs:         $TOTAL_CPUS"
echo "Workers:      $N_WORKERS  (${THREADS_PER_WORKER} threads each)"
echo "=============================================="
echo "Fixed config:"
echo "  SHIFT_EPSILON:      $SHIFT_EPSILON"
echo "  MAD:                $MAD"
echo "  STEP_CONTROL:       $STEP_CONTROL"
echo "  NOISE_SEED:         $NOISE_SEED"
echo "  N_STEPS:            $N_STEPS"
echo "  MAX_SAMPLES:        $MAX_SAMPLES"
echo "  FORCE_CONVERGED:    $FORCE_CONVERGED"
echo "  LOG_SPECTRUM_K:     $LOG_SPECTRUM_K"
echo "Configs:  16 (4 relaxed thresholds x 4 noise levels)"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

COMBO=0
FAILED=0

# =============================================================================
# Helper: run one combination
# Usage: run_v9 <noise_level> <tag> [extra_flags...]
# =============================================================================
run_v9() {
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
            --max-atom-disp "$MAD" \
            --force-converged "$FORCE_CONVERGED" \
            --min-interatomic-dist "$MIN_INTERATOMIC_DIST" \
            --project-gradient-and-v \
            --log-spectrum-k "$LOG_SPECTRUM_K" \
            --shift-epsilon "$SHIFT_EPSILON" \
            --step-control "$STEP_CONTROL" \
            "$@"; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# =============================================================================
# Grid: 4 thresholds x 4 noise levels = 16 combos
# =============================================================================

for NOISE in "0.5" "1.0" "1.5" "2.0"; do

    # Config 1: Strict baseline (n_neg==0, no relaxation)
    run_v9 "$NOISE" "n${NOISE}_mad${MAD}_se${SHIFT_EPSILON}_sctr_re0_ar0_pgtrue_phfalse"

    # Config 2: Tight relaxed (threshold=0.001)
    run_v9 "$NOISE" "n${NOISE}_mad${MAD}_se${SHIFT_EPSILON}_sctr_re0.001_ar1_pgtrue_phfalse" \
        --relaxed-eval-threshold 0.001 \
        --accept-relaxed

    # Config 3: Moderate relaxed (threshold=0.005)
    run_v9 "$NOISE" "n${NOISE}_mad${MAD}_se${SHIFT_EPSILON}_sctr_re0.005_ar1_pgtrue_phfalse" \
        --relaxed-eval-threshold 0.005 \
        --accept-relaxed

    # Config 4: Standard relaxed (threshold=0.01)
    run_v9 "$NOISE" "n${NOISE}_mad${MAD}_se${SHIFT_EPSILON}_sctr_re0.01_ar1_pgtrue_phfalse" \
        --relaxed-eval-threshold 0.01 \
        --accept-relaxed

done

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "v9 grid completed at $(date)"
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
    --top-k 16 > "${ANALYSIS_DIR}/report.txt" 2>&1 || true

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
