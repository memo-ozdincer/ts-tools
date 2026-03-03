#!/bin/bash
#SBATCH --job-name=min_pic_arc_v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=09:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/min_pic_arc_v1_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/min_pic_arc_v1_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# PIC-ARC Minimization — v1
# =============================================================================
#
# Clean-sheet two-phase optimizer:
#   Phase A (FLOW): Preconditioned steepest descent + trust region
#   Phase B (ARC):  Subspace cubic regularization + trust region
#
# Design informed by v1-v8 experimental record:
#   - Trust region in both phases (v7 proved TR > line search)
#   - Molecule-aware PD metric (Lindh-type stretch Hessian)
#   - Subspace ARC with negative eigenvectors (shifted Newton insight)
#   - Multi-channel termination (STRICT / RELAXED / STALLED)
#   - No escape mechanisms (all v3/v4 escape attempts failed)
#   - Conservative FLOW->ARC switching (stay first-order if unsure)
#
# Grid: 6 configs x 4 noise levels = 24 combos x 300 samples each
#
# | # | sigma_init | kappa_thresh | accept_relaxed | refresh | Description           |
# |---|-----------|-------------|----------------|---------|------------------------|
# | 1 | 1.0       | 1e6         | false          | 0       | Default baseline       |
# | 2 | 0.1       | 1e6         | false          | 0       | Low sigma (Newton-like)|
# | 3 | 10.0      | 1e6         | false          | 0       | High sigma (regularized)|
# | 4 | 1.0       | 1e4         | false          | 0       | Tight kappa (more FLOW)|
# | 5 | 1.0       | 1e6         | true           | 0       | Accept relaxed conv    |
# | 6 | 1.0       | 1e6         | false          | 500     | Metric refresh         |
#
# Noise levels: 0.5, 1.0, 1.5, 2.0 A
# Total: 24 combos x 300 samples = 7200 trajectories
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
OUT_DIR="${SCRATCH_DIR}/runs/min_pic_arc_v1_${SLURM_JOB_ID}"

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
MAD="1.3"

# Single seed from SLURM_JOB_ID
NOISE_SEED="${SLURM_JOB_ID:-$(date +%s)}"

# Parallel worker config
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "=============================================="
echo "PIC-ARC Minimization — v1"
echo "=============================================="
echo "Date:         $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "CPUs:         $TOTAL_CPUS"
echo "Workers:      $N_WORKERS  (${THREADS_PER_WORKER} threads each)"
echo "=============================================="
echo "Fixed config:"
echo "  MAD:                $MAD"
echo "  NOISE_SEED:         $NOISE_SEED"
echo "  N_STEPS:            $N_STEPS"
echo "  MAX_SAMPLES:        $MAX_SAMPLES"
echo "  FORCE_CONVERGED:    $FORCE_CONVERGED"
echo "  LOG_SPECTRUM_K:     $LOG_SPECTRUM_K"
echo "Configs:  24 (6 PIC-ARC configs x 4 noise levels)"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

COMBO=0
FAILED=0

# =============================================================================
# Helper: run one combination
# Usage: run_pic_arc_v1 <noise_level> <tag> [extra_flags...]
# =============================================================================
run_pic_arc_v1() {
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
            --method pic_arc \
            --max-atom-disp "$MAD" \
            --force-converged "$FORCE_CONVERGED" \
            --min-interatomic-dist "$MIN_INTERATOMIC_DIST" \
            --project-gradient-and-v \
            --log-spectrum-k "$LOG_SPECTRUM_K" \
            "$@"; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# =============================================================================
# Grid: 6 configs x 4 noise levels = 24 combos
# =============================================================================

for NOISE in "0.5" "1.0" "1.5" "2.0"; do

    # Config 1: Default baseline (sigma=1.0, kappa=1e6)
    run_pic_arc_v1 "$NOISE" "n${NOISE}_si1.0_kt1e6_ar0_mr0"

    # Config 2: Low sigma — more Newton-like ARC steps
    run_pic_arc_v1 "$NOISE" "n${NOISE}_si0.1_kt1e6_ar0_mr0" \
        --sigma-init 0.1

    # Config 3: High sigma — more regularized ARC steps
    run_pic_arc_v1 "$NOISE" "n${NOISE}_si10.0_kt1e6_ar0_mr0" \
        --sigma-init 10.0

    # Config 4: Tight kappa — spends more time in FLOW
    run_pic_arc_v1 "$NOISE" "n${NOISE}_si1.0_kt1e4_ar0_mr0" \
        --kappa-threshold 1e4

    # Config 5: Accept relaxed convergence
    run_pic_arc_v1 "$NOISE" "n${NOISE}_si1.0_kt1e6_ar1_mr0" \
        --accept-relaxed

    # Config 6: Metric refresh every 500 steps
    run_pic_arc_v1 "$NOISE" "n${NOISE}_si1.0_kt1e6_ar0_mr500" \
        --metric-refresh-every 500

done

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "PIC-ARC v1 grid completed at $(date)"
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
    --result-glob "*/minimization_pic_arc_*_results.json" \
    --output-dir "$ANALYSIS_DIR" \
    --top-k 24 > "${ANALYSIS_DIR}/report.txt" 2>&1 || true

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
