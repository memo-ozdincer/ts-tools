#!/bin/bash
#SBATCH --job-name=min_nr_v12q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v12q_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v12q_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Newton-Raphson Minimization — v12 Quick Smoke Test
# =============================================================================
#
# Quick sanity check: 300 samples, 3 configs, noise=2.0 only (hardest case).
#
# | # | Config                    | Steps | Purpose                |
# |---|--------------------------|-------|------------------------|
# | 1 | RFO + PLS (baseline)     | 20k   | Control                |
# | 2 | RFO + PLS + both kicks   | 20k   | Do kicks fire?         |
# | 3 | RFO + PLS + both(eig=1)  | 20k   | Ablation fires too?    |
#
# =============================================================================

# Note: no set -e — individual combo failures are caught by run_config().

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
OUT_DIR="${SCRATCH_DIR}/runs/min_nr_v12_quick_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Quick test config: 20 samples, noise=2.0 only
# =============================================================================
MAX_SAMPLES=300
SCINE_FUNCTIONAL="DFTB0"
FORCE_CONVERGED="1e-4"
MIN_INTERATOMIC_DIST="0.5"
LOG_SPECTRUM_K=10
MAD="1.3"
RELAXED_EVAL_THRESHOLD="0.01"
NOISE_SEED="${SLURM_JOB_ID:-$(date +%s)}"
NOISE="2.0"

THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "=============================================="
echo "Newton-Raphson — v12 Quick Smoke Test"
echo "=============================================="
echo "Date:         $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "CPUs:         $TOTAL_CPUS"
echo "Workers:      $N_WORKERS  (${THREADS_PER_WORKER} threads each)"
echo "MAX_SAMPLES:  $MAX_SAMPLES"
echo "NOISE:        $NOISE"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

COMBO=0
FAILED=0

run_config() {
    local tag="$1"
    local n_steps="$2"
    shift 2
    local extra_flags="$*"

    local start_from="midpoint_rt_noise${NOISE}A"
    local run_out="${OUT_DIR}/${tag}"
    mkdir -p "$run_out"

    COMBO=$((COMBO + 1))
    echo "[combo ${COMBO}]  ${tag}  (noise=${NOISE}A, steps=${n_steps})"

    if python "$PROJECT_DIR/src/noisy/v2_tests/runners/run_minimization_parallel.py" \
            --h5-path "$H5_PATH" \
            --out-dir "$run_out" \
            --scine-functional "$SCINE_FUNCTIONAL" \
            --n-steps "$n_steps" \
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
            --optimizer-mode rfo \
            --polynomial-linesearch \
            --trust-radius-floor "0.01" \
            --relaxed-eval-threshold "$RELAXED_EVAL_THRESHOLD" \
            --accept-relaxed \
            $extra_flags; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# =============================================================================
# 3 configs, noise=2.0 only
# =============================================================================

# Config 1: Baseline (control)
run_config "n${NOISE}_rfo_pls_20k" "20000"

# Config 2: Both kicks
run_config "n${NOISE}_rfo_pls_both_kicks_20k" "20000" \
    --osc-kick \
    --osc-kick-scale 0.1 \
    --osc-kick-patience 3 \
    --osc-kick-cooldown 50 \
    --blind-kick \
    --blind-kick-scale 0.5 \
    --blind-kick-overlap-thresh 0.1 \
    --blind-kick-force-thresh 0.1 \
    --blind-kick-patience 100

# Config 3: Both kicks, eigvec=1 ablation
run_config "n${NOISE}_rfo_pls_both_kicks_eigvec1_20k" "20000" \
    --osc-kick \
    --osc-kick-scale 0.1 \
    --osc-kick-patience 3 \
    --osc-kick-cooldown 50 \
    --blind-kick \
    --blind-kick-scale 0.5 \
    --blind-kick-overlap-thresh 0.1 \
    --blind-kick-force-thresh 0.1 \
    --blind-kick-patience 100 \
    --kick-eigvec-index 1

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "v12 quick test completed at $(date)"
echo "  Total combinations: $COMBO"
echo "  Successful:         $((COMBO - FAILED))"
echo "  Failed:             $FAILED"
echo "  Noise seed:         $NOISE_SEED"
echo "  Results:            $OUT_DIR"
echo "=============================================="

# Quick analysis
ANALYSIS_DIR="${OUT_DIR}/analysis"
mkdir -p "$ANALYSIS_DIR"

echo "Running grid analysis ..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_minimization_nr_grid.py" \
    --grid-dir "$OUT_DIR" \
    --result-glob "*/minimization_newton_raphson_*_results.json" \
    --output-dir "$ANALYSIS_DIR" \
    --top-k 3 > "${ANALYSIS_DIR}/report.txt" 2>&1 || true

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
