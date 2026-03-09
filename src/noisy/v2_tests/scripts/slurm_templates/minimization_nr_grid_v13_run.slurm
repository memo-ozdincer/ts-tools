#!/bin/bash
#SBATCH --job-name=min_nr_v13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v13_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v13_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Newton-Raphson Minimization — v13: 50k + v12b Kicks Combined
# =============================================================================
#
# Summary of all breakthroughs to date:
#   v10c: PLS (polynomial line search)         → +7 to +34 pts
#   v11:  20k steps                             → +6 to +21 pts
#   v12:  50k steps (no kicks)                  → +6 to +13 pts (n≤1.5)
#   v12b: probe kicks + late escape             → +12 pts at n=2.0
#
# Best results so far:
#   n=0.5: 95.1% (v12, 50k)
#   n=1.0: 95.8% (v12, 50k)
#   n=1.5: 92.7% (v12, 50k)
#   n=2.0: 76.0% (v12b, 20k, probe+late) — 50k never tested
#
# v13 hypothesis: 50k + v12b kicks are orthogonal wins.
#   - 50k rescues slow convergers (pure superset of 20k, zero losses)
#   - v12b kicks rescue stuck-in-saddle samples (probe 2.4:1, late 3.2:1)
#   - Combined should push n=2.0 past 80%, possibly toward 90%.
#
# Grid: 3 main configs × 4 noise + 3 ablations at n=2.0 = 15 combos
#
# | # | Config                              | Steps | Noise     | Purpose                    |
# |---|-------------------------------------|-------|-----------|----------------------------|
# | 1 | RFO+PLS baseline                    | 20k   | all       | Same-seed control          |
# | 2 | RFO+PLS baseline                    | 50k   | all       | Step-count reference       |
# | 3 | Full v12b (adaptive+probe+late)     | 50k   | all       | Main experiment            |
# | 4 | Full v12b (adaptive+probe+late)     | 20k   | 2.0 only  | Kick effect control        |
# | 5 | Adaptive+probe (no late)            | 50k   | 2.0 only  | Does 50k replace late?     |
# | 6 | Probe+late (no adaptive)            | 50k   | 2.0 only  | Is adaptive needed at 50k? |
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
OUT_DIR="${SCRATCH_DIR}/runs/min_nr_v13_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Fixed configuration
# =============================================================================
MAX_SAMPLES=300
SCINE_FUNCTIONAL="DFTB0"
FORCE_CONVERGED="1e-4"
MIN_INTERATOMIC_DIST="0.5"
LOG_SPECTRUM_K=10
MAD="1.3"
RELAXED_EVAL_THRESHOLD="0.01"
NOISE_SEED="${SLURM_JOB_ID:-$(date +%s)}"

# Parallel worker config
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "=============================================="
echo "Newton-Raphson Minimization — v13"
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
echo "  MAX_SAMPLES:        $MAX_SAMPLES"
echo "  FORCE_CONVERGED:    $FORCE_CONVERGED"
echo "  LOG_SPECTRUM_K:     $LOG_SPECTRUM_K"
echo "  RELAXED_THRESHOLD:  $RELAXED_EVAL_THRESHOLD"
echo "Configs:  15 (3 x 4 noise + 3 ablations at n=2.0)"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

COMBO=0
FAILED=0

# =============================================================================
# Helper: run RFO + PLS config with optional kicks
# Usage: run_config <noise> <tag> <n_steps> <extra_flags...>
# =============================================================================
run_config() {
    local noise_level="$1"
    local tag="$2"
    local n_steps="$3"
    shift 3
    local extra_flags="$*"

    local start_from="midpoint_rt_noise${noise_level}A"
    local run_out="${OUT_DIR}/${tag}"
    mkdir -p "$run_out"

    COMBO=$((COMBO + 1))
    echo "[combo ${COMBO}]  ${tag}  (noise=${noise_level}A, steps=${n_steps})"

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

# Full v12b kick flags (adaptive + probe + late escape)
FULL_V12B_FLAGS="\
    --osc-kick \
    --osc-kick-patience 3 \
    --osc-kick-cooldown 50 \
    --blind-kick \
    --blind-kick-overlap-thresh 0.1 \
    --blind-kick-force-thresh 0.1 \
    --blind-kick-patience 100 \
    --adaptive-kick-scale \
    --adaptive-kick-C 0.1 \
    --blind-kick-probe \
    --late-escape \
    --late-escape-after 15000 \
    --late-escape-alpha 0.1 \
    --late-escape-cooldown 500"

# =============================================================================
# Main grid: 3 configs × 4 noise levels = 12 combos
# =============================================================================

for NOISE in "0.5" "1.0" "1.5" "2.0"; do

    # Config 1: RFO+PLS baseline, 20k (same-seed control)
    run_config "$NOISE" "n${NOISE}_rfo_pls_20k" \
        "20000"

    # Config 2: RFO+PLS baseline, 50k (step-count reference)
    run_config "$NOISE" "n${NOISE}_rfo_pls_50k" \
        "50000"

    # Config 3: Full v12b at 50k (THE main experiment)
    run_config "$NOISE" "n${NOISE}_rfo_pls_both_kicks_adaptive_probe_late_50k" \
        "50000" \
        $FULL_V12B_FLAGS

done

# =============================================================================
# n=2.0 ablations: 3 configs
# =============================================================================

NOISE="2.0"

# Config 4: Full v12b at 20k (kick effect control, reproduces v12b)
run_config "$NOISE" "n${NOISE}_rfo_pls_both_kicks_adaptive_probe_late_20k" \
    "20000" \
    $FULL_V12B_FLAGS

# Config 5: Adaptive + probe, NO late escape, 50k
# Question: does 50k make late escape unnecessary?
run_config "$NOISE" "n${NOISE}_rfo_pls_both_kicks_adaptive_probe_50k" \
    "50000" \
    --osc-kick \
    --osc-kick-patience 3 \
    --osc-kick-cooldown 50 \
    --blind-kick \
    --blind-kick-overlap-thresh 0.1 \
    --blind-kick-force-thresh 0.1 \
    --blind-kick-patience 100 \
    --adaptive-kick-scale \
    --adaptive-kick-C 0.1 \
    --blind-kick-probe

# Config 6: Probe + late, NO adaptive scaling, 50k
# Question: is adaptive scaling needed at 50k? (was marginal at 20k)
run_config "$NOISE" "n${NOISE}_rfo_pls_both_kicks_probe_late_50k" \
    "50000" \
    --osc-kick \
    --osc-kick-scale 0.1 \
    --osc-kick-patience 3 \
    --osc-kick-cooldown 50 \
    --blind-kick \
    --blind-kick-scale 0.5 \
    --blind-kick-overlap-thresh 0.1 \
    --blind-kick-force-thresh 0.1 \
    --blind-kick-patience 100 \
    --blind-kick-probe \
    --late-escape \
    --late-escape-after 15000 \
    --late-escape-alpha 0.1 \
    --late-escape-cooldown 500

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "v13 grid completed at $(date)"
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
    --top-k 15 > "${ANALYSIS_DIR}/report.txt" 2>&1 || true

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
