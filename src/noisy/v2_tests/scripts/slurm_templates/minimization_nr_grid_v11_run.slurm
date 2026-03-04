#!/bin/bash
#SBATCH --job-name=min_nr_v11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v11_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v11_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Newton-Raphson Minimization — v11 PLS-Focused Grid
# =============================================================================
#
# v10c results (champion: RFO+Schlegel+PLS):
#   | Noise | GQT   | GQT+PLS | RFO+Schl+PLS |
#   |-------|-------|---------|--------------|
#   | 0.5 A | 76%   | 83%     | 89%          |
#   | 1.0 A | 55%   | 67%     | 78%          |
#   | 1.5 A | 49%   | 56%     | 63%          |
#   | 2.0 A | 27%   | 45%     | 42%          |
#
# v10c conclusions:
#   - ARC: catastrophic (dropped)
#   - GDIIS: zero effect (dropped)
#   - PLS: breakthrough (+7 to +34 pts)
#   - RFO+Schlegel+PLS: champion at 0.5-1.5A; GQT+PLS wins at 2.0A
#
# v11 focus: explore remaining degrees of freedom with PLS as baseline
#   - Untested: GQT+Schlegel+PLS
#   - Trust radius floor: 0.01 may be too conservative for blind modes
#   - More steps: do almost_converged samples just need more time?
#
# Grid: 8 configs x 4 noise levels = 32 combos x 300 samples each
#
# | # | Algorithm                  | TR floor | Steps | Notes               |
# |---|---------------------------|----------|-------|---------------------|
# | 1 | RFO + Schlegel + PLS      | 0.01     | 10k   | Champion (control)  |
# | 2 | GQT + PLS                 | 0.01     | 10k   | High-noise alt      |
# | 3 | GQT + Schlegel + PLS      | 0.01     | 10k   | NEW: untested combo |
# | 4 | RFO + PLS (no Schlegel)   | 0.01     | 10k   | Isolate Schlegel    |
# | 5 | RFO + Schlegel + PLS      | 0.005    | 10k   | Lower TR floor      |
# | 6 | GQT + Schlegel + PLS      | 0.005    | 10k   | Lower TR floor      |
# | 7 | RFO + Schlegel + PLS      | 0.01     | 20k   | More steps          |
# | 8 | GQT + Schlegel + PLS      | 0.01     | 20k   | More steps          |
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
OUT_DIR="${SCRATCH_DIR}/runs/min_nr_v11_${SLURM_JOB_ID}"

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

# Relaxed convergence for all configs
RELAXED_EVAL_THRESHOLD="0.01"

# Single seed from SLURM_JOB_ID
NOISE_SEED="${SLURM_JOB_ID:-$(date +%s)}"

# Parallel worker config
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "=============================================="
echo "Newton-Raphson Minimization — v11 PLS-Focused"
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
echo "Configs:  32 (8 configs x 4 noise levels)"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

COMBO=0
FAILED=0

# =============================================================================
# Helper: run RFO + PLS config
# Usage: run_rfo_pls <noise> <tag> <schlegel_tr> <tr_floor> <n_steps>
# =============================================================================
run_rfo_pls() {
    local noise_level="$1"
    local tag="$2"
    local schlegel_tr="$3"
    local tr_floor="$4"
    local n_steps="$5"

    local start_from="midpoint_rt_noise${noise_level}A"
    local run_out="${OUT_DIR}/${tag}"
    mkdir -p "$run_out"

    COMBO=$((COMBO + 1))
    echo "[combo ${COMBO}]  ${tag}  (noise=${noise_level}A, steps=${n_steps}, tr_floor=${tr_floor})"

    local extra_flags=""
    if [ "$schlegel_tr" = "1" ]; then
        extra_flags="--schlegel-trust-update"
    fi

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
            --trust-radius-floor "$tr_floor" \
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
# Helper: run GQT + PLS config
# Usage: run_gqt_pls <noise> <tag> <schlegel_tr> <tr_floor> <n_steps>
# =============================================================================
run_gqt_pls() {
    local noise_level="$1"
    local tag="$2"
    local schlegel_tr="$3"
    local tr_floor="$4"
    local n_steps="$5"

    local start_from="midpoint_rt_noise${noise_level}A"
    local run_out="${OUT_DIR}/${tag}"
    mkdir -p "$run_out"

    COMBO=$((COMBO + 1))
    echo "[combo ${COMBO}]  ${tag}  (noise=${noise_level}A, steps=${n_steps}, tr_floor=${tr_floor})"

    local extra_flags=""
    if [ "$schlegel_tr" = "1" ]; then
        extra_flags="--schlegel-trust-update"
    fi

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
            --shift-epsilon "1e-3" \
            --step-control trust_region \
            --polynomial-linesearch \
            --trust-radius-floor "$tr_floor" \
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
# Grid: 8 configs x 4 noise levels = 32 combos
# =============================================================================

for NOISE in "0.5" "1.0" "1.5" "2.0"; do

    # Config 1: RFO + Schlegel + PLS (champion control, 10k steps)
    run_rfo_pls "$NOISE" "n${NOISE}_rfo_str_re${RELAXED_EVAL_THRESHOLD}_ar1_pls_pgtrue_phfalse" \
        "1" "0.01" "10000"

    # Config 2: GQT + PLS (high-noise alternative, 10k steps)
    run_gqt_pls "$NOISE" "n${NOISE}_mad${MAD}_se1e-3_sctr_re${RELAXED_EVAL_THRESHOLD}_ar1_pls_pgtrue_phfalse" \
        "0" "0.01" "10000"

    # Config 3: GQT + Schlegel + PLS (NEW: untested combo, 10k steps)
    run_gqt_pls "$NOISE" "n${NOISE}_mad${MAD}_se1e-3_sctr_str_re${RELAXED_EVAL_THRESHOLD}_ar1_pls_pgtrue_phfalse" \
        "1" "0.01" "10000"

    # Config 4: RFO + PLS, no Schlegel (isolate Schlegel contribution, 10k steps)
    run_rfo_pls "$NOISE" "n${NOISE}_rfo_re${RELAXED_EVAL_THRESHOLD}_ar1_pls_pgtrue_phfalse" \
        "0" "0.01" "10000"

    # Config 5: RFO + Schlegel + PLS, lower TR floor (0.005A, 10k steps)
    run_rfo_pls "$NOISE" "n${NOISE}_rfo_str_trf0.005_re${RELAXED_EVAL_THRESHOLD}_ar1_pls_pgtrue_phfalse" \
        "1" "0.005" "10000"

    # Config 6: GQT + Schlegel + PLS, lower TR floor (0.005A, 10k steps)
    run_gqt_pls "$NOISE" "n${NOISE}_mad${MAD}_se1e-3_sctr_str_trf0.005_re${RELAXED_EVAL_THRESHOLD}_ar1_pls_pgtrue_phfalse" \
        "1" "0.005" "10000"

    # Config 7: RFO + Schlegel + PLS, 20k steps (do almost_converged need more time?)
    run_rfo_pls "$NOISE" "n${NOISE}_rfo_str_20k_re${RELAXED_EVAL_THRESHOLD}_ar1_pls_pgtrue_phfalse" \
        "1" "0.01" "20000"

    # Config 8: GQT + Schlegel + PLS, 20k steps
    run_gqt_pls "$NOISE" "n${NOISE}_mad${MAD}_se1e-3_sctr_str_20k_re${RELAXED_EVAL_THRESHOLD}_ar1_pls_pgtrue_phfalse" \
        "1" "0.01" "20000"

done

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "v11 grid completed at $(date)"
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
    --top-k 32 > "${ANALYSIS_DIR}/report.txt" 2>&1 || true

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
