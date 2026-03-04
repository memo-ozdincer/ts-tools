#!/bin/bash
#SBATCH --job-name=min_nr_v10c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v10c_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v10c_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Newton-Raphson Minimization — v10c Future Work Validation
# =============================================================================
#
# v9 results:
#   | Noise | Strict | Relaxed(-0.01) |
#   |-------|--------|----------------|
#   | 0.5 A | 80.5%  | 80.5%          |
#   | 1.0 A | 59.9%  | 59.9%          |
#   | 1.5 A | 46.3%  | 46.3%          |
#   | 2.0 A | 34.5%  | 34.5%          |
#
#   Root cause: trust radius collapse. Near-zero modes get 1000x amplification
#   -> oversized steps -> trust shrinks 0.25x per bad step -> floor at 0.01A.
#   User proved 100% is achievable via brute-force "v2-kicking".
#
# v10 approach: ARC + RFO + Schlegel TR fixes + GDIIS late-stage
#   - ARC: replaces shifted Newton + trust region (O(eps^{-3/2}))
#   - RFO: augmented Hessian, adaptive shift (chemistry standard)
#   - Schlegel TR: boundary-check growth + step-anchored shrink
#   - GDIIS: oscillation breaking + late-stage acceleration (Schlegel 2011)
#
# v10c additions:
#   - Polynomial line-search refinement near inflection crossings (trust-region path)
#   - Transition1x minima verification script (DFTB0 Hessians at DFT minima)
#   - SCINE convergence investigation script
#
# Grid: 14 configs x 4 noise levels = 56 combos x 300 samples each
#
# | # | Algorithm            | sigma_init | gamma1 | GDIIS | Notes              |
# |---|---------------------|------------|--------|-------|--------------------|
# | 1 | v9 baseline         | -          | -      | off   | Control (SN+TR)    |
# | 2 | ARC                 | 0.1        | 2.0    | off   | Low regularization |
# | 3 | ARC                 | 1.0        | 2.0    | off   | Default            |
# | 4 | ARC                 | 10.0       | 2.0    | off   | High regularization|
# | 5 | ARC                 | 1.0        | 3.0    | off   | Aggressive sigma   |
# | 6 | ARC + GDIIS-8       | 1.0        | 2.0    | 8     | Oscillation damp   |
# | 7 | ARC + GDIIS-12      | 1.0        | 2.0    | 12    | Larger buffer      |
# | 8 | RFO + TR            | -          | -      | off   | Chemistry standard |
# | 9 | RFO + Schlegel TR   | -          | -      | off   | RFO + fixed TR     |
# |10 | RFO + SchlTR + GDIIS| -          | -      | 8+late| Full RFO           |
# |11 | ARC + GDIIS-late    | 1.0        | 2.0    | 8+late| ARC late accel     |
# |12 | v9 + Schlegel TR    | -          | -      | off   | TR fix only        |
# |13 | v9 + polynomial LS  | -          | -      | off   | Schlegel inflection fix |
# |14 | RFO + SchlTR + PLS  | -          | -      | off   | RFO + polynomial refinement |
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
OUT_DIR="${SCRATCH_DIR}/runs/min_nr_v10c_${SLURM_JOB_ID}"

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

# Relaxed convergence for all configs
RELAXED_EVAL_THRESHOLD="0.01"

# Single seed from SLURM_JOB_ID
NOISE_SEED="${SLURM_JOB_ID:-$(date +%s)}"

# Parallel worker config
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "=============================================="
echo "Newton-Raphson Minimization — v10c Future Work"
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
echo "  RELAXED_THRESHOLD:  $RELAXED_EVAL_THRESHOLD"
echo "Configs:  56 (14 configs x 4 noise levels)"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

COMBO=0
FAILED=0

# =============================================================================
# Helper: run v9 baseline (shifted Newton + trust region)
# Usage: run_v9_baseline <noise_level> <tag> [extra_flags...]
# =============================================================================
run_v9_baseline() {
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
            --shift-epsilon "1e-3" \
            --step-control trust_region \
            --relaxed-eval-threshold "$RELAXED_EVAL_THRESHOLD" \
            --accept-relaxed \
            "$@"; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# =============================================================================
# Helper: run ARC config
# Usage: run_arc <noise_level> <tag> <sigma_init> <gamma1> <gdiis_buffer>
# =============================================================================
run_arc() {
    local noise_level="$1"
    local tag="$2"
    local sigma_init="$3"
    local gamma1="$4"
    local gdiis_buffer="$5"

    local start_from="midpoint_rt_noise${noise_level}A"
    local run_out="${OUT_DIR}/${tag}"
    mkdir -p "$run_out"

    COMBO=$((COMBO + 1))
    echo "[combo ${COMBO}]  ${tag}  (noise=${noise_level}A, seed=${NOISE_SEED})"

    local gdiis_flags=""
    if [ "$gdiis_buffer" -gt 0 ]; then
        gdiis_flags="--gdiis-buffer-size $gdiis_buffer --gdiis-every 5"
    fi

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
            --optimizer-mode arc \
            --arc-sigma-init "$sigma_init" \
            --arc-gamma1 "$gamma1" \
            --relaxed-eval-threshold "$RELAXED_EVAL_THRESHOLD" \
            --accept-relaxed \
            $gdiis_flags; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# =============================================================================
# Helper: run RFO config
# Usage: run_rfo <noise_level> <tag> <gdiis_buffer> <gdiis_late_force> <schlegel_tr> <poly_ls>
# =============================================================================
run_rfo() {
    local noise_level="$1"
    local tag="$2"
    local gdiis_buffer="$3"
    local gdiis_late_force="$4"
    local schlegel_tr="$5"
    local poly_ls="$6"

    local start_from="midpoint_rt_noise${noise_level}A"
    local run_out="${OUT_DIR}/${tag}"
    mkdir -p "$run_out"

    COMBO=$((COMBO + 1))
    echo "[combo ${COMBO}]  ${tag}  (noise=${noise_level}A, seed=${NOISE_SEED})"

    local extra_flags=""
    if [ "$gdiis_buffer" -gt 0 ]; then
        extra_flags="--gdiis-buffer-size $gdiis_buffer --gdiis-every 5"
    fi
    if [ "$gdiis_late_force" != "0" ]; then
        extra_flags="$extra_flags --gdiis-late-force-threshold $gdiis_late_force"
    fi
    if [ "$schlegel_tr" = "1" ]; then
        extra_flags="$extra_flags --schlegel-trust-update"
    fi
    if [ "$poly_ls" = "1" ]; then
        extra_flags="$extra_flags --polynomial-linesearch"
    fi

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
            --optimizer-mode rfo \
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
# Helper: run v9 baseline with Schlegel TR fix
# Usage: run_v9_schlegel <noise_level> <tag>
# =============================================================================
run_v9_schlegel() {
    local noise_level="$1"
    local tag="$2"

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
            --shift-epsilon "1e-3" \
            --step-control trust_region \
            --schlegel-trust-update \
            --relaxed-eval-threshold "$RELAXED_EVAL_THRESHOLD" \
            --accept-relaxed; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# =============================================================================
# Helper: run ARC with late-stage GDIIS
# Usage: run_arc_late <noise_level> <tag> <sigma_init> <gamma1> <gdiis_buffer> <gdiis_late_force>
# =============================================================================
run_arc_late() {
    local noise_level="$1"
    local tag="$2"
    local sigma_init="$3"
    local gamma1="$4"
    local gdiis_buffer="$5"
    local gdiis_late_force="$6"

    local start_from="midpoint_rt_noise${noise_level}A"
    local run_out="${OUT_DIR}/${tag}"
    mkdir -p "$run_out"

    COMBO=$((COMBO + 1))
    echo "[combo ${COMBO}]  ${tag}  (noise=${noise_level}A, seed=${NOISE_SEED})"

    local gdiis_flags=""
    if [ "$gdiis_buffer" -gt 0 ]; then
        gdiis_flags="--gdiis-buffer-size $gdiis_buffer --gdiis-every 5"
    fi

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
            --optimizer-mode arc \
            --arc-sigma-init "$sigma_init" \
            --arc-gamma1 "$gamma1" \
            --gdiis-late-force-threshold "$gdiis_late_force" \
            --relaxed-eval-threshold "$RELAXED_EVAL_THRESHOLD" \
            --accept-relaxed \
            $gdiis_flags; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# =============================================================================
# Grid: 14 configs x 4 noise levels = 56 combos
# =============================================================================

for NOISE in "0.5" "1.0" "1.5" "2.0"; do

    # Config 1: v9 baseline (shifted Newton + trust region)
    run_v9_baseline "$NOISE" "n${NOISE}_mad${MAD}_se1e-3_sctr_re${RELAXED_EVAL_THRESHOLD}_ar1_pgtrue_phfalse"

    # Config 2: ARC, sigma_init=0.1, gamma1=2.0, no GDIIS
    run_arc "$NOISE" "n${NOISE}_arc_si0.1_g12.0_re${RELAXED_EVAL_THRESHOLD}_ar1_pgtrue_phfalse" \
        "0.1" "2.0" "0"

    # Config 3: ARC, sigma_init=1.0, gamma1=2.0, no GDIIS
    run_arc "$NOISE" "n${NOISE}_arc_si1.0_g12.0_re${RELAXED_EVAL_THRESHOLD}_ar1_pgtrue_phfalse" \
        "1.0" "2.0" "0"

    # Config 4: ARC, sigma_init=10.0, gamma1=2.0, no GDIIS
    run_arc "$NOISE" "n${NOISE}_arc_si10.0_g12.0_re${RELAXED_EVAL_THRESHOLD}_ar1_pgtrue_phfalse" \
        "10.0" "2.0" "0"

    # Config 5: ARC, sigma_init=1.0, gamma1=3.0, no GDIIS
    run_arc "$NOISE" "n${NOISE}_arc_si1.0_g13.0_re${RELAXED_EVAL_THRESHOLD}_ar1_pgtrue_phfalse" \
        "1.0" "3.0" "0"

    # Config 6: ARC + GDIIS-8, sigma_init=1.0, gamma1=2.0
    run_arc "$NOISE" "n${NOISE}_arc_si1.0_g12.0_gd8_re${RELAXED_EVAL_THRESHOLD}_ar1_pgtrue_phfalse" \
        "1.0" "2.0" "8"

    # Config 7: ARC + GDIIS-12, sigma_init=1.0, gamma1=2.0
    run_arc "$NOISE" "n${NOISE}_arc_si1.0_g12.0_gd12_re${RELAXED_EVAL_THRESHOLD}_ar1_pgtrue_phfalse" \
        "1.0" "2.0" "12"

    # Config 8: RFO + trust region (chemistry standard, no GDIIS)
    run_rfo "$NOISE" "n${NOISE}_rfo_re${RELAXED_EVAL_THRESHOLD}_ar1_pgtrue_phfalse" \
        "0" "0" "0" "0"

    # Config 9: RFO + Schlegel trust region (RFO + corrected TR rules)
    run_rfo "$NOISE" "n${NOISE}_rfo_str_re${RELAXED_EVAL_THRESHOLD}_ar1_pgtrue_phfalse" \
        "0" "0" "1" "0"

    # Config 10: RFO + Schlegel TR + GDIIS-8 late-stage (full RFO)
    run_rfo "$NOISE" "n${NOISE}_rfo_gd8_glf0.01_str_re${RELAXED_EVAL_THRESHOLD}_ar1_pgtrue_phfalse" \
        "8" "0.01" "1" "0"

    # Config 11: ARC + GDIIS-8 late-stage (ARC with late acceleration, force < 0.01)
    run_arc_late "$NOISE" "n${NOISE}_arc_si1.0_g12.0_gd8_glf0.01_re${RELAXED_EVAL_THRESHOLD}_ar1_pgtrue_phfalse" \
        "1.0" "2.0" "8" "0.01"

    # Config 12: v9 baseline + Schlegel TR fix (control for TR fix)
    run_v9_schlegel "$NOISE" "n${NOISE}_mad${MAD}_se1e-3_sctr_str_re${RELAXED_EVAL_THRESHOLD}_ar1_pgtrue_phfalse"

    # Config 13: v9 baseline + polynomial line-search refinement (v10c)
    run_v9_baseline "$NOISE" "n${NOISE}_mad${MAD}_se1e-3_sctr_re${RELAXED_EVAL_THRESHOLD}_ar1_pls_pgtrue_phfalse" \
        --polynomial-linesearch

    # Config 14: RFO + Schlegel TR + polynomial line-search refinement (v10c)
    run_rfo "$NOISE" "n${NOISE}_rfo_str_re${RELAXED_EVAL_THRESHOLD}_ar1_pls_pgtrue_phfalse" \
        "0" "0" "1" "1"

done

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "v10c grid completed at $(date)"
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
    --top-k 56 > "${ANALYSIS_DIR}/report.txt" 2>&1 || true

echo "Running failure autopsy ..."
AUTOPSY_DIR="${OUT_DIR}/autopsy"
mkdir -p "$AUTOPSY_DIR"
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_nr_failure_autopsy.py" \
    --grid-dir "$OUT_DIR" \
    --output-dir "$AUTOPSY_DIR" > "${AUTOPSY_DIR}/autopsy_report.txt" 2>&1 || true

echo "Running Transition1x minima verification ..."
VERIFY_DIR="${OUT_DIR}/verify_transition1x"
mkdir -p "$VERIFY_DIR"
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/verify_transition1x_minima.py" \
    --h5-path "$H5_PATH" \
    --out-dir "$VERIFY_DIR" \
    --split test \
    --max-samples "$MAX_SAMPLES" \
    --scine-functional "$SCINE_FUNCTIONAL" \
    --results-dir "$OUT_DIR" \
    --n-workers "$N_WORKERS" \
    --threads-per-worker "$THREADS_PER_WORKER" > "${VERIFY_DIR}/verify_report.txt" 2>&1 || true

echo "Running SCINE convergence investigation ..."
SCF_INVEST_DIR="${OUT_DIR}/scine_convergence"
mkdir -p "$SCF_INVEST_DIR"
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/investigate_scine_convergence.py" \
    --h5-path "$H5_PATH" \
    --out-dir "$SCF_INVEST_DIR" \
    --split test \
    --max-samples 10 \
    --scine-functional "$SCINE_FUNCTIONAL" > "${SCF_INVEST_DIR}/investigation_report.txt" 2>&1 || true

echo "=============================================="
echo "All done.  Results in: $OUT_DIR"
echo "  Grid report:    ${ANALYSIS_DIR}/report.txt"
echo "  Autopsy report: ${AUTOPSY_DIR}/autopsy_report.txt"
echo "  Minima verify:  ${VERIFY_DIR}/verify_transition1x_minima_report.txt"
echo "  SCINE study:    ${SCF_INVEST_DIR}/investigate_scine_convergence_report.txt"
echo "=============================================="
