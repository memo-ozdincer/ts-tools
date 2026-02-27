#!/bin/bash
#SBATCH --job-name=min_nr_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=09:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v3_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v3_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Newton-Raphson Minimization — v3 Grid Search
# =============================================================================
#
# PROBLEM SUMMARY (from v2 results):
#   Best v2 config: LM μ=0.002, 9/30 converged (30%).
#   Cascade analysis reveals TWO failure populations:
#     Population A (~50%): optimizer gets close but tiny negative eigenvalues
#       (λ ∈ (-0.002, 0)) persist. These are "almost minima."
#     Population B (~33%): 10 samples never converge at any eval threshold.
#       Genuinely stuck, need structural escape.
#
# ROOT CAUSE: The NR step uses |λ| (absolute value Newton), which treats
# negative eigenvalue modes identically to positive ones. It has no mechanism
# to specifically target negative modes. Near convergence, gradient projections
# onto the negative modes are tiny, so progress stalls.
#
# v3 NEW ALGORITHMS:
#
# 1. SHIFTED NEWTON (Phase A)
#    step_i = (g·v_i) / (λ_i + σ), where σ = max(0, -λ_min) + ε
#    Standard Levenberg shift. All effective eigenvalues become positive.
#    Key difference from |λ|: negative modes get LARGER weights than
#    equally-sized positive modes:
#      λ=-0.01, σ=0.011 → weight = 1/0.001 = 1000  (aggressive!)
#      λ=+0.01, σ=0.011 → weight = 1/0.021 ≈ 48    (normal)
#    This makes the step MORE aggressive along the problematic modes.
#    Sweep shift_epsilon ∈ {1e-4, 5e-4, 1e-3, 2e-3}.
#
# 2. LM + ADAPTIVE μ ANNEALING (Phase B)
#    Start with μ=0.002 (best from v2). When close to convergence
#    (n_neg ≤ 2, |λ_min| < 5e-3), reduce μ by anneal_factor (0.1 or 0.5).
#    Lower μ near convergence → more aggressive steps along flat/negative
#    modes without the early-optimization explosion risk.
#
# 3. LM + STAGNATION ESCAPE (Phase C)
#    Use LM μ=0.002 (best from v2) with stagnation detection + targeted
#    negative-mode perturbation. When n_neg unchanged for stagnation_window
#    steps, perturb along the negative eigenvectors (escape_alpha max disp).
#    Optional: negative-mode line search scans along v_neg to find where
#    curvature changes sign.
#
# 4. SHIFTED NEWTON + STAGNATION ESCAPE (Phase D)
#    Combine the best shifted Newton ε with stagnation escape.
#    The shifted Newton handles Population A (aggressive on negative modes);
#    the stagnation escape handles Population B (genuinely stuck).
#
# 5. DIAGNOSTIC RUN (Phase E)
#    Best LM config from v2 with extended steps (50k) and full logging.
#    Purpose: trajectory-level autopsy of all failures.
#
# ── Grid layout ───────────────────────────────────────────────────────────
#
# Phase A — Shifted Newton:
#   4 (shift_epsilon) × 1 (pg) × 1 (ph)  = 4 combos
#
# Phase B — LM + adaptive μ annealing:
#   2 (lm_mu) × 2 (anneal_factor)  = 4 combos
#
# Phase C — LM + stagnation escape:
#   1 (lm_mu=2e-3) × 3 (stagnation_window) × 2 (escape_alpha)  = 6 combos
#
# Phase D — Shifted Newton + stagnation escape:
#   2 (shift_epsilon) × 2 (stagnation_window+escape_alpha)  = 4 combos
#
# Phase E — Diagnostic (extended steps):
#   1 combo (LM μ=0.002, 50k steps, full logging)
#
# Total: 4 + 4 + 6 + 4 + 1 = 19 combinations
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
OUT_DIR="${SCRATCH_DIR}/runs/min_nr_v3_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Fixed configuration
# =============================================================================
N_STEPS="${N_STEPS:-10000}"
MAX_SAMPLES="${MAX_SAMPLES:-15}"
START_FROM="${START_FROM:-midpoint_rt_noise2.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
SCINE_FUNCTIONAL="${SCINE_FUNCTIONAL:-DFTB0}"
FORCE_CONVERGED="${FORCE_CONVERGED:-1e-4}"
MIN_INTERATOMIC_DIST="${MIN_INTERATOMIC_DIST:-0.5}"
LOG_SPECTRUM_K="${LOG_SPECTRUM_K:-10}"
TRUST_RADIUS_FLOOR="${TRUST_RADIUS_FLOOR:-0.01}"

# Parallel worker config
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "=============================================="
echo "Newton-Raphson Minimization — v3 Grid Search"
echo "=============================================="
echo "Date:         $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "CPUs:         $TOTAL_CPUS"
echo "Workers:      $N_WORKERS  (${THREADS_PER_WORKER} threads each)"
echo "=============================================="
echo "Fixed config:"
echo "  N_STEPS:            $N_STEPS"
echo "  MAX_SAMPLES:        $MAX_SAMPLES"
echo "  START_FROM:         $START_FROM"
echo "  FORCE_CONVERGED:    $FORCE_CONVERGED"
echo "  LOG_SPECTRUM_K:     $LOG_SPECTRUM_K"
echo "  TRUST_RADIUS_FLOOR: $TRUST_RADIUS_FLOOR"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

COMBO=0
FAILED=0

# =============================================================================
# Helper: run one combination and track pass/fail
# =============================================================================
run_combo() {
    local tag="$1"
    shift
    local run_out="${OUT_DIR}/${tag}"
    mkdir -p "$run_out"

    COMBO=$((COMBO + 1))
    echo "[combo ${COMBO}]  ${tag}"

    if python "$PROJECT_DIR/src/noisy/v2_tests/runners/run_minimization_parallel.py" \
            --h5-path "$H5_PATH" \
            --out-dir "$run_out" \
            --scine-functional "$SCINE_FUNCTIONAL" \
            --n-steps "$N_STEPS" \
            --max-samples "$MAX_SAMPLES" \
            --start-from "$START_FROM" \
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
# Phase A — Shifted Newton sweep
# step_i = (g·v_i) / (λ_i + σ), σ = max(0, -λ_min) + shift_epsilon
# Negative modes get LARGER weights → more aggressive escape from shallow
# saddle points. This directly addresses Population A.
# =============================================================================
echo "=============================================="
echo "Phase A — Shifted Newton (shift_epsilon sweep)"
echo "  Axes: shift_epsilon ∈ {1e-4, 5e-4, 1e-3, 2e-3}"
echo "=============================================="

for SE in "1e-4" "5e-4" "1e-3" "2e-3"; do
    run_combo "mad1.3_se${SE}_pgtrue_phfalse" \
        --shift-epsilon "$SE"
done

# =============================================================================
# Phase B — LM + adaptive μ annealing
# Start with proven-best μ values; reduce μ when close to convergence
# (n_neg ≤ 2, |λ_min| < 5e-3) to be more aggressive on the last mile.
# =============================================================================
echo "=============================================="
echo "Phase B — LM + adaptive μ annealing"
echo "  Axes: lm_mu ∈ {2e-3, 5e-3} × anneal_factor ∈ {0.1, 0.5}"
echo "=============================================="

for MU in "2e-3" "5e-3"; do
    for AF in "0.1" "0.5"; do
        run_combo "mad1.3_lmmu${MU}_af${AF}_pgtrue_phfalse" \
            --lm-mu "$MU" \
            --lm-mu-anneal-factor "$AF"
    done
done

# =============================================================================
# Phase C — LM + stagnation escape
# Use best LM μ=0.002 with stagnation detection. When n_neg is stuck for
# stagnation_window steps, apply targeted perturbation along negative modes.
# Addresses both Population A (escape from shallow saddle) and Population B
# (genuinely stuck — break out with larger perturbation).
# =============================================================================
echo "=============================================="
echo "Phase C — LM + stagnation escape"
echo "  Axes: stag_window ∈ {50, 100, 200} × escape_alpha ∈ {0.05, 0.1}"
echo "  Fixed: lm_mu=2e-3"
echo "=============================================="

for SW in "50" "100" "200"; do
    for EA in "0.05" "0.1"; do
        run_combo "mad1.3_lmmu2e-3_sw${SW}_ea${EA}_pgtrue_phfalse" \
            --lm-mu "2e-3" \
            --stagnation-window "$SW" \
            --escape-alpha "$EA"
    done
done

# =============================================================================
# Phase D — Shifted Newton + stagnation escape
# Combine shifted Newton (aggressive on negative modes) with stagnation
# escape (break out when stuck). Best of both worlds.
# =============================================================================
echo "=============================================="
echo "Phase D — Shifted Newton + stagnation escape"
echo "  Axes: shift_epsilon ∈ {5e-4, 1e-3} × (sw=100, ea=0.1)"
echo "  Also test with negative-mode line search enabled"
echo "=============================================="

for SE in "5e-4" "1e-3"; do
    run_combo "mad1.3_se${SE}_sw100_ea0.1_pgtrue_phfalse" \
        --shift-epsilon "$SE" \
        --stagnation-window "100" \
        --escape-alpha "0.1"

    run_combo "mad1.3_se${SE}_sw100_ea0.1_ls_pgtrue_phfalse" \
        --shift-epsilon "$SE" \
        --stagnation-window "100" \
        --escape-alpha "0.1" \
        --neg-mode-line-search
done

# =============================================================================
# Phase E — Diagnostic run (extended steps)
# Best v2 config with 50k steps for trajectory-level failure autopsy.
# =============================================================================
echo "=============================================="
echo "Phase E — Diagnostic run (LM μ=0.002, 50k steps)"
echo "=============================================="

run_combo "mad1.3_lmmu2e-3_diag50k_pgtrue_phfalse" \
    --lm-mu "2e-3" \
    --n-steps "50000"

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "Grid search completed at $(date)"
echo "  Total combinations attempted: $COMBO"
echo "  Completed successfully:       $((COMBO - FAILED))"
echo "  Failed:                       $FAILED"
echo "  Results directory:            $OUT_DIR"
echo "=============================================="

# =============================================================================
# Analysis — main grid summary + 2D cascade table
# =============================================================================
ANALYSIS_OUT="${OUT_DIR}/analysis"
mkdir -p "$ANALYSIS_OUT"
echo "Running analysis script (includes 2D cascade table) ..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_minimization_nr_grid.py" \
    --grid-dir "$OUT_DIR" \
    --result-glob "*/minimization_newton_raphson_*_results.json" \
    --output-dir "$ANALYSIS_OUT" \
    --top-k 20 > "${ANALYSIS_OUT}/report.txt" 2>&1
echo "Analysis complete.  Report: ${ANALYSIS_OUT}/report.txt"

# =============================================================================
# Failure autopsy
# =============================================================================
AUTOPSY_OUT="${OUT_DIR}/autopsy"
mkdir -p "$AUTOPSY_OUT"
echo "Running failure autopsy ..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_nr_failure_autopsy.py" \
    --grid-dir "$OUT_DIR" \
    --output-dir "$AUTOPSY_OUT" > "${AUTOPSY_OUT}/autopsy_report.txt" 2>&1
echo "Autopsy report: ${AUTOPSY_OUT}/autopsy_report.txt"

# =============================================================================
# Trajectory plots
# =============================================================================
PLOTS_OUT="${OUT_DIR}/plots"
mkdir -p "$PLOTS_OUT"
echo "Generating trajectory plots ..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/plot_nr_grid_trajectories.py" \
    --grid-dir "$OUT_DIR" \
    --output-dir "$PLOTS_OUT"
echo "Plots saved to: $PLOTS_OUT"

echo "=============================================="
echo "All done.  Results in: $OUT_DIR"
echo "=============================================="
