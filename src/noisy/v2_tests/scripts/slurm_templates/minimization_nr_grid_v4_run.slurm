#!/bin/bash
#SBATCH --job-name=min_nr_v4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=09:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v4_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v4_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Newton-Raphson Minimization — v4 Grid Search
# =============================================================================
#
# PROBLEM SUMMARY (from v3 results):
#   Best v3 config: Shifted Newton ε=5e-4, 12/15 converged (80%).
#   All 132 failures across 19 configs fall into two categories:
#     - oscillating (34.7%): trust radius collapses to floor, then thousands
#       of useless micro-steps dominated by positive-mode components while
#       negative eigenvalues bounce around near zero.
#     - almost_converged (11.6%): very close but can't push the last
#       tiny negative eigenvalues to zero.
#   Zero genuinely_stuck failures.
#
# THREE FAILURE POPULATIONS:
#
#   Population 1 — "Blind modes" (samples 000, 010, 007, 003):
#     5 near-zero negative eigenvalues, gradient overlap ≈ 0.001 for all modes.
#     NR step component ∝ (g·v_i) ≈ 0 → optimizer literally blind to these modes.
#
#   Population 2 — "Crushed by trust radius" (samples 005, 002, 009, 004, 001):
#     n_neg=1-3, moderate gradient overlap (0.13-0.16), but trust radius at
#     floor (0.01) kills the already-small neg-mode step component.
#
#   Population 3 — "True saddle" (sample 012):
#     Large |λ_min| ≈ 0.148, gradient orthogonal to both negative modes.
#
# v4 NEW FEATURES (all default off → backward compatible with v3):
#
# 1. SEPARATE NEG-MODE TRUST RADIUS (--neg-trust-floor)
#    Decomposes NR step into neg/pos eigenvalue subspace components and caps
#    each with independent trust radii. Neg-mode trust radius has its own
#    floor (default 0.05 Å) and grows/shrinks based on eigenvalue improvement.
#    Addresses Pop 1+2: prevents trust collapse from crushing neg-mode steps.
#
# 2. BLIND-MODE CORRECTION (--blind-mode-threshold, --blind-correction-alpha)
#    For negative modes with |g·v_i|/|g| < threshold, adds fixed-magnitude
#    perturbation with alternating sign per step. Zero extra evaluations.
#    Addresses Pop 1: provides exploration along modes invisible to NR.
#
# 3. AGGRESSIVE TRUST RECOVERY (--aggressive-trust-recovery)
#    Softer trust-radius shrink (×0.5 instead of ×0.25) near convergence.
#    Auto-recovery: resets trust to 0.5×max when n_neg decreases; grows ×2
#    when 50-step eigenvalue window shows improvement.
#    Addresses Pop 2: breaks the trust-radius collapse → flat pattern.
#
# 4. BIDIRECTIONAL STAGNATION ESCAPE (--escape-bidirectional)
#    Probes both ±v_0 for the most negative mode, evaluates Hessian at both,
#    picks direction with less negative eigenvalue. Conditional acceptance:
#    only moves if min_eval doesn't worsen. 2-3 extra evals per escape event.
#    Addresses Pop 1+2: smarter escape vs v3's sign(g·v_i) (which is random
#    when overlap ≈ 0).
#
# 5. MODE-FOLLOWING (--mode-follow-eval-threshold, --mode-follow-alpha)
#    For true saddle points (large |λ_min|): bidirectional probe along most
#    negative eigenvector, pick direction that improves eigenvalue. 2 extra
#    evals per event. Only triggers after --mode-follow-after-steps.
#    Addresses Pop 3: handles large negative eigenvalues that NR can't fix.
#
# ── Grid layout ───────────────────────────────────────────────────────────
#
# Phase A — Extended SN baseline (50k steps, ε sweep):         4 combos
# Phase B — Neg-mode trust radius sweep:                       6 combos
# Phase C — Blind correction sweep (with ntf=0.05):            6 combos
# Phase D — Aggressive trust recovery (with ntf+bmt+bca):      2 combos
# Phase E — Stagnation escape overhaul:                         4 combos
# Phase F — Mode-following (with all above):                    2 combos
# Phase G — Kitchen sink (all features, best params):           1 combo
#
# Total: 4 + 6 + 6 + 2 + 4 + 2 + 1 = 25 combinations
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
OUT_DIR="${SCRATCH_DIR}/runs/min_nr_v4_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Fixed configuration
# =============================================================================
N_STEPS="${N_STEPS:-10000}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
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
echo "Newton-Raphson Minimization — v4 Grid Search"
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
# Phase A — Extended Shifted Newton baseline (50k steps)
# Re-run ε sweep from v3 with 50k steps to see if more steps help.
# Also narrows the ε range around the v3 sweet spot (5e-4).
# =============================================================================
echo "=============================================="
echo "Phase A — Shifted Newton baseline (ε sweep, 50k steps)"
echo "  Axes: shift_epsilon ∈ {2e-4, 3e-4, 5e-4, 7e-4}"
echo "=============================================="

for SE in "2e-4" "3e-4" "5e-4" "7e-4"; do
    run_combo "mad1.3_se${SE}_pgtrue_phfalse" \
        --shift-epsilon "$SE"
done

# =============================================================================
# Phase B — Neg-mode trust radius sweep
# Separate trust radius for the negative-eigenvalue subspace, with its own
# floor. Prevents trust-radius collapse from crushing neg-mode step components.
# Sweep ntf (neg-trust-floor) for two best ε values.
# =============================================================================
echo "=============================================="
echo "Phase B — Neg-mode trust radius (ntf sweep)"
echo "  Axes: shift_epsilon ∈ {3e-4, 5e-4} × ntf ∈ {0.03, 0.05, 0.1}"
echo "=============================================="

for SE in "3e-4" "5e-4"; do
    for NTF in "0.03" "0.05" "0.1"; do
        run_combo "mad1.3_se${SE}_ntf${NTF}_pgtrue_phfalse" \
            --shift-epsilon "$SE" \
            --neg-trust-floor "$NTF"
    done
done

# =============================================================================
# Phase C — Blind correction sweep
# For negative modes with near-zero gradient overlap, adds alternating-sign
# perturbations. Combined with ntf=0.05 to protect corrections from trust
# collapse. Sweep blind-correction-alpha.
# bmt (blind-mode-threshold) fixed at 0.05 (modes with overlap < 5%).
# =============================================================================
echo "=============================================="
echo "Phase C — Blind correction (bca sweep, ntf=0.05)"
echo "  Axes: shift_epsilon ∈ {3e-4, 5e-4} × bca ∈ {0.01, 0.02, 0.05}"
echo "  Fixed: ntf=0.05, bmt=0.05"
echo "=============================================="

for SE in "3e-4" "5e-4"; do
    for BCA in "0.01" "0.02" "0.05"; do
        run_combo "mad1.3_se${SE}_ntf0.05_bmt0.05_bca${BCA}_pgtrue_phfalse" \
            --shift-epsilon "$SE" \
            --neg-trust-floor "0.05" \
            --blind-mode-threshold "0.05" \
            --blind-correction-alpha "$BCA"
    done
done

# =============================================================================
# Phase D — Aggressive trust recovery
# Add softer trust shrink + auto-recovery on top of ntf + blind correction.
# Tests whether breaking the trust-radius flat pattern helps convergence.
# =============================================================================
echo "=============================================="
echo "Phase D — Aggressive trust recovery (ntf+bmt+bca+atr)"
echo "  Axes: shift_epsilon ∈ {3e-4, 5e-4}"
echo "  Fixed: ntf=0.05, bmt=0.05, bca=0.02, atr=on"
echo "=============================================="

for SE in "3e-4" "5e-4"; do
    run_combo "mad1.3_se${SE}_ntf0.05_bmt0.05_bca0.02_atr_pgtrue_phfalse" \
        --shift-epsilon "$SE" \
        --neg-trust-floor "0.05" \
        --blind-mode-threshold "0.05" \
        --blind-correction-alpha "0.02" \
        --aggressive-trust-recovery
done

# =============================================================================
# Phase E — Stagnation escape overhaul
# Test bidirectional escape (v4) vs original (v3) with all features enabled.
# Sweep escape_alpha with stagnation_window=100.
# =============================================================================
echo "=============================================="
echo "Phase E — Stagnation escape overhaul"
echo "  Axes: escape_alpha ∈ {0.03, 0.05} × {ebd, no-ebd}"
echo "  Fixed: se=5e-4, ntf=0.05, bmt=0.05, bca=0.02, atr=on, sw=100"
echo "=============================================="

for EA in "0.03" "0.05"; do
    # v3-style escape (sign(g·v_i))
    run_combo "mad1.3_se5e-4_ntf0.05_bmt0.05_bca0.02_atr_sw100_ea${EA}_pgtrue_phfalse" \
        --shift-epsilon "5e-4" \
        --neg-trust-floor "0.05" \
        --blind-mode-threshold "0.05" \
        --blind-correction-alpha "0.02" \
        --aggressive-trust-recovery \
        --stagnation-window "100" \
        --escape-alpha "$EA"

    # v4 bidirectional escape
    run_combo "mad1.3_se5e-4_ntf0.05_bmt0.05_bca0.02_atr_sw100_ea${EA}_ebd_pgtrue_phfalse" \
        --shift-epsilon "5e-4" \
        --neg-trust-floor "0.05" \
        --blind-mode-threshold "0.05" \
        --blind-correction-alpha "0.02" \
        --aggressive-trust-recovery \
        --stagnation-window "100" \
        --escape-alpha "$EA" \
        --escape-bidirectional
done

# =============================================================================
# Phase F — Mode-following for true saddle points (Pop 3)
# Bidirectional probe along most negative eigenvector for large |λ_min|.
# Test two eval thresholds. Uses all other v4 features.
# =============================================================================
echo "=============================================="
echo "Phase F — Mode-following (mft sweep)"
echo "  Axes: mode_follow_eval_threshold ∈ {0.01, 0.05}"
echo "  Fixed: se=5e-4, ntf=0.05, bmt=0.05, bca=0.02, atr=on, sw=100, ea=0.05, ebd=on"
echo "=============================================="

for MFT in "0.01" "0.05"; do
    run_combo "mad1.3_se5e-4_ntf0.05_bmt0.05_bca0.02_atr_sw100_ea0.05_ebd_mft${MFT}_pgtrue_phfalse" \
        --shift-epsilon "5e-4" \
        --neg-trust-floor "0.05" \
        --blind-mode-threshold "0.05" \
        --blind-correction-alpha "0.02" \
        --aggressive-trust-recovery \
        --stagnation-window "100" \
        --escape-alpha "0.05" \
        --escape-bidirectional \
        --mode-follow-eval-threshold "$MFT" \
        --mode-follow-alpha "0.15"
done

# =============================================================================
# Phase G — Kitchen sink (all features, best guessed params)
# Single run with everything enabled at what we think are optimal values.
# =============================================================================
echo "=============================================="
echo "Phase G — Kitchen sink (all v4 features)"
echo "=============================================="

run_combo "mad1.3_se5e-4_ntf0.05_bmt0.05_bca0.02_atr_sw100_ea0.05_ebd_mft0.01_mfa0.15_pgtrue_phfalse" \
    --shift-epsilon "5e-4" \
    --neg-trust-floor "0.05" \
    --blind-mode-threshold "0.05" \
    --blind-correction-alpha "0.02" \
    --aggressive-trust-recovery \
    --stagnation-window "100" \
    --escape-alpha "0.05" \
    --escape-bidirectional \
    --mode-follow-eval-threshold "0.01" \
    --mode-follow-alpha "0.15" \
    --mode-follow-after-steps "2000"

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
    --top-k 25 > "${ANALYSIS_OUT}/report.txt" 2>&1
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

# =============================================================================
# Eigenvalue tolerance justification
# =============================================================================
EIGVAL_OUT="${OUT_DIR}/eigenvalue_justification"
mkdir -p "$EIGVAL_OUT"
echo "Running eigenvalue tolerance justification analysis ..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_nr_eigenvalue_justification.py" \
    --grid-dir "$OUT_DIR" --output-dir "$EIGVAL_OUT" \
    > "${EIGVAL_OUT}/report.txt" 2>&1
echo "Eigenvalue justification report: ${EIGVAL_OUT}/report.txt"

# =============================================================================
# Surface forensics
# =============================================================================
FORENSICS_OUT="${OUT_DIR}/surface_forensics"
mkdir -p "$FORENSICS_OUT"
echo "Running surface forensics analysis ..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_nr_surface_forensics.py" \
    --grid-dir "$OUT_DIR" --output-dir "$FORENSICS_OUT" \
    > "${FORENSICS_OUT}/report.txt" 2>&1
echo "Surface forensics report: ${FORENSICS_OUT}/report.txt"

echo "=============================================="
echo "All done.  Results in: $OUT_DIR"
echo "=============================================="
