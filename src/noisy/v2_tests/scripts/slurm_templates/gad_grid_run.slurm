#!/bin/bash
#SBATCH --job-name=gad_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/gad_grid_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/gad_grid_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# GAD (Gradient Ascent Deflation) — Algorithmic Improvements Grid Search (v2)
# =============================================================================
#
# GAD dynamics:
#   ĝ_GAD = F_GAD / ||F_GAD||,   F_GAD = F + 2*(F·v)v  (Eckart-projected)
#   x_{k+1} = x_k + Δ_k * ĝ_GAD(x_k)
#
#   where Δ_k is the adaptive trust radius (initialized to max_atom_disp).
#   Trust radius adapted via |ρ| = |dE_actual / dE_pred|:
#     |ρ| > 0.75  → grow  Δ by 1.5x  (good quadratic fit)
#     |ρ| < 0.25  → shrink Δ by 0.5x (poor fit)
#     |ρ| < 0.1   → reject step, shrink by 0.25x, retry up to 10 times
#
# Convergence criterion (GAD ≠ NR minimization):
#   eig_product = λ_0 * λ_1 < -ts_eps
#   Requires exactly one negative mode (λ_0 < 0) and one positive (λ_1 > 0),
#   so n_neg == 1, i.e. Morse index 1 = true first-order saddle point (TS).
#
# ── What changed in v2 ────────────────────────────────────────────────────
#
# 1. CASCADING EVALUATION (mandatory, always on)
#    Every successful convergence (and failure) records n_neg_at_<T> for
#    thresholds T ∈ [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2].
#
#    GAD-specific cascade logic:
#      "success at eval T" means n_neg_at_T == 1  (exactly Morse-1 at the
#       level of accepting eigenvalues down to -T as "zero")
#
#    Two sub-tables are reported:
#      rate_eq1  — n_neg_at_T == 1  (true TS, not overshot)
#      rate_le1  — n_neg_at_T <= 1  (would also count minimums)
#    The rate_le1 > rate_eq1 gap reveals overshooting (algorithm passed
#    through the TS into a minimum without stopping).
#    The rate_eq1[T=2e-3] >> rate_eq1[T=0] gap reveals false rejection
#    (algorithm found a TS but residual flat mode caused eig_product to miss).
#
#    Pure diagnostics — never changes trust radius, step, or convergence gate.
#
# 2. NEGATIVE EIGENVALUE MAGNITUDE DIAGNOSTICS (mandatory, always on)
#    At every convergence (and failure) the runner logs:
#      lambda_0        — most-negative eigenvalue (the climbing/TS mode)
#      lambda_1        — second eigenvalue (first of positive ladder)
#      abs_lambda_0    — |lambda_0|
#      lambda_gap_ratio — |lambda_0| / |lambda_1|
#      bottom_spectrum — sorted bottom-10 vibrational eigenvalues
#
#    KEY SCIENCE QUESTION:  Is λ_0 meaningfully different from the noise?
#    gap_ratio >> 1  → TS mode is well-separated; convergence is reliable.
#    gap_ratio ≈ 1   → TS mode buried in noise; "converter" may be spurious.
#    Reported per-combo and in the cascade CSV.
#
# 3. ts_eps SWEEP — relaxed convergence criterion (Phase B)
#    ts_eps controls how strict the eig_product gate is.
#    ts_eps = 1e-5 (default): very strict; λ_0 and λ_1 must be well-separated.
#    ts_eps = 1e-3: more permissive; accepts shallower sign separation.
#    Combined with cascade analysis this distinguishes:
#      "tight ts_eps rejects cases where the algorithm converged but λ_1 is small"
#      from "the algorithm genuinely hasn't reached a clean Morse-1 geometry".
#
# 4. FULL SPECTRUM LOGGING (--log-spectrum-k 10, always on)
#    Bottom-10 vibrational eigenvalues logged at convergence/failure.
#
# ── Grid layout ────────────────────────────────────────────────────────────
#
# Fixed (from NR study / prior GAD runs):
#   - project_gradient_and_v = true  (Eckart, mandatory)
#   - purify_hessian = false          (no measurable benefit)
#   - START_FROM = midpoint_rt_noise2.0A  (same as NR grid)
#   - Best MAD from prior run used as the primary; full sweep in Phase A.
#
# Phase A — Cascade baseline (replays prior best configs, adds cascade):
#   4 (mad) × 2 (baseline) × 1 (tr, diagnostic only) = 8 combos
#   Outcome: cascade table reveals false-rejection vs genuine failures.
#
# Phase B — ts_eps sweep (relaxed convergence criterion):
#   best mad × 2 (baseline) × 5 (ts_eps) = 10 combos
#   Outcome: shows how strict/permissive the eig_product gate should be.
#
# Phase C — tr_filter_eig sweep (legacy mode, mad=1.3, ts_eps=1e-5):
#   best mad × 2 (baseline) × 4 (tr_threshold) = 8 combos
#   Outcome: recovers old performance, quantifies soft-mode contamination in λ_1.
#
# Total: 8 + 10 + 8 = 26 combinations
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
OUT_DIR="${SCRATCH_DIR}/runs/gad_grid_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Fixed configuration
# =============================================================================
N_STEPS="${N_STEPS:-10000}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
START_FROM="${START_FROM:-midpoint_rt_noise2.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
SCINE_FUNCTIONAL="${SCINE_FUNCTIONAL:-DFTB0}"
MIN_INTERATOMIC_DIST="${MIN_INTERATOMIC_DIST:-0.5}"
LOG_SPECTRUM_K="${LOG_SPECTRUM_K:-10}"

# TR diagnostic threshold — NOT an algorithmic param, NOT swept.
# Logged per-step for residual-mode monitoring only.
TR_DIAGNOSTIC="${TR_DIAGNOSTIC:-8e-3}"

# Eckart projection is mandatory (from NR findings: critical for stability)
PROJECT_GRADIENT_FLAG="--project-gradient-and-v"
# Purification has no measurable impact; skip to save compute
PURIFY_FLAG=""

# Parallel worker config
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "=============================================="
echo "GAD — v2 Grid Search"
echo "=============================================="
echo "Date:          $(date)"
echo "SLURM_JOB_ID:  $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "CPUs:          $TOTAL_CPUS"
echo "Workers:       $N_WORKERS  (${THREADS_PER_WORKER} threads each)"
echo "=============================================="
echo "Fixed config:"
echo "  N_STEPS:           $N_STEPS"
echo "  MAX_SAMPLES:       $MAX_SAMPLES"
echo "  START_FROM:        $START_FROM"
echo "  LOG_SPECTRUM_K:    $LOG_SPECTRUM_K"
echo "  TR_DIAGNOSTIC:     $TR_DIAGNOSTIC  (logging only, not algorithmic)"
echo "  project-gradient:  always on (Eckart mandatory)"
echo "  purify-hessian:    off (no measurable benefit from NR study)"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

COMBO=0
FAILED=0

# =============================================================================
# Helper: run one combination and track pass/fail.
# Usage: run_combo <tag> <baseline> <mad> <ts_eps> [extra args...]
# Extra args are passed directly to the python runner (e.g. --tr-filter-eig,
# --tr-threshold <val>).
# =============================================================================
run_combo() {
    local tag="$1"
    local baseline="$2"
    local mad="$3"
    local ts_eps="$4"
    shift 4
    local run_out="${OUT_DIR}/${tag}"
    mkdir -p "$run_out"

    COMBO=$((COMBO + 1))
    echo "[combo ${COMBO}]  ${tag}"

    if python "$PROJECT_DIR/src/noisy/v2_tests/runners/run_gad_baselines_parallel.py" \
            --h5-path "$H5_PATH" \
            --out-dir "$run_out" \
            --scine-functional "$SCINE_FUNCTIONAL" \
            --n-steps "$N_STEPS" \
            --max-samples "$MAX_SAMPLES" \
            --start-from "$START_FROM" \
            --noise-seed "$NOISE_SEED" \
            --n-workers "$N_WORKERS" \
            --threads-per-worker "$THREADS_PER_WORKER" \
            --baseline "$baseline" \
            --max-atom-disp "$mad" \
            --min-interatomic-dist "$MIN_INTERATOMIC_DIST" \
            --ts-eps "$ts_eps" \
            --tr-threshold "$TR_DIAGNOSTIC" \
            --projection-mode eckart_full \
            --log-spectrum-k "$LOG_SPECTRUM_K" \
            $PROJECT_GRADIENT_FLAG \
            $PURIFY_FLAG \
            "$@"; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# Helper: run_combo_tr_filter — same as run_combo but always passes
# --tr-filter-eig and allows overriding the tr-threshold.
# Usage: run_combo_tr_filter <tag> <baseline> <mad> <ts_eps> <tr_thr>
run_combo_tr_filter() {
    local tag="$1"
    local baseline="$2"
    local mad="$3"
    local ts_eps="$4"
    local tr_thr="$5"
    shift 5
    local run_out="${OUT_DIR}/${tag}"
    mkdir -p "$run_out"

    COMBO=$((COMBO + 1))
    echo "[combo ${COMBO}]  ${tag}"

    if python "$PROJECT_DIR/src/noisy/v2_tests/runners/run_gad_baselines_parallel.py" \
            --h5-path "$H5_PATH" \
            --out-dir "$run_out" \
            --scine-functional "$SCINE_FUNCTIONAL" \
            --n-steps "$N_STEPS" \
            --max-samples "$MAX_SAMPLES" \
            --start-from "$START_FROM" \
            --noise-seed "$NOISE_SEED" \
            --n-workers "$N_WORKERS" \
            --threads-per-worker "$THREADS_PER_WORKER" \
            --baseline "$baseline" \
            --max-atom-disp "$mad" \
            --min-interatomic-dist "$MIN_INTERATOMIC_DIST" \
            --ts-eps "$ts_eps" \
            --tr-threshold "$tr_thr" \
            --tr-filter-eig \
            --projection-mode eckart_full \
            --log-spectrum-k "$LOG_SPECTRUM_K" \
            $PROJECT_GRADIENT_FLAG \
            $PURIFY_FLAG \
            "$@"; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# =============================================================================
# Phase A — Cascade baseline (new code, no tr_filter_eig)
# Replays the prior grid (mad × baseline sweep) with cascade logging ON.
# Uses new get_vib_evals_evecs path: raw vibrational eigenvalues, no filtering.
# Fixed ts_eps=1e-5 (same as before).
# Cascade table reveals: false-rejection vs genuine failures.
# Neg-eigenvalue gap table reveals: is the TS mode actually resolved?
# =============================================================================
echo "=============================================="
echo "Phase A — Cascade baseline (mad × baseline, ts_eps=1e-5, raw eigs)"
echo "  Axes: mad ∈ {0.5, 1.0, 1.3, 1.5} × baseline ∈ {plain, mode_tracked}"
echo "  CASCADE & SPECTRUM LOGGING: always on"
echo "  PURPOSE: characterize false-rejection and TS-mode resolution"
echo "=============================================="

for MAD in "0.5" "1.0" "1.3" "1.5"; do
    for BL in "plain" "mode_tracked"; do
        run_combo "mad${MAD}_tr${TR_DIAGNOSTIC}_bl${BL}_pgtrue" \
            "$BL" "$MAD" "1e-5"
    done
done

# =============================================================================
# Phase B — ts_eps sweep (relaxed convergence criterion, raw eigs)
# Fixed to best-performing parameters from Phase A (expected: mad=1.3, mode_tracked).
# ts_eps controls how strict the eig_product gate is:
#   1e-5 (original): λ_0 * λ_1 < -1e-5, needs clean sign separation
#   1e-3: more permissive, accepts smaller positive λ_1
#   1e-2: very permissive, will accept even near-zero λ_1
# Cascade analysis separates "ts_eps too strict" from "not at Morse-1 at all".
# Run for BOTH baselines so Phase B can also confirm plain vs mode_tracked.
# =============================================================================
echo "=============================================="
echo "Phase B — ts_eps sweep (relaxed convergence, mad=1.3, raw eigs)"
echo "  Axes: ts_eps ∈ {1e-4, 1e-3, 5e-3, 1e-2} × baseline ∈ {plain, mode_tracked}"
echo "  (ts_eps=1e-5 already covered in Phase A)"
echo "  PURPOSE: find if tighter/looser ts_eps changes success rate"
echo "           combined with cascade: distinguish false-rejection from failure"
echo "=============================================="

for TSE in "1e-4" "1e-3" "5e-3" "1e-2"; do
    for BL in "plain" "mode_tracked"; do
        run_combo "mad1.3_tr${TR_DIAGNOSTIC}_bl${BL}_pgtrue_tse${TSE}" \
            "$BL" "1.3" "$TSE"
    done
done

# =============================================================================
# Phase C — Legacy tr_filter_eig (matches old implementation)
# The old code filtered near-zero modes (|λ| < tr_threshold) from the eigenvalue
# array before computing eig_product = evals_vib[0] * evals_vib[1].  This prevents
# soft Eckart-leakage modes from making λ_1 ≈ 0, which suppresses eig_product and
# causes false non-convergence even when the geometry is a clean Morse-1 saddle.
#
# Phase C sweeps tr_threshold (the filter cutoff) at the best mad and both baselines
# to recover the old performance AND compare it to Phase A (unfiltered).
#
# Both unfiltered AND filtered λ_0, λ_1 are logged simultaneously:
#   lambda_0, lambda_1        — raw two smallest (always logged)
#   filt_lambda_0, filt_lambda_1 — post-filter two smallest (only when tr_filter_eig)
# This lets you directly compare how much the filtering changes the effective eig_product.
# =============================================================================
echo "=============================================="
echo "Phase C — Legacy tr_filter_eig (old behaviour, mad=1.3)"
echo "  Axes: tr_threshold ∈ {1e-3, 5e-3, 8e-3, 2e-2} × baseline ∈ {plain, mode_tracked}"
echo "  --tr-filter-eig ON: near-zero modes excluded from eig_product gate"
echo "  Both raw and filtered λ_0, λ_1 logged for direct comparison"
echo "  PURPOSE: recover old success rate, quantify soft-mode contamination"
echo "=============================================="

for TR_F in "1e-3" "5e-3" "8e-3" "2e-2"; do
    for BL in "plain" "mode_tracked"; do
        run_combo_tr_filter "mad1.3_trf${TR_F}_bl${BL}_pgtrue_tse1e-5" \
            "$BL" "1.3" "1e-5" "$TR_F"
    done
done

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
# Analysis — grid summary + 2D cascade table + neg-eigenvalue gap
# =============================================================================
ANALYSIS_OUT="${OUT_DIR}/analysis"
mkdir -p "$ANALYSIS_OUT"
echo "Running analysis script (cascade table + eigenvalue gap) ..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_gad_grid.py" \
    --grid-dir "$OUT_DIR" \
    --result-glob "*/gad_*_parallel_*_results.json" \
    --output-dir "$ANALYSIS_OUT" \
    --top-k 10 > "${ANALYSIS_OUT}/report.txt" 2>&1
echo "Analysis complete."
echo "  Report:               ${ANALYSIS_OUT}/report.txt"
echo "  Cascade 2D table:     ${ANALYSIS_OUT}/gad_grid_cascade_table.csv"
echo "  Neg-eval gap:         ${ANALYSIS_OUT}/gad_grid_neg_eval_gap.csv"
echo "  Ranked configs:       ${ANALYSIS_OUT}/gad_grid_ranked.csv"

# =============================================================================
# Trajectory plots
# =============================================================================
PLOTS_OUT="${OUT_DIR}/plots"
mkdir -p "$PLOTS_OUT"
echo "Generating trajectory plots ..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/plot_gad_grid_trajectories.py" \
    --grid-dir "$OUT_DIR" \
    --output-dir "$PLOTS_OUT"
echo "Plots saved to: $PLOTS_OUT"

# =============================================================================
# Trajectory-level statistics
# =============================================================================
TRAJ_STATS_OUT="${OUT_DIR}/traj_stats"
mkdir -p "$TRAJ_STATS_OUT"
echo "Running trajectory-level statistics ..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_gad_trajectory_stats.py" \
    --grid-dir "$OUT_DIR" \
    --output-dir "$TRAJ_STATS_OUT" \
    --top-k 10 > "${TRAJ_STATS_OUT}/traj_stats_report.txt" 2>&1
echo "Trajectory stats: ${TRAJ_STATS_OUT}/traj_stats_report.txt"
echo "  • gad_traj_stats_per_sample.csv — one row per trajectory"
echo "  • gad_traj_stats_per_combo.csv  — aggregated per hyperparameter combo"
echo "  • gad_traj_stats_summary.json   — machine-readable summary"
echo "=============================================="
echo "All done.  Results in: $OUT_DIR"
echo "=============================================="
