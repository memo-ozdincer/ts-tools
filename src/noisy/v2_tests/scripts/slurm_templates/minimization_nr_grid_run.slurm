#!/bin/bash
#SBATCH --job-name=min_nr_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/min_nr_grid_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/min_nr_grid_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Newton-Raphson Minimization — Algorithmic Improvements Grid Search (v2)
# =============================================================================
#
# x_{k+1} = x_k - H(x_k)^{-1} * grad E(x_k)
#
# The Hessian pseudoinverse is computed in the reduced vibrational basis
# (exact 3N-k dimensional subspace via Eckart projection), so TR zero-modes
# are removed by construction.
#
# ── What changed in v2 ────────────────────────────────────────────────────
#
# 1. CASCADING EVALUATION (mandatory, always on)
#    Every trajectory step now stores n_neg_at_<threshold> for
#    thresholds [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2].
#    The analysis script builds a 2D table (optimizer_threshold × eval_threshold)
#    to separate "optimizer failed" from "evaluation too strict".
#    This is PURE DIAGNOSTICS — it never changes the step or convergence logic.
#
# 2. LEVENBERG-MARQUARDT DAMPING (new grid axis: LM_MU_GRID)
#    step_i = (g·v_i) * |λ_i| / (λ_i² + μ²)
#    |λ|>>μ → 1/|λ| (pure Newton); |λ|<<μ → 0 (flat modes vanish smoothly).
#    No hard cutoff, no discontinuity when modes drift across the threshold.
#    Run with the same μ values as nr_threshold to make the cascade comparison
#    directly interpretable.
#    Folder tag: mad<v>_lmmu<v>_pg<bool>_ph<bool>
#
# 3. TWO-PHASE THRESHOLD ANNEALING (new grid axis: ANNEAL_GRID)
#    Phase 1: hard-filter NR with nr_threshold (bulk, proven safe at 8e-3).
#    Phase 2: once force_norm < anneal_force_threshold, switch to
#             cleanup_nr_threshold (default 0 = full pseudoinverse) for
#             cleanup_max_steps steps. Near a stationary point the Hessian is
#             better-conditioned, so lower thresholds won't cause explosion.
#    Folder tag: mad<v>_nrt<v>_pg<bool>_ph<bool>_af<anneal_force>
#
# 4. SPECTRAL GAP DIAGNOSTIC (always on)
#    |λ_min| / |λ_second_min| logged every step.  A large ratio (or inf when
#    exactly 1 mode is negative) signals an isolated dominant negative mode
#    (TS-like character).  Also logs n_neg on the full (unprojected) Hessian
#    for comparison with the vibrational-subspace n_neg.
#
# 5. FULL SPECTRUM LOGGING (always on, --log-spectrum-k 10)
#    Bottom-10 vibrational eigenvalues stored in each trajectory step.
#    Invaluable post-hoc: know exactly what's happening at failure/convergence.
#
# ── Grid layout ───────────────────────────────────────────────────────────
#
# Phase A — Cascade baseline (replays proven best config, adds cascade data):
#   1 (mad) × 4 (nrt) × 1 (pg) × 1 (ph)  = 4 combos
#
# Phase B — Two-phase annealing (anneal_force sweep, best nrt=8e-3):
#   1 (mad) × 1 (nrt=8e-3) × 1 (pg) × 1 (ph) × 4 (anneal_force) = 4 combos
#
# Phase C — LM damping (μ sweep):
#   1 (mad) × 4 (lm_mu) × 1 (pg) × 1 (ph)  = 4 combos
#
# Total: 4 + 4 + 4 = 12 combinations
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
OUT_DIR="${SCRATCH_DIR}/runs/min_nr_grid_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Fixed configuration
# =============================================================================
N_STEPS="${N_STEPS:-20000}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
START_FROM="${START_FROM:-midpoint_rt_noise2.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
SCINE_FUNCTIONAL="${SCINE_FUNCTIONAL:-DFTB0}"
FORCE_CONVERGED="${FORCE_CONVERGED:-1e-4}"
MIN_INTERATOMIC_DIST="${MIN_INTERATOMIC_DIST:-0.5}"
LOG_SPECTRUM_K="${LOG_SPECTRUM_K:-10}"

# Parallel worker config
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "=============================================="
echo "Newton-Raphson Minimization — v2 Grid Search"
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
            "$@"; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# =============================================================================
# Phase A — Cascade baseline
# Replays the proven NR hard-filter grid (nrt sweep) with cascade logging.
# The cascade table will reveal whether failures are "eval too strict" vs
# "optimizer failed". This is the most important diagnostic.
# =============================================================================
echo "=============================================="
echo "Phase A — Cascade baseline (hard-filter NR, nrt sweep)"
echo "  Axes: nrt ∈ {2e-3, 5e-3, 8e-3, 1e-2}"
echo "  Note: n_neg logged at 8 eval thresholds per step (pure diagnostics)"
echo "=============================================="

for NRT in "2e-3" "5e-3" "8e-3" "1e-2"; do
    run_combo "mad1.3_nrt${NRT}_pgtrue_phfalse" \
        --nr-threshold "$NRT"
done

# =============================================================================
# Phase B — Two-phase threshold annealing
# Bulk phase uses nrt=8e-3 (proven best).
# Cleanup phase uses nrt=0 (full pseudoinverse) for up to 50 steps once
# force_norm < anneal_force_threshold.
# If near-minimum Hessian is better-conditioned, the tighter threshold is safe.
# We sweep anneal_force_threshold: the force level at which we transition.
# =============================================================================
echo "=============================================="
echo "Phase B — Two-phase annealing (anneal_force sweep, nrt=8e-3)"
echo "  Axes: anneal_force ∈ {5e-4, 1e-3, 5e-3, 1e-2}"
echo "  cleanup_nr_threshold=0  (full pseudoinverse in cleanup)"
echo "  cleanup_max_steps=50"
echo "=============================================="

for AF in "5e-4" "1e-3" "5e-3" "1e-2"; do
    run_combo "mad1.3_nrt8e-3_pgtrue_phfalse_af${AF}" \
        --nr-threshold "8e-3" \
        --anneal-force-threshold "$AF" \
        --cleanup-nr-threshold "0" \
        --cleanup-max-steps "50"
done

# =============================================================================
# Phase C — Levenberg-Marquardt damping
# step_i = (g·v_i) * |λ_i| / (λ_i² + μ²)
# No hard cutoff; flat modes shrink smoothly to zero (vs hard-filter which
# removes them entirely, vs clamping which was tested in the TeX and failed).
# Sweep μ over the same values as the NR threshold grid in Phase A to make
# Phase A vs Phase D comparison straightforward on the cascade table.
# =============================================================================
echo "=============================================="
echo "Phase C — Levenberg-Marquardt damping (lm_mu sweep)"
echo "  Axes: lm_mu ∈ {2e-3, 5e-3, 8e-3, 1e-2}"
echo "  Note: all modes included; flat modes → near-zero contribution"
echo "=============================================="

for MU in "2e-3" "5e-3" "8e-3" "1e-2"; do
    run_combo "mad1.3_lmmu${MU}_pgtrue_phfalse" \
        --lm-mu "$MU"
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
# Analysis — main grid summary + 2D cascade table
# =============================================================================
ANALYSIS_OUT="${OUT_DIR}/analysis"
mkdir -p "$ANALYSIS_OUT"
echo "Running analysis script (includes 2D cascade table) ..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_minimization_nr_grid.py" \
    --grid-dir "$OUT_DIR" \
    --result-glob "*/minimization_newton_raphson_*_results.json" \
    --output-dir "$ANALYSIS_OUT" \
    --top-k 10 > "${ANALYSIS_OUT}/report.txt" 2>&1
echo "Analysis complete.  Report: ${ANALYSIS_OUT}/report.txt"
echo "Cascade 2D table:   ${ANALYSIS_OUT}/nr_grid_cascade_table.csv"

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
# Trajectory-level statistics
# =============================================================================
TRAJ_STATS_OUT="${OUT_DIR}/traj_stats"
mkdir -p "$TRAJ_STATS_OUT"
echo "Running trajectory-level statistics ..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_nr_trajectory_stats.py" \
    --grid-dir "$OUT_DIR" \
    --output-dir "$TRAJ_STATS_OUT" \
    --top-k 10 > "${TRAJ_STATS_OUT}/traj_stats_report.txt" 2>&1
echo "Trajectory stats: ${TRAJ_STATS_OUT}/traj_stats_report.txt"
echo "  • nr_traj_stats_per_sample.csv — one row per trajectory"
echo "  • nr_traj_stats_per_combo.csv  — aggregated per hyperparameter combo"
echo "  • nr_traj_stats_summary.json   — machine-readable summary"
echo "=============================================="
echo "All done.  Results in: $OUT_DIR"
echo "=============================================="
