#!/bin/bash
#SBATCH --job-name=gad_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/gad_v3_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/gad_v3_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# GAD v3 — Newton-Preconditioned GAD Grid Search
# =============================================================================
#
# Problem statement (from v2 cascade analysis):
#   - Best v2 config: 10% strict success (3/30), ts_eps=1e-5, mode_tracked
#   - Cascade revealed THREE failure modes:
#     (1) 25% never converge: stuck at Morse index 4-8 after 10k steps
#         → 1st-order GAD too slow.  FIX: Newton-preconditioned step.
#     (2) 40% overshoot: passed through TS into a minimum (le1-eq1 gap at T=0.01)
#         → no mechanism to detect/slow at the TS.  FIX: anti-overshoot mechanism.
#     (3) 27% false rejection: at true Morse-1 but tiny negative eigenvalues
#         (|λ|<0.005) cause strict T=0 failure.
#         → FIX: filtered convergence gate + post-convergence cleanup.
#
# Newton-GAD step:
#   Instead of the v2 unit-vector step (ĝ_GAD / ||ĝ_GAD||) × trust_radius:
#     1. Decompose gradient into vibrational eigenmodes
#     2. For tracked TS mode (λ₀ < 0): standard Newton → −g₀/λ₀ (toward saddle)
#     3. For positive modes (λᵢ > 0): standard Newton → −gᵢ/λᵢ (minimization)
#     4. For non-tracked negative modes: abs-pseudoinverse → −gᵢ/|λᵢ| (GD from max)
#     5. Filter noise modes: |λᵢ| < step_filter_threshold → zeroed out
#     6. Cap total step by trust radius
#
#   This gives second-order convergence (curvature-aware step sizing per mode)
#   while preserving GAD's direction logic and the proven noise-mode filtering.
#
# Anti-overshoot:
#   Track Morse index at T=0.002 per step.  When it drops to ≤1:
#     - Cap trust radius to 25% of max_atom_disp (prevent large steps near TS)
#     - If it drops to 0 (overshot to min), cap to 10%
#   Once in the TS region, limit trust radius growth to 50% of max_atom_disp.
#
# Filtered convergence gate:
#   The optimizer's eig_product check uses eigenvalues filtered by
#   step_filter_threshold — this prevents near-zero λ₁ from suppressing the
#   product (the main cause of overshoot in v2: algorithm couldn't detect
#   when it reached Morse-1 because λ₁ ≈ 0 made eig_product ≈ 0 > -ts_eps).
#   The CASCADE table always reports UNFILTERED n_neg for honest evaluation.
#
# Post-convergence cleanup:
#   After the convergence gate triggers, take N additional NR minimization steps
#   in the subspace ORTHOGONAL to the TS mode (eigenvector v₀).
#   This pushes residual small negative modes positive without disturbing the
#   saddle geometry, directly addressing the T=0 false-rejection problem.
#
# ── Grid layout ────────────────────────────────────────────────────────────
#
# Fixed:
#   - baseline = mode_tracked (clear v2 winner: 5.8% vs 1.7%)
#   - project_gradient_and_v = true (mandatory for stability)
#   - ts_eps = 1e-5 (strict; proven best in v2)
#   - START_FROM = midpoint_rt_noise2.0A
#
# Phase A — Newton step validation (compare 1st-order vs Newton):
#   2 (step_mode) × 2 (mad) = 4 combos
#   All with anti_overshoot=true, converge_on_filtered=true, cleanup=0
#
# Phase B — Step filter threshold sweep (Newton only):
#   4 (sft) × 1 = 4 combos
#   mad=1.5, anti_overshoot=true, converge_on_filtered=true, cleanup=0
#
# Phase C — Feature ablation (Newton, sft=8e-3, mad=1.5):
#   6 feature combinations (see below)
#
# Total: 4 + 4 + 6 = 14 combos × 30 samples = 420 runs
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
OUT_DIR="${SCRATCH_DIR}/runs/gad_v3_${SLURM_JOB_ID}"

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
TS_EPS="1e-5"

# TR diagnostic threshold (logging only, not used in Newton step)
TR_DIAGNOSTIC="8e-3"

# Always-on settings
PROJECT_GRADIENT_FLAG="--project-gradient-and-v"
PURIFY_FLAG=""
BASELINE="mode_tracked"

# Parallel worker config
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "=============================================="
echo "GAD v3 — Newton-Preconditioned Grid Search"
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
echo "  TS_EPS:            $TS_EPS"
echo "  baseline:          $BASELINE  (v2 winner)"
echo "  project-gradient:  always on (Eckart mandatory)"
echo "  purify-hessian:    off"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

COMBO=0
FAILED=0

# =============================================================================
# Helper: run one v3 combination.
# Usage: run_v3 <tag> <mad> <step_mode> <sft> <anti_overshoot> <conv_filtered> <cleanup>
# =============================================================================
run_v3() {
    local tag="$1"
    local mad="$2"
    local step_mode="$3"
    local sft="$4"
    local anti_os="$5"       # "true" or "false"
    local conv_filt="$6"     # "true" or "false"
    local cleanup="$7"
    shift 7
    local run_out="${OUT_DIR}/${tag}"
    mkdir -p "$run_out"

    COMBO=$((COMBO + 1))
    echo "[combo ${COMBO}]  ${tag}"

    # Build optional flags
    local EXTRA_FLAGS=""
    if [ "$anti_os" = "true" ]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --anti-overshoot"
    fi
    # converge-on-filtered removed: produced false convergences by
    # masking negative eigenvalues below step_filter_threshold.

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
            --baseline "$BASELINE" \
            --max-atom-disp "$mad" \
            --min-interatomic-dist "$MIN_INTERATOMIC_DIST" \
            --ts-eps "$TS_EPS" \
            --tr-threshold "$TR_DIAGNOSTIC" \
            --projection-mode eckart_full \
            --log-spectrum-k "$LOG_SPECTRUM_K" \
            --step-mode "$step_mode" \
            --step-filter-threshold "$sft" \
            --cleanup-steps "$cleanup" \
            $PROJECT_GRADIENT_FLAG \
            $PURIFY_FLAG \
            $EXTRA_FLAGS \
            "$@"; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# =============================================================================
# Phase A — Newton step validation: 1st-order vs Newton-GAD
# Both with anti-overshoot + converge-on-filtered (best-guess defaults).
# This confirms the Newton step is actually better before sweeping further.
# =============================================================================
echo "=============================================="
echo "Phase A — Step mode comparison (first_order vs newton_gad)"
echo "  Axes: step_mode × mad ∈ {1.0, 1.5}"
echo "  Fixed: anti_overshoot=T, converge_on_filtered=T, sft=8e-3, cleanup=0"
echo "=============================================="

for STEP_MODE in "first_order" "newton_gad"; do
    for MAD in "1.0" "1.5"; do
        TAG="${STEP_MODE}_mad${MAD}_sft8e-3_aoT_cfT_cl0"
        run_v3 "$TAG" "$MAD" "$STEP_MODE" "8e-3" "true" "true" "0"
    done
done

# =============================================================================
# Phase B — Step filter threshold sweep (Newton only)
# Determines the optimal noise-mode cutoff for the Newton step.
# =============================================================================
echo "=============================================="
echo "Phase B — Step filter threshold sweep (Newton, mad=1.5)"
echo "  Axes: sft ∈ {2e-3, 5e-3, 8e-3, 1.5e-2}"
echo "  Fixed: anti_overshoot=T, converge_on_filtered=T, cleanup=0"
echo "=============================================="

for SFT in "2e-3" "5e-3" "8e-3" "1.5e-2"; do
    TAG="newton_gad_mad1.5_sft${SFT}_aoT_cfT_cl0"
    run_v3 "$TAG" "1.5" "newton_gad" "$SFT" "true" "true" "0"
done

# =============================================================================
# Phase C — Feature ablation (Newton, sft=8e-3, mad=1.5)
# Tests each improvement individually and combined.
# =============================================================================
echo "=============================================="
echo "Phase C — Feature ablation (what helps?)"
echo "  Base: Newton, sft=8e-3, mad=1.5"
echo "=============================================="

# C1: Nothing (pure Newton, no anti-overshoot, no conv-filter, no cleanup)
run_v3 "newton_gad_mad1.5_sft8e-3_aoF_cfF_cl0" \
    "1.5" "newton_gad" "8e-3" "false" "false" "0"

# C2: Anti-overshoot only
run_v3 "newton_gad_mad1.5_sft8e-3_aoT_cfF_cl0" \
    "1.5" "newton_gad" "8e-3" "true" "false" "0"

# C3: Converge-on-filtered only
run_v3 "newton_gad_mad1.5_sft8e-3_aoF_cfT_cl0" \
    "1.5" "newton_gad" "8e-3" "false" "true" "0"

# C4: Cleanup=10 only (with conv-filter to actually trigger convergence)
run_v3 "newton_gad_mad1.5_sft8e-3_aoF_cfT_cl10" \
    "1.5" "newton_gad" "8e-3" "false" "true" "10"

# C5: Cleanup=20 only
run_v3 "newton_gad_mad1.5_sft8e-3_aoF_cfT_cl20" \
    "1.5" "newton_gad" "8e-3" "false" "true" "20"

# C6: Everything on (anti-overshoot + conv-filter + cleanup=10)
run_v3 "newton_gad_mad1.5_sft8e-3_aoT_cfT_cl10" \
    "1.5" "newton_gad" "8e-3" "true" "true" "10"

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "v3 grid search completed at $(date)"
echo "  Total combinations attempted: $COMBO"
echo "  Completed successfully:       $((COMBO - FAILED))"
echo "  Failed:                       $FAILED"
echo "  Results directory:            $OUT_DIR"
echo "=============================================="

# =============================================================================
# Analysis
# =============================================================================
ANALYSIS_OUT="${OUT_DIR}/analysis"
mkdir -p "$ANALYSIS_OUT"
echo "Running analysis script ..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_gad_grid.py" \
    --grid-dir "$OUT_DIR" \
    --result-glob "*/gad_*_parallel_*_results.json" \
    --output-dir "$ANALYSIS_OUT" \
    --top-k 14 > "${ANALYSIS_OUT}/report.txt" 2>&1
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
    --top-k 14 > "${TRAJ_STATS_OUT}/traj_stats_report.txt" 2>&1
echo "Trajectory stats: ${TRAJ_STATS_OUT}/traj_stats_report.txt"
echo "=============================================="
echo "All done.  Results in: $OUT_DIR"
echo "=============================================="
