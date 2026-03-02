#!/bin/bash
#SBATCH --job-name=min_nr_v5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v5_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v5_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Newton-Raphson Minimization — v5 Multi-Seed Statistical Experiment
# =============================================================================
#
# MOTIVATION:
#   v4 results showed 53% convergence (16/30) with plain shifted Newton, but
#   the same 7 samples ALWAYS fail — we're overfitting to 30 fixed starting
#   geometries (noise seed 42). All v4 features (neg-trust-floor, blind
#   correction, etc.) hurt performance and introduced errors.
#
# EXPERIMENTAL DESIGN:
#   Instead of tuning optimizer parameters on fixed starting points, we run
#   a properly-controlled experiment with:
#
#   1. NOISE MAGNITUDE SWEEP: {0.5, 1.0, 1.5, 2.0} Å
#      Maps the difficulty curve — how convergence rate degrades with
#      distance from the PES minimum.
#
#   2. NOISE SEED SWEEP: {42, 123, 456, 789, 1024}
#      5 independent noise realizations per noise level → breaks the
#      overfitting to specific starting geometries. Gives mean ± std
#      convergence rates with statistical significance.
#
#   3. OPTIMIZER CONFIGS (slim — only what works):
#      a) SN ε=5e-4:  Best v4 config (53% at 2.0Å)
#      b) SN ε=2e-4:  Tied best, slower convergence
#      c) SPDN:       New v5 spectral partitioned optimizer (default params)
#      d) SPDN-tuned: SPDN with momentum + larger tau_hard
#
# DIRECTORY STRUCTURE:
#   <OUT_DIR>/
#     noise_0.5A/
#       seed_42/
#         mad1.3_se5e-4_pgtrue_phfalse/   <- combo directories
#         mad1.3_se2e-4_pgtrue_phfalse/
#         mad1.3_spdn_pgtrue_phfalse/
#         mad1.3_spdn_tuned_pgtrue_phfalse/
#       seed_123/
#         ...
#     noise_1.0A/
#       ...
#
# GRID SIZE:
#   4 noise levels × 5 seeds × 4 optimizer configs = 80 combinations
#   50 samples per combo → 4,000 total trajectories
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
OUT_DIR="${SCRATCH_DIR}/runs/min_nr_v5_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Fixed configuration
# =============================================================================
N_STEPS="${N_STEPS:-10000}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
SCINE_FUNCTIONAL="${SCINE_FUNCTIONAL:-DFTB0}"
FORCE_CONVERGED="${FORCE_CONVERGED:-1e-4}"
MIN_INTERATOMIC_DIST="${MIN_INTERATOMIC_DIST:-0.5}"
LOG_SPECTRUM_K="${LOG_SPECTRUM_K:-10}"
TRUST_RADIUS_FLOOR="${TRUST_RADIUS_FLOOR:-0.01}"

# Parallel worker config
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

# Sweep axes
NOISE_LEVELS="0.5 1.0 1.5 2.0"
NOISE_SEEDS="42 123 456 789 1024"

echo "=============================================="
echo "Newton-Raphson Minimization — v5 Multi-Seed Experiment"
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
echo "  FORCE_CONVERGED:    $FORCE_CONVERGED"
echo "  LOG_SPECTRUM_K:     $LOG_SPECTRUM_K"
echo "  TRUST_RADIUS_FLOOR: $TRUST_RADIUS_FLOOR"
echo "Sweep axes:"
echo "  NOISE_LEVELS:       $NOISE_LEVELS"
echo "  NOISE_SEEDS:        $NOISE_SEEDS"
echo "  OPTIMIZERS:         SN-5e4, SN-2e4, SPDN, SPDN-tuned"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

COMBO=0
FAILED=0

# =============================================================================
# Helper: run one combination
# =============================================================================
run_combo() {
    local noise_level="$1"
    local seed="$2"
    local tag="$3"
    shift 3

    local start_from="midpoint_rt_noise${noise_level}A"
    local run_out="${OUT_DIR}/noise_${noise_level}A/seed_${seed}/${tag}"
    mkdir -p "$run_out"

    COMBO=$((COMBO + 1))
    echo "[combo ${COMBO}]  noise=${noise_level}A seed=${seed} ${tag}"

    if python "$PROJECT_DIR/src/noisy/v2_tests/runners/run_minimization_parallel.py" \
            --h5-path "$H5_PATH" \
            --out-dir "$run_out" \
            --scine-functional "$SCINE_FUNCTIONAL" \
            --n-steps "$N_STEPS" \
            --max-samples "$MAX_SAMPLES" \
            --start-from "$start_from" \
            --noise-seed "$seed" \
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
# Main grid: noise_level × seed × optimizer
# =============================================================================

for NOISE in $NOISE_LEVELS; do
    echo "=============================================="
    echo "Noise level: ${NOISE} Å"
    echo "=============================================="

    for SEED in $NOISE_SEEDS; do
        echo "--- seed ${SEED} ---"

        # (a) Shifted Newton ε=5e-4
        run_combo "$NOISE" "$SEED" "mad1.3_se5e-4_pgtrue_phfalse" \
            --shift-epsilon "5e-4"

        # (b) Shifted Newton ε=2e-4
        run_combo "$NOISE" "$SEED" "mad1.3_se2e-4_pgtrue_phfalse" \
            --shift-epsilon "2e-4"

        # (c) SPDN default (tau_hard=0.01, tau_soft=1e-4, diis_size=8, diis_every=5)
        run_combo "$NOISE" "$SEED" "mad1.3_spdn_pgtrue_phfalse" \
            --optimizer-mode "spdn" \
            --spdn-tau-hard "0.01" \
            --spdn-tau-soft "1e-4" \
            --spdn-diis-size "8" \
            --spdn-diis-every "5"

        # (d) SPDN tuned (tau_hard=0.02, momentum=0.2, diis_size=8, diis_every=3)
        run_combo "$NOISE" "$SEED" "mad1.3_spdn_tuned_pgtrue_phfalse" \
            --optimizer-mode "spdn" \
            --spdn-tau-hard "0.02" \
            --spdn-tau-soft "1e-4" \
            --spdn-diis-size "8" \
            --spdn-diis-every "3" \
            --spdn-momentum "0.2"

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
# Per-seed analysis — run existing analysis for each noise_level/seed pair
# =============================================================================
echo "Running per-seed analyses ..."
for NOISE in $NOISE_LEVELS; do
    for SEED in $NOISE_SEEDS; do
        SEED_DIR="${OUT_DIR}/noise_${NOISE}A/seed_${SEED}"
        if [ ! -d "$SEED_DIR" ]; then continue; fi

        SEED_ANALYSIS="${SEED_DIR}/analysis"
        mkdir -p "$SEED_ANALYSIS"

        # Grid summary (reuses existing analysis script with default glob)
        python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_minimization_nr_grid.py" \
            --grid-dir "$SEED_DIR" \
            --result-glob "*/minimization_newton_raphson_*_results.json" \
            --output-dir "$SEED_ANALYSIS" \
            --top-k 10 > "${SEED_ANALYSIS}/report.txt" 2>&1 || true

        # Failure autopsy
        SEED_AUTOPSY="${SEED_DIR}/autopsy"
        mkdir -p "$SEED_AUTOPSY"
        python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_nr_failure_autopsy.py" \
            --grid-dir "$SEED_DIR" \
            --output-dir "$SEED_AUTOPSY" > "${SEED_AUTOPSY}/autopsy_report.txt" 2>&1 || true

        echo "  Analysis done for noise=${NOISE}A seed=${SEED}"
    done
done

# =============================================================================
# Cross-seed aggregation — the key new analysis
# =============================================================================
CROSS_SEED_OUT="${OUT_DIR}/cross_seed_analysis"
mkdir -p "$CROSS_SEED_OUT"
echo "Running cross-seed aggregation analysis ..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_nr_cross_seed.py" \
    --grid-dir "$OUT_DIR" \
    --output-dir "$CROSS_SEED_OUT" \
    > "${CROSS_SEED_OUT}/report.txt" 2>&1
echo "Cross-seed report: ${CROSS_SEED_OUT}/report.txt"

# =============================================================================
# SPDN diagnostics (across all seeds)
# =============================================================================
SPDN_OUT="${OUT_DIR}/spdn_diagnostics"
mkdir -p "$SPDN_OUT"
echo "Running SPDN diagnostics analysis ..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_nr_spdn_diagnostics.py" \
    --grid-dir "$OUT_DIR" --output-dir "$SPDN_OUT" \
    --traj-glob "*/*/*/diagnostics/*_trajectory.json" \
    > "${SPDN_OUT}/report.txt" 2>&1
echo "SPDN diagnostics report: ${SPDN_OUT}/report.txt"

echo "=============================================="
echo "All done.  Results in: $OUT_DIR"
echo "=============================================="
