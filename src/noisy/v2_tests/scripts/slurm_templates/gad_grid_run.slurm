#!/bin/bash
#SBATCH --job-name=gad_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/gad_grid_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/gad_grid_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# GAD (Gradient Ascent Deflation) - Grid Search
# =============================================================================
#
# GAD dynamics:
#   ĝ_GAD = F_GAD / ||F_GAD||,   F_GAD = F + 2*(F·v)v  (Eckart-projected)
#   x_{k+1} = x_k + Δ_k * ĝ_GAD(x_k)
#
#   where Δ_k is the adaptive trust radius (initialized to max_atom_disp).
#   Trust radius is adapted via |ρ| = |dE_actual / dE_pred|:
#     |ρ| > 0.75  → grow  Δ by 1.5x  (good quadratic fit)
#     |ρ| < 0.25  → shrink Δ by 0.5x (poor fit)
#     |ρ| < 0.1   → reject step, shrink by 0.25x, retry up to 10 times
#
# NOTE: unlike NR minimization, the Newton preconditioner (H_vib^{-1} * F_GAD)
# is intentionally NOT used here. At Morse index > 1 the second-most-negative
# mode has larger 1/|λ| amplification than the tracked climbing mode, causing
# the algorithm to ascend along the wrong mode and get stuck. The GAD force
# direction + trust radius is the correct formulation.
#
# Key hyperparameters informed by NR minimization findings:
#   1. max_atom_disp  - Trust radius cap (Å). Bigger → larger confident steps.
#                       Sweep: 0.5, 1.0, 1.3, 1.5 Å.
#   2. baseline       - plain (always lowest eigenvec) vs mode_tracked (tracks
#                       eigenvec across steps for smoother TS search).
#
# NOTE on tr_threshold: the GAD step and convergence check use the reduced
# vibrational basis (exact 3N-k eigenvalues, no threshold). The --tr-threshold
# arg is accepted by the runner and forwarded to the trajectory logger ONLY
# for TR residual diagnostics (monitoring how many modes look near-zero).
# It does NOT affect the GAD direction, trust radius, or convergence criterion.
# The sweep below is kept for diagnostic sensitivity analysis; do not interpret
# it as an algorithmic hyperparameter sweep.
#
# Fixed (from NR findings):
#   - project_gradient_and_v=true (Eckart projection: mandatory)
#   - purify_hessian=false       (no measurable impact, saves compute)
#
# Grid: 4 (mad) x 2 (baseline) = 8 algorithmic combinations x 4 (tr diagnostic) x 30 samples
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
TOTAL_CPUS="${SLURM_CPUS_ON_NODE:-32}"
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
TS_EPS="${TS_EPS:-1e-5}"

# Eckart projection is mandatory (from NR findings: critical for stability)
PROJECT_GRADIENT_AND_V="true"
# Purification has no measurable impact (from NR findings); skip to save compute
PURIFY_HESSIAN="false"

# =============================================================================
# Grid search axes
# =============================================================================
# Trust radius cap (Å). NR finding: bigger is significantly better (1.3 dominated).
# Sweep the full range for GAD to characterize the trust-radius dynamics.
MAX_ATOM_DISP_GRID=("0.5" "1.0" "1.3" "1.5")

# Logger-only diagnostic threshold. Does NOT affect GAD algorithm behaviour.
# Sweep kept to characterize TR residual sensitivity in logged diagnostics.
TR_THRESHOLD_GRID=("2e-3" "5e-3" "8e-3" "1e-2")

# plain: always picks lowest eigenvector
# mode_tracked: tracks v_1 across steps for continuity (expected to dominate)
BASELINE_GRID=("plain" "mode_tracked")

echo "=============================================="
echo "GAD - Grid Search"
echo "=============================================="
echo "Date: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs available: $TOTAL_CPUS"
echo "=============================================="
echo "Fixed config:"
echo "  N_STEPS: $N_STEPS"
echo "  MAX_SAMPLES: $MAX_SAMPLES"
echo "  START_FROM: $START_FROM"
echo "  NOISE_SEED: $NOISE_SEED"
echo "  TS_EPS: $TS_EPS"
echo "  project_gradient_and_v: $PROJECT_GRADIENT_AND_V (Eckart, always on)"
echo "  purify_hessian: $PURIFY_HESSIAN (no measurable impact from NR study)"
echo "=============================================="
echo "Grid axes:"
echo "  max_atom_disp: ${MAX_ATOM_DISP_GRID[*]}"
echo "  tr_threshold: ${TR_THRESHOLD_GRID[*]}"
echo "  baseline: ${BASELINE_GRID[*]}"
TOTAL_COMBOS=$(( ${#MAX_ATOM_DISP_GRID[@]} * ${#TR_THRESHOLD_GRID[@]} * ${#BASELINE_GRID[@]} ))
echo "  Total combinations: $TOTAL_COMBOS"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "Parallel config:"
echo "  Workers: $N_WORKERS"
echo "  Threads per worker: $THREADS_PER_WORKER"
echo "=============================================="

COMBO=0
FAILED=0

for MAD in "${MAX_ATOM_DISP_GRID[@]}"; do
    for TR in "${TR_THRESHOLD_GRID[@]}"; do
        for BL in "${BASELINE_GRID[@]}"; do
            COMBO=$((COMBO + 1))
            RUN_TAG="mad${MAD}_tr${TR}_bl${BL}_pg${PROJECT_GRADIENT_AND_V}"
            RUN_OUT_DIR="${OUT_DIR}/${RUN_TAG}"
            mkdir -p "$RUN_OUT_DIR"

            echo "[${COMBO}/${TOTAL_COMBOS}] Running: $RUN_TAG"

            CMD=(
                python "$PROJECT_DIR/src/noisy/v2_tests/runners/run_gad_baselines_parallel.py"
                --h5-path "$H5_PATH"
                --out-dir "$RUN_OUT_DIR"
                --scine-functional "$SCINE_FUNCTIONAL"
                --n-steps "$N_STEPS"
                --max-samples "$MAX_SAMPLES"
                --start-from "$START_FROM"
                --noise-seed "$NOISE_SEED"
                --n-workers "$N_WORKERS"
                --threads-per-worker "$THREADS_PER_WORKER"
                --baseline "$BL"
                --max-atom-disp "$MAD"
                --min-interatomic-dist "$MIN_INTERATOMIC_DIST"
                --ts-eps "$TS_EPS"
                --tr-threshold "$TR"
                --projection-mode eckart_full
            )

            if [ "$PROJECT_GRADIENT_AND_V" = "true" ]; then
                CMD+=(--project-gradient-and-v)
            fi
            if [ "$PURIFY_HESSIAN" = "true" ]; then
                CMD+=(--purify-hessian)
            fi

            if "${CMD[@]}"; then
                echo "  -> Completed: $RUN_TAG"
            else
                echo "  -> FAILED: $RUN_TAG (exit code $?)"
                FAILED=$((FAILED + 1))
            fi
            echo ""
        done
    done
done

echo "=============================================="
echo "Grid search completed at $(date)"
echo "  Total combinations: $TOTAL_COMBOS"
echo "  Completed: $((COMBO - FAILED))"
echo "  Failed: $FAILED"
echo "  Results in: $OUT_DIR"
echo "=============================================="

# =============================================================================
# Run analysis and plotting
# =============================================================================
ANALYSIS_OUT="${OUT_DIR}/analysis"
mkdir -p "$ANALYSIS_OUT"
echo "Running analysis script..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_gad_grid.py" \
    --grid-dir "$OUT_DIR" \
    --output-dir "$ANALYSIS_OUT" \
    --top-k 10 > "${ANALYSIS_OUT}/report.txt"

echo "Analysis complete. Report saved to ${ANALYSIS_OUT}/report.txt"

PLOTS_OUT="${OUT_DIR}/plots"
mkdir -p "$PLOTS_OUT"
echo "Generating trajectory plots..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/plot_gad_grid_trajectories.py" \
    --grid-dir "$OUT_DIR" \
    --output-dir "$PLOTS_OUT"

echo "Plotting complete. PNGs saved to ${PLOTS_OUT}"

TRAJ_STATS_OUT="${OUT_DIR}/traj_stats"
mkdir -p "$TRAJ_STATS_OUT"
echo "Running trajectory-level statistics..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_gad_trajectory_stats.py" \
    --grid-dir "$OUT_DIR" \
    --output-dir "$TRAJ_STATS_OUT" \
    --top-k 10 > "${TRAJ_STATS_OUT}/traj_stats_report.txt"

echo "Trajectory stats written to ${TRAJ_STATS_OUT}/traj_stats_report.txt"
echo "  • gad_traj_stats_per_sample.csv — one row per trajectory"
echo "  • gad_traj_stats_per_combo.csv  — aggregated per hyperparameter combo"
echo "  • gad_traj_stats_summary.json   — machine-readable summary"
echo "=============================================="
