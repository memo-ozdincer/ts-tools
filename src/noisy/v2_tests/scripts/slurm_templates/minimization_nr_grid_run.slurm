#!/bin/bash
#SBATCH --job-name=min_nr_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=0
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/min_nr_grid_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/min_nr_grid_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Newton-Raphson Minimization - Grid Search
# =============================================================================
#
# x_{k+1} = x_k - H(x_k)^{-1} * grad E(x_k)
#
# Full Hessian is computed at every step. The NR step and convergence check
# both use the reduced vibrational basis (exact 3N-k dimensional subspace via
# Eckart projection), so TR zero-modes are removed by construction — no
# threshold filtering, no blind zone for soft negative modes.
#
# Most impactful hyperparameters for NR:
#   1. max_atom_disp  - Trust radius; caps the NR step per atom.
#                       Too small = slow convergence, too large = overshoot.
#   2. project_gradient_and_v - Eckart-project gradient before NR step.
#   3. purify_hessian - Enforce translational sum rules on Hessian.
#
# NOTE: --tr-threshold is accepted by the CLI for legacy compatibility but is
# NOT forwarded to the algorithm. TR mode removal is handled by construction
# through the reduced basis; no threshold is needed or used.
#
# Refinement grid (based on previous 80-combo sweep):
#   - Best region was around max_atom_disp=1.3.
#   - purify_hessian produced identical outcomes across all matched pairs.
#
# Grid: 1 (mad) x 1 (project_grad) x 1 (purify) = 1 combination
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

# =============================================================================
# Grid search axes (focused around best-performing region with Trust Region)
# =============================================================================
# Trust radius acts as the *initial* trust radius and absolute cap in the TR method.
MAX_ATOM_DISP_GRID=("1.3")

# Eckart projection is mandatory for stability; no measurable benefit to sweeping.
PROJECT_GRAD_GRID=("true")

# Previously no measurable effect in this benchmark; keep fixed to reduce combinatorics.
PURIFY_HESS_GRID=("false")

echo "=============================================="
echo "Newton-Raphson Minimization - Grid Search"
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
echo "  FORCE_CONVERGED: $FORCE_CONVERGED"
echo "=============================================="
echo "Grid axes:"
echo "  max_atom_disp: ${MAX_ATOM_DISP_GRID[*]}"
echo "  project_gradient: ${PROJECT_GRAD_GRID[*]}"
echo "  purify_hessian: ${PURIFY_HESS_GRID[*]}"
TOTAL_COMBOS=$(( ${#MAX_ATOM_DISP_GRID[@]} * ${#PROJECT_GRAD_GRID[@]} * ${#PURIFY_HESS_GRID[@]} ))
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
    for PG in "${PROJECT_GRAD_GRID[@]}"; do
        for PH in "${PURIFY_HESS_GRID[@]}"; do
            COMBO=$((COMBO + 1))
            RUN_TAG="mad${MAD}_pg${PG}_ph${PH}"
            RUN_OUT_DIR="${OUT_DIR}/${RUN_TAG}"
            mkdir -p "$RUN_OUT_DIR"

            echo "[${COMBO}/${TOTAL_COMBOS}] Running: $RUN_TAG"

            # Build command
            CMD=(
                python "$PROJECT_DIR/src/noisy/v2_tests/runners/run_minimization_parallel.py"
                --h5-path "$H5_PATH"
                --out-dir "$RUN_OUT_DIR"
                --scine-functional "$SCINE_FUNCTIONAL"
                --n-steps "$N_STEPS"
                --max-samples "$MAX_SAMPLES"
                --start-from "$START_FROM"
                --noise-seed "$NOISE_SEED"
                --n-workers "$N_WORKERS"
                --threads-per-worker "$THREADS_PER_WORKER"
                --method newton_raphson
                --max-atom-disp "$MAD"
                --force-converged "$FORCE_CONVERGED"
                --min-interatomic-dist "$MIN_INTERATOMIC_DIST"
            )

            if [ "$PG" = "true" ]; then
                CMD+=(--project-gradient-and-v)
            fi
            if [ "$PH" = "true" ]; then
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
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_minimization_nr_grid.py" \
    --grid-dir "$OUT_DIR" \
    --output-dir "$ANALYSIS_OUT" \
    --top-k 10 > "${ANALYSIS_OUT}/report.txt"

echo "Analysis complete. Report saved to ${ANALYSIS_OUT}/report.txt"

PLOTS_OUT="${OUT_DIR}/plots"
mkdir -p "$PLOTS_OUT"
echo "Generating trajectory plots..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/plot_nr_grid_trajectories.py" \
    --grid-dir "$OUT_DIR" \
    --output-dir "$PLOTS_OUT"

echo "Plotting complete. PNGs saved to ${PLOTS_OUT}"

TRAJ_STATS_OUT="${OUT_DIR}/traj_stats"
mkdir -p "$TRAJ_STATS_OUT"
echo "Running trajectory-level statistics..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_nr_trajectory_stats.py" \
    --grid-dir "$OUT_DIR" \
    --output-dir "$TRAJ_STATS_OUT" \
    --top-k 10 > "${TRAJ_STATS_OUT}/traj_stats_report.txt"

echo "Trajectory stats written to ${TRAJ_STATS_OUT}/traj_stats_report.txt"
echo "  • nr_traj_stats_per_sample.csv — one row per trajectory"
echo "  • nr_traj_stats_per_combo.csv  — aggregated per hyperparameter combo"
echo "  • nr_traj_stats_summary.json   — machine-readable summary"
echo "=============================================="
