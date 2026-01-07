#!/bin/bash
#SBATCH --job-name=scine_sella_hpo
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=hip_sella_1A
#SBATCH --output=logs/hip_sella_1A_%j.out
#SBATCH --error=logs/hip_sella_1A_%j.err

# =============================================================================
# SCINE Sella Hyperparameter Optimization
# =============================================================================
#
# This script performs a grid search over Sella hyperparameters to find the
# best configuration for converging to true first-order saddle points (TS).
#
# Based on: Wander et al. (2024) "Accessing Numerical Energy Hessians with 
# Graph Neural Network Potentials" (arXiv:2410.01650v2)
#
# Key insights from the paper:
# - Using exact Hessians improved TS convergence from ~65% to 93%
# - Trust radius management critical for stability
# - Force convergence threshold affects saddle point quality
#
# FIXED parameters from paper (NOT grid-searched):
# - use_exact_hessian: True (always)
# - diag_every_n: 1 (always - Hessian at every step)
# - delta0: 4.8E-2 = 0.048 (initial trust radius, included in grid)
# - rho_inc: 1.035 (trust radius increase threshold)
# - rho_dec: 5.0 (trust radius decrease threshold)
# - sigma_inc: 1.15 (trust radius increase factor)
# - sigma_dec: 0.65 (trust radius decrease factor)
#
# Primary metric: eigenvalue_ts_rate (fraction with exactly 1 negative eigenvalue)
#
# NOTE: No GPU needed for SCINE - it runs on CPU only
# =============================================================================

mkdir -p logs
source .venv/bin/activate

# Ensure CPU-side linear algebra can use allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Use non-interactive matplotlib backend
export MPLBACKEND=Agg

H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
OUT_DIR="$HOME/scratch/ts-tools-output/hpo"

# W&B setup
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"
export WANDB_MODE=online
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set (required for WANDB_MODE=online)." >&2
    exit 2
fi

export WANDB_RUN_GROUP="sella-hpo-scine"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running SCINE Sella HPO experiment..."

# =============================================================================
# HPO Grid Search Parameters
# =============================================================================
# Based on paper recommendations (Wander et al. 2024):
#
# FMAX values (force convergence threshold in eV/Å):
#   Very tight values for precise saddle point convergence
#   - 0.00005: Extremely tight (5e-5)
#   - 0.0001: Very tight (1e-4)
#   - 0.0003: Tight (3e-4)
#   - 0.0005: Moderate (5e-4)
#
# NOTE: eta (finite difference step size, paper=7.0E-4) is NOT used since
# we use exact analytical Hessians from SCINE (not finite differences).
# =============================================================================

# Configurable HPO parameters (can override via environment variables)
FMAX_VALUES="${FMAX_VALUES:-0.00005 0.0001 0.0003 0.0005}"
# DELTA0 is FIXED at paper value 0.048 (not grid-searched)

# Experiment parameters
MAX_STEPS="${MAX_STEPS:-150}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
START_FROM="${START_FROM:-midpoint_rt_noise1.0A}"
NOISE_SEED="${NOISE_SEED:-42}"

# Whether to also test Cartesian coordinates (default: internal only)
TEST_CARTESIAN="${TEST_CARTESIAN:-false}"

WANDB_NAME="sella-hpo-scine_${START_FROM}_seed${NOISE_SEED}_job${SLURM_JOB_ID}"

echo ">>> HPO Grid Search Configuration:"
echo "    FMAX values: $FMAX_VALUES"
echo "    MAX_STEPS: $MAX_STEPS"
echo "    MAX_SAMPLES: $MAX_SAMPLES"
echo "    START_FROM: $START_FROM"
echo "    TEST_CARTESIAN: $TEST_CARTESIAN"
echo ""
echo ">>> Fixed parameters (from paper):"
echo "    delta0: 0.048"
echo "    use_exact_hessian: True"
echo "    diag_every_n: 1"
echo "    rho_inc: 1.035, rho_dec: 5.0"
echo "    sigma_inc: 1.15, sigma_dec: 0.65"

# Build command arguments
CARTESIAN_ARG=""
if [ "$TEST_CARTESIAN" = "true" ]; then
    CARTESIAN_ARG="--test-cartesian"
fi

python -m src.experiments.Sella.scine_sella_hpo \
    --fmax-values $FMAX_VALUES \
    --max-steps "$MAX_STEPS" \
    --max-samples "$MAX_SAMPLES" \
    --scine-functional DFTB0 \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    $CARTESIAN_ARG \
    --verbose \
    --wandb \
    --wandb-project "sella-hpo" \
    --wandb-entity "memo-ozdincer-university-of-toronto" \
    --wandb-name "$WANDB_NAME"

echo "Job completed successfully."
echo ""
echo "Results saved to: $OUT_DIR/hpo_results.json"
echo "Best config saved to: $OUT_DIR/best_config.json"
