#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=hip_sella_grid
#SBATCH --output=logs/hip_sella_grid_%j_%a.out
#SBATCH --error=logs/hip_sella_grid_%j_%a.err
#SBATCH --array=0-3

# =============================================================================
# HIP Sella Grid Search - Trust Radius Parameters
# =============================================================================
# Testing different trust radius configurations:
#   0: Default (baseline)
#   1: Tolerant (larger steps, less shrinking)
#   2: Aggressive (much larger steps, very tolerant)
#   3: Very Aggressive (extreme)
#
# Uses DIRECT HIP Hessians (no Eckart projection, no autograd)
# Uses Sella's INTERNAL coordinates
# =============================================================================

mkdir -p logs
source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MPLBACKEND=Agg

H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CKPT_PATH="/project/6101772/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output/grid_${SLURM_ARRAY_TASK_ID}"

export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"
export WANDB_MODE=online
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set." >&2
    exit 2
fi

export WANDB_RUN_GROUP="Experiment-sella-hip-gridsearch"

# Common parameters
START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42
MAX_STEPS=150
MAX_SAMPLES=10
FMAX=0.03

# Grid search configurations
case $SLURM_ARRAY_TASK_ID in
    0)
        # Default (baseline from Sella paper)
        CONFIG_NAME="default"
        DELTA0=0.048
        RHO_DEC=5.0
        SIGMA_DEC=0.65
        SIGMA_INC=1.15
        ;;
    1)
        # Tolerant - less aggressive shrinking
        CONFIG_NAME="tolerant"
        DELTA0=0.1
        RHO_DEC=10.0
        SIGMA_DEC=0.8
        SIGMA_INC=1.2
        ;;
    2)
        # Aggressive - larger steps, much more tolerant
        CONFIG_NAME="aggressive"
        DELTA0=0.2
        RHO_DEC=20.0
        SIGMA_DEC=0.85
        SIGMA_INC=1.3
        ;;
    3)
        # Very Aggressive - extreme tolerance
        CONFIG_NAME="very_aggressive"
        DELTA0=0.5
        RHO_DEC=50.0
        SIGMA_DEC=0.9
        SIGMA_INC=1.5
        ;;
esac

echo `date`: Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID is allocated resources.
echo "=============================================="
echo "HIP Sella Grid Search - Config: $CONFIG_NAME"
echo "=============================================="
echo "delta0=$DELTA0, rho_dec=$RHO_DEC"
echo "sigma_dec=$SIGMA_DEC, sigma_inc=$SIGMA_INC"
echo "=============================================="
echo "Using DIRECT HIP Hessians (no projection)"
echo "Using Sella INTERNAL coordinates"
echo "=============================================="

WANDB_NAME="sella-grid | HIP | ${CONFIG_NAME} | d0=${DELTA0} | job${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

python -m src.experiments.Sella.hip_sella \
    --fmax "$FMAX" \
    --max-steps "$MAX_STEPS" \
    --delta0 "$DELTA0" \
    --order 1 \
    --rho-inc 1.035 \
    --rho-dec "$RHO_DEC" \
    --sigma-inc "$SIGMA_INC" \
    --sigma-dec "$SIGMA_DEC" \
    --eig-interval 1 \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --use-exact-hessian \
    --diag-every-n 1 \
    --gamma 0.0 \
    --diagnostic \
    --verbose \
    --wandb \
    --wandb-project "experiment-sella" \
    --wandb-entity "memo-ozdincer-university-of-toronto" \
    --wandb-name "$WANDB_NAME"

# NOTE: No --no-internal = uses internal coordinates (default)

echo ""
echo "=============================================="
echo "Config $CONFIG_NAME completed at `date`"
echo "=============================================="
