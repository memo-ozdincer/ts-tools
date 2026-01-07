#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=hip_sella_cart
#SBATCH --output=logs/hip_sella_cartesian_%j.out
#SBATCH --error=logs/hip_sella_cartesian_%j.err

# =============================================================================
# HIP Sella TS Refinement - CARTESIAN COORDINATES (internal=False)
# =============================================================================
# Testing hypothesis: Internal coordinate conversion may be causing issues
# with HIP's predicted Hessian. This runs with Cartesian coordinates only.
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
OUT_DIR="$HOME/scratch/ts-tools-output"

export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"
export WANDB_MODE=online
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set." >&2
    exit 2
fi

export WANDB_RUN_GROUP="Experiment-sella-hip-cartesian"

echo ">>> Running HIP Sella with CARTESIAN coordinates (internal=False)..."

START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42
MAX_STEPS=100    # Reduced for diagnostic run
MAX_SAMPLES=3    # Just a few samples for diagnosis
FMAX=0.03

WANDB_NAME="sella | HIP | CARTESIAN | ${START_FROM} | seed${NOISE_SEED} | job${SLURM_JOB_ID}"

python -m src.experiments.Sella.hip_sella \
    --fmax "$FMAX" \
    --max-steps "$MAX_STEPS" \
    --delta0 0.048 \
    --order 1 \
    --rho-inc 1.035 \
    --rho-dec 5.0 \
    --sigma-inc 1.15 \
    --sigma-dec 0.65 \
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
    --no-internal \
    --diagnostic \
    --verbose \
    --wandb \
    --wandb-project "experiment-sella" \
    --wandb-entity "memo-ozdincer-university-of-toronto" \
    --wandb-name "$WANDB_NAME"

echo "Job completed."
