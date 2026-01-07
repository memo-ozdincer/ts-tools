#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=hip_sella_1A
#SBATCH --output=logs/hip_sella_1A_%j.out
#SBATCH --error=logs/hip_sella_1A_%j.err

mkdir -p logs
source .venv/bin/activate

# Ensure CPU-side ops can use the allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Use non-interactive matplotlib backend to avoid tkinter dependency
export MPLBACKEND=Agg

H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CKPT_PATH="/project/6101772/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

# W&B setup
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"
export WANDB_MODE=online
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set (required for WANDB_MODE=online)." >&2
    exit 2
fi

export WANDB_RUN_GROUP="Experiment-sella-hip-scine"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running HIP Sella TS refinement experiment..."

START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42

# Experiment parameters
MAX_STEPS=300
MAX_SAMPLES=20
FMAX=0.03

# Sella-specific parameters
# fmax: Force convergence threshold (eV/A)
# delta0: Initial trust radius
# order: Saddle point order (1 for TS)
FMAX=${FMAX:-0.03}
DELTA0=${DELTA0:-0.1}
ORDER=${ORDER:-1}

# Hessian computation parameters
# use_exact_hessian: Use exact Hessian from HIP at every step (default: true)
#   This is crucial for ML potentials - avoids iterative approximation
# diag_every_n: How often to recompute Hessian (default: 1 = every step)
# gamma: Tolerance for iterative eigensolver when NOT using exact Hessian (default: 0.0)
USE_EXACT_HESSIAN=${USE_EXACT_HESSIAN:-true}
DIAG_EVERY_N=${DIAG_EVERY_N:-1}
GAMMA=${GAMMA:-0.0}

WANDB_NAME="experiment | sella | HIP | ${START_FROM} | seed${NOISE_SEED} | job${SLURM_JOB_ID}"

echo ">>> HIP Sella TS refinement, start-from=$START_FROM, noise-seed=$NOISE_SEED"
echo ">>> Hessian mode: exact=$USE_EXACT_HESSIAN, diag_every_n=$DIAG_EVERY_N, gamma=$GAMMA"

# Build Hessian arguments
HESSIAN_ARGS=""
if [ "$USE_EXACT_HESSIAN" = "true" ]; then
    HESSIAN_ARGS="--use-exact-hessian"
else
    HESSIAN_ARGS="--no-exact-hessian"
fi
HESSIAN_ARGS="$HESSIAN_ARGS --diag-every-n $DIAG_EVERY_N --gamma $GAMMA"

python -m src.experiments.Sella.hip_sella \
    --fmax "$FMAX" \
    --max-steps "$MAX_STEPS" \
    --delta0 "$DELTA0" \
    --order "$ORDER" \
    --eig-interval 10 \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    $HESSIAN_ARGS \
    --verbose \
    --wandb \
    --wandb-project "experiment-sella" \
    --wandb-entity "memo-ozdincer-university-of-toronto" \
    --wandb-name "$WANDB_NAME"

echo "Job completed successfully."
