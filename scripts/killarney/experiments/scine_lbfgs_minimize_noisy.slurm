#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=scine_lbfgs_noisy_2A_30
#SBATCH --output=logs/scine_lbfgs_noisy_2A_30_%j.out
#SBATCH --error=logs/scine_lbfgs_noisy_2A_30_%j.err

# NOTE: No GPU needed for SCINE - it runs on CPU only

mkdir -p logs
source .venv/bin/activate

H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
OUT_DIR="$HOME/scratch/ts-tools-output"

# W&B setup
# - Do NOT hardcode API keys in SLURM scripts; set WANDB_API_KEY in your shell or job env.
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"
export WANDB_MODE=online
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set (required for WANDB_MODE=online)." >&2
    exit 2
fi

export WANDB_RUN_GROUP="Experiment-lbfgs-precondition-hip-scine"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running SCINE L-BFGS energy minimizer (noisy precondition)..."

# Noise level for this experiment
START_FROM="reactant_noise2A"
NOISE_SEED=42

# Minimizer controls
MAX_SAMPLES=10
MAX_ITERATIONS=15000
MAX_STEP=0.5
TARGET_NEG_EIG=1
EIG_CHECK_FREQ=1
SCINE_FUNCTIONAL="DFTB0"

WANDB_NAME="experiment | lbfgs minimize | SCINE | ${SCINE_FUNCTIONAL} | ${START_FROM} | seed${NOISE_SEED} | job${SLURM_JOB_ID}"

echo ">>> SCINE L-BFGS, functional=$SCINE_FUNCTIONAL, start-from=$START_FROM, noise-seed=$NOISE_SEED"
python -m src.lbfgs_energy_minimizer \
    --calculator scine \
    --scine-functional "$SCINE_FUNCTIONAL" \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --max-iterations "$MAX_ITERATIONS" \
    --max-step "$MAX_STEP" \
    --target-neg-eig "$TARGET_NEG_EIG" \
    --eigenvalue-check-freq "$EIG_CHECK_FREQ" \
    --wandb \
    --wandb-project "experiment-lbfgs-minimize" \
    --wandb-name "$WANDB_NAME"

echo "✅ Job completed successfully."
