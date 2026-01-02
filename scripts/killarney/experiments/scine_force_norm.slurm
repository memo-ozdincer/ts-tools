#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=scine_gad_norm
#SBATCH --output=logs/scine_gad_norm_%j.out
#SBATCH --error=logs/scine_gad_norm_%j.err

mkdir -p logs
source .venv/bin/activate

# Ensure CPU-side ops can use the allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
OUT_DIR="$HOME/scratch/ts-tools-output"

# W&B setup
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"
export WANDB_MODE=online
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set (required for WANDB_MODE=online)." >&2
    exit 2
fi

export WANDB_RUN_GROUP="Experiment-gad-norm-scine"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running SCINE GAD vector normalization experiment (fixed dt)..."

START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42

# Experiment parameters
N_STEPS=15000
MAX_SAMPLES=100
DT=0.001

# GAD vector normalization threshold
# When |gad| < this, normalize gad to unit length so step = dt * (gad/|gad|)
GAD_NORM_THRESHOLD=${GAD_NORM_THRESHOLD:-0.01}

WANDB_NAME="experiment | gad-norm | SCINE | ${START_FROM} | gadnorm${GAD_NORM_THRESHOLD} | dt${DT} | seed${NOISE_SEED} | job${SLURM_JOB_ID}"

echo ">>> SCINE Euler + GAD normalization (FIXED dt=$DT), start-from=$START_FROM, noise-seed=$NOISE_SEED"
python -m src.experiments.multi_mode_force_norm \
    --calculator scine \
    --method euler \
    --gad-norm-threshold "$GAD_NORM_THRESHOLD" \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --n-steps "$N_STEPS" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --stop-at-ts \
    --dt "$DT" \
    --wandb \
    --wandb-project "experiment-gad-norm" \
    --wandb-name "$WANDB_NAME"

echo "Job completed successfully."
