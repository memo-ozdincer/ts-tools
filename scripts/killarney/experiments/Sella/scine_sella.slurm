#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=scine_sella_1A
#SBATCH --output=logs/scine_sella_1A_%j.out
#SBATCH --error=logs/scine_sella_1A_%j.err

# NOTE: No GPU needed for SCINE - it runs on CPU only

mkdir -p logs
source .venv/bin/activate

# Ensure CPU-side linear algebra can use allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Use non-interactive matplotlib backend to avoid tkinter dependency
export MPLBACKEND=Agg

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

export WANDB_RUN_GROUP="Experiment-sella-hip-scine"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running SCINE Sella TS refinement experiment..."

START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42

# Experiment parameters
MAX_STEPS=100
MAX_SAMPLES=20

# Sella-specific parameters
# fmax: Force convergence threshold (eV/A)
# delta0: Initial trust radius
# order: Saddle point order (1 for TS)
FMAX=${FMAX:-0.03}
DELTA0=${DELTA0:-0.1}
ORDER=${ORDER:-1}

WANDB_NAME="experiment | sella | SCINE | ${START_FROM} | seed${NOISE_SEED} | job${SLURM_JOB_ID}"

echo ">>> SCINE Sella TS refinement, start-from=$START_FROM, noise-seed=$NOISE_SEED"
python -m src.experiments.Sella.scine_sella \
    --fmax "$FMAX" \
    --max-steps "$MAX_STEPS" \
    --delta0 "$DELTA0" \
    --order "$ORDER" \
    --eig-interval 10 \
    --scine-functional DFTB0 \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --verbose \
    --wandb \
    --wandb-project "experiment-sella" \
    --wandb-entity "memo-ozdincer-university-of-toronto" \
    --wandb-name "$WANDB_NAME"

echo "Job completed successfully."
