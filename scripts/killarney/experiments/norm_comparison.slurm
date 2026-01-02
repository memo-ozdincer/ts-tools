#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=force_norm
#SBATCH --output=logs/force_norm_%j.out
#SBATCH --error=logs/force_norm_%j.err

mkdir -p logs
source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

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

# All experiments go into the same W&B group
export WANDB_RUN_GROUP="force-norm-${SLURM_JOB_ID}"

echo "=============================================="
echo "Force Normalization Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "W&B Group: $WANDB_RUN_GROUP"
echo "=============================================="

START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42
N_STEPS=1000
MAX_SAMPLES=100
DT=0.001
NORM_THRESHOLD=0.01
WANDB_PROJECT="memo-ozdincer-university-of-toronto"

# ============================================
# Experiment 1: HIP + normalize force
# ============================================
echo ""
echo ">>> [1/2] HIP + normalize force"
echo "=============================================="
python -m src.experiments.multi_mode_force_norm \
    --calculator hip \
    --method euler \
    --normalize-target force \
    --norm-threshold "$NORM_THRESHOLD" \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --n-steps "$N_STEPS" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --stop-at-ts \
    --dt "$DT" \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "HIP_norm-force_thresh${NORM_THRESHOLD}_dt${DT}"

echo ">>> [1/2] Complete"

# ============================================
# Experiment 2: SCINE + normalize force
# ============================================
echo ""
echo ">>> [2/2] SCINE + normalize force"
echo "=============================================="
python -m src.experiments.multi_mode_force_norm \
    --calculator scine \
    --method euler \
    --normalize-target force \
    --norm-threshold "$NORM_THRESHOLD" \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --n-steps "$N_STEPS" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --stop-at-ts \
    --dt "$DT" \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "SCINE_norm-force_thresh${NORM_THRESHOLD}_dt${DT}"

echo ">>> [2/2] Complete"

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "W&B Group: $WANDB_RUN_GROUP"
echo "=============================================="
