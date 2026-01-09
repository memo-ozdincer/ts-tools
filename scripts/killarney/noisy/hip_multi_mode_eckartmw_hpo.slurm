#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=20:00:00
# NOTE: GAD is sequential per-sample, so 1 GPU is sufficient
# Multiple GPUs would only help if we parallelize across samples (not implemented)
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96GB
#SBATCH --job-name=hip_multi_mode_hpo
#SBATCH --output=logs/hip_multi_mode_hpo_%j.out
#SBATCH --error=logs/hip_multi_mode_hpo_%j.err

mkdir -p logs
source .venv/bin/activate

# Ensure CPU-side ops can use the allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CKPT_PATH="/project/6101772/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output/hpo"

# W&B setup
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"
export WANDB_MODE=online
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set (required for WANDB_MODE=online)." >&2
    exit 2
fi

export WANDB_RUN_GROUP="HPO-multi-mode-eckartmw"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running HIP Multi-Mode Eckart-MW HPO..."

# Run parameters (with early stopping at 500 steps of no progress)
# Reduced samples for faster HPO trials (30 is statistically meaningful)
N_STEPS=4000
MAX_SAMPLES=${MAX_SAMPLES:-30}
START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42

# HPO configuration
N_TRIALS=${N_TRIALS:-50}
N_STARTUP_TRIALS=${N_STARTUP_TRIALS:-5}
STUDY_NAME=${STUDY_NAME:-"hip_multi_mode_hpo_${SLURM_JOB_ID}"}

# Resume flag (set RESUME=1 to resume existing study)
RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then
    RESUME_FLAG="--resume"
fi

# Skip verification flag (set SKIP_VERIFY=1 to skip)
SKIP_VERIFY_FLAG=""
if [[ "${SKIP_VERIFY:-0}" == "1" ]]; then
    SKIP_VERIFY_FLAG="--skip-verification"
fi

echo ">>> HPO Configuration:"
echo "    Study: $STUDY_NAME"
echo "    Trials: $N_TRIALS"
echo "    Startup trials: $N_STARTUP_TRIALS"
echo "    Resume: ${RESUME:-0}"
echo "    Skip verification: ${SKIP_VERIFY:-0}"

python -m src.noisy.hip_multi_mode_eckartmw_hpo \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --n-steps "$N_STEPS" \
    --max-samples "$MAX_SAMPLES" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --n-trials "$N_TRIALS" \
    --n-startup-trials "$N_STARTUP_TRIALS" \
    --study-name "$STUDY_NAME" \
    $RESUME_FLAG \
    $SKIP_VERIFY_FLAG \
    --wandb \
    --wandb-project "hip-multi-mode-hpo"

echo "Job completed successfully."
