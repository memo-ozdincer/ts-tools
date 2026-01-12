#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --job-name=scine_mm_hpo_informed
#SBATCH --output=logs/scine_multi_mode_hpo_informed_%j.out
#SBATCH --error=logs/scine_multi_mode_hpo_informed_%j.err

# NOTE: No GPU needed for SCINE - it runs on CPU only

# =============================================================================
# SCINE Multi-Mode Eckart-MW HPO - INFORMED SEARCH
# =============================================================================
# Uses narrowed search ranges based on 800-step HPO results (job 1809794).
# 
# Key findings from previous HPO (trial 14):
#   - adaptive_delta=FALSE works best for SCINE (opposite of HIP!)
#   - dt ~0.00213, dt_max ~0.0229
#   - plateau_patience ~9, plateau_boost ~2.67
#   - escape_window ~35
#
# This script runs 30 trials with focused search ranges.
# =============================================================================

mkdir -p logs
source .venv/bin/activate

# Ensure CPU-side linear algebra can use allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
OUT_DIR="$HOME/scratch/ts-tools-output/hpo_informed"

# W&B setup
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"
export WANDB_MODE=online
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set (required for WANDB_MODE=online)." >&2
    exit 2
fi

export WANDB_RUN_GROUP="HPO-multi-mode-eckartmw-informed"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running SCINE Multi-Mode Eckart-MW HPO - INFORMED SEARCH..."

# Run parameters
N_STEPS=4000
MAX_SAMPLES=100
START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42
SCINE_FUNCTIONAL="DFTB0"

# HPO configuration - fewer trials needed with informed ranges
N_TRIALS=${N_TRIALS:-30}
N_STARTUP_TRIALS=${N_STARTUP_TRIALS:-3}
STUDY_NAME=${STUDY_NAME:-"scine_multi_mode_hpo_informed_${SLURM_JOB_ID}"}

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

echo ">>> INFORMED HPO Configuration:"
echo "    Study: $STUDY_NAME"
echo "    Trials: $N_TRIALS (reduced - informed search)"
echo "    Startup trials: $N_STARTUP_TRIALS"
echo "    N_STEPS: $N_STEPS"
echo "    Resume: ${RESUME:-0}"
echo "    Skip verification: ${SKIP_VERIFY:-0}"
echo "    NOTE: Search ranges narrowed based on trial 14 from 800-step HPO"
echo "    NOTE: adaptive_delta=False FIXED (SCINE-specific finding!)"

python -m src.noisy.scine_multi_mode_eckartmw_hpo_informed \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --scine-functional "$SCINE_FUNCTIONAL" \
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
    --wandb-project "scine-multi-mode-hpo-informed"

echo ""
echo `date`: Job $SLURM_JOB_ID INFORMED HPO complete.
