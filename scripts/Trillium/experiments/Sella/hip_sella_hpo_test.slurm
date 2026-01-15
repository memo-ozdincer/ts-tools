#!/bin/bash
#SBATCH --job-name=hip_sella_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/hip_sella_test_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/hip_sella_test_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# HIP Sella HPO - LOW-SAMPLE TEST VERSION
# =============================================================================
# 
# Quick test version with minimal samples to verify the pipeline works.
# Uses: 5 samples, 3 trials, 50 max steps
# Expected runtime: ~10-20 minutes
# =============================================================================

set -e

mkdir -p /scratch/memoozd/ts-tools-scratch/logs

PROJECT_DIR="/project/rrg-aspuru/memoozd/ts-tools"
SCRATCH_DIR="/scratch/memoozd/ts-tools-scratch"

cd "$SCRATCH_DIR"

# Clear pycache to avoid stale bytecode issues
echo "Clearing Python bytecode cache..."
find "$PROJECT_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$PROJECT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

module purge
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6

source "$PROJECT_DIR/.venv/bin/activate"

# Cache directories
export TMPDIR="${SCRATCH_DIR}/tmp"
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export MPLCONFIGDIR="${SCRATCH_DIR}/.config/matplotlib"
export FONTCONFIG_PATH=/etc/fonts
export FONTCONFIG_FILE=/etc/fonts/fonts.conf

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

export OMP_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export MKL_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24
export VECLIB_MAXIMUM_THREADS=24
export MPLBACKEND=Agg

# Data paths
H5_PATH="/project/rrg-aspuru/memoozd/data/transition1x.h5"
CKPT_PATH="/project/rrg-aspuru/memoozd/models/hip_v2.ckpt"
OUT_DIR="${SCRATCH_DIR}/hpo/hip_sella_test_${SLURM_JOB_ID}"
DB_DIR="${SCRATCH_DIR}/dbs"

mkdir -p "$OUT_DIR" "$DB_DIR"

export WANDB_MODE=offline
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"

# ========== LOW-SAMPLE TEST CONFIG ==========
N_TRIALS=3
MAX_STEPS=50
MAX_SAMPLES=5
PRESCREEN=0  # Skip prescreening for test
START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42
OPTUNA_SEED=42
STUDY_NAME="hip_sella_TEST_${SLURM_JOB_ID}"
# ============================================

echo "=============================================="
echo "HIP Sella HPO - LOW SAMPLE TEST"
echo "=============================================="
echo "Date: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=============================================="
echo "TEST CONFIG (low-sample):"
echo "  N_TRIALS: $N_TRIALS"
echo "  MAX_STEPS: $MAX_STEPS"
echo "  MAX_SAMPLES: $MAX_SAMPLES"
echo "  STUDY_NAME: $STUDY_NAME"
echo "=============================================="

nvidia-smi

python -m src.experiments.Sella.hip_sella_hpo \
    --n-trials "$N_TRIALS" \
    --max-steps "$MAX_STEPS" \
    --max-samples "$MAX_SAMPLES" \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$DB_DIR" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --optuna-seed "$OPTUNA_SEED" \
    --study-name "$STUDY_NAME" \
    --verbose \
    --wandb \
    --wandb-project "sella-hpo-test" \
    --wandb-entity "memo-ozdincer-university-of-toronto" \
    --wandb-name "hip_sella_TEST_${SLURM_JOB_ID}"

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "TEST PASSED at $(date)"
else
    echo "TEST FAILED with code $EXIT_CODE at $(date)"
fi
echo "=============================================="

exit $EXIT_CODE
