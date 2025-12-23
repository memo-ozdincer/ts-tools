#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32GB
#SBATCH --job-name=lbfgs_minimize
#SBATCH --output=logs/lbfgs_minimize_%j.out
#SBATCH --error=logs/lbfgs_minimize_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# W&B setup (online)
export WANDB_MODE=online
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set (required for WANDB_MODE=online)." >&2
    exit 2
fi

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running L-BFGS Energy Minimization for pre-conditioning noisy geometries..."

# Killarney paths
H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CKPT_PATH="/project/6101772/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

# === L-BFGS ENERGY MINIMIZATION ===
# Pre-condition noisy starting geometries by descending to a minimum or
# low-order saddle point before running GAD/eigenvalue descent.
#
# Strategy:
#   1. Start from noisy geometry (e.g., reactant + 2Å Gaussian noise)
#   2. Run L-BFGS to minimize energy until neg_eig_count <= 1
#   3. Output: geometries ready for TS-finding algorithms

# --- Test with 2Å noise (common challenging case) ---
echo ""
echo "=== Running L-BFGS with 2Å noise ==="
python -m src.lbfgs_energy_minimizer \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 30 \
    --start-from reactant_noise2A \
    --max-iterations 200 \
    --max-step 0.5 \
    --target-neg-eig 1 \
    --eigenvalue-check-freq 1 \
    --noise-seed 42 \
    --wandb \
    --wandb-project "lbfgs-precondition"

# --- Test with 5Å noise (more extreme) ---
echo ""
echo "=== Running L-BFGS with 5Å noise ==="
python -m src.lbfgs_energy_minimizer \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 30 \
    --start-from reactant_noise5A \
    --max-iterations 300 \
    --max-step 1.0 \
    --target-neg-eig 1 \
    --eigenvalue-check-freq 1 \
    --noise-seed 42 \
    --wandb \
    --wandb-project "lbfgs-precondition"

# --- Test with 10Å noise (extreme case) ---
echo ""
echo "=== Running L-BFGS with 10Å noise ==="
python -m src.lbfgs_energy_minimizer \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 30 \
    --start-from reactant_noise10A \
    --max-iterations 500 \
    --max-step 2.0 \
    --target-neg-eig 1 \
    --eigenvalue-check-freq 1 \
    --noise-seed 42 \
    --wandb \
    --wandb-project "lbfgs-precondition"

echo ""
echo "✅ L-BFGS minimization job completed successfully."

