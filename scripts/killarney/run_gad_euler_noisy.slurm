#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/aip-aspuru/memooz/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32GB
#SBATCH --job-name=gad_euler_noisy
#SBATCH --output=/scratch/memoozd/ts-tools-output/slurm-%j.txt
#SBATCH --error=/scratch/memoozd/ts-tools-output/slurm-%j.txt

# Activate virtual environment
source .venv/bin/activate

# Set up W&B API key for logging
export WANDB_API_KEY="ed8d3e45635321a86859f77337bdb4dfb5ceb113"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running GAD-Euler Search with noisy starting geometries..."

# Killarney paths
H5_PATH="$HOME/projects/aip-aspuru/memooz/data/transition1x.h5"
CKPT_PATH="$HOME/projects/aip-aspuru/memooz/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

# === EIGPRODUCT MINIMIZATION EXPERIMENTS ===
# Testing two approaches to escape weird geometries when GAD stalls:
#   1. Larger step sizes (gradient descent with bigger max_step)
#   2. L-BFGS-B optimization (more robust, can take larger steps)

# --- EXPERIMENT 1: Larger step size (gradient descent) ---
echo ">>> Running Experiment 1: Larger step size (gradient descent)"
python -m src.gad_gad_euler_rmsd \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 30 \
    --start-from reactant_noise2A \
    --enable-minimization-fallback \
    --eig-descent-lr 0.01 \
    --eig-descent-max-step 0.5 \
    --stop-at-ts \
    --n-steps 500 \
    --dt 0.005 \
    --wandb \
    --wandb-project "gad-noise-experiments"

# --- EXPERIMENT 2: L-BFGS-B optimization ---
echo ">>> Running Experiment 2: L-BFGS-B optimization"
python -m src.gad_gad_euler_rmsd \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 30 \
    --start-from reactant_noise2A \
    --enable-minimization-fallback \
    --use-bfgs \
    --bfgs-maxiter 50 \
    --bfgs-gtol 1e-5 \
    --bfgs-max-step 0.5 \
    --stop-at-ts \
    --n-steps 500 \
    --dt 0.005 \
    --wandb \
    --wandb-project "gad-noise-experiments"

echo "✅ Job completed successfully."
