#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/aip-aspuru/memoozd/ts-tools
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
H5_PATH="$HOME/projects/aip-aspuru/memoozd/data/transition1x.h5"
CKPT_PATH="$HOME/projects/aip-aspuru/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

# === GAD + EIGPROD MINIMIZATION EXPERIMENTS ===
# Testing three approaches:
#   1. GAD with gradient-based eigproduct fallback (larger step sizes)
#   2. GAD with BFGS-based eigproduct fallback
#   3. Standalone GAD BFGS (saddle point optimization)

# --- EXPERIMENT 1: GAD + Eigprod (gradient descent) ---
# Increases max_step from 0.1 to 0.5 Å to escape weird geometries
# echo ">>> Running Experiment 1: Larger step size (gradient descent)"
# python -m src.gad_gad_euler_rmsd \
#     --h5-path "$H5_PATH" \
#     --checkpoint-path "$CKPT_PATH" \
#     --out-dir "$OUT_DIR" \
#     --max-samples 30 \
#     --start-from reactant_noise2A \
#     --enable-minimization-fallback \
#     --eig-descent-lr 0.01 \
#     --eig-descent-max-step 0.5 \
#     --stop-at-ts \
#     --n-steps 1000 \
#     --dt 0.005 \
#     --wandb \
#     --wandb-project "gad-noise-experiments"

# --- EXPERIMENT 2: GAD + Eigprod (L-BFGS-B fallback) ---
# Uses scipy L-BFGS-B for more robust eigproduct minimization when GAD stalls
echo ">>> Running Experiment 2: Eigprod BFGS (fallback mode)"
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
    --n-steps 1000 \
    --dt 0.005 \
    --wandb \
    --wandb-project "gad-noise-experiments"

# --- EXPERIMENT 3: GAD BFGS (standalone saddle point optimization) ---
# Uses L-BFGS-B with GAD vector as gradient to directly find saddle points
echo ">>> Running Experiment 3: GAD BFGS (standalone)"
python -m src.gad_gad_euler_rmsd \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 30 \
    --start-from reactant_noise2A \
    --use-gad-bfgs \
    --gad-bfgs-maxiter 100 \
    --gad-bfgs-gtol 1e-5 \
    --gad-bfgs-max-step 1.0 \
    --wandb \
    --wandb-project "gad-noise-experiments"

echo "✅ Job completed successfully."
