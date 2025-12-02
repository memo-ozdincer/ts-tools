#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/aip-aspuru/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32GB
#SBATCH --job-name=eig_desc
#SBATCH --output=/scratch/memoozd/ts-tools-output/slurm-%j.txt
#SBATCH --error=/scratch/memoozd/ts-tools-output/slurm-%j.txt

# Activate virtual environment
source .venv/bin/activate

# Set up W&B API key for logging
export WANDB_API_KEY="ed8d3e45635321a86859f77337bdb4dfb5ceb113"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running Eigenvalue Descent script..."

# Killarney paths
H5_PATH="$HOME/projects/aip-aspuru/memoozd/data/transition1x.h5"
CKPT_PATH="$HOME/projects/aip-aspuru/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

# Each type of loss f'n. Uncomment exactly one command below.
# python -m src.gad_eigenvalue_descent \
#     --h5-path "$H5_PATH" \
#     --checkpoint-path "$CKPT_PATH" \
#     --out-dir "$OUT_DIR" \
#     --max-samples 500 \
#     --n-steps-opt 500 \
#     --lr 0.01 \
#     --start-from reactant \
#     --loss-type targeted_magnitude \
#     --target-eig0 -0.05 \
#     --target-eig1 0.10
#
# python -m src.gad_eigenvalue_descent \
#     --h5-path "$H5_PATH" \
#     --checkpoint-path "$CKPT_PATH" \
#     --out-dir "$OUT_DIR" \
#     --max-samples 300 \
#     --n-steps-opt 200 \
#     --lr 0.01 \
#     --start-from reactant_noise1A \
#     --loss-type sign_enforcer \
#     --sign-neg-target -0.01 \
#     --sign-pos-floor 0.1 \
#     --use-line-search \
#     --adaptive-max-displacement \
#     --initial-max-displacement 2.0 \
#     --final-max-displacement 0.1 \
#     --wandb \
#     --wandb-project "gad-ts-search"

echo "✅ Job completed."
