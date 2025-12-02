#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/aip-aspuru/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32GB
#SBATCH --job-name=gad_bfgs
#SBATCH --output=/scratch/memoozd/ts-tools-output/slurm-%j.txt
#SBATCH --error=/scratch/memoozd/ts-tools-output/slurm-%j.txt

# Activate virtual environment
source .venv/bin/activate

# Set up W&B API key for logging
export WANDB_API_KEY="ed8d3e45635321a86859f77337bdb4dfb5ceb113"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running GAD BFGS Optimization..."

# Killarney paths
H5_PATH="$HOME/projects/aip-aspuru/memoozd/data/transition1x.h5"
CKPT_PATH="$HOME/projects/aip-aspuru/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output"

# === GAD BFGS OPTIMIZATION ===
# Standalone saddle point optimization using L-BFGS-B with GAD vector as gradient.
# Finds points where GAD vector = 0 (saddle points).

echo ">>> Running GAD BFGS (standalone saddle point optimization)"
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
