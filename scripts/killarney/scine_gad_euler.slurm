#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=scine_gad_euler
#SBATCH --output=logs/scine_gad_euler_%j.out
#SBATCH --error=logs/scine_gad_euler_%j.err

# NOTE: No GPU needed for SCINE - it runs on CPU only

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Set up W&B API key for logging
export WANDB_API_KEY="ed8d3e45635321a86859f77337bdb4dfb5ceb113"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running GAD Euler with SCINE Sparrow (DFTB0)..."

# Killarney paths
H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
OUT_DIR="$HOME/scratch/ts-tools-output"

# === SCINE DFTB0 - GAD Euler Search ===
python -m src.runners.gad_euler_core \
    --calculator scine \
    --scine-functional DFTB0 \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 30 \
    --n-steps 150 \
    --start-from midpoint_rt \
    --stop-at-ts \
    --dt 0.001 \
    --wandb \
    --wandb-project "gad-ts-search-scine"

# --- Alternative: Use PM6 functional ---
# python -m src.gad_gad_euler_rmsd \
#     --calculator scine \
#     --scine-functional PM6 \
#     --h5-path "$H5_PATH" \
#     --out-dir "$OUT_DIR" \
#     --max-samples 20 \
#     --n-steps 150 \
#     --start-from reactant \
#     --stop-at-ts \
#     --dt 0.001 \
#     --wandb \
#     --wandb-project "gad-ts-search-scine"

# --- Example with kick mechanism and minimization fallback ---
# python -m src.gad_gad_euler_rmsd \
#     --calculator scine \
#     --scine-functional DFTB0 \
#     --h5-path "$H5_PATH" \
#     --out-dir "$OUT_DIR" \
#     --max-samples 100 \
#     --n-steps 500 \
#     --start-from reactant \
#     --stop-at-ts \
#     --dt 0.001 \
#     --enable-kick \
#     --kick-force-threshold 0.015 \
#     --kick-magnitude 0.1 \
#     --enable-minimization-fallback \
#     --use-bfgs \
#     --wandb \
#     --wandb-project "gad-ts-search-scine"

echo "✅ Job completed successfully."
