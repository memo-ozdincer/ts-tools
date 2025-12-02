#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/aip-aspuru/memooz/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32GB
#SBATCH --job-name=gad_euler
#SBATCH --output=/scratch/memoozd/ts-tools-output/slurm-%j.txt
#SBATCH --error=/scratch/memoozd/ts-tools-output/slurm-%j.txt

# Activate virtual environment
source .venv/bin/activate

# Set up W&B API key for logging
export WANDB_API_KEY="ed8d3e45635321a86859f77337bdb4dfb5ceb113"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running Euler RMSD script..."
python -m src.gad_gad_euler_rmsd \
    --max-samples 20 \
    --n-steps 150 \
    --start-from midpoint_rt \
    --stop-at-ts \
    --dt 0.001 \
    --wandb \
    --wandb-project "gad-ts-search"

# --- Example with kick mechanism enabled ---
# Uncomment to enable kick mechanism to escape local minima:
# python -m src.gad_gad_euler_rmsd \
#     --max-samples 300 \
#     --n-steps 1000 \
#     --start-from reactant \
#     --stop-at-ts \
#     --dt 0.001 \
#     --enable-kick \
#     --kick-force-threshold 0.015 \
#     --kick-magnitude 0.1

# python -m src.gad_gad_euler_rmsd \
#     --max-samples 300 \
#     --start-from three_quarter_rt

echo "✅ Job completed successfully."