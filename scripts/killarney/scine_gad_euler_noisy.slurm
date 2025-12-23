#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=scine_gad_euler_noisy
#SBATCH --output=logs/scine_gad_euler_noisy_%j.out
#SBATCH --error=logs/scine_gad_euler_noisy_%j.err

# NOTE: No GPU needed for SCINE - it runs on CPU only

mkdir -p logs
source .venv/bin/activate

# W&B setup (online)
export WANDB_MODE=online
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set (required for WANDB_MODE=online)." >&2
    exit 2
fi

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running GAD Euler (core) with SCINE Sparrow on noisy geometries..."

# Killarney paths
H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
OUT_DIR="$HOME/scratch/ts-tools-output"

# Clean Euler-GAD from a noisy reactant geometry.
# Noise is injected by `src/common_utils.parse_starting_geometry`.
python -m src.runners.gad_euler_core \
    --calculator scine \
    --scine-functional DFTB0 \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 20 \
    --n-steps 300 \
    --start-from reactant_noise2A \
    --stop-at-ts \
    --dt 0.001 \
    --wandb \
    --wandb-project "gad-noise-experiments-scine"

echo "✅ Job completed successfully."
