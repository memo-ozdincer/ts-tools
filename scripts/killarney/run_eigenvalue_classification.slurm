#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=eigenvalue_classification
#SBATCH --output=logs/eigenvalue_classification_%j.out
#SBATCH --error=logs/eigenvalue_classification_%j.err

# This script compares HIP and SCINE eigenvalue classification of starting geometries.
# It counts negative eigenvalues for: midpoint_rt, reactant, product, noise1A, noise2A

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Ensure CPU-side linear algebra can use allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running Eigenvalue Classification (HIP vs SCINE)..."

# Killarney paths
H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CHECKPOINT="/project/6101772/memoozd/ckpt/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output/eigenvalue_classification"

# Run the classification experiment
python -m src.experiments.eigenvalue_classification \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CHECKPOINT" \
    --out-dir "$OUT_DIR" \
    --max-samples 100 \
    --scine-functional DFTB0 \
    --noise-seed 42

echo ">>> Done."
echo `date`: Job $SLURM_JOB_ID complete.
