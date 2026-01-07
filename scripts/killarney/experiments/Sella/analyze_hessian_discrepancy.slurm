#!/bin/bash
#SBATCH --job-name=hess-discrepancy
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=logs/hess_discrepancy_%j.out
#SBATCH --error=logs/hess_discrepancy_%j.err

# ==============================================================================
# Analyze discrepancy between HIP's predicted Hessian and autograd Hessian
#
# This script compares:
#   1. Predicted Hessian: Direct neural network output (what HIP predicts)
#   2. Autograd Hessian: Computed as H = -dF/dx via automatic differentiation
#
# For Sella's trust radius mechanism to work correctly, the Hessian must be
# consistent with the forces (H = -dF/dx). Large discrepancies explain why
# Sella+HIP has erratic trust radius behavior while Sella+SCINE works well.
# ==============================================================================

set -e

# Change to project root
cd /home/user/ts-tools

# Create logs directory
mkdir -p logs

# Activate environment
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate ts-tools

# Configuration
CHECKPOINT_PATH="ckpt/hip_v2.ckpt"
H5_PATH="data/transition1x.h5"
MAX_SAMPLES=20
SPLIT="test"

echo "=============================================="
echo "HIP Hessian Discrepancy Analysis"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Dataset: $H5_PATH"
echo "Max samples: $MAX_SAMPLES"
echo "Split: $SPLIT"
echo "=============================================="

python -m src.experiments.Sella.analyze_hessian_discrepancy \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --h5-path "$H5_PATH" \
    --max-samples $MAX_SAMPLES \
    --split "$SPLIT" \
    --device cuda

echo ""
echo "=============================================="
echo "Analysis complete!"
echo "=============================================="
