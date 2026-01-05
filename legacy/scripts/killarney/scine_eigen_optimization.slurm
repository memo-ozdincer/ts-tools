#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=scine_eigen_opt
#SBATCH --output=logs/scine_eig_desc_%j.out
#SBATCH --error=logs/scine_eig_desc_%j.err

# NOTE: No GPU needed for SCINE - it runs on CPU only

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Set up W&B API key for logging


echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running Eigenvalue Descent with SCINE Sparrow..."

# Killarney paths
H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
OUT_DIR="$HOME/scratch/ts-tools-output"

# === EXPERIMENT 1: Linear Eigproduct Descent with SCINE DFTB0 ===
echo ">>> Running Linear Eigproduct Descent with SCINE DFTB0"
python -m src.gad_eigenvalue_descent \
    --calculator scine \
    --scine-functional DFTB0 \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 30 \
    --start-from reactant_noise2A \
    --loss-type eig_product \
    --n-steps-opt 200 \
    --lr 0.01 \
    --use-line-search \
    --adaptive-max-displacement \
    --initial-max-displacement 2.0 \
    --final-max-displacement 0.1 \
    --wandb \
    --wandb-project "gad-scine-experiments"

# === EXPERIMENT 2: Sign Enforcer with SCINE DFTB0 ===
echo ">>> Running Sign Enforcer with SCINE DFTB0"
python -m src.gad_eigenvalue_descent \
    --calculator scine \
    --scine-functional DFTB0 \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples 30 \
    --start-from reactant_noise2A \
    --loss-type sign_enforcer \
    --sign-neg-target -5e-3 \
    --sign-pos-floor 1e-3 \
    --n-steps-opt 200 \
    --lr 0.01 \
    --use-line-search \
    --wandb \
    --wandb-project "gad-scine-experiments"

# === EXPERIMENT 3: Targeted Magnitude with SCINE PM6 ===
# echo ">>> Running Targeted Magnitude Gradient Descent with SCINE PM6"
# python -m src.gad_eigenvalue_descent \
#     --calculator scine \
#     --scine-functional PM6 \
#     --h5-path "$H5_PATH" \
#     --out-dir "$OUT_DIR" \
#     --max-samples 30 \
#     --start-from reactant \
#     --loss-type targeted_magnitude \
#     --target-eig0 -0.05 \
#     --target-eig1 0.10 \
#     --n-steps-opt 200 \
#     --lr 0.01 \
#     --use-line-search \
#     --wandb \
#     --wandb-project "gad-scine-experiments"

# === EXPERIMENT 4: Compare different SCINE functionals ===
# Test DFTB0, PM6, and AM1 on the same dataset
# for FUNCTIONAL in DFTB0 PM6 AM1; do
#     echo ">>> Running with $FUNCTIONAL"
#     python -m src.gad_eigenvalue_descent \
#         --calculator scine \
#         --scine-functional $FUNCTIONAL \
#         --h5-path "$H5_PATH" \
#         --out-dir "$OUT_DIR" \
#         --max-samples 10 \
#         --start-from reactant \
#         --loss-type eig_product \
#         --n-steps-opt 200 \
#         --lr 0.01 \
#         --wandb \
#         --wandb-project "gad-scine-functional-comparison"
# done

echo "✅ Job completed successfully."
