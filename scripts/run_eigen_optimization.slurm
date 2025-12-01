#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=proj_eig_desc
#SBATCH --output=logs/proj_eig_desc_%j.out

set -e
PROJECT_DIR="/project/memo/code/ts-tools" # Adjust if needed
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_VERSION="3.11" 

cd ${PROJECT_DIR}
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load python/${PYTHON_VERSION}
source ${VENV_DIR}/bin/activate

# Set up W&B API key for logging
export WANDB_API_KEY="ed8d3e45635321a86859f77337bdb4dfb5ceb113"

echo ">>> Running Eigenvalue Descent script..."

# Each type of loss f'n. Uncomment exactly one command below.
# python -m src.gad_eigenvalue_descent \
#     --max-samples 500 \
#     --n-steps-opt 500 \
#     --lr 0.01 \
#     --start-from reactant \
#     --loss-type targeted_magnitude \
#     --target-eig0 -0.05 \
#     --target-eig1 0.10
#
# python -m src.gad_eigenvalue_descent \
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
# #
# python -m src.gad_eigenvalue_descent \
#     --max-samples 500 \
#     --n-steps-opt 500 \
#     --lr 0.01 \
#     --start-from reactant \
#     --loss-type midpoint_squared
#
# python -m src.gad_eigenvalue_descent \
#     --max-samples 30 \
#     --n-steps-opt 150 \
#     --lr 0.01 \
#     --start-from midpoint_rt \
#     --loss-type eig_product \
#     --early-stop-eig-product 5e-4 \
#     --wandb \
#     --wandb-project "gad-ts-search"
#
#
# python -m src.gad_eigenvalue_descent \
#     --max-samples 1000 \
#     --n-steps-opt 500 \
#     --lr 0.01 \
#     --start-from reactant \
#     --loss-type relu

# --- Optional: Adaptive target relaxation for targeted magnitude ---
# python -m src.gad_eigenvalue_descent \
#     --max-samples 500 \
#     --n-steps-opt 500 \
#     --lr 0.01 \
#     --start-from reactant \
#     --loss-type targeted_magnitude \
#     --target-eig0 -0.10 \
#     --target-eig1 0.15 \
#     --adaptive-targets \
#     --adaptive-relax-steps 50 \
#     --adaptive-final-eig0 -0.02 \
#     --adaptive-final-eig1 0.05

# --- Adaptive Step Size for Noisy Starting Geometries ---
# The following parameters help with noisy starting geometries (noise levels: 0.5, 1, 2 Å):
#
# --adaptive-step-sizing: Enable saddle-order-aware step sizing (RECOMMENDED for noisy starts)
# --higher-order-multiplier: Scale factor for higher-order saddles (default: 5.0)
# --ts-multiplier: Scale factor near TS for refinement (default: 0.5)
# --use-line-search: Tests multiple step sizes [0.25×, 0.5×, 1×, 2×, 4×] and chooses the best
# --adaptive-max-displacement: Gradually reduces max displacement from initial to final
# --initial-max-displacement 2.0: Start with large steps (2.0 Å) for noisy geometries
# --final-max-displacement 0.1: End with small steps (0.1 Å) for fine-tuning
#
# Example with different noise levels:
# python -m src.gad_eigenvalue_descent \
#     --max-samples 30 \
#     --n-steps-opt 150 \
#     --lr 0.01 \
#     --start-from reactant_noise2A \
#     --loss-type eig_product \
#     --adaptive-step-sizing \
#     --higher-order-multiplier 10.0 \
#     --ts-multiplier 0.5 \
#     --use-line-search \
#     --adaptive-max-displacement \
#     --initial-max-displacement 3.0 \
#     --final-max-displacement 0.1 \
#     --wandb \
#     --wandb-project "gad-ts-search"
python -m src.gad_eigenvalue_descent \
    --adaptive-step-sizing \
    --higher-order-multiplier 10.0 \
    --ts-multiplier 0.5 \
    --max-samples 30 \
    --n-steps-opt 150 \
    --start-from reactant_noise2A \
    --loss-type eig_product \
    --use-line-search \
    --wandb \
    --wandb-project "gad-noise-experiments"

echo "✅ Job completed."
