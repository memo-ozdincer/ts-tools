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

echo ">>> Running Eigenvalue Descent script..."

# Each type of loss f'n.
# python -m src.gad_eigenvalue_descent \
#     --max-samples 150 \
#     --n-steps-opt 50 \
#     --lr 0.01 \
#     --start-from midpoint_rt \
#     --loss-type targeted_magnitude \
#     --target-eig0 -0.05 \
#     --target-eig1 0.10 \
#
# python -m src.gad_eigenvalue_descent \
#     --max-samples 150 \
#     --n-steps-opt 50 \
#     --lr 0.01 \
#     --start-from midpoint_rt \
#     --loss-type midpoint_squared
#
# python -m src.gad_eigenvalue_descent \
#     --max-samples 150 \
#     --n-steps-opt 50 \
#     --lr 0.01 \
#     --start-from midpoint_rt \
#     --loss-type eig_product \
#     --early-stop-eig-product 5e-4
#
# python -m src.gad_eigenvalue_descent \
#     --max-samples 150 \
#     --n-steps-opt 200 \
#     --lr 0.01 \
#     --start-from midpoint_rt \
#     --loss-type relu

# --- Optional: Adaptive target relaxation for targeted magnitude ---
python -m src.gad_eigenvalue_descent \
    --max-samples 150 \
    --n-steps-opt 250 \
    --lr 0.01 \
    --start-from reactant \
    --loss-type targeted_magnitude \
    --target-eig0 -0.10 \
    --target-eig1 0.15 \
    --adaptive-targets \
    --adaptive-relax-steps 50 \
    --adaptive-final-eig0 -0.02 \
    --adaptive-final-eig1 0.05

echo "✅ Job completed."
