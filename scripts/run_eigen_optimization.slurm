#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=relu_manual_desc  # CHANGED: More descriptive job name
#SBATCH --output=logs/relu_manual_desc_%j.out # CHANGED: Log file matches job name

set -e
PROJECT_DIR="/project/memo/code/ts-tools" # Adjust if needed
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_VERSION="3.11" 

cd ${PROJECT_DIR}
source /cvfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load python/${PYTHON_VERSION}
source ${VENV_DIR}/bin/activate

echo ">>> Running MANUAL ReLU Eigenvalue Descent script..."
python -m src.gad_relu_manual_descent \
    --max-samples 30 \
    --n-steps-opt 40 \
    --lr 0.005 \
    --start-from midpoint_rt

echo "✅ Job completed."