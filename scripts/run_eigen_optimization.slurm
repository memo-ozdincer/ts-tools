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

echo ">>> Running eigenvalue descent script..."
python -m src.gad_eigenvalue_descent \
    --max-samples 30 \
    --n-steps-opt 200 \
    --start-from midpoint_rt

# python -m src.gad_direct_gradient \
    # --max-samples 30 \
    # --start-from reactant \
    # --loss-function relu_sum

echo "✅ Job completed."