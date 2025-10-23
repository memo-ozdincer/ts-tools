#!/bin/bash
#SBATCH --job-name=eigen_opt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eigen_opt_%j.log

set -e
PROJECT_DIR="/project/memo/code/ts-tools"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_VERSION="3.11"

cd ${PROJECT_DIR}

source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load python/${PYTHON_VERSION}
source ${VENV_DIR}/bin/activate

echo ">>> Running Eigenvalue Product Optimization script..."

python -m src.gad_eigen_optimization \
    --max-samples 300 \
    --n-steps 200 \
    --lr 0.01 \
    --start-from reactant \
    --stop-at-ts

echo "✅ Job completed successfully."