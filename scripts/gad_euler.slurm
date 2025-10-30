#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=gad_euler
#SBATCH --output=logs/gad_euler_%j.out
#SBATCH --error=logs/gad_euler_%j.err

# --- Configuration ---
set -e
PROJECT_DIR="/project/memo/code/ts-tools"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_VERSION="3.11"

# --- Script Start ---
echo ">>> Starting job on node $(hostname)"
echo ">>> Project Directory: ${PROJECT_DIR}"

cd ${PROJECT_DIR}

echo ">>> Loading Python ${PYTHON_VERSION} module..."
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load python/${PYTHON_VERSION}

echo ">>> Activating virtual environment..."
source ${VENV_DIR}/bin/activate

echo ">>> Running Euler RMSD script..."
# ARGS R HERE NOW
python -m src.gad_gad_euler_rmsd \
    --max-samples 300 \
    --n-steps 1000 \
    --start-from midpoint_rt \
    --stop-at-ts \
    --dt 0.001

# python -m src.gad_gad_euler_rmsd \
#     --max-samples 300 \
#     --start-from three_quarter_rt

# python -m src.gad_gad_euler_rmsd \
#     --mode eigprod \
#     --start-from reactant \
#     --stop-at-ts \
#     --n-steps 200 \
#     --max-samples 10 \
    
echo "✅ Job completed successfully."