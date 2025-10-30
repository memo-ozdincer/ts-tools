#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=gad_rk45
#SBATCH --output=logs/gad_rk45_%j.out
#SBATCH --error=logs/gad_rk45_%j.err

# --- Configuration ---
set -e
PROJECT_DIR="/project/memo/code/ts-tools"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_VERSION="3.11"

# --- Script Start ---
echo ">>> Starting job on node $(hostname)"
cd ${PROJECT_DIR}

echo ">>> Loading Python ${PYTHON_VERSION} module..."
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load python/${PYTHON_VERSION}

echo ">>> Activating virtual environment..."
source ${VENV_DIR}/bin/activate

echo ">>> Running GAD-RK45 Search script (No Kick)..."

# --- Main Experiment: Start from Reactant and search for TS ---
# This version uses the standard GAD dynamics with the RK45 solver.
python -m src.gad_rk45_search \
    --max-samples 30 \
    --start-from reactant \
    --stop-at-ts \
    --t-end 2.0

echo "✅ Job completed successfully."