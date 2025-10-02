#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4  # Increased CPUs slightly, often helpful with GPU data loading
#SBATCH --job-name=gad_euler
#SBATCH --output=logs/gad_euler_%j.out
#SBATCH --error=logs/gad_euler_%j.err

# --- Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e

# Define project paths. Changing them here updates the whole script.
PROJECT_DIR="/project/memo/code/ts-tools"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_VERSION="3.11" # Must match the version used in your setup script

# --- Script Start ---
echo ">>> Starting job on node $(hostname)"
echo ">>> Project Directory: ${PROJECT_DIR}"

# Navigate to the project directory. This is good practice.
cd ${PROJECT_DIR}

# 1. Load the same Python module you used to build the environment
# echo ">>> Loading Python ${PYTHON_VERSION} module..."
# source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load python/${PYTHON_VERSION}

# 2. Activate the virtual environment
echo ">>> Activating virtual environment..."
source ${VENV_DIR}/bin/activate

# 3. Run your Python script
echo ">>> Running Euler RMSD script..."
python scripts/gad_gad_euler_rmsd.py

echo "✅ Job completed successfully."