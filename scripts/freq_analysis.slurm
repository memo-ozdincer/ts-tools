#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=freq_analysis
#SBATCH --output=logs/freq_analysis_%j.out
#SBATCH --error=logs/freq_analysis_%j.err

# --- Configuration ---
set -e
PROJECT_DIR="/project/memo/code/ts-tools"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_VERSION="3.10" # Must match the version used in your setup script

# --- Script Start ---
echo ">>> Starting job on node $(hostname)"
echo ">>> Project Directory: ${PROJECT_DIR}"

cd ${PROJECT_DIR}

# 1. Load the same Python module
echo ">>> Loading Python ${PYTHON_VERSION} module..."
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load python/${PYTHON_VERSION}

# 2. Activate the virtual environment
echo ">>> Activating virtual environment..."
source ${VENV_DIR}/bin/activate

# 3. Run your Python script
echo ">>> Running frequency analysis script..."
python scripts/gad_frequency_analysis.py

echo "✅ Job completed successfully."