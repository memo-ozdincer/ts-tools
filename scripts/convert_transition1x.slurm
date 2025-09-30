#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=convert_t1x
#SBATCH --output=logs/convert_t1x_%j.out
#SBATCH --error=logs/convert_t1x_%j.err

# --- Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e

# Define project paths
CODE_DIR="/project/memo/code/ts-tools"
VENV_DIR="${CODE_DIR}/.venv"
PYTHON_VERSION="3.10" # Must match the version used in your setup script

# --- Script Start ---
echo ">>> Starting data conversion job on node $(hostname)"
echo ">>> Project Directory: ${CODE_DIR}"

# Navigate to the code directory
cd ${CODE_DIR}

# 1. Load the same Python module you used to build the environment
echo ">>> Loading Python ${PYTHON_VERSION} module..."
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load python/${PYTHON_VERSION}

# 2. Activate the virtual environment
echo ">>> Activating virtual environment..."
source ${VENV_DIR}/bin/activate

# 3. Run your Python script
echo ">>> Running the data conversion script..."
python scripts/convert_t1x_to_lmdb.py --limit 100

echo "✅ Job completed successfully."