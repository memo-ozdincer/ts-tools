#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=setup_project_env
#SBATCH --output=logs/setup_env_%j.out
#SBATCH --error=logs/setup_env_%j.err

# --- Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e

# Define the base directory for your project in /project space
BASE_DIR="/project/memo"
TS_TOOLS_DIR="${BASE_DIR}/code/ts-tools"
VENV_DIR="${TS_TOOLS_DIR}/.venv"
PYTHON_VERSION="3.10" # As you requested

# --- Script Start ---
echo ">>> Starting environment setup script..."
echo ">>> Project Base Directory: ${BASE_DIR}"
echo ">>> Python Version: ${PYTHON_VERSION}"

# Navigate to the main project directory where the script is located
# This is also where the logs directory should be.
cd ${TS_TOOLS_DIR}

# 1. Load Compute Canada environment stack
echo ">>> Loading system modules..."
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load python/${PYTHON_VERSION}

# 2. Create a fresh virtual environment using uv
echo ">>> Creating fresh virtual environment at ${VENV_DIR}"
rm -rf ${VENV_DIR}
uv venv ${VENV_DIR} --python python${PYTHON_VERSION}
source ${VENV_DIR}/bin/activate
echo ">>> Virtual environment activated."

# 3. Upgrade pip and install core dependencies
echo ">>> Installing PyTorch and PyG (for CUDA 12.1)..."

# Upgrade pip itself using uv
uv pip install --upgrade pip

# Install PyTorch for CUDA 12.1 (cu121 is the correct identifier)
uv pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Install PyG dependencies matching torch 2.3.1 and CUDA 12.1
uv pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
uv pip install torch-geometric

# 4. Install project-specific repositories
echo ">>> Installing 'hip' and 'ts-tools' dependencies..."

# Install the 'hip' repository located at the same level as ts-tools
# The path '../hip' is correct because we are inside the ts-tools directory.
uv pip install -e ../hip

# Install requirements for ts-tools itself (if it has any)
# If you have a requirements.txt in your ts-tools folder, uncomment the next line:
# uv pip install -r requirements.txt

echo "✅ Environment setup complete. You can now use the environment at ${VENV_DIR}"