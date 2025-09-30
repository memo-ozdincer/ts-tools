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

BASE_DIR="/project/memo"
TS_TOOLS_DIR="${BASE_DIR}/code/ts-tools"
VENV_DIR="${TS_TOOLS_DIR}/.venv"
PYTHON_VERSION="3.10"

# --- Script Start ---
echo ">>> Starting environment setup script..."
echo ">>> Project Base Directory: ${BASE_DIR}"
echo ">>> Python Version: ${PYTHON_VERSION}"

cd ${TS_TOOLS_DIR}

# For python >3.9
echo ">>> Loading system modules..."
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load python/${PYTHON_VERSION}

# uv venv
echo ">>> Creating fresh virtual environment at ${VENV_DIR}"
rm -rf ${VENV_DIR}
uv venv ${VENV_DIR} --python python${PYTHON_VERSION}
source ${VENV_DIR}/bin/activate
echo ">>> Virtual environment activated."

echo ">>> Installing PyTorch and PyG (for CUDA 12.1)..."

uv pip install --upgrade pip

# pytorch for cuda, cu121 is the correct identifier
uv pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Install PyG dependencies matching torch 2.3.1 and CUDA 12.1
uv pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
uv pip install torch-geometric

# 4. Install project-specific repositories
echo ">>> Installing 'hip' and 'ts-tools' dependencies..."

uv pip install -e ../hip

uv pip install -r requirements.txt

echo "Environment setup complete. You can now use the environment at ${VENV_DIR}"