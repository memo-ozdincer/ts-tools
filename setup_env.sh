#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=setup_project_env
#SBATCH --output=logs/setup_env_%j.out
#SBATCH --error=logs/setup_env_%j.err

# --- Configuration ---
set -e
CODE_DIR="/project/memo/code/ts-tools"
VENV_DIR="${CODE_DIR}/.venv"
PYTHON_VERSION="3.10"

# --- Script Start ---
echo ">>> Starting environment setup script..."
cd ${CODE_DIR}

# 1. Load system modules
echo ">>> Loading Python ${PYTHON_VERSION} module..."
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load python/${PYTHON_VERSION}

# 2. Create a fresh virtual environment
echo ">>> Creating fresh virtual environment at ${VENV_DIR}"
rm -rf ${VENV_DIR}
uv venv ${VENV_DIR} --python python${PYTHON_VERSION}
source ${VENV_DIR}/bin/activate

# 3. Install complex compiled dependencies (PyTorch & PyG)
echo ">>> Installing PyTorch and PyG dependencies (v2.3.1 for CUDA 12.1)..."
PYTORCH_VERSION="2.3.1"
CUDA_VERSION="cu121"
TORCH_PYG_URL="https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html"

uv pip install torch==${PYTORCH_VERSION} --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
uv pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f ${TORCH_PYG_URL}
uv pip install torch-geometric

# 4. Install dependencies from local packages' requirements files
echo ">>> Installing dependencies from requirements.txt files..."
# --- THIS IS THE CORRECT WAY TO INCLUDE YOUR REQUIREMENTS ---
uv pip install -r ../hip/requirements.txt
uv pip install -r requirements.txt # This installs the new, clean ts-tools requirements

# If Transition1x has a requirements.txt, add it here too:
# if [ -f ../Transition1x/requirements.txt ]; then
#     uv pip install -r ../Transition1x/requirements.txt
# fi

# 5. Install the local packages themselves in editable mode
echo ">>> Linking local packages (hip, Transition1x, ts-tools)..."
uv pip install -e ../hip
uv pip install -e ../Transition1x
uv pip install -e . # Install the current directory (ts-tools) as an editable package

# 6. Verification Step
echo ">>> Verifying installation..."
echo ">>> Attempting to import critical libraries..."
python -c "import torch; import torch_scatter; import transition1x; import hip; print('✅✅✅ All critical modules imported successfully!')"

echo "✅ Environment setup and verification complete."