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

# --- UPDATED PATH ---
# The Transition1x package is now a sibling to hip and ts-tools
TRANSITION1X_PACKAGE_DIR="../Transition1x" 

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

# 3. Install core dependencies
echo ">>> Installing PyTorch and PyG dependencies..."
uv pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
uv pip install torch-geometric

# 4. Install your project-specific local packages
echo ">>> Installing local repositories..."
uv pip install -e ../hip
uv pip install -e ${TRANSITION1X_PACKAGE_DIR}

# 5. Verification Step
echo ">>> Verifying installation..."
echo ">>> Final list of installed packages:"
uv pip list
echo ">>> Attempting to import 'transition1x' to confirm success..."
python -c "import transition1x; print('✅✅✅ transition1x module imported successfully!')"

echo "✅ Environment setup and verification complete."