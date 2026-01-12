#!/bin/bash
# =============================================================================
# Trillium Virtual Environment Setup Script
# =============================================================================
#
# Creates a Python virtual environment with correct PyTorch/CUDA wheels for
# Trillium's H100 GPUs (CUDA 12.x).
#
# Usage:
#   ./setup_venv.sh [venv_name]
#
# Default venv name: .venv
#
# Run this from the ts-tools project root on a Trillium login node.
# For GPU jobs, run from trig-login01 (GPU login node).
# =============================================================================

set -e  # Exit on error

VENV_NAME="${1:-.venv}"
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

echo "=============================================="
echo "Trillium Virtual Environment Setup"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Venv name: $VENV_NAME"
echo "=============================================="

# Load required modules
echo ""
echo ">>> Loading modules..."
module purge
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6

echo "Loaded modules:"
module list

# Create virtual environment
echo ""
echo ">>> Creating virtual environment..."
cd "$PROJECT_ROOT"

if [ -d "$VENV_NAME" ]; then
    echo "Warning: $VENV_NAME already exists."
    read -p "Delete and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_NAME"
        python -m venv "$VENV_NAME"
    fi
else
    python -m venv "$VENV_NAME"
fi

# Activate the venv
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo ""
echo ">>> Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 support (compatible with CUDA 12.6)
# H100 requires sm_90 architecture, supported in PyTorch 2.1+
echo ""
echo ">>> Installing PyTorch with CUDA 12.4..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA availability
echo ""
echo ">>> Verifying PyTorch CUDA installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Install common scientific packages
echo ""
echo ">>> Installing scientific packages..."
pip install numpy scipy h5py ase

# Install Optuna for HPO
echo ""
echo ">>> Installing Optuna..."
pip install optuna optuna-dashboard

# Install W&B (will run in offline mode on Trillium)
echo ""
echo ">>> Installing wandb..."
pip install wandb

# Install the project in editable mode
echo ""
echo ">>> Installing ts-tools in editable mode..."
pip install -e .

# Install HIP repository (sibling directory)
if [ -d "$PROJECT_ROOT/../hip" ]; then
    echo ""
    echo ">>> Installing HIP repository (../hip)..."
    pip install -e "$PROJECT_ROOT/../hip"
else
    echo "Warning: ../hip not found, skipping"
fi

# Install transition1x repository (sibling directory)
if [ -d "$PROJECT_ROOT/../transition1x" ]; then
    echo ""
    echo ">>> Installing transition1x repository (../transition1x)..."
    pip install -e "$PROJECT_ROOT/../transition1x"
else
    echo "Warning: ../transition1x not found, skipping"
fi

# Install Sella repository if present (inside ts-tools)
if [ -d "$PROJECT_ROOT/sella_repository" ]; then
    echo ""
    echo ">>> Installing Sella repository..."
    pip install -e "$PROJECT_ROOT/sella_repository"
fi

# Print summary
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "For GPU jobs, submit from scratch directory:"
echo "  cd \$SCRATCH/ts-tools"
echo "  sbatch scripts/Trillium/experiments/Sella/hip_sella_hpo.slurm"
echo ""
echo "Remember:"
echo "  - Trillium has no internet access on compute nodes"
echo "  - W&B must run in offline mode"
echo "  - Job output must go to \$SCRATCH"
echo "  - Home/project are read-only on compute nodes"
echo "=============================================="
