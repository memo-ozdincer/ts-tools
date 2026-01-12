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

# Upgrade pip first
echo ""
echo ">>> Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA 12.4 support (compatible with CUDA 12.6)
# H100 requires sm_90 architecture, supported in PyTorch 2.1+
# Install PyTorch FIRST before anything else to avoid conflicts
echo ""
echo ">>> Installing PyTorch with CUDA 12.4 (FIRST, before other packages)..."
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
pip install wandb==0.21.0

# =============================================================================
# Install HIP dependencies (based on HIP's requirements.txt)
# These are installed AFTER torch to avoid version conflicts
# =============================================================================
echo ""
echo ">>> Installing HIP dependencies (from HIP requirements)..."

# Core scientific computing
pip install scipy scikit-learn pandas

# Chemistry and molecular modeling
pip install ase rdkit rmsd pyscf openbabel-wheel
# dxtb[libcint] can be tricky, try without libcint first
pip install dxtb || echo "Warning: dxtb install failed, continuing..."

# Visualization and plotting
pip install plotly imageio seaborn kaleido nglview py3Dmol==2.5.0

# Development and formatting
pip install ruff pydantic==2.11.4

# Progress and utilities
pip install tqdm "progressbar==2.5"

# Machine learning (torch-dependent - install carefully)
pip install einops torchmetrics pyarrow fastparquet pytorch_warmup
pip install lightning==2.5.1.post0
pip install triton==3.3.0 || echo "Warning: triton install failed, continuing..."
pip install opt-einsum-fx==0.1.4
pip install e3nn==0.5.1

# Jupyter and notebook support
pip install ipykernel nbformat

# Configuration management
pip install toml omegaconf pyyaml
pip install "hydra-core>=1.0,<2.0" hydra-submitit-launcher

# Experiment tracking and cloud
pip install datasets huggingface_hub kagglehub

# Job submission and distributed computing
pip install submitit joblib==1.5.1 networkx==3.4.2

# Database and storage
pip install lmdb==1.5.1 h5py

# Muon optimizer (skip on HPC - requires git access)
# pip install git+https://github.com/KellerJordan/Muon || echo "Warning: Muon install failed, continuing..."

# =============================================================================
# Install local repositories in editable mode
# =============================================================================

# Install transition1x repository (sibling directory) - BEFORE hip
if [ -d "$PROJECT_ROOT/../transition1x" ]; then
    echo ""
    echo ">>> Installing transition1x repository (../transition1x)..."
    pip install -e "$PROJECT_ROOT/../transition1x" --no-deps
else
    echo "Warning: ../transition1x not found at $PROJECT_ROOT/../transition1x"
    echo "Expected structure:"
    echo "  parent_dir/"
    echo "    ├── ts-tools/"
    echo "    ├── hip/"
    echo "    └── transition1x/"
fi

# Install HIP repository (sibling directory)
if [ -d "$PROJECT_ROOT/../hip" ]; then
    echo ""
    echo ">>> Installing HIP repository (../hip)..."
    pip install -e "$PROJECT_ROOT/../hip" --no-deps
else
    echo "Warning: ../hip not found at $PROJECT_ROOT/../hip"
    echo "Expected structure:"
    echo "  parent_dir/"
    echo "    ├── ts-tools/"
    echo "    ├── hip/"
    echo "    └── transition1x/"
fi

# Install Sella repository if present (inside ts-tools)
if [ -d "$PROJECT_ROOT/sella_repository" ]; then
    echo ""
    echo ">>> Installing Sella repository..."
    pip install -e "$PROJECT_ROOT/sella_repository"
fi

# Install ts-tools in editable mode (LAST)
echo ""
echo ">>> Installing ts-tools in editable mode..."
pip install -e . --no-deps

# Final verification
echo ""
echo ">>> Final verification..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
try:
    import hip
    print('HIP: OK')
except ImportError as e:
    print(f'HIP: FAILED - {e}')
try:
    import transition1x
    print('transition1x: OK')
except ImportError as e:
    print(f'transition1x: FAILED - {e}')
try:
    import sella
    print('Sella: OK')
except ImportError as e:
    print(f'Sella: FAILED - {e}')
"

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
