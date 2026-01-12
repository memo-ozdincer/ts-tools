#!/bin/bash
# =============================================================================
# Download Transition1x dataset for Trillium
# =============================================================================
# Run this from a Trillium login node (which has internet access)
# =============================================================================

set -e

# Target directory
DATA_DIR="/project/rrg-aspuru/memoozd/data"
mkdir -p "$DATA_DIR"

echo "=============================================="
echo "Downloading Transition1x Dataset"
echo "=============================================="
echo "Target: $DATA_DIR/transition1x.h5"
echo "=============================================="

# Load Python
module load StdEnv/2023
module load python/3.11.5

# Create temp venv for download
TEMP_VENV=$(mktemp -d)
python -m venv "$TEMP_VENV"
source "$TEMP_VENV/bin/activate"

# Clone and install transition1x
cd /tmp
rm -rf Transition1x
git clone https://gitlab.com/matschreiner/Transition1x.git
cd Transition1x
pip install .

# Download the dataset
echo ""
echo ">>> Downloading transition1x.h5..."
python download_t1x.py "$DATA_DIR"

# Cleanup
rm -rf "$TEMP_VENV"
rm -rf /tmp/Transition1x

echo ""
echo "=============================================="
echo "Download complete!"
echo "Dataset: $DATA_DIR/transition1x.h5"
ls -lh "$DATA_DIR/transition1x.h5"
echo "=============================================="
