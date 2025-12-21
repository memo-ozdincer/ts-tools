#!/bin/bash
# Setup script for downloading Transition1x dataset on Killarney

set -e

echo "=== Transition1x Dataset Download for Killarney ==="
echo ""

# Define paths
DATA_DIR="/project/6101772/memoozd/data"
TEMP_DIR="$HOME/scratch/tmp_t1x_download"

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$TEMP_DIR"

# Clone Transition1x repo
echo ">>> Cloning Transition1x repository..."
cd "$TEMP_DIR"
if [ -d "Transition1x" ]; then
    echo "Transition1x directory already exists, removing..."
    rm -rf Transition1x
fi
git clone https://gitlab.com/matschreiner/Transition1x
cd Transition1x

# Install transition1x package (should already be in your venv)
echo ">>> Installing transition1x package..."
source "/project/6101772/memoozd/ts-tools/.venv/bin/activate"
pip install .

# Download the dataset
echo ">>> Downloading Transition1x HDF5 file to $DATA_DIR..."
python download_t1x.py "$DATA_DIR"

# Verify download
if [ -f "$DATA_DIR/transition1x.h5" ]; then
    echo ""
    echo "✅ Success! Dataset downloaded to: $DATA_DIR/transition1x.h5"
    ls -lh "$DATA_DIR/transition1x.h5"

    # Check dataset info
    echo ""
    echo ">>> Dataset info:"
    python -c "
from transition1x import Dataloader
import os

h5_path = os.path.join('$DATA_DIR', 'transition1x.h5')
print(f'Path: {h5_path}')
print('')

# Count samples in each split
for split in ['train', 'val', 'test']:
    loader = Dataloader(h5_path, datasplit=split, only_final=True)
    count = sum(1 for _ in loader)
    print(f'{split.upper():5s} set: {count:6d} reactions')
"
else
    echo "❌ Error: Dataset file not found at $DATA_DIR/transition1x.h5"
    exit 1
fi

# Cleanup
echo ""
echo ">>> Cleaning up temporary files..."
cd "$HOME"
rm -rf "$TEMP_DIR"

echo ""
echo "=== Setup complete! ==="
echo "Dataset location: $DATA_DIR/transition1x.h5"
echo ""
echo "Note: The full dataset contains train/val/test splits."
echo "      You can specify which split to use with --split argument in your scripts."
