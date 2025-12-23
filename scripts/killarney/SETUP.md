# Killarney Setup Guide

Complete setup instructions for running ts-tools on the Killarney cluster.

## 1. Initial Directory Setup

```bash
# SSH to Killarney (i forget the command)
# Create directory structure
mkdir -p /project/6101772/memoozd/{data,models}
mkdir -p ~/scratch/ts-tools-output

# Clone repositories
cd /project/6101772/memoozd
git clone <your-ts-tools-repo-url> ts-tools
git clone <your-HIP-repo-url> HIP
```

## 2. Environment Setup

```bash
cd /project/6101772/memoozd/ts-tools

# Create virtual environment with UV
uv venv .venv
source .venv/bin/activate

# Install packages in editable mode
uv pip install -e .
uv pip install -e ../HIP
```

## 3. Download Transition1x Dataset

**Option A: Automatic download (recommended)**

```bash
# Make setup script executable
chmod +x scripts/killarney/setup_data.sh

# Run setup script (downloads to /project/6101772/memoozd/data/)
./scripts/killarney/setup_data.sh
```

**Option B: Manual download**

```bash
# Clone Transition1x
cd ~/scratch
git clone https://gitlab.com/matschreiner/Transition1x
cd Transition1x

# Activate your venv
source /project/6101772/memoozd/ts-tools/.venv/bin/activate

# Install transition1x
pip install .

# Download dataset to project directory
python download_t1x.py /project/6101772/memoozd/data
```

**Dataset Info:**
- Full HDF5 file: ~1-2 GB
- Contains: train, validation, and test splits
- Each split has thousands of reactions with reactant → TS → product geometries
- Default scripts use `--split test` but you can specify `train` or `val`

## 4. Download Model Checkpoint

You'll need to manually download the HIP model checkpoint (`hip_v2.ckpt`) and place it in:
```
/project/6101772/memoozd/models/hip_v2.ckpt
```

If you have it on another cluster or locally, you can transfer it:

```bash
# From your local machine or other cluster:
scp /path/to/hip_v2.ckpt killarney:/project/6101772/memoozd/models/
```

## 5. Verify Setup

```bash
cd /project/6101772/memoozd/ts-tools
source .venv/bin/activate

# Check files exist
ls -lh /project/6101772/memoozd/data/transition1x.h5
ls -lh /project/6101772/memoozd/models/hip_v2.ckpt

# Test data loading
python -c "
from transition1x import Dataloader
loader = Dataloader('/project/6101772/memoozd/data/transition1x.h5',
                    datasplit='test', only_final=True)
mol = next(iter(loader))
print('✅ Data loader works!')
print(f\"Sample reaction: {mol['transition_state']['rxn']}\")
"
```

## 6. Submit Your First Job

```bash
# Make sure you're in the ts-tools directory
cd /project/6101772/memoozd/ts-tools

# Submit a test job (small sample size)
sbatch scripts/killarney/gad_euler.slurm

# Check job status
squeue -u $USER

# Monitor output (e.g., for gad_euler)
tail -f logs/gad_euler_<JOBID>.out
```

## Directory Structure After Setup

```
/project/6101772/memoozd/
├── ts-tools/              # This repository
│   ├── .venv/            # Virtual environment
│   ├── src/
│   ├── scripts/
│   │   └── killarney/   # Killarney SLURM scripts
│   ├── logs/             # Job output logs
│   └── ...
├── HIP/                  # HIP dependency
├── data/
│   └── transition1x.h5   # Dataset (~1-2GB)
└── models/
    └── hip_v2.ckpt       # Model checkpoint

~/scratch/
└── ts-tools-output/      # Simulation results
```

## Using Different Dataset Splits

The Transition1x dataset has three splits:
- **train**: Largest split, most reactions
- **val**: Validation split
- **test**: Test split (default in our scripts)

To use a different split, modify the `--split` argument in your SLURM script or when running directly:

```bash
# Use training set (largest)
python -m src.gad_euler_rmsd \
    --split train \
    --max-samples 1000 \
    ...

# Use validation set
python -m src.gad_euler_rmsd \
    --split val \
    --max-samples 500 \
    ...
```

## Troubleshooting

**Issue**: `transition1x` module not found
- **Solution**: Make sure you installed it: `pip install git+https://gitlab.com/matschreiner/Transition1x`

**Issue**: Dataset file not found
- **Solution**: Check path is correct: `ls /project/6101772/memoozd/data/transition1x.h5`

**Issue**: GPU not allocated
- **Solution**: Check SLURM output logs, may need to wait in queue for L40S GPUs

**Issue**: Out of memory
- **Solution**: Reduce `--max-samples` or increase `--mem` in SLURM script
