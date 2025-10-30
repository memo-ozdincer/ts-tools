# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains research code for computational chemistry transition state (TS) search and analysis using Gradient Ascent Dynamics (GAD). The code uses machine-learned force fields (specifically Equiformer models from the HIP package) to locate and analyze transition states in chemical reactions.

## Environment Setup

This project is designed to run on Compute Canada clusters with SLURM scheduling. It depends on several external packages:
- `hip`: Provides the EquiformerTorchCalculator and frequency analysis utilities
- `transition1x`: Dataset loader for transition state data from HDF5 files
- `nets`: Provides atom symbol mappings and prediction utilities

### Initial Setup

```bash
# Setup script creates virtual environment and installs dependencies
sbatch setup_env.sh

# Or manually:
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load python/3.11
source .venv/bin/activate
```

### Installing Package

This package can be installed in editable mode:
```bash
uv pip install -e .
```

## Running Jobs

All main scripts are run as SLURM jobs. Scripts are located in `scripts/` directory:

```bash
# Euler integration GAD search
sbatch scripts/gad_euler.slurm

# RK45 adaptive integration GAD search
sbatch scripts/run_gad_rk45.slurm

# Eigenvalue descent optimization
sbatch scripts/run_eigen_optimization.slurm

# Frequency analysis
sbatch scripts/freq_analysis.slurm
```

### Running Scripts Directly

Scripts can also be run directly (e.g., for debugging):

```bash
python -m src.gad_gad_euler_rmsd --max-samples 30 --n-steps 1000 --start-from midpoint_rt --stop-at-ts --dt 0.001

python -m src.gad_rk45_search --max-samples 30 --start-from reactant --stop-at-ts --t-end 2.0

python -m src.gad_eigenvalue_descent --max-samples 30 --start-from reactant --n-steps-opt 200
```

## Code Architecture

### Core Components

**`src/common_utils.py`**: Shared infrastructure
- `Transition1xDataset`: PyTorch Dataset that loads transition state data from HDF5, storing both reactant and TS geometries
- `UsePos`: Transform to set which geometry (reactant vs TS) is active
- `add_common_args()`: Standard CLI arguments for data paths, device, sample counts
- `setup_experiment()`: Boilerplate setup returning (calculator, dataloader, device, output_dir)

**`src/differentiable_projection.py`**: Differentiable vibrational analysis
- `eckart_B_massweighted_torch()`: Constructs 6 Eckart generators (3 translation + 3 rotation) in mass-weighted coordinates
- `eckartprojection_torch()`: Projects out translations/rotations from Hessian eigenspaces
- `differentiable_massweigh_and_eckartprojection_torch()`: Full differentiable frequency analysis pipeline

### Search Methods

**`src/gad_gad_euler_rmsd.py`**: Euler integration GAD search
- Implements simple Euler integration: `pos += dt * forces`
- Two modes: standard GAD (`mode='gad'`) or eigenvalue product minimization (`mode='eigprod'`)
- Optional early stopping when TS is found (`--stop-at-ts`)
- Tracks RMSD to true TS throughout trajectory
- Starting points: `reactant`, `midpoint_rt`, `three_quarter_rt`

**`src/gad_rk45_search.py`**: Adaptive RK45 integration GAD search
- Uses custom RK45 (Runge-Kutta-Fehlberg) adaptive ODE solver for higher accuracy
- Tracks statistics: forces, RMSD, eigenvalues, energy along trajectory
- Event detection for stopping criteria
- More robust than Euler for stiff dynamics

**`src/gad_eigenvalue_descent.py`**: Direct eigenvalue optimization
- Gradient descent to minimize eigenvalue product: `loss = λ₀ × λ₁` where λ₀, λ₁ are smallest projected eigenvalues
- Uses L-BFGS optimizer with line search
- Goal: find geometries where eigenvalue product crosses zero (TS signature)

**`src/gad_frequency_analysis.py`**: Frequency analysis utilities
- Standalone script for analyzing vibrational frequencies at geometries
- Uses `analyze_frequencies_torch()` from HIP package

### Key Patterns

**Data Flow**:
1. Load molecule from Transition1xDataset (stores `pos_reactant`, `pos_transition`, `z`, `energy`, `forces`)
2. Apply `UsePos` transform to set active geometry
3. Convert to PyG batch for model inference
4. Calculator returns `{energy, forces, hessian}`
5. Frequency analysis projects Hessian → eigenvalues/eigenvectors

**RMSD Calculation**:
- `align_ordered_and_get_rmsd()`: Kabsch alignment + RMSD
- Always converts torch tensors to numpy before alignment
- Used to measure convergence to true TS

**Coordinate Systems**:
- Cartesian coordinates: (N_atoms, 3) positions in Angstroms
- Mass-weighted coordinates: Used for Eckart projection and normal modes
- All Hessians are mass-weighted before eigendecomposition

## Data Paths

Standard data locations (on Compute Canada):
- Dataset: `/project/memo/large-files/data/transition1x.h5`
- Model checkpoint: `/project/memo/large-files/ckpt/hip_v2.ckpt`
- Output directory: `/project/memo/large-files/graphs/out` or `results/` (symlink to `/Users/memoozdincer/large-files/out`)

Symbolic links in repo:
- `data/` → `/project/memo/large-files/data`
- `models/` → `/project/memo/large-files/ckpt`
- `results/` → `/Users/memoozdincer/large-files/out`

## Output Format

All scripts output JSON files with per-molecule results:
- Trajectory snapshots: positions, energies, forces, eigenvalues, RMSD
- Summary statistics: final RMSD, number of steps, convergence status
- Metadata: reaction ID, formula, starting point, hyperparameters

Plots (matplotlib PNG) are also generated showing:
- RMSD vs steps
- Energy vs steps
- Eigenvalue products vs steps
