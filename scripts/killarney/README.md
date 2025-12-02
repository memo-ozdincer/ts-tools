# Killarney Cluster SLURM Scripts

This directory contains SLURM scripts configured for the Killarney cluster.

## Key Differences from Compute Canada Scripts

1. **Account**: Uses `-A aip-aspuru` allocation
2. **Working Directory**: `/project/aip-aspuru/memoozd/ts-tools`
3. **GPU Type**: Requests L40S GPUs (`--gres=gpu:l40s:1`)
4. **Output Directory**: `/scratch/memoozd/ts-tools-output/` (high I/O, not backed up)
5. **Environment**: Uses UV-managed virtual environment (no module loading)

## Data Paths

The scripts expect:
- **Dataset**: `~/projects/aip-aspuru/memoozd/data/transition1x.h5`
- **Model checkpoint**: `~/projects/aip-aspuru/memoozd/models/hip_v2.ckpt`
- **Output directory**: `~/scratch/ts-tools-output/`

## Usage

From the ts-tools root directory:

```bash
# Submit eigenvalue descent job
sbatch scripts/killarney/run_eigen_optimization.slurm

# Submit GAD Euler integration
sbatch scripts/killarney/gad_euler.slurm

# Submit RK45 adaptive integration
sbatch scripts/killarney/run_gad_rk45.slurm

# Frequency analysis
sbatch scripts/killarney/freq_analysis.slurm

# Starting geometry statistics
sbatch scripts/killarney/run_starting_geom_stats.slurm
```

## Initial Setup on Killarney

Before running jobs, ensure:

1. **Virtual environment is set up**:
   ```bash
   cd ~/projects/aip-aspuru/memoozd/ts-tools
   uv venv .venv
   source .venv/bin/activate
   uv pip install -e .
   uv pip install -e ../HIP
   ```

2. **Output directory exists**:
   ```bash
   mkdir -p ~/scratch/ts-tools-output
   ```

3. **Data files are downloaded**:
   - Place `transition1x.h5` in `~/projects/aip-aspuru/memoozd/data/`
   - Place `hip_v2.ckpt` in `~/projects/aip-aspuru/memoozd/models/`
