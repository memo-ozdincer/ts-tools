# Trillium SLURM Scripts

Scripts for running experiments on the Trillium HPC cluster (SciNet).

## Cluster Information

- **CPU Subcluster**: 192 cores per node, 755 GiB RAM
- **GPU Subcluster**: 4× NVIDIA H100 (SXM) GPUs per node, 96 cores, 755 GiB RAM
- **Storage**: 29 PB VAST NVMe unified storage

### Key Differences from Killarney

1. **No internet on compute nodes** - W&B runs in offline mode
2. **Output must go to `$SCRATCH`** - Home/project are read-only on compute nodes
3. **Account**: Use `--account=rrg-aspuru` instead of `-A aip-aspuru`
4. **GPU requests**: Use `--gpus-per-node=1` or `--gpus-per-node=4` (no other values)
5. **CPU jobs**: Use `--gpus-per-node=0` for CPU-only jobs

## Directory Structure

```
scripts/Trillium/
├── setup_venv.sh                    # Venv setup with PyTorch/CUDA for H100
├── README.md                        # This file
├── experiments/
│   └── Sella/
│       ├── hip_sella_hpo.slurm      # HIP Sella HPO (GPU)
│       └── scine_sella_hpo.slurm    # SCINE Sella HPO (CPU)
└── noisy/
    ├── hip_multi_mode_eckartmw_hpo.slurm   # HIP Multi-Mode HPO (GPU)
    └── scine_multi_mode_eckartmw_hpo.slurm # SCINE Multi-Mode HPO (CPU)
```

## Setup

### 1. Initial Setup (Run Once)

From the ts-tools project root on a Trillium login node:

```bash
# For GPU experiments, use the GPU login node
ssh trillium-gpu.scinet.utoronto.ca

# Run the venv setup script
chmod +x scripts/Trillium/setup_venv.sh
./scripts/Trillium/setup_venv.sh

# Verify GPU access (on GPU login node)
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Copy Project to Scratch (Required for Jobs)

Jobs must write to scratch, and it's faster to run from scratch:

```bash
# Create scratch directories
mkdir -p /scratch/memoozd/ts-tools-scratch/logs

# Optional: sync project to scratch (for faster I/O)
rsync -av /project/rrg-aspuru/memoozd/ts-tools/ /scratch/memoozd/ts-tools/
```

## Running Jobs

### HIP Sella HPO (GPU)

```bash
# From GPU login node (trig-login01)
cd /scratch/memoozd/ts-tools-scratch
sbatch /project/rrg-aspuru/memoozd/ts-tools/scripts/Trillium/experiments/Sella/hip_sella_hpo.slurm

# With custom settings
N_TRIALS=100 MAX_SAMPLES=50 sbatch scripts/Trillium/experiments/Sella/hip_sella_hpo.slurm

# Resume a previous run
RESUME=1 STUDY_NAME=hip_sella_hpo_job12345 sbatch scripts/Trillium/experiments/Sella/hip_sella_hpo.slurm
```

### SCINE Sella HPO (CPU)

```bash
# From CPU login node
cd /scratch/memoozd/ts-tools-scratch
sbatch /project/rrg-aspuru/memoozd/ts-tools/scripts/Trillium/experiments/Sella/scine_sella_hpo.slurm
```

### Multi-Mode HPO

```bash
# HIP (GPU login node)
sbatch scripts/Trillium/noisy/hip_multi_mode_eckartmw_hpo.slurm

# SCINE (CPU login node)
sbatch scripts/Trillium/noisy/scine_multi_mode_eckartmw_hpo.slurm
```

## Configuration Variables

All scripts support environment variable overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `N_TRIALS` | 50 | Number of Optuna trials |
| `MAX_STEPS` | 100 | Max optimization steps per sample |
| `MAX_SAMPLES` | 30 | Samples per trial |
| `START_FROM` | `midpoint_rt_noise1.0A` | Starting geometry |
| `NOISE_SEED` | 42 | Random seed for noise |
| `OPTUNA_SEED` | 42 | Optuna sampler seed |
| `STUDY_NAME` | Auto-generated | Study name for resume |
| `RESUME` | 0 | Set to 1 to resume existing study |
| `SKIP_VERIFY` | 0 | Set to 1 to skip verification (multi-mode) |
| `SCINE_FUNCTIONAL` | DFTB0 | SCINE functional (SCINE scripts) |

## Output Structure

All output goes to scratch:

```
/scratch/memoozd/ts-tools-scratch/
├── logs/
│   ├── hip_sella_hpo_12345.out
│   └── hip_sella_hpo_12345.err
└── hpo/
    └── hip_sella_12345/
        ├── hip_sella_hpo_job12345.db     # SQLite database (main results)
        ├── hip_hpo_results.json          # JSON summary
        ├── best_config.json              # Best hyperparameters
        └── wandb/
            └── offline-run-*/            # W&B offline data
```

## SQLite Database

Results are saved to a clean SQLite database with Optuna's schema:

- **trials**: Trial number, state, objective value
- **trial_params**: Hyperparameter values per trial
- **trial_user_attributes**: Detailed metrics (success_rate, avg_steps, etc.)

### Analyzing Results

```python
import sqlite3
import json

db_path = "/scratch/memoozd/ts-tools-scratch/hpo/hip_sella_12345/hip_sella_hpo_job12345.db"
conn = sqlite3.connect(db_path)

# Get best trial
cursor = conn.cursor()
cursor.execute("""
    SELECT t.number, tv.value 
    FROM trials t 
    JOIN trial_values tv ON t.trial_id = tv.trial_id 
    WHERE tv.value = (SELECT MAX(value) FROM trial_values)
""")
print(cursor.fetchone())

# Get user attributes for best trial
cursor.execute("""
    SELECT key, value_json 
    FROM trial_user_attributes 
    WHERE trial_id = (SELECT trial_id FROM trial_values WHERE value = (SELECT MAX(value) FROM trial_values))
""")
for key, val_json in cursor.fetchall():
    print(f"{key}: {json.loads(val_json)}")
```

Or use the analysis script:
```bash
python scripts/analyze_hpo_results.py
```

## W&B Offline Mode

Trillium has no internet on compute nodes. W&B runs offline and saves data locally.

### Syncing W&B Data

After job completion, sync from a login node (which has internet):

```bash
# From login node
source /project/rrg-aspuru/memoozd/ts-tools/.venv/bin/activate
wandb sync /scratch/memoozd/ts-tools-scratch/hpo/hip_sella_12345/wandb/offline-run-*
```

## Monitoring Jobs

```bash
# Check queue
squeue -u $USER

# Job details
squeue -j <JOBID>

# Cancel job
scancel <JOBID>

# GPU usage (while connected to compute node)
nvidia-smi

# Check job performance
jobperf <JOBID>
```

## Troubleshooting

### Job fails immediately
- Check logs in `/scratch/memoozd/ts-tools-scratch/logs/`
- Ensure data files exist in `/project/rrg-aspuru/memoozd/data/`
- Verify venv activation works

### CUDA errors
- Ensure you submit GPU jobs from `trig-login01` (GPU login node)
- Check module loads: `module load cuda/12.6`

### W&B errors
- W&B should be in offline mode (`WANDB_MODE=offline`)
- Check `WANDB_DIR` exists and is writable

### Resume not working
- Use exact same `STUDY_NAME` as original run
- Database file must exist in `OUT_DIR`
