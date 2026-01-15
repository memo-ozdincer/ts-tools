# HPO Results Sharing Workflow

## Overview

HPO results are stored as Optuna SQLite databases and can be shared via W&B Artifacts.

## Workflow

### 1. On Compute Node (no internet)
HPO runs save results to SQLite and W&B offline:
```
/scratch/memoozd/ts-tools-scratch/dbs/*.db      # Optuna DBs
/scratch/memoozd/ts-tools-scratch/hpo/*/wandb/  # Offline W&B runs
```

The HPO scripts automatically log the SQLite DB as a W&B artifact when the run completes.

### 2. From Login Node (has internet)
Sync everything to W&B:
```bash
./scripts/sync_hpo_results.sh
```

Options:
```bash
./scripts/sync_hpo_results.sh --dbs-only    # Only upload DBs
./scripts/sync_hpo_results.sh --runs-only   # Only sync W&B runs
./scripts/sync_hpo_results.sh --job 201956  # Sync specific job
```

### 3. View Results
- **W&B Dashboard**: https://wandb.ai/memo-ozdincer-university-of-toronto
- **Download artifacts**:
  ```bash
  python scripts/download_hpo_artifacts.py --list
  python scripts/download_hpo_artifacts.py --name scine_sella_hpo_job833394
  ```
- **Optuna Dashboard** (local):
  ```bash
  optuna-dashboard sqlite:///hpo_results/scine_sella_hpo_job833394.db
  ```

### 4. Analyze Locally
```bash
source .venv/bin/activate
python hpo_results/analyze_hpo_results.py
```

## Files

| File | Description |
|------|-------------|
| `*.db` | Optuna SQLite databases |
| `analyze_hpo_results.py` | Reproducible analysis script |
| `hpo_results.tex` | LaTeX summary document |
| `figures/` | Generated plots |

## Sharing with Supervisor

1. Run `./scripts/sync_hpo_results.sh` from login node
2. Share W&B project link: https://wandb.ai/memo-ozdincer-university-of-toronto/sella-hpo
3. They can view trials, compare hyperparameters, and download artifacts
