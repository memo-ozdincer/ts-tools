# Bayesian Hyperparameter Optimization for Multi-Mode Eckart-MW GAD

This directory contains Optuna-based Bayesian hyperparameter optimization (HPO) scripts for improving convergence rates of the multi-mode Eckart-MW GAD algorithm on both HIP and SCINE calculators.

## Overview

**Goal**: Maximize convergence to first-order saddle points (1 negative eigenvalue) across all samples.

**Current Performance**:
- HIP: ~74% convergence rate
- SCINE: ~92% convergence rate

**Strategy**: Use Bayesian optimization (TPE sampler) to find hyperparameters that help the remaining non-converging samples converge while maintaining or improving performance on already-converging samples.

## Key Features

- **W&B Logging**: Detailed per-trial hyperparameter and metric tracking (no plots, just stats)
- **Crash Recovery**: SQLite storage saves every trial immediately - resume on failure
- **Graceful Error Handling**: Saves partial results on KeyboardInterrupt or exceptions
- **Difficult Sample Focus**: Pre-screens samples to identify hard cases, focuses HPO on those

## Files

### Python Scripts
- `hip_multi_mode_eckartmw_hpo.py` - HPO for HIP calculator
- `scine_multi_mode_eckartmw_hpo.py` - HPO for SCINE calculator

### SLURM Scripts
- `hip_multi_mode_eckartmw_hpo.slurm` - Cluster job for HIP HPO
- `scine_multi_mode_eckartmw_hpo.slurm` - Cluster job for SCINE HPO

## Hyperparameters Optimized

| Parameter | Description | Search Range | Log Scale |
|-----------|-------------|--------------|-----------|
| `dt_min` | Minimum time step | 1e-7 to 1e-5 | Yes |
| `dt_max` | Maximum time step | 0.01 to 0.1 | Yes |
| `plateau_patience` | Steps before dt adjustment | 3 to 20 | No |
| `plateau_boost` | dt increase factor on improvement | 1.2 to 3.0 | No |
| `plateau_shrink` | dt decrease factor on plateau | 0.3 to 0.7 | No |
| `escape_disp_threshold` | Displacement threshold for plateau detection (Å) | 1e-5 to 1e-3 | Yes |
| `escape_window` | Window size for plateau averaging | 10 to 50 | No |
| `escape_neg_vib_std` | Max std(neg_vib) for stable saddle | 0.1 to 1.0 | No |
| `escape_delta` | Base perturbation magnitude (Å) | 0.05 to 0.5 | No |
| `adaptive_delta_scale` | Adaptive delta scaling factor | 0.0 to 2.0 | No |
| `trust_radius_max` | Max atom displacement per step (Å) | 0.1 to 0.5 | No |

## Algorithm Details

### Sample Selection Strategy

The HPO focuses on **difficult samples** that fail to converge with baseline parameters:

1. **Baseline Test**: Run all samples (up to 100) for 200 steps with default parameters
2. **Difficulty Scoring**: 
   - Non-converged samples get difficulty = 1000
   - Converged samples get difficulty = steps_to_converge
3. **Selection**: Take top 50% most difficult samples (configurable via `--difficulty-threshold`)
4. **Focused Optimization**: Run HPO trials only on these difficult samples

This approach:
- Reduces computational cost (fewer samples per trial)
- Focuses optimization on problematic cases
- Maintains performance on already-working samples (monitored but not optimized)

### Objective Function

```python
score = convergence_rate - 0.01 * normalized_steps_to_converge
```

Where:
- `convergence_rate` = fraction of samples reaching 1 negative eigenvalue
- `normalized_steps_to_converge` = mean(steps) / n_steps_max for converged samples
- The 0.01 coefficient makes convergence rate the primary objective (~100x more important)

This heavily prioritizes getting samples to converge, with convergence speed as a secondary tiebreaker.

### Optuna Configuration

- **Sampler**: TPE (Tree-structured Parzen Estimator) - Bayesian optimization
- **Direction**: Maximize (higher score = better)
- **Startup Trials**: 10 random trials for initial exploration
- **Storage**: SQLite database for persistence and resumability

## Usage

### Basic Usage (SLURM)

```bash
# HIP optimization
sbatch hip_multi_mode_eckartmw_hpo.slurm

# SCINE optimization
sbatch scine_multi_mode_eckartmw_hpo.slurm
```

### Crash Recovery / Resume

Progress is saved to SQLite after each trial. To resume a crashed or interrupted job:

```bash
# Resume from previous study
RESUME=--resume STUDY_NAME=hip-gad-hpo-12345 sbatch hip_multi_mode_eckartmw_hpo.slurm
```

### Custom Configuration

Modify environment variables in SLURM script or set before submission:

```bash
export N_TRIALS=200              # Number of optimization trials (default: 100)
export N_STEPS_PER_SAMPLE=1000   # Steps per sample per trial (default: 800)
export N_SAMPLES=20              # Samples per trial (default: 15)
export DIFFICULTY_THRESHOLD=0.3  # Fraction of samples to use (default: 0.5)

sbatch hip_multi_mode_eckartmw_hpo.slurm
```

### W&B Logging

W&B logging is enabled automatically if `WANDB_API_KEY` is set. Metrics logged:

- **Per-trial**: All hyperparameters (`hparams/*`), convergence rate, mean steps, score
- **Summary**: Best trial info, best hyperparameters, trial statistics

View results at: https://wandb.ai/memo-ozdincer-university-of-toronto/gad-hpo

### Running Locally (for testing)

```bash
# HIP
python -m src.experiments.2025.hip_multi_mode_eckartmw_hpo \
    --h5-path /path/to/transition1x.h5 \
    --checkpoint-path /path/to/hip_v2.ckpt \
    --out-dir ./hpo_output \
    --max-samples 50 \
    --n-trials 20 \
    --n-steps-per-sample 500 \
    --n-samples 10

# SCINE
python -m src.experiments.2025.scine_multi_mode_eckartmw_hpo \
    --scine-functional DFTB0 \
    --h5-path /path/to/transition1x.h5 \
    --out-dir ./hpo_output \
    --max-samples 50 \
    --n-trials 20 \
    --n-steps-per-sample 500 \
    --n-samples 10
```

## Output

### Files Generated

1. **`hpo_results.json`** - Complete results including:
   - Best hyperparameters
   - Best convergence rate
   - All trial results with scores

2. **`{hip|scine}_hpo_study.db`** - Optuna SQLite database:
   - Full study history
   - Can be loaded to resume optimization
   - Can be analyzed with Optuna visualization tools

### Example Results JSON

```json
{
  "best_trial": 42,
  "best_score": 0.8234,
  "best_params": {
    "dt_min": 2.3e-6,
    "dt_max": 0.067,
    "plateau_patience": 7,
    "plateau_boost": 2.1,
    "plateau_shrink": 0.45,
    "escape_disp_threshold": 2.1e-4,
    "escape_window": 25,
    "escape_neg_vib_std": 0.35,
    "escape_delta": 0.15,
    "adaptive_delta_scale": 1.2,
    "trust_radius_max": 0.28
  },
  "best_convergence_rate": 0.833,
  "best_mean_steps": 423.5
}
```

## Analyzing Results

### Using Optuna Visualization

```python
import optuna

# Load study
study = optuna.load_study(
    study_name="hip-multi-mode-eckartmw-hpo-12345",
    storage="sqlite:///path/to/hip_hpo_study.db"
)

# Importance plot (which parameters matter most)
optuna.visualization.plot_param_importances(study).show()

# Optimization history
optuna.visualization.plot_optimization_history(study).show()

# Parallel coordinate plot
optuna.visualization.plot_parallel_coordinate(study).show()

# Contour plots (2D parameter relationships)
optuna.visualization.plot_contour(study, params=["dt_max", "escape_delta"]).show()
```

### Applying Best Parameters

After finding optimal parameters, update your experiment SLURM script:

```bash
# In hip_multi_mode_eckartmw.slurm or scine_multi_mode_eckartmw.slurm
export ESCAPE_DISP_THRESHOLD=2.1e-4  # From HPO results
export ESCAPE_WINDOW=25
export ESCAPE_DELTA=0.15
# ... etc
```

Or pass directly to the experiment script:

```bash
python -m src.experiments.hip_multi_mode_eckartmw \
    --dt-min 2.3e-6 \
    --dt-max 0.067 \
    --plateau-patience 7 \
    --plateau-boost 2.1 \
    --plateau-shrink 0.45 \
    --escape-disp-threshold 2.1e-4 \
    --escape-window 25 \
    --escape-neg-vib-std 0.35 \
    --escape-delta 0.15 \
    --adaptive-delta \
    --max-atom-disp 0.28 \
    # ... other args
```

## Performance Considerations

### Computational Cost

- **Per Trial**: ~15 samples × 800 steps × (inference + Hessian + eigendecomp)
- **HIP**: ~30-60 seconds per trial (GPU)
- **SCINE**: ~5-10 minutes per trial (CPU, slower due to quantum chemistry)
- **100 Trials**:
  - HIP: ~1-2 hours total
  - SCINE: ~10-15 hours total

### Resource Requirements

**HIP**:
- GPU: 1× L40S (or equivalent)
- CPUs: 8 cores
- Memory: 32GB
- Time: 24 hours (safe upper bound)

**SCINE**:
- GPUs: None (CPU-only)
- CPUs: 8 cores
- Memory: 32GB
- Time: 24 hours (safe upper bound)

## Tips for Effective HPO

1. **Start Small**: Test with 10-20 trials locally before full cluster run
2. **Monitor Progress**: Check logs to ensure optimization is improving over time
3. **Adjust Search Space**: If best params hit boundaries, expand ranges
4. **Multiple Runs**: Consider running 2-3 studies with different seeds for robustness
5. **Validation**: After finding best params, validate on full dataset (not just difficult samples)

## Troubleshooting

### All Trials Fail
- Check data paths in SLURM script
- Verify checkpoint/model paths
- Test base experiment first without HPO

### No Improvement in Convergence
- Try increasing `N_TRIALS` (more exploration)
- Expand hyperparameter search ranges
- Check if difficult samples are truly representative

### Study Database Locked
- Only one process can write to SQLite at a time
- Use different `--study-name` for parallel runs
- Or use PostgreSQL storage for true parallel optimization

## Advanced: Parallel HPO

For faster optimization, run multiple workers in parallel:

```bash
# Terminal 1
sbatch --job-name=hpo_worker1 hip_multi_mode_eckartmw_hpo.slurm

# Terminal 2 (same study name, same storage)
sbatch --job-name=hpo_worker2 hip_multi_mode_eckartmw_hpo.slurm
```

Both workers will coordinate through the shared SQLite database.

## References

- Optuna Documentation: https://optuna.readthedocs.io/
- TPE Sampler: Bergstra et al., "Algorithms for Hyper-Parameter Optimization" (2011)
- Multi-mode GAD: See main experiment documentation in `multi_mode_eckartmw.py`
