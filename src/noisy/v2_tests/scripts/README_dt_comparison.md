# Comparing dt_eff Between Path-Based and State-Based Methods

## Overview

This guide explains how to compare the adaptive timestep (`dt_eff`) evolution between two different adaptive timestep strategies:

1. **Path-based adaptive control** ([gad_plain_run.slurm](slurm_templates/gad_plain_run.slurm))
   - Adapts `dt_eff` based on displacement history
   - Increases by 1.05× when steps are successful
   - Decreases by 0.8× or 0.5× when constraints are violated

2. **State-based adaptive strategies** ([gad_plain_run_nopath.slurm](slurm_templates/gad_plain_run_nopath.slurm))
   - Adapts `dt_eff` based only on current state (no path history)
   - Three strategies: `none` (fixed), `gradient` (1/grad_norm), `eigenvalue` (1/|λ₀|)
   - Default: `eigenvalue` adaptation

## Running Comparisons

### Step 1: Run Both SLURM Jobs

First, run both types of simulations:

```bash
# Path-based adaptive control
cd /scratch/memoozd/ts-tools-scratch
sbatch /project/rrg-aspuru/memoozd/ts-tools/src/noisy/v2_tests/scripts/slurm_templates/gad_plain_run.slurm

# State-based adaptive (eigenvalue strategy)
sbatch /project/rrg-aspuru/memoozd/ts-tools/src/noisy/v2_tests/scripts/slurm_templates/gad_plain_run_nopath.slurm
```

Both jobs will create output directories with trajectory diagnostics:
- Path-based: `/scratch/memoozd/ts-tools-scratch/runs/gad_plain_<JOB_ID>/diagnostics/`
- State-based: `/scratch/memoozd/ts-tools-scratch/runs/gad_plain_<JOB_ID>/diagnostics/`

### Step 2: Compare dt_eff Evolution

Once both jobs complete, run the comparison script:

```bash
cd /project/rrg-aspuru/memoozd/ts-tools

# Activate environment
source .venv/bin/activate

# Run comparison
python src/noisy/v2_tests/scripts/compare_dt_eff.py \
    --path-dir /scratch/memoozd/ts-tools-scratch/runs/gad_plain_<PATH_JOB_ID>/diagnostics \
    --nopath-dir /scratch/memoozd/ts-tools-scratch/runs/gad_plain_<NOPATH_JOB_ID>/diagnostics \
    --output-dir /scratch/memoozd/ts-tools-scratch/analysis/dt_eff_comparison
```

Replace `<PATH_JOB_ID>` and `<NOPATH_JOB_ID>` with the actual SLURM job IDs.

## Output Files

The comparison script generates:

### 1. Per-Trajectory Plots
File: `<sample_id>_dt_eff_comparison.png`

Each plot shows:
- **Top panel**: dt_eff evolution over time for both methods (log scale)
- **Bottom panel**: Ratio of path-based to state-based dt_eff (log scale)

### 2. Aggregate Comparison Plot
File: `aggregate_dt_eff_comparison.png`

Shows:
- **Left panel**: Bar chart comparing mean dt_eff across all trajectories
- **Right panel**: Scatter plot showing correlation between methods

### 3. Summary Statistics
File: `dt_eff_comparison_summary.json`

Contains for each trajectory:
```json
{
  "sample_id": "sample_000",
  "n_steps_path": 10000,
  "n_steps_nopath": 10000,
  "n_steps_compared": 10000,
  "path_method": {
    "mean_dt": 0.0234,
    "std_dt": 0.0089,
    "min_dt": 0.000001,
    "max_dt": 0.08,
    "final_dt": 0.0456
  },
  "state_method": {
    "mean_dt": 0.0198,
    "std_dt": 0.0123,
    "min_dt": 0.000001,
    "max_dt": 0.08,
    "final_dt": 0.0234
  }
}
```

## Key Differences to Look For

### Path-Based Method (adaptive control)
- **Advantages**: Smooth, stable adaptation based on recent trajectory behavior
- **Disadvantages**: Can be slow to respond to sudden changes in landscape
- **Expected behavior**: Gradual increase/decrease, less variance

### State-Based Method (eigenvalue strategy)
- **Advantages**: Responds immediately to changes in local curvature
- **Disadvantages**: Can be more volatile, especially near singularities
- **Expected behavior**: More dynamic adaptation, higher variance near TS

### Questions to Answer
1. Which method converges faster?
2. Which method is more stable (lower variance in dt_eff)?
3. How do they behave near transition states (high curvature regions)?
4. Does eigenvalue-based adaptation avoid singularities better?

## Advanced Usage

### Compare Different State-Based Strategies

You can run the state-based method with different strategies:

```bash
# Gradient-based strategy
DT_ADAPTATION=gradient sbatch gad_plain_run_nopath.slurm

# Fixed timestep (no adaptation)
DT_ADAPTATION=none sbatch gad_plain_run_nopath.slurm
```

### Adjust Scale Factor

The state-based method uses a scale factor to tune adaptation sensitivity:

```bash
# More aggressive adaptation (smaller dt_eff)
DT_SCALE_FACTOR=0.5 sbatch gad_plain_run_nopath.slurm

# Less aggressive adaptation (larger dt_eff)
DT_SCALE_FACTOR=2.0 sbatch gad_plain_run_nopath.slurm
```

## Example Analysis Workflow

```bash
# 1. Set up paths
SCRATCH="/scratch/memoozd/ts-tools-scratch"
PROJECT="/project/rrg-aspuru/memoozd/ts-tools"

# 2. Run both methods (note the job IDs)
cd $SCRATCH
PATH_JOB=$(sbatch --parsable $PROJECT/src/noisy/v2_tests/scripts/slurm_templates/gad_plain_run.slurm)
NOPATH_JOB=$(sbatch --parsable $PROJECT/src/noisy/v2_tests/scripts/slurm_templates/gad_plain_run_nopath.slurm)

echo "Path-based job: $PATH_JOB"
echo "State-based job: $NOPATH_JOB"

# 3. Wait for jobs to complete (check with squeue)
# ...

# 4. Run comparison
cd $PROJECT
source .venv/bin/activate

python src/noisy/v2_tests/scripts/compare_dt_eff.py \
    --path-dir $SCRATCH/runs/gad_plain_${PATH_JOB}/diagnostics \
    --nopath-dir $SCRATCH/runs/gad_plain_${NOPATH_JOB}/diagnostics \
    --output-dir $SCRATCH/analysis/dt_comparison_${PATH_JOB}_vs_${NOPATH_JOB}

# 5. View results
ls -lh $SCRATCH/analysis/dt_comparison_${PATH_JOB}_vs_${NOPATH_JOB}/
```

## Troubleshooting

### No common sample IDs found
- Ensure both runs used the same `--max-samples`, `--start-from`, and `--noise-seed` parameters
- Check that both diagnostics directories contain `*_trajectory.json` files

### Missing dt_eff data
- Verify that both runs completed successfully (check SLURM output logs)
- Ensure the TrajectoryLogger was enabled (it is by default)

### Import errors
- Make sure you've activated the virtual environment: `source .venv/bin/activate`
- Verify matplotlib and numpy are installed: `pip list | grep -E "matplotlib|numpy"`
