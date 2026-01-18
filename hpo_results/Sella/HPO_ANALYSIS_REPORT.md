# HPO Analysis Report: Sella Optimizer Hyperparameter Optimization

## Executive Summary

Two HPO studies were analyzed: SCINE (500 trials) and HIP (468 trials) using Bayesian optimization with TPESampler. Both studies suffered from **extremely high pruning rates** (97.4% for SCINE, 88% for HIP), which limited the effectiveness of the optimization. Despite this, valuable insights were extracted about optimal parameter ranges.

---

## 1. Study Configuration Summary

| Parameter | SCINE Study | HIP Study |
|-----------|-------------|-----------|
| Total Trials | 500 | 468 |
| Completed | 13 (2.6%) | 55 (11.8%) |
| Pruned | 487 (97.4%) | 412 (88.0%) |
| max_steps | 500 | 300 |
| max_samples | 30 | 30 (25 with prescreen) |
| prune_after_n | 10 | 10 |
| n_startup_trials | 5 | 5 |

### Hyperparameters Optimized (Both Studies)
- `delta0`: Trust radius initial value
- `rho_dec`: Trust radius decrease factor
- `rho_inc`: Trust radius increase factor
- `sigma_dec`: Step size decrease factor
- `sigma_inc`: Step size increase factor
- `fmax`: Force convergence tolerance
- `apply_eckart`: Whether to apply Eckart projection

---

## 2. Key Findings

### 2.1 Pruning Analysis (Critical Issue)

**The median pruner was too aggressive**, causing potentially excellent trials to be discarded:

| Metric | SCINE | HIP |
|--------|-------|-----|
| Max score (COMPLETED) | 0.639 | 0.366 |
| Max score (PRUNED) | 0.710 | 0.307 |
| Avg score (COMPLETED) | 0.524 | 0.233 |
| Avg score (PRUNED) | 0.429 | 0.109 |

**Root Cause**: Early completed trials had high intermediate scores (0.71-1.01 at step 10), setting an unrealistically high bar. Subsequent trials with intermediate scores of 0.70 were pruned despite being potentially competitive.

**Example**: SCINE Trial 385 (best completed):
- Intermediate score at step 10: **1.005**
- Final score at step 30: **0.639**

This 36% drop from intermediate to final demonstrates that 10-sample intermediate scores have high variance and set artificially high pruning thresholds.

### 2.2 Transition State Success Rates

| Calculator | Best TS Rate | Mean TS Rate | Target |
|------------|--------------|--------------|--------|
| SCINE | 63.3% | 51.8% | 100% |
| HIP | 36.0% | 22.8% | 100% |

**SCINE significantly outperforms HIP** in finding transition states with exactly 1 negative eigenvalue.

### 2.3 Step Count Analysis

| Calculator | max_steps | Avg Steps | Max Steps Used | Hitting Limit (>90%) |
|------------|-----------|-----------|----------------|---------------------|
| SCINE | 500 | 232.9 | 342.1 | 0 |
| HIP | 300 | 132.2 | 237.2 | 0 |

**max_steps appears adequate** - no trials are hitting the step limit at 90% threshold. However, increasing max_steps to 5000 as planned is reasonable for robustness.

### 2.4 Negative Eigenvalue Distribution

**SCINE (390 samples from completed trials)**:
- 1 neg eig (TS): 51.8%
- 2 neg eigs: 22.3%
- 3+ neg eigs: 24.6%
- 0 neg eigs: 1.3%

**HIP (1375 samples from completed trials)**:
- 1 neg eig (TS): 22.8%
- 2-3 neg eigs: 27.0%
- 4+ neg eigs: 48.3%
- 0 neg eigs: 0.9%

**HIP has serious convergence issues** - nearly half of samples end with 4+ negative eigenvalues, indicating failure to reach a saddle point.

### 2.5 apply_eckart Analysis

| Calculator | eckart=True avg score | eckart=False avg score | Recommendation |
|------------|----------------------|------------------------|----------------|
| SCINE | 0.5395 | 0.4719 | **Use True** |
| HIP | 0.2306 | 0.2391 | Slight edge for False |

**For SCINE**: `apply_eckart=True` clearly performs better (+14% relative improvement)
**For HIP**: Marginal difference; `apply_eckart=False` has a slight edge

---

## 3. Optimal Parameter Ranges (From Top 10 Trials)

### SCINE Top Performers

| Parameter | Current Range | Top 10 Range | Recommended New Range |
|-----------|--------------|--------------|----------------------|
| delta0 | [0.03, 0.8] log | [0.04, 0.30] | **[0.05, 0.40]** log |
| rho_dec | [3.0, 80.0] | [43.4, 79.9] | **[40.0, 90.0]** |
| rho_inc | [1.01, 1.1] | [1.02, 1.10] | [1.01, 1.10] (keep) |
| sigma_dec | [0.5, 0.95] | [0.63, 0.93] | **[0.60, 0.95]** |
| sigma_inc | [1.1, 1.8] | [1.22, 1.71] | [1.1, 1.8] (keep) |
| fmax | [1e-4, 1e-2] log | [2e-4, 5e-3] | **[1e-4, 1e-2]** (keep) |
| apply_eckart | [T, F] | Mostly True | **Fix to True** |

### HIP Top Performers

| Parameter | Current Range | Top 10 Range | Recommended New Range |
|-----------|--------------|--------------|----------------------|
| delta0 | [0.15, 0.8] log | [0.29, 0.78] | **[0.25, 0.80]** log |
| rho_dec | [15.0, 80.0] | [35.8, 62.1] | **[30.0, 70.0]** |
| rho_inc | [1.01, 1.1] | [1.02, 1.10] | [1.01, 1.10] (keep) |
| sigma_dec | [0.75, 0.95] | [0.78, 0.95] | **[0.75, 0.95]** (keep) |
| sigma_inc | [1.1, 1.8] | [1.20, 1.55] | **[1.15, 1.60]** |
| fmax | [1e-4, 1e-2] log | [1.4e-3, 6.1e-3] | **[5e-4, 1e-2]** log |
| apply_eckart | [T, F] | Mixed | Keep both |

---

## 4. Top 5 Configurations Reference

### SCINE Best Configs

| Rank | Score | TS Rate | delta0 | rho_dec | sigma_dec | fmax | eckart |
|------|-------|---------|--------|---------|-----------|------|--------|
| 1 | 0.639 | 63.3% | 0.302 | 73.2 | 0.753 | 2.6e-4 | True |
| 2 | 0.607 | 60.0% | 0.101 | 78.7 | 0.733 | 4.4e-4 | True |
| 3 | 0.606 | 60.0% | 0.105 | 79.9 | 0.725 | 5.3e-3 | True |
| 4 | 0.605 | 60.0% | 0.081 | 43.4 | 0.631 | 1.9e-4 | True |
| 5 | 0.542 | 53.3% | 0.100 | 79.7 | 0.725 | 1.9e-3 | True |

### HIP Best Configs

| Rank | Score | TS Rate | delta0 | rho_dec | sigma_dec | fmax | eckart |
|------|-------|---------|--------|---------|-----------|------|--------|
| 1 | 0.366 | 36.0% | 0.518 | 53.5 | 0.950 | 1.4e-3 | False |
| 2 | 0.328 | 32.0% | 0.286 | 35.8 | 0.804 | 5.8e-3 | True |
| 3 | 0.328 | 32.0% | 0.496 | 49.9 | 0.783 | 2.8e-3 | True |
| 4 | 0.327 | 32.0% | 0.352 | 49.5 | 0.784 | 2.3e-3 | False |
| 5 | 0.326 | 32.0% | 0.490 | 50.6 | 0.800 | 6.1e-3 | True |

---

## 5. Recommendations for Next HPO Study

### 5.1 Critical: Fix Pruning Strategy

**Option A (Recommended)**: Disable pruning entirely
```python
pruner = None  # or optuna.pruners.NopPruner()
```

**Option B**: Use more lenient pruner
```python
pruner = optuna.pruners.PercentilePruner(
    percentile=25.0,  # Only prune bottom 25%
    n_startup_trials=20,  # Let 20 trials complete first
    n_warmup_steps=15,   # Don't prune until 15 samples
)
```

**Option C**: Increase evaluation before pruning
```python
prune_after_n = 20  # Instead of 10
n_startup_trials = 15  # Instead of 5
```

### 5.2 Increase max_steps

```python
MAX_STEPS = 5000  # Current: 300 (HIP), 500 (SCINE)
```

While current avg_steps don't hit limits, 5000 provides safety margin for harder cases.

### 5.3 Fixed Parameters (Remove from HPO)

**SCINE**: Fix `apply_eckart=True` - clear winner
**Both**: Keep `internal=True`, `use_exact_hessian=True`, `diag_every_n=1`, `gamma=0.0`, `order=1`

### 5.4 Updated Parameter Ranges (EXPANDED for Exploration)

**SCINE New Ranges** (expanded to explore unexplored territory):
```python
delta0 = trial.suggest_float("delta0", 0.01, 1.0, log=True)    # Was [0.03, 0.8]
rho_dec = trial.suggest_float("rho_dec", 20.0, 150.0)          # Was [3.0, 80.0] - top performers hit 80!
rho_inc = trial.suggest_float("rho_inc", 1.005, 1.15)          # Was [1.01, 1.1]
sigma_dec = trial.suggest_float("sigma_dec", 0.4, 0.98)        # Was [0.5, 0.95]
sigma_inc = trial.suggest_float("sigma_inc", 1.05, 2.5)        # Was [1.1, 1.8]
fmax = trial.suggest_float("fmax", 1e-5, 5e-2, log=True)       # Was [1e-4, 1e-2]
# apply_eckart = True (FIXED - clear winner)
```

**HIP New Ranges** (expanded to explore unexplored territory):
```python
delta0 = trial.suggest_float("delta0", 0.1, 1.5, log=True)     # Was [0.15, 0.8] - HIP likes higher delta0
rho_dec = trial.suggest_float("rho_dec", 15.0, 100.0)          # Was [15.0, 80.0]
rho_inc = trial.suggest_float("rho_inc", 1.005, 1.15)          # Was [1.01, 1.1]
sigma_dec = trial.suggest_float("sigma_dec", 0.6, 0.98)        # Was [0.75, 0.95]
sigma_inc = trial.suggest_float("sigma_inc", 1.05, 2.0)        # Was [1.1, 1.8]
fmax = trial.suggest_float("fmax", 1e-4, 2e-2, log=True)       # Was [1e-4, 1e-2]
apply_eckart = trial.suggest_categorical("apply_eckart", [True, False])
```

**Rationale for Expansion**:
- **rho_dec**: Top SCINE performers clustered near 73-80, hitting the upper bound - need to explore higher
- **delta0**: Both extremes showed promise in different trials - expand to explore
- **sigma_inc/sigma_dec**: More aggressive step size adjustments might help convergence
- **fmax**: Tighter tolerance (1e-5) might improve TS quality; looser (5e-2) might help convergence rate

### 5.5 Study Continuation

**Can resume?** Partially - if you only change ranges slightly and don't change fixed params, you can use `load_if_exists=True`. However, if you change the pruning strategy or fix `apply_eckart`, start a new study.

**Recommended**: Start fresh with the new configuration to get clean data.

---

## 6. Additional Levers to Consider

### 6.1 Sella-specific Parameters Not Currently Optimized

From Sella documentation, these could be valuable:
- `eig`: Number of eigenvalues to target (currently fixed at 1)
- `constraints`: Whether to use constraints
- `trajectory_logging_interval`: For debugging

### 6.2 Calculator-specific Considerations

**HIP**: The low TS rate (22.8% avg) suggests the neural network potential may not provide accurate enough Hessians for saddle point optimization. Consider:
- Using higher-fidelity calculator for Hessian computation
- Investigating specific failure modes (samples ending with many neg eigs)

**SCINE**: Better performance suggests analytical Hessians from DFTB are more reliable for this task.

---

## 7. Summary Action Items

1. **Fix pruning** - Disable or significantly relax the pruner
2. **Increase max_steps to 5000** - Provides safety margin
3. **Fix apply_eckart=True for SCINE** - Clear performance benefit
4. **Narrow parameter ranges** - Focus on regions where top performers cluster
5. **Increase n_trials** - With pruning disabled, you'll need more trials but get better data
6. **Start fresh study** - Don't resume the current study with these changes

---

*Report generated from analysis of hpo_results/scine_sella_1.db and hpo_results/hip_sella_1.db*
