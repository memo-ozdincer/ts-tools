# Newton-Raphson Minimization v3 — Changes & Rationale

## Problem Statement

v2 grid search (best config: LM μ=0.002) achieved only **30% strict convergence** (9/30 samples).
The cascade evaluation table revealed two distinct failure populations:

| Population | Size | Description |
|---|---|---|
| **A — "Almost there"** | ~50% of samples | Final min eigenvalue in (-0.002, 0). Optimizer gets close but can't push the last negative eigenvalue(s) to zero. |
| **B — "Genuinely stuck"** | ~33% (10 samples) | 0/12 convergence across ALL configs, even at relaxed eval thresholds (eval≥-0.01). |

### Root Cause

The NR step uses `|λ|` (absolute value Newton):
```
step_i = -(g·v_i) / |λ_i|
```
This treats negative eigenvalue modes identically to positive ones. It has **no mechanism to specifically target negative modes**. Near convergence:
- The negative eigenvalue is small → curvature is nearly flat
- The gradient projection onto the negative mode is small → NR step barely moves along it
- Other well-converged modes dominate the overall displacement

---

## New Algorithms

### 1. Shifted Newton (`--shift-epsilon`)

**File:** `minimization.py` → `_nr_step_shifted_newton()`

Standard Levenberg shift:
```
step_i = (g·v_i) / (λ_i + σ)
where σ = max(0, -λ_min) + shift_epsilon
```

All effective eigenvalues `(λ_i + σ)` become positive. The key insight: **negative modes get LARGER weights than equally-sized positive modes**:

| Mode | Weight with `|λ|` | Weight with shifted (ε=1e-3) |
|---|---|---|
| λ = -0.01 | 1/0.01 = 100 | 1/0.001 = **1000** |
| λ = +0.01 | 1/0.01 = 100 | 1/0.021 = 48 |

The step is ~20× more aggressive along the negative mode. This directly addresses Population A.

**Priority:** Takes precedence over LM and hard-filter when `shift_epsilon > 0`.

### 2. Stagnation Escape (`--stagnation-window`, `--escape-alpha`)

**File:** `minimization.py` → `_stagnation_escape_perturbation()`

When `n_neg` is unchanged for `stagnation_window` consecutive steps AND negative eigenvalues are small (`|λ_min| < 0.02`):

1. **Targeted perturbation** along negative eigenvectors:
   ```
   perturbation = α · Σ_i sign(g·v_i) · v_i    (for negative modes i)
   ```
   Normalized so max atom displacement = `escape_alpha`. The sign is chosen to move downhill along each mode.

2. **Optional line search** (`--neg-mode-line-search`): scan along the most negative eigenvector at multiple trial displacements (0.02–0.3 Å), pick the trial point where the most-negative eigenvalue is closest to zero.

After escape, the trust radius resets to `max_atom_disp`.

**Why this is justified vs global v₂ kicks:** The perturbation is exclusively along the modes that define the problem. It's equivalent to a mode-following escape step — not a geometry reset.

### 3. Adaptive LM μ Annealing (`--lm-mu-anneal-factor`)

**File:** `minimization.py`, inside `run_newton_raphson()` loop

When close to convergence (`n_neg ≤ lm_mu_anneal_n_neg_leq` AND `|λ_min| < lm_mu_anneal_eval_leq`), multiply μ by `lm_mu_anneal_factor`:

```
effective_μ = μ × anneal_factor    (e.g., 0.002 × 0.1 = 0.0002)
```

Lower μ near convergence → the LM weight `|λ|/(λ²+μ²)` approaches `1/|λ|` → more aggressive steps along flat/negative modes. Early in optimization, the full μ prevents explosion from truly flat modes.

### 4. Trust-Region Floor (`--trust-radius-floor`)

**File:** `minimization.py`, trust-region update logic

All trust-radius reductions now clamp to `max(new_radius, trust_radius_floor)` (default 0.01 Å). This prevents the optimizer from shrinking to useless displacement magnitudes when the quadratic model is a poor fit.

Previously, after repeated rejections the trust radius could shrink to `0.001 * 0.25^N` — effectively zero.

---

## Diagnostic Additions

### Per-Step Negative-Mode Diagnostics

**File:** `minimization.py` → `_neg_mode_diagnostics()`

Logged in every trajectory step (when `n_neg > 0`) under key `"neg_mode_diag"`:

| Field | What it tells you |
|---|---|
| `neg_mode_grad_overlaps` | `\|g·v_i\|/\|g\|` for each negative mode. If near zero → NR step literally cannot address that mode (gradient is orthogonal). |
| `neg_mode_eigenvalues` | The negative eigenvalues themselves (sorted, most negative first). |
| `step_along_neg_frac` | Fraction of `\|\|dx\|\|²` in the negative-eigenvalue subspace. |
| `step_along_pos_frac` | Fraction in the positive-eigenvalue subspace. |
| `max_neg_grad_overlap` | Largest overlap (easiest mode to fix). |
| `min_neg_grad_overlap` | Smallest overlap (bottleneck mode). |

### Per-Step Stagnation Tracking

| Field | Description |
|---|---|
| `stagnation_counter` | Consecutive steps with unchanged `n_neg` |
| `energy_plateau` | True if energy range over last 10 steps < 1e-8 |
| `effective_lm_mu` | Current μ value (shows when annealing kicks in) |
| `escape_triggered` | True on steps where stagnation escape fired |
| `line_search_info` | Dict with line search trial results (when enabled) |

---

## New Scripts

### Failure Autopsy (`analyze_nr_failure_autopsy.py`)

Reads every trajectory JSON and classifies each failed trajectory:

| Classification | Criteria |
|---|---|
| `almost_converged` | `n_neg ≤ 3` and `\|λ_min\| < 0.002` |
| `oscillating` | Eigenvalues bouncing up/down in last 100 steps |
| `energy_plateau` | Energy range < 1e-6 over last 100 steps |
| `genuinely_stuck` | `n_neg` unchanged for >50% of all steps |
| `slow_convergence` | Eigenvalues still improving but ran out of steps |
| `drifting` | None of the above |

**Output:**
- `failure_autopsy.csv` — one row per trajectory
- `failure_autopsy.json` — structured summary with per-combo breakdown and hardest-sample rankings

Reports for each failed sample: bottom eigenvalues, gradient-mode overlaps, stagnation fraction, trust-radius history, escape event counts.

### SLURM Grid Script (`minimization_nr_grid_v3_run.slurm`)

19 combinations across 5 phases:

| Phase | What | Combos |
|---|---|---|
| **A** | Shifted Newton sweep: ε ∈ {1e-4, 5e-4, 1e-3, 2e-3} | 4 |
| **B** | LM + adaptive μ annealing: μ ∈ {2e-3, 5e-3} × factor ∈ {0.1, 0.5} | 4 |
| **C** | LM + stagnation escape: window ∈ {50, 100, 200} × α ∈ {0.05, 0.1} | 6 |
| **D** | Shifted Newton + stagnation escape (± line search) | 4 |
| **E** | Diagnostic run (LM μ=0.002, 50k steps) | 1 |

---

## Updated Analysis Script (`analyze_minimization_nr_grid.py`)

- **New regex patterns** for v3 folder tags (`se*` for shifted Newton, `sw*_ea*` for stagnation, `af*` for anneal factor, `ls` for line search, `diag*` for diagnostic runs)
- **Extended `ComboRecord`** with v3 fields: `shift_epsilon`, `stagnation_window`, `escape_alpha`, `lm_mu_anneal_factor`, `neg_mode_line_search`, `total_escapes`, `total_line_searches`
- **Cascade table** includes shifted Newton rows (labeled `SN ε=...`)
- **Main effects** summary includes `shift_epsilon`, `stagnation_window`, `lm_mu_anneal_factor`
- **CSV output** includes all v3 columns

---

## Files Changed

| File | Change |
|---|---|
| `src/noisy/v2_tests/baselines/minimization.py` | New step mode, escape, annealing, diagnostics, trust-region floor |
| `src/noisy/v2_tests/runners/run_minimization_parallel.py` | 8 new CLI args, v3 fields in results/trajectory JSON |
| `src/noisy/v2_tests/scripts/analyze_minimization_nr_grid.py` | v3 regex, extended ComboRecord, cascade table, main effects |
| `src/noisy/v2_tests/scripts/analyze_nr_trajectory_stats.py` | Broadened combo regex to accept all tag formats |
| `src/noisy/v2_tests/scripts/analyze_nr_failure_autopsy.py` | **New** — failure classification and autopsy |
| `src/noisy/v2_tests/scripts/slurm_templates/minimization_nr_grid_v3_run.slurm` | **New** — v3 grid search (19 combos, 5 phases) |

---

## How to Interpret v3 Results

### Reading the Cascade Table

```
optimizer                        eval≥-0.0  eval≥-0.001  eval≥-0.005  strict
SN  ε=0.001                          0.600      0.800        0.933    0.600
LM  μ=0.002                          0.300      0.733        0.900    0.300
```

- **Gap between `eval≥0.0` and `eval≥0.002`** → Population A size (false-rejection from tiny eigenvalues)
- **Gap shrinking in v3** → shifted Newton / escape is eliminating the last-mile problem
- **`eval≥0.005` near 1.0 but `eval≥0.0` still low** → still a residual last-mile issue
- **`eval≥0.005` itself low** → genuine optimizer failure (Population B)

### Reading the Autopsy Report

Focus on the **hardest samples** section. For each sample:
- If `min_grad_overlap ≈ 0`: the gradient is orthogonal to the negative mode — NR fundamentally cannot fix this. Need escape perturbation.
- If `stagnation_frac > 0.5`: optimizer spent most of its time stuck — stagnation escape should help.
- If `classification = "oscillating"`: the optimizer is bouncing — shifted Newton (smoother step) may help.
- If `classification = "energy_plateau"`: optimizer has stopped making progress — need escape or more steps.

### What "Good" Looks Like

Target: all 30 samples converging with strict n_neg==0. The ideal v3 config should show:
- Cascade table: `eval≥0.0` column ≈ 1.0 (no false rejections)
- Autopsy: no `genuinely_stuck` or `oscillating` failures
- Escape count > 0 but not excessive (escapes should be rare, targeted interventions)
