# v10c Implementation Prompt — Future Work Items

## Overview

Implement three features in the Newton-Raphson minimization codebase, plus one new diagnostic script. These were identified from a literature review (Schlegel 2011, Packwood 2018, Transition1x paper) as potentially impactful additions that were deferred from v10/v10b.

All changes must be backward-compatible (new parameters default to off).

---

## Context Documents (read these first, in order)

1. **Memory files** (concise summaries of everything):
   - `/Users/memoozdincer/.claude/projects/-Users-memoozdincer-Documents-Research-Guzik-ts-tools/memory/MEMORY.md` — master index
   - `/Users/memoozdincer/.claude/projects/-Users-memoozdincer-Documents-Research-Guzik-ts-tools/memory/technical.md` — algorithms, equations, key parameters
   - `/Users/memoozdincer/.claude/projects/-Users-memoozdincer-Documents-Research-Guzik-ts-tools/memory/experiments.md` — v1-v9 results
   - `/Users/memoozdincer/.claude/projects/-Users-memoozdincer-Documents-Research-Guzik-ts-tools/memory/diagnostics.md` — failure modes, cascade data

2. **Literature synthesis TeX** (comprehensive, ~700 lines):
   - `src/noisy/v2_tests/LITERATURE_REVIEW_AND_SYNTHESIS.tex` — Sections 7 (dataset), 9 (not incorporated), 10 (open questions)

3. **Core code files**:
   - `src/noisy/v2_tests/baselines/minimization.py` — the optimizer (~2700 lines). Key functions: `run_newton_raphson()` (line ~1648), trust region block (line ~2519), step-building chain (line ~2276)
   - `src/noisy/v2_tests/runners/run_minimization_parallel.py` — CLI and parallelization (~860 lines)
   - `src/noisy/v2_tests/scripts/analyze_minimization_nr_grid.py` — grid analysis with regex tag parsing
   - `src/noisy/v2_tests/scripts/slurm_templates/minimization_nr_grid_v10_run.slurm` — current 48-combo grid

4. **Existing diagnostic script** (pattern to follow for new script):
   - `src/noisy/v2_tests/scripts/diagnostic_from_minima.py` — runs DFTB0 analysis at DFT-labeled minima

---

## Task 1: Polynomial Line Search Near Inflection Points

**Source**: Schlegel 2011, Section on "step size control"

**What**: When the trust region step crosses a Hessian eigenvalue sign change (inflection point), the quadratic model is especially poor. A cubic or quartic polynomial fit using energy and gradient at the current AND previous geometry can find a better step length at zero extra evaluation cost.

**Where**: `minimization.py`, inside the legacy trust region block (line ~2519, the `else:` branch of step control)

**New parameter on `run_newton_raphson()`**:
```python
polynomial_linesearch: bool = False,  # v10c: cubic interpolation near inflection
```

**Logic** (insert after step acceptance, before trust radius update):
```
if polynomial_linesearch and step > 0:
    # We have (E_prev, g_prev) and (E_curr, g_curr) at x_prev and x_curr
    # Along the step direction d = (x_curr - x_prev), fit cubic:
    #   p(t) = a*t^3 + b*t^2 + c*t + d
    # where t=0 is x_prev, t=1 is x_curr
    # p(0) = E_prev, p(1) = E_curr
    # p'(0) = g_prev . d_hat, p'(1) = g_curr . d_hat
    #
    # If the cubic has a minimum in (0, 1), use it to refine the step.
    # This is especially useful when eigenvalues cross zero between steps.
    #
    # Solve for minimum: 3a*t^2 + 2b*t + c = 0
    # Only accept if minimum is in (0.1, 0.9) and predicted energy < E_curr
```

You'll need to store `prev_energy`, `prev_grad`, `prev_coords` from the previous accepted step (already partially available via trajectory). The cubic fit uses 4 data points (2 energies, 2 directional derivatives) to determine 4 coefficients.

**Important**: This should NOT replace the trust region. It's a refinement: after the trust region accepts a step, check if a cubic interpolation along the last step direction suggests a better point. If so, evaluate there. One extra predict_fn call per step (only when triggered).

**Runner**: Add `--polynomial-linesearch` flag (store_true).
**Analysis**: No regex change needed (it's a boolean, not in the tag — or add `_pls` to tag if you prefer).

---

## Task 2: Transition1x Frequency Verification Script

**Source**: Transition1x paper — no frequency checks on minima, only force convergence to 0.01 eV/Å

**What**: New standalone script that loads the Transition1x dataset, evaluates DFTB0 Hessian at each DFT-labeled minimum, and reports:
- n_neg distribution (how many "minima" are actually saddle points under DFTB0)
- Eigenvalue spectrum statistics
- Force norm at DFT minima under DFTB0
- Which samples are "genuine minima" (n_neg=0) vs "DFTB0 saddle points" (n_neg>0)
- Correlation between n_neg at DFT minimum and optimizer failure rate

**Where**: New file `src/noisy/v2_tests/scripts/verify_transition1x_minima.py`

**Pattern to follow**: `diagnostic_from_minima.py` already does something very similar. Read it first — it loads the H5 file, creates a SCINE calculator, evaluates Hessians at DFT minima, and reports eigenvalue statistics. The new script should:

1. Load Transition1x H5 file (same as existing scripts)
2. For each reaction's reactant and product geometries:
   - Evaluate DFTB0 energy, forces, Hessian
   - Compute vibrational eigenvalues via `get_vib_evals_evecs()`
   - Record: n_neg, min_eval, force_norm, eigenvalue spectrum
3. Output a JSON summary + human-readable report
4. Cross-reference with optimizer results (if `--results-dir` provided): for each sample, compare "is this a DFTB0 minimum?" with "did the optimizer converge?"

**CLI args**:
```
--h5-path           Path to transition1x.h5
--out-dir            Output directory
--max-samples        Limit samples (default: all)
--scine-functional   Calculator (default: DFTB0)
--results-dir        Optional: cross-reference with optimizer results
--n-workers          Parallel workers
--threads-per-worker Threads per worker
```

**Key insight to validate**: Our diagnostics.md says only 10% of DFT reactant minima have n_neg=0 under DFTB0. This script makes that analysis reproducible and extends it to products.

---

## Task 3: SCINE SCF Convergence Investigation

**Source**: Our Hessian noise floor is ~8e-3 (from nr_threshold parameter). Tighter SCF convergence might reduce this.

**What**: This is an investigation, not a permanent code change. Create a small test script that:

1. Takes a few molecules (e.g., 10 from Transition1x)
2. Evaluates DFTB0 Hessian at the DFT minimum with default SCINE settings
3. Evaluates again with tighter settings (if SCINE exposes them)
4. Compares eigenvalue spectra — does the noise floor change?

**Where**: New file `src/noisy/v2_tests/scripts/investigate_scine_convergence.py`

**Challenge**: SCINE/Sparrow is "zero-configuration" — it may not expose SCF convergence parameters through the Python API. The script should:
1. First, enumerate what settings ARE available via `scine_utilities` and `scine_sparrow`
2. If convergence settings exist, test them
3. If not, document that finding (this is also a valid result)

**This task has lower priority than Tasks 1 and 2.**

---

## Task 4 (optional): Add Grid Entries for New Features

If Task 1 (polynomial line search) is implemented, add configs to the SLURM grid:

| # | Algorithm | polynomial_ls | Notes |
|---|-----------|--------------|-------|
| 13 | v9 baseline + polynomial LS | on | Does interpolation help GQT? |
| 14 | RFO + Schlegel TR + polynomial LS | on | Full chemistry stack |

That's 2 more configs × 4 noise = 8 more combos → 56 total. Update `--top-k` in the analysis call accordingly.

---

## Verification Checklist

After implementation:
1. `python3 -c "import ast; ast.parse(open('src/noisy/v2_tests/baselines/minimization.py').read())"` — syntax OK
2. `python3 -c "import ast; ast.parse(open('src/noisy/v2_tests/runners/run_minimization_parallel.py').read())"` — syntax OK
3. `bash -n src/noisy/v2_tests/scripts/slurm_templates/minimization_nr_grid_v10_run.slurm` — SLURM syntax OK
4. New scripts should be runnable with `--help` without crashing
5. Existing configs (1-12) must produce identical results (backward compatibility)
