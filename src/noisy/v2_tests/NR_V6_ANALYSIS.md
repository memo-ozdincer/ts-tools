# NR Minimization v6: 300-Sample Analysis & Trust Radius Diagnosis

**Run**: `min_nr_v6_1090871` | **Date**: March 2026
**Setup**: 300 samples requested, 287 loaded, 0.5 A noise, seed = SLURM_JOB_ID
**Configs**: SN e=1e-3, SN e=5e-4 | **Steps**: 10,000 | **Calculator**: DFTB0

---

## 1. Top-Line Results

| Config | Converged | Valid | Rate | Errors | Mean Steps | Wall Time |
|--------|-----------|-------|------|--------|------------|-----------|
| SN e=1e-3 | 219 | 278 | 78.8% | 9 | 1232 | 55.5s |
| SN e=5e-4 | 208 | 278 | 74.8% | 9 | 1005 | 53.7s |

- 13 samples skipped at load (bad geometry), 9 crash during optimization = 22/300 (7.3%) never run
- e=1e-3 converges 11 more samples than e=5e-4, but takes slightly more steps
- At 0.5 A noise (the easiest setting), convergence should be 90%+. 76% is a problem.

## 2. Failure Mode Breakdown (556 trajectories total)

| Classification | Count | % of All | % of Failures |
|---------------|-------|----------|---------------|
| converged | 427 | 76.8% | -- |
| almost_converged | 64 | 11.5% | 49.6% |
| oscillating | 57 | 10.3% | 44.2% |
| ghost_modes | 7 | 1.3% | 5.4% |
| drifting | 1 | 0.2% | 0.8% |

Per-config failures:
- e=1e-3: 29 almost_converged, 27 oscillating, 2 ghost, 1 drifting
- e=5e-4: 35 almost_converged, 30 oscillating, 5 ghost

## 3. Root Cause: Trust Radius Collapse

**Every single failed trajectory (129/129) ends with trust_radius = 0.01 A (the floor).**

The collapse mechanism:
1. Shifted Newton computes weight = 1/(lambda + sigma), max = 1/epsilon = 1000x
2. Near-zero eigenvalue modes get amplified by 1000x, producing oversized steps
3. Oversized steps violate the quadratic model: rho < 0.25 -> trust shrinks by 0.25x
4. After ~4 bad steps: 1.3 -> 0.325 -> 0.081 -> 0.020 -> 0.01 (floor)
5. At the floor, the entire step is uniformly scaled down
6. Negative-mode displacement = trust x grad_overlap = 0.01 x 0.05 = 0.0005 A/step
7. To flip an eigenvalue requires ~0.01-0.1 A displacement -> 20-200 steps of monotonic progress
8. But eigenvalues oscillate, so monotonic progress never happens -> stuck forever

Recovery is nearly impossible: trust grows by 1.5x per good step, needs ~13 consecutive
good steps to get back to 0.3. But any meaningful step triggers collapse again, creating
the period-8 oscillation cycle documented in the v4 surface forensics.

## 4. Three Failure Populations

### Population A: "Blind Modes" (almost_converged, 49.6% of failures)

Characteristics:
- Tiny eigenvalues: min_eval between -0.0002 and -0.001
- High forces: 0.034 to 0.092 eV/A (200-900x above threshold)
- Very low gradient overlap with negative modes: < 0.1 (usually < 0.05)
- Trust at floor for entire trajectory

Representative samples:

| Sample | n_neg | min_eval | force_norm | max grad_overlap | stagnation |
|--------|-------|----------|------------|------------------|------------|
| 002 | 3 | -0.00022 | 0.069 | 0.184 | 0.0% |
| 016 | 1 | -0.00135 | 0.041 | 0.015 | 80.9% |
| 024 | 1 | -0.00048 | 0.068 | 0.090 | 0.7% |
| 028 | 1 | -0.00044 | 0.002 | 0.033 | 22.2% |
| 038 | 2 | -0.00022 | 0.064 | 0.019 | 0.0% |
| 098 | 1 | -0.00074 | 0.092 | 0.051 | 99.6% |

Key insight: "almost_converged" is misleading. Most have force_norm >> 1e-4, meaning they
are NOT at a minimum. The eigenvalue is small because it oscillates near zero, not because
the geometry is close to a minimum. Only sample_028 (force=0.002) is genuinely close.

### Population B: "Oscillation-Collapse" (oscillating, 44.2% of failures)

Characteristics:
- Larger eigenvalues: min_eval = -0.001 to -0.01
- High forces: 0.1 to 0.37 eV/A (nowhere near a minimum)
- Grad overlaps 0.02-0.10 (some visibility but trust-crushed)
- Significant stagnation in some cases

| Sample | n_neg | min_eval | force_norm | grad_overlap | stagnation |
|--------|-------|----------|------------|--------------|------------|
| 034 | 4 | -0.0100 | 0.180 | 0.018, 0.002, 0.006, 0.002 | 0.4% |
| 041 | 1 | -0.0070 | 0.260 | 0.098 | 47.1% |
| 050 | 4 | -0.0012 | 0.106 | 0.051, 0.120, 0.062, 0.003 | 0.0% |
| 109 | 3 | -0.0041 | 0.374 | 0.075, 0.047, 0.026 | 9.7% |

sample_034: 4 negative modes, ALL grad_overlaps < 0.02 — completely blind to every
direction it needs to fix. This is the hardest failure type.

sample_109: force_norm = 0.374, the highest of any failure. Not even in the right basin.

### Population C: "Stagnation Zombies" (subset of both A and B)

| Sample | stagnation % | Status |
|--------|-------------|--------|
| 098 | 99.6% | Frozen for ~9960 of 10000 steps |
| 016 | 80.9% | Grad overlap 0.015, blind, dead |
| 103 | 80.0% | Same pattern |
| 041 | 47.1% | Partially unstuck but can't converge |

sample_098 is the extreme: 99.6% of steps accomplish nothing. The optimizer is taking
0.01 A steps for 9960 steps, making 0.0005 A of progress along the negative mode.

## 5. Cascade Analysis: What's Fixable vs Irreducible

For SN e=1e-3, reading cascade as "convergence rate with eigenvalue tolerance T":

| Tolerance T | Conv Rate | Delta | Population Captured |
|-------------|-----------|-------|---------------------|
| strict (0) | 76.3% | -- | -- |
| -1e-4 | 78.4% | +2.1% | Ghost modes only |
| -5e-4 | 81.2% | +4.9% | + very small negatives |
| -1e-3 | 84.3% | +8.0% | + most "almost_converged" |
| -2e-3 | 86.4% | +10.1% | + moderate borderline |
| -5e-3 | 89.9% | +13.6% | + all but genuine stuck |
| -1e-2 | 94.1% | +17.8% | + some oscillating |

The irreducible failure rate is ~6% — samples with final eigenvalues worse than -0.01
that no eigenvalue tolerance can fix. These are the genuine oscillation-collapse cases.

But 17.8% of samples (76.3% -> 94.1%) are "fixable" — they reach near-minimum
geometries but have residual negative eigenvalues between 0 and -0.01. The question
is whether fixing the trust radius collapse would push these into full convergence
rather than requiring a tolerance hack.

## 6. The 9 Errors

9/296 = 3% of attempted samples crash during optimization (min_interatomic_dist
violation). At 0.5 A noise this likely means the midpoint_rt starting geometry
already has atoms close together, and the noise pushes them into overlap.

## 7. Diagnosis & Next Step

The trust region is the bottleneck, not the step computation. The shifted Newton step
direction is sound — it correctly gives negative modes more weight. But the trust region
mechanism punishes the oversized steps that this asymmetric weighting produces.

**Fix**: Replace the trust region with Armijo backtracking line search for shifted Newton
mode. The line search:
- Starts alpha=1.0 fresh every step (no persistent state to collapse)
- Guarantees energy decrease (Armijo condition)
- Already implemented in the SPDN path (lines 1944-1998 of minimization.py)
- SPDN failed because of its spectral step builder + GDIIS, not the line search

Additionally, capping the maximum NR weight (e.g., at 200x instead of 1000x) prevents
the oversized steps that trigger problems in the first place.

This is implemented in v7.
