# Comprehensive Analysis Guide for mar8 Results

## Your Task

Analyze the experimental results in this directory (`mar8/`) with **no detail spared**. This contains results from a Newton-Raphson minimization project on the DFTB0 potential energy surface. There are three runs: v10, v10c, and v11. The analysis must be thorough, precise, and publication-quality.

**Output**: Write all analysis directly into the TeX document at `src/noisy/v2_tests/LITERATURE_REVIEW_AND_SYNTHESIS.tex`. Do NOT write an intermediate markdown file — go straight to TeX. This avoids token-doubling and produces higher quality output (TeX forces precise table formatting).

---

## 1. Project Background

**Goal**: Find energy minima on a DFTB0 PES starting from displaced geometries near transition states (Transition1x dataset, 287 organic reactions). Full analytical Hessian computed at every step. Newton-Raphson in Cartesian coordinates with Eckart projection.

**Key metric**: Convergence rate = fraction of 287 samples reaching strict convergence (n_neg=0, force < 1e-4 eV/A) within N steps.

**Noise levels**: Displacement from transition state midpoint: 0.5, 1.0, 1.5, 2.0 Angstrom.

**Prior best results** (tsMar4 runs, different seed):
| Noise | v9 GQT | tsMar4 GQT+PLS | tsMar4 RFO+Schl+PLS |
|-------|--------|----------------|---------------------|
| 0.5A  | 80%    | 83%            | 89%                 |
| 1.0A  | 60%    | 67%            | 78%                 |
| 1.5A  | 46%    | 56%            | 63%                 |
| 2.0A  | 29%    | 45%            | 42%                 |

---

## 2. Tag Decoding

Every trial folder name is a config tag. Here's how to read them:

### GQT (Goldfeld-Quandt-Trotter = Shifted Newton) tags:
`n<noise>_mad<max_atom_disp>_se<shift_epsilon>_sc<step_control>[_str][_trf<v>][_20k]_re<relax_thresh>_ar<accept_relaxed>[_pls]_pg<project_grad>_ph<purify_hessian>`

- `mad1.3` = max atom displacement 1.3A
- `se1e-3` = shift epsilon 1e-3
- `sctr` = step control = trust region
- `_str` = Schlegel trust radius fixes enabled
- `_trf0.005` = trust radius floor lowered to 0.005A (default 0.01A)
- `_20k` = 20,000 steps (default 10,000)
- `_pls` = polynomial line search enabled
- `re0.01` = relaxed eval threshold -0.01
- `ar1` = accept relaxed convergence
- `pgtrue` = project gradient (Eckart)
- `phfalse` = don't purify Hessian

### RFO (Rational Function Optimization) tags:
`n<noise>_rfo[_gd<buffer>][_glf<force>][_str][_trf<v>][_20k]_re<threshold>_ar<0|1>[_pls]_pg<bool>_ph<bool>`

- `_gd8` = GDIIS buffer size 8
- `_glf0.01` = GDIIS late force threshold 0.01
- `_str` = Schlegel trust radius
- Other fields same as GQT

### ARC (Adaptive Regularization with Cubics) tags:
`n<noise>_arc_si<sigma_init>_g1<gamma1>[_gd<buffer>][_glf<force>]_re<threshold>_ar<0|1>_pg<bool>_ph<bool>`

- `si1.0` = initial sigma regularization parameter
- `g12.0` = gamma1 = 2.0 (adaptation rate)
- `g13.0` = gamma1 = 3.0

---

## 3. What Each Run Tests

### v10 (48 configs = 12 algorithms x 4 noise)
**Same configs as previous tsMar4/v10.2 run, different SLURM seed.** Tests:
1. GQT baseline (shifted Newton + trust region)
2. GQT + Schlegel TR
3. ARC with sigma_init in {0.1, 1.0, 10.0}, gamma1 in {2.0, 3.0}
4. ARC + GDIIS (buffer 8, 12)
5. ARC + GDIIS late-stage
6. RFO (plain, +Schlegel, +Schlegel+GDIIS-late)

**Key question**: Does this reproduce tsMar4/v10.2 results with a different seed?

### v10c (56 configs = 14 algorithms x 4 noise)
**Same configs as previous tsMar4/v10c run, different seed.** Tests everything in v10 PLUS:
- GQT + PLS (polynomial line search)
- RFO + Schlegel + PLS

**Key question**: Does this reproduce the PLS breakthrough (tsMar4 showed +7 to +34 pts)?

### v11 (32 configs = 8 algorithms x 4 noise)
**NEW experiments.** All configs use PLS. Tests:
1. RFO + Schlegel + PLS (champion control, 10k steps)
2. GQT + PLS (no Schlegel, 10k steps)
3. **GQT + Schlegel + PLS** (NEVER TESTED BEFORE)
4. RFO + PLS (no Schlegel, 10k steps)
5. RFO + Schlegel + PLS, trust floor = 0.005A (default 0.01A)
6. GQT + Schlegel + PLS, trust floor = 0.005A
7. RFO + Schlegel + PLS, **20k steps** (do almost_converged samples just need more time?)
8. GQT + Schlegel + PLS, **20k steps**

---

## 4. Files to Read

For each run (v10, v10c, v11):

### Primary analysis files:
- `<run>/analysis/report.txt` — Ranked config list, main effects, hardest samples
- `<run>/autopsy/autopsy_report.txt` — Failure classification per combo + hardest sample forensics
- `<run>/analysis/nr_grid_ranked.csv` — Machine-readable ranked configs
- `<run>/analysis/nr_grid_sample_hardness.csv` — Per-sample convergence rates across all configs

### v10c extras:
- `v10c/scine_convergence/investigation_report.txt` — SCINE SCF investigation (crashed with NameError)
- `v10c/verify_transition1x/verify_transition1x_minima_report.txt` — DFTB0 frequency check on DFT-labeled minima

### Per-trial diagnostics (sample a few):
- `<run>/n<tag>/minimization_newton_raphson_*_results.json` — Per-sample results
- `<run>/n<tag>/diagnostics/*.json` — Detailed trajectory data (1-2 sampled per config)

**NOTE**: The summary JSONs (`nr_grid_summary.json`) are 16-28 MB each. Use them for per-sample cross-referencing but don't try to read them in full — extract specific fields as needed.

---

## 5. Analysis Checklist — Do ALL of These

### A. Per-Run Summary Tables
For each run, create a clean convergence table organized by algorithm family:

```
| Config | 0.5A | 1.0A | 1.5A | 2.0A | Mean Steps | Notes |
```

Group by: optimizer (GQT vs RFO vs ARC), then by features (±PLS, ±Schlegel, ±GDIIS, etc.)

### B. Reproducibility Check (v10 mar8 vs tsMar4/v10.2, v10c mar8 vs tsMar4/v10c)
Compare matching configs across seeds. Quantify:
- Mean absolute difference in convergence rates
- Are the same configs ranked similarly?
- Do the same samples fail?

Prior tsMar4 results for comparison:
- tsMar4/v10.2 GQT best: 75/53/44/27%
- tsMar4/v10.2 RFO best: 67/48/32/12%
- tsMar4/v10.2 ARC best: 20/4/1/0%
- tsMar4/v10c RFO+Schl+PLS: 89/78/63/42%
- tsMar4/v10c GQT+PLS: 83/67/56/45%

### C. v11 Factor Analysis (the most important section)

Answer EACH of these precisely:

1. **GQT+Schlegel+PLS (config 3) — the never-tested combo**:
   How does it compare to GQT+PLS (config 2) and RFO+Schlegel+PLS (config 1)?
   Does Schlegel help GQT the way it helps RFO?

2. **20k steps effect** (configs 7,8 vs 1,3):
   - Exact convergence gain at each noise level
   - Which failure class benefits? (oscillating? almost_converged? ghost_modes?)
   - Cost: wall-time increase vs convergence gain
   - Are the extra converged samples from almost_converged or from oscillating?

3. **Trust radius floor 0.005A** (configs 5,6 vs 1,3):
   - Does lowering the floor help or hurt?
   - Which failure class changes?
   - Interaction with optimizer (RFO vs GQT)?

4. **Schlegel TR effect** (config 1 vs 4, config 3 vs 2):
   - Is Schlegel helpful, harmful, or neutral when PLS is already on?
   - Interaction with optimizer choice?

5. **RFO vs GQT** (with PLS on both):
   - At which noise levels does each win?
   - Is there a crossover point?
   - Step count differences?

### D. Failure Autopsy Deep Dive

For each run's autopsy:
1. Report the overall classification distribution
2. For non-ARC configs, report the failure breakdown
3. Compare v11 failure distribution to v10c (expecting improvement from 20k steps)
4. Identify the **hardest samples** that fail across ALL runs
5. For the top 5-10 hardest samples, report their spectral characteristics (min_eval, n_neg, grad_overlap, force_norm)

### E. Cross-Run Hardest Samples

Using the sample_hardness CSVs across all three runs:
- Which samples NEVER converge in any config of any run?
- Which samples converge only with 20k steps?
- What distinguishes the "rescued by 20k" samples from the "permanently hard" ones?

### F. SCINE & Transition1x Extras

- Note the SCINE investigation crashed (NameError: defaultdict not defined)
- Report the Transition1x frequency verification results
- Weak correlation (r≈0.22) means optimizer failure is NOT primarily driven by starting-point PES character

### G. Consolidated Best-Ever Table

Across ALL runs (mar8 + tsMar4), what is the best convergence rate at each noise level, and which config achieves it?

---

## 6. Theoretical Context for Interpretation

### Why PLS works
Polynomial line search fits a cubic p(t) = at^3 + bt^2 + ct + d using energy + gradient at current and previous geometry. Zero extra evaluations. It refines step LENGTH, not direction. The trust region already picked the direction; PLS fixes the distance. Critical in the eigenvalue-crossing regime where the quadratic model is poor.

### Why ARC fails
ARC's sigma-adaptation was designed for smooth optimization. On our noisy PES, even good steps can have poor quality ratio rho (due to Hessian noise ~8e-3), causing sigma to ratchet upward. Steps shrink to near-zero, energy stalls → "energy_plateau."

### Why GDIIS has zero effect
GDIIS extrapolation assumes error vectors span a low-dimensional subspace. In our ~60-DOF molecules, oscillation involves too many coupled modes for 8-12 history vectors to capture.

### The epsilon^{1/2} theorem
Boumal et al. proves: ||g|| <= epsilon implies lambda_min(H) >= -sqrt(epsilon). Our force threshold epsilon=1e-4 eV/A implies allowed eigenvalue ~ -0.01. This is exactly our relaxed threshold. Our method (GQT) is implicit cubic regularization per Benson-Shanno equivalence.

### Failure populations
- **Population A (blind modes)**: lambda_min in [-1e-3, -2e-4], gradient overlap < 0.1 with negative eigenvector. Optimizer sees the negative eigenvalue but gradient has no component in that direction. Shows up as `almost_converged` and `ghost_modes` in autopsy.
- **Population B (oscillation-collapse)**: lambda_min in [-1e-2, -1e-3], forces 0.1-0.37 eV/A. Period-~8 oscillation. PLS halves this population.

---

## 7. Output Format

Write ALL analysis directly into `src/noisy/v2_tests/LITERATURE_REVIEW_AND_SYNTHESIS.tex`. Do NOT create an intermediate markdown file. TeX is the final target and produces higher quality tables and formatting. Token-doubling (MD then TeX conversion) wastes context and degrades quality.

**What to add/update in the TeX:**

1. Update the version history table with v11 results
2. Add a new section `\section{mar8 Experimental Results}` (or update the existing v10/v10c section) containing:
   - **Reproducibility assessment** (mar8 vs tsMar4 at matching configs)
   - **v10 results table** with autopsy summary
   - **v10c results table** with PLS effect quantified, autopsy summary
   - **v11 results** — the main event:
     - Full convergence table (8 configs × 4 noise)
     - Factor analysis subsections: 20k steps, trust floor, Schlegel effect, RFO vs GQT, GQT+Schlegel+PLS
     - Autopsy comparison (v11 vs v10c failure distributions)
   - **Cross-run hardest samples** analysis
   - **SCINE & Transition1x** notes
   - **Consolidated best-ever table** across ALL runs
3. Update the `\section{Open Questions}` with new findings
4. Update the recommended configuration
5. Add any new references if needed

Use proper LaTeX: `\begin{table}`, `\toprule`/`\midrule`/`\bottomrule`, `\textbf{}` for emphasis, `\label{}`/`\ref{}` for cross-references.

---

## 8. Prior Results — Full Reference (tsMar4 runs, different random seed)

### tsMar4/v9 — Relaxed Convergence Baseline
16 configs: GQT with 4 relaxed thresholds × 4 noise levels. **ALL relaxed thresholds give IDENTICAL counts.**

| Noise | Conv rate | Conv/Total | Steps |
|-------|-----------|------------|-------|
| 0.5A  | 80%       | 231/287    | 963   |
| 1.0A  | 60%       | 172/287    | 1791  |
| 1.5A  | 46%       | 133/287    | 3041  |
| 2.0A  | 29%       | 84/287     | 3044  |

### tsMar4/v10 — First ARC Test
28 configs. GQT baseline + ARC grid. GQT best: 82/55/47/37%. **ARC catastrophic**: 23/2/0/0% at best (σ₀=1, γ₁=3, n=0.5). Autopsy: 80.3% energy_plateau.

### tsMar4/v10.2 — Added RFO + Schlegel + GDIIS
48 configs (no PLS). Key results:
- GQT: 75/53/44/27%
- GQT+Schlegel: 74/55/47/31%
- RFO: 67/48/32/12%
- RFO+Schlegel: 65/46/29/13%
- RFO+Schlegel+GDIIS: 65/46/29/13% (=RFO+Schlegel, GDIIS zero effect)
- ARC best: 20/4/1/0%

### tsMar4/v10c — PLS Breakthrough
56 configs. Added PLS to GQT and RFO+Schlegel. Key results:
- **RFO+Schlegel+PLS: 89/78/63/42%** (champion at 0.5-1.5A)
- GQT+PLS: 83/67/56/45% (wins at 2.0A)
- GQT+Schlegel (no PLS): 78/57/47/29%
- GQT (no PLS): 76/55/49/27%
- RFO+Schlegel (no PLS): 67/44/31/14%
- RFO (no PLS): 65/46/32/13%
- ARC unchanged: 20/2/1/0%

PLS factor effect: +7 to +34 pts. Halves oscillating failures.

### tsMar4/v10c Autopsy (non-ARC failure breakdown):
| Classification | Overall | Non-ARC |
|---------------|---------|---------|
| converged     | 27.7%   | —       |
| energy_plateau| 47.4%   | 0% (ARC only) |
| oscillating   | 16.9%   | ~50% of failures |
| almost_converged| 6.2%  | ~30% of failures |
| ghost_modes   | 1.5%    | ~10% of failures |

At n=0.5, RFO+PLS: 18 oscillating vs RFO: 61 oscillating (PLS halves).
At n=0.5, GQT+PLS: 12 oscillating vs GQT: 32 oscillating.

### Version History Summary (v1-v8)
| Ver | Key Idea | Result | Verdict |
|-----|----------|--------|---------|
| v1 | Hard-filter NR + TR | Baseline | — |
| v2 | LM damping | Inferior to SN | Diagnostics kept |
| v3 | **Shifted Newton** + escape | 80% at 1.0A (15 samples) | Best step |
| v4 | 5 targeted failure fixes | All hurt (53→10%) | All reverted |
| v5 | SPDN (partition + GDIIS + LS) | 0% convergence | Catastrophic |
| v6 | Back to basics: SN + TR | 76-79% at 0.5A (300) | Confirmed |
| v7 | Line search vs trust region | LS worse at all noise | TR wins |
| v8 | iHiSD crossover (GD→NR) | Crossover hurts | Pure SN best |
| PIC-ARC | First-order + cubic reg | 0% convergence | First-order dead |

---

## 9. Complete File Index

### mar8/ (NEW results — this analysis)
```
mar8/
  ANALYSIS_GUIDE.md              ← This file
  v10/
    analysis/report.txt          ← 48-config ranked list + main effects
    analysis/nr_grid_ranked.csv  ← Machine-readable rankings
    analysis/nr_grid_sample_hardness.csv  ← Per-sample convergence rates
    analysis/nr_grid_summary.json  ← 24 MB per-sample detail (use selectively)
    autopsy/autopsy_report.txt   ← Failure classification per combo
  v10c/
    analysis/report.txt          ← 56-config ranked list + main effects
    analysis/nr_grid_ranked.csv
    analysis/nr_grid_sample_hardness.csv
    analysis/nr_grid_summary.json  ← 28 MB
    autopsy/autopsy_report.txt
    scine_convergence/investigation_report.txt  ← Crashed (NameError)
    verify_transition1x/verify_transition1x_minima_report.txt
    verify_transition1x/verify_report.txt
    verify_transition1x/verify_transition1x_minima_samples.csv
    verify_transition1x/verify_transition1x_minima.json
  v11/
    analysis/report.txt          ← 32-config ranked list + main effects
    analysis/nr_grid_ranked.csv
    analysis/nr_grid_sample_hardness.csv
    analysis/nr_grid_summary.json  ← 16 MB
    autopsy/autopsy_report.txt
```

### Prior results (tsMar4/ — for reproducibility comparison)
```
src/noisy/v2_tests/tsMar4/
  v9/report.txt                 ← 16 configs, GQT relaxed threshold sweep
  v9/autopsy_report.txt
  v10/report.txt                ← 28 configs, first ARC test
  v10/autopsy_report.txt
  v10.2/report.txt              ← 48 configs, +RFO +Schlegel +GDIIS
  v10.2/autopsy_report.txt
  v10c/report.txt               ← 56 configs, +PLS
  v10c/autopsy_report.txt
```

### Key documentation
```
src/noisy/v2_tests/LITERATURE_REVIEW_AND_SYNTHESIS.tex  ← Main TeX doc (~1355 lines)
memory/MEMORY.md                ← Master index
memory/experiments.md           ← Full experiment record
memory/technical.md             ← Equations and algorithms
memory/diagnostics.md           ← Cascade data and failure modes
memory/literature.md            ← Literature synthesis
```

Note: "memory/" refers to `/Users/memoozdincer/.claude/projects/-Users-memoozdincer-Documents-Research-Guzik-ts-tools/memory/`

---

## 10. Recommended Workflow

**No subagents. Read everything yourself.** The total data is ~1200 lines across 6 report/autopsy files — well within context. Reading it all directly gives you full fidelity for cross-referencing.

### Step 1: Read this guide, then the TeX document
1. This guide (`mar8/ANALYSIS_GUIDE.md`)
2. `src/noisy/v2_tests/LITERATURE_REVIEW_AND_SYNTHESIS.tex` — existing TeX you'll update

### Step 2: Read prior results (tsMar4) for comparison baselines
- `src/noisy/v2_tests/tsMar4/v9/report.txt`
- `src/noisy/v2_tests/tsMar4/v10.2/report.txt`
- `src/noisy/v2_tests/tsMar4/v10c/report.txt`
- `src/noisy/v2_tests/tsMar4/v10c/autopsy_report.txt`

### Step 3: Read all mar8 results
- `mar8/v10/analysis/report.txt` + `mar8/v10/autopsy/autopsy_report.txt`
- `mar8/v10c/analysis/report.txt` + `mar8/v10c/autopsy/autopsy_report.txt`
- `mar8/v10c/scine_convergence/investigation_report.txt`
- `mar8/v10c/verify_transition1x/verify_transition1x_minima_report.txt`
- `mar8/v11/analysis/report.txt` + `mar8/v11/autopsy/autopsy_report.txt`

### Step 4: Analyze and write directly to TeX
Work through EVERY item in the analysis checklist (sections 5A–5G). Write results directly into the TeX document.

### Step 5 (if needed): Sample-level cross-referencing
If you need to identify which samples are rescued by 20k steps vs permanently hard, read the sample_hardness CSVs:
- `mar8/v11/analysis/nr_grid_sample_hardness.csv`
- `mar8/v10c/analysis/nr_grid_sample_hardness.csv`

---

## 11. Critical Reminders

- **287 samples per config** (some configs may show fewer due to worker errors)
- **Convergence = strict n_neg=0, force < 1e-4 eV/A**
- All configs use `accept_relaxed=True` with `relaxed_eval_threshold=-0.01`
- Compare RATES (%), not raw counts, when configs have different N_total
- The `errors` field in reports counts worker crashes (SCINE failures), not optimizer failures
- When comparing to tsMar4, note these are DIFFERENT random seeds — some variation is expected
- PLS = polynomial_linesearch (cubic interpolation, zero extra evaluations)
- GQT = Shifted Newton with epsilon=1e-3 (also called "v9 baseline", "SN", or "mad1.3_se1e-3_sctr")
- "str" in tags = Schlegel trust radius update (NOT "string")
- The v10 mar8 data has 48 configs (same as tsMar4/v10.2, not tsMar4/v10 which had 28)
- The trial folders (n*/) are mostly empty — raw data stayed on the Compute Canada scratch filesystem. All aggregated results are in analysis/ and autopsy/
- The sample_hardness CSV has one row per sample with columns: sample_idx, n_converged, n_total, convergence_rate, best_converged_step, best_combo_tag
- The ranked CSV has 37 columns but does NOT include `schlegel_trust_update`, `trust_radius_floor`, or `n_steps_20k` — read these from the tag names instead (e.g., `_str` = Schlegel, `_trf0.005` = floor 0.005, `_20k` = 20k steps)
- For cross-run comparisons, note that v10 mar8 has the SAME 48 configs as tsMar4/v10.2 (not tsMar4/v10 which had 28 configs). The v10c mar8 matches tsMar4/v10c exactly (56 configs)
