# Newton-Raphson Minimization — Changes & Rationale (v2 → v3 → v4)

---

## Table of Contents

1. [v2 Baseline & Problem Statement](#v2-baseline--problem-statement)
2. [v3 Algorithms](#v3-algorithms)
3. [v3 Results & Failure Analysis](#v3-results--failure-analysis)
4. [v4 Root Cause Analysis](#v4-root-cause-analysis)
5. [v4 Algorithms](#v4-algorithms)
6. [v4 Diagnostic Additions](#v4-diagnostic-additions)
7. [v4 Grid Search Design](#v4-grid-search-design)
8. [Updated Scripts & Analysis](#updated-scripts--analysis)
9. [Files Changed](#files-changed)
10. [How to Interpret Results](#how-to-interpret-results)

---

## v2 Baseline & Problem Statement

v2 grid search (best config: LM μ=0.002) achieved only **30% strict convergence** (9/30 samples).
The cascade evaluation table revealed two distinct failure populations:

| Population | Size | Description |
|---|---|---|
| **A — "Almost there"** | ~50% of samples | Final min eigenvalue in (-0.002, 0). Optimizer gets close but can't push the last negative eigenvalue(s) to zero. |
| **B — "Genuinely stuck"** | ~33% (10 samples) | 0/12 convergence across ALL configs, even at relaxed eval thresholds (eval≥-0.01). |

### Root Cause (v2)

The NR step uses `|λ|` (absolute value Newton):
```
step_i = -(g·v_i) / |λ_i|
```
This treats negative eigenvalue modes identically to positive ones. It has **no mechanism to specifically target negative modes**. Near convergence:
- The negative eigenvalue is small → curvature is nearly flat
- The gradient projection onto the negative mode is small → NR step barely moves along it
- Other well-converged modes dominate the overall displacement

---

## v3 Algorithms

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

### 3. Adaptive LM μ Annealing (`--lm-mu-anneal-factor`)

When close to convergence (`n_neg ≤ lm_mu_anneal_n_neg_leq` AND `|λ_min| < lm_mu_anneal_eval_leq`), multiply μ by `lm_mu_anneal_factor`:

```
effective_μ = μ × anneal_factor    (e.g., 0.002 × 0.1 = 0.0002)
```

Lower μ near convergence → the LM weight `|λ|/(λ²+μ²)` approaches `1/|λ|` → more aggressive steps along flat/negative modes.

### 4. Trust-Region Floor (`--trust-radius-floor`)

All trust-radius reductions now clamp to `max(new_radius, trust_radius_floor)` (default 0.01 Å). Prevents the optimizer from shrinking to useless displacement magnitudes when the quadratic model is a poor fit.

---

## v3 Results & Failure Analysis

### Best v3 Configuration

**Shifted Newton ε=5e-4** achieved **80% strict convergence** (12/15 samples), up from 30% in v2.

### v3 Cascade Evaluation Table

```
optimizer                        eval≥-0.0  eval≥-0.001  eval≥-0.005  eval≥-0.01  strict
SN  ε=0.0005                       0.778      0.911        0.911       0.911      0.778
SN  ε=0.001                        0.756      0.867        0.911       0.911      0.756
SN  ε=0.0002                       0.711      0.867        0.911       0.911      0.711
SN  ε=0.0001                       0.667      0.822        0.911       0.911      0.667
LM  μ=0.002                        0.733      0.822        0.911       0.933      0.600
LM  μ=0.005                        0.567      0.700        0.900       0.933      0.467
```

Key observations:
- **ε=5e-4 is the sweet spot**: 77.8% strict, 91.1% at eval≥-0.001
- **13.3% gap** between strict and eval≥-0.001 → these samples have eigenvalues in (-0.001, 0) that are tantalizingly close
- **eval≥-0.01 plateau at 91–93%** across ALL optimizers → 7–9% are genuinely hard (true saddle character)
- Shifted Newton dominates LM at strict evaluation; LM catches up only at relaxed thresholds

### v3 Failure Autopsy (All 132 Failed Trajectories)

Across all 19 v3 configurations:

| Classification | Count | Fraction |
|---|---|---|
| `oscillating` | 81 | 34.7% |
| `almost_converged` | 27 | 11.6% |
| `energy_plateau` | 12 | 5.1% |
| `genuinely_stuck` | 8 | 3.4% |
| `slow_convergence` | 4 | 1.7% |

**Critical finding: zero genuinely_stuck failures** for the best shifted Newton configs. The remaining failures are almost entirely `oscillating` (trust radius collapse → micro-steps → eigenvalues bounce) and `almost_converged` (very close but can't push the last tiny eigenvalue to zero).

### Per-Sample Failure Deep Dive

Detailed numerical analysis of the 3 samples that never converge (0/19 across all configs):

#### sample_000 — "Blind modes"
- Final state: 5 negative eigenvalues, all ~1e-5 to 1e-6
- Gradient overlaps: `[0.003, 0.001, 0.001, 0.001, 0.001]`
- The gradient is orthogonal to ALL 5 negative modes
- NR step component ∝ `(g·v_i)` ≈ 0 → optimizer is literally blind to these modes
- 65% stagnation fraction — the optimizer does nothing for most of the run

#### sample_003 — "Single blind mode"
- Final state: 1 negative eigenvalue at -8e-8
- Gradient overlap: `[0.004]`
- 65% stagnation — optimizer stuck with overlap near zero
- Trust radius at floor (0.01 Å) for last 57,000 steps

#### sample_012 — "True double saddle"
- Final state: 2 large negative eigenvalues `[-0.148, -0.147]`
- Gradient overlaps: `[0.002, 0.0001]`
- Genuine saddle-point character — these are NOT near zero
- No amount of shifted Newton can fix this: the gradient is orthogonal to both large negative modes

#### samples_001, 002, 004, 005, 009 — "Crushed by trust radius"
- n_neg = 1–3, eigenvalues 1e-5 to 3.7e-4
- Gradient overlaps moderate: 0.13–0.16 (workable for NR)
- Trust radius stuck at floor (0.01 Å) → kills the already-small neg-mode step component
- sample_001: 55 escape events + 55 line searches — all ineffective because `sign(g·v_i)` is random when overlap ≈ 0.002

### Trust Radius Collapse Pattern

From trajectory graphs, a universal pathological pattern:
1. **Active phase** (~2000 steps): condition number oscillates 10^7 ↔ 10^12, trust radius adjusts actively
2. **First plateau** (steps 2500–60000): trust radius collapses to floor, condition number range narrows to tiny window
3. **Brief revival**: occasional spike of activity, trust radius grows slightly
4. **Second plateau** (rest of run): flat again, condition number locked in 5×10^8 to 10^9

**Root cause**: ρ = actual_dE/pred_dE is unreliable when the Hessian has near-zero eigenvalues (the quadratic model predicts near-zero energy change, so ρ swings wildly). Trust radius ratchets down via ×0.25 shrink on each bad ρ → hits floor → never recovers because ρ stays unreliable at the floor displacement scale.

---

## v4 Root Cause Analysis

The v3 failures decompose into **three distinct populations** with clear mechanistic causes:

### Population 1 — "Blind Modes" (samples 000, 010, 007, 003)

| Property | Typical Value |
|---|---|
| n_neg | 1–5 |
| eigenvalues | -1e-5 to -1e-6 |
| gradient overlap `\|g·v_i\|/\|g\|` | 0.001–0.004 |
| stagnation fraction | >50% |
| trust radius | at floor (0.01 Å) |

**Mechanism**: The gradient is orthogonal to all negative eigenvectors. Since every NR variant computes step components as `f(g·v_i)`, when `g·v_i ≈ 0` the step has essentially zero projection onto the problematic modes. **No gradient-based optimizer can fix this.** The stagnation escape also fails because its direction choice `sign(g·v_i)` is random when overlap ≈ 0.

**Implication**: Need gradient-independent exploration along negative modes.

### Population 2 — "Crushed by Trust Radius" (samples 005, 002, 009, 004, 001)

| Property | Typical Value |
|---|---|
| n_neg | 1–3 |
| eigenvalues | -1e-5 to -3.7e-4 |
| gradient overlap | 0.13–0.16 (moderate) |
| trust radius | at floor (0.01 Å) |
| step neg-mode fraction | <5% |

**Mechanism**: The gradient overlap is workable — shifted Newton could generate a useful neg-mode step. But the trust radius has collapsed to floor (0.01 Å), and the cap operation treats the full step vector equally. The neg-mode component (already small due to small eigenvalue) gets proportionally crushed by the cap. The positive-mode components (well-converged, large eigenvalues, strong gradients) dominate the capped step.

**Implication**: Need separate trust radii for pos/neg eigenvalue subspaces.

### Population 3 — "True Saddle" (sample 012)

| Property | Value |
|---|---|
| n_neg | 2 |
| eigenvalues | -0.148, -0.147 |
| gradient overlap | 0.002, 0.0001 |
| classification | genuine double-saddle |

**Mechanism**: Large negative eigenvalues with gradient orthogonal to both modes. This is a true saddle point — the geometry is sitting on a ridge in the PES. NR (which follows the gradient) cannot step off the ridge because the ridge direction is orthogonal to the gradient.

**Implication**: Need explicit mode-following along the most negative eigenvector, probing both directions.

---

## v4 Algorithms

All v4 features default to off (zero or False), so the optimizer is exactly backward-compatible with v3 when no v4 flags are passed.

### 1. Separate Neg-Mode Trust Radius (`--neg-trust-floor`)

**File:** `minimization.py` → `_cap_displacement_split()`

**Addresses:** Population 1 + 2 (trust collapse crushing neg-mode steps)

```python
def _cap_displacement_split(
    step_disp, evals_vib, evecs_vib_3N,
    pos_trust_radius, neg_trust_radius,
) -> torch.Tensor:
```

Decomposes the NR step vector into two orthogonal subspaces:
```
neg_component = Σ_{λ_i < 0} (step · v_i) v_i
pos_component = step - neg_component
```

Each component is capped independently:
```
capped_neg = _cap_displacement(neg_component, neg_trust_radius)
capped_pos = _cap_displacement(pos_component, pos_trust_radius)
result = capped_neg + capped_pos
```

**Neg-mode trust radius dynamics** (per-step update in main loop):
- Eigenvalue improved (less negative): `neg_trust_radius *= 1.5` (capped at `max_atom_disp`)
- Eigenvalue worsened: `neg_trust_radius *= 0.7` (floored at `neg_trust_floor`)
- n_neg decreased or reached zero: reset to `max_atom_disp`

**Why this works**: Even when `pos_trust_radius` has collapsed to 0.01 Å (due to unreliable ρ), the neg-mode trust radius stays at its own floor (e.g. 0.05 Å), preserving the neg-mode step component. The two trust radii evolve independently.

### 2. Blind-Mode Correction (`--blind-mode-threshold`, `--blind-correction-alpha`)

**File:** `minimization.py` → `_blind_mode_correction()`

**Addresses:** Population 1 (gradient orthogonal to negative modes)

```python
def _blind_mode_correction(
    delta_x, grad, evals_vib, evecs_vib_3N,
    blind_threshold, correction_alpha, step_number,
) -> Tuple[torch.Tensor, Dict]:
```

For each negative mode where the gradient overlap is below threshold:
```
overlap_i = |g · v_i| / |g|
if overlap_i < blind_threshold:
    sign_i = +1 if step_number is even, else -1
    correction += sign_i * v_i
```

The accumulated correction is normalized so max atom displacement = `correction_alpha`, then added to `delta_x` **before** trust-region capping.

**Key design choices:**
- **Alternating sign per step**: explores both directions along each blind mode without requiring extra evaluations. On even steps, pushes along +v_i; on odd steps, pushes along -v_i.
- **Applied before capping**: combined with Feature 1, the neg-mode trust radius (floor 0.05 Å) protects this correction from being crushed by pos-mode trust collapse.
- **Zero extra evaluations**: this is pure geometry modification, no Hessian probes needed.

**Diagnostics logged per step** (under key `"blind_correction"`):
- `n_blind_modes`: number of modes corrected
- `blind_modes`: list of `{eval, overlap, sign}` per corrected mode

### 3. Aggressive Trust Recovery (`--aggressive-trust-recovery`)

**File:** `minimization.py`, trust-region update logic in `run_newton_raphson()` main loop

**Addresses:** Population 2 (trust-radius collapse → flat pattern)

Three mechanisms:

#### 3a. Softer shrink near zero eigenvalues
When `aggressive_trust_recovery` is True and `|λ_min| < 0.01`:
```
trust_radius *= 0.5    (instead of 0.25)
```
In both the ρ < 0 branch and the step-rejection branch. This halves the rate of trust radius decay in the near-convergence regime where ρ is unreliable.

#### 3b. Reset on n_neg decrease
When `n_neg` decreases compared to the previous step:
```
trust_radius = 0.5 × max_atom_disp
```
A decrease in n_neg means the optimizer made real structural progress — reward it with a useful trust radius.

#### 3c. 50-step eigenvalue improvement window
Maintains a rolling 50-step window of `min_vib_eval` values. When the eigenvalue at the end of the window is better (less negative) than at the start:
```
trust_radius = min(trust_radius × 2.0, max_atom_disp)
```
This catches slow but steady eigenvalue improvement that the ρ-based trust logic misses (because individual ρ values are noisy when eigenvalues are near zero).

### 4. Bidirectional Stagnation Escape (`--escape-bidirectional`)

**File:** `minimization.py` → `_stagnation_escape_v4()`

**Addresses:** Population 1 + 2 (v3 escape ineffective when gradient overlap ≈ 0)

```python
def _stagnation_escape_v4(
    predict_fn, coords, atomic_nums, atomsymbols, grad,
    evals_vib, evecs_vib_3N, escape_alpha,
    *, blind_threshold=0.05, purify_hessian=False, min_interatomic_dist=0.5,
) -> Tuple[torch.Tensor, Dict]:
```

**Most negative mode** — bidirectional probe:
1. Evaluate Hessian at `coords + escape_alpha × v_0` (2 extra evals total)
2. Evaluate Hessian at `coords - escape_alpha × v_0`
3. Pick direction where min eigenvalue is less negative (closer to zero)

**Other negative modes** — informed sign selection:
- If `overlap_i > blind_threshold`: use `sign(g · v_i)` (gradient-informed, same as v3)
- Else: random sign (since gradient provides no information)

**Conditional acceptance** (in main loop, not in the helper):
After computing the escape geometry, evaluate the Hessian at the proposed point:
```
if min_eval_at_proposed >= min_eval_current - 1e-6:
    accept (move coords, reset trust radius, reset stagnation)
else:
    reject (don't move, log rejection reason)
```

This prevents escapes from making things worse, which was a problem in v3 where 55 escapes could fire with zero improvement (sample_001).

**Diagnostics**: `escape_accepted`, `escape_rejected`, `escape_info` (with probe results) logged per step.

### 5. Mode-Following (`--mode-follow-eval-threshold`, `--mode-follow-alpha`)

**File:** `minimization.py` → `_mode_follow_step()`

**Addresses:** Population 3 (true saddle points with large |λ_min|)

```python
def _mode_follow_step(
    predict_fn, coords, atomic_nums, atomsymbols,
    evals_vib, evecs_vib_3N, mode_follow_alpha,
    *, purify_hessian=False, min_interatomic_dist=0.5,
) -> Tuple[Optional[torch.Tensor], Dict, Optional[Dict]]:
```

Bidirectional probe along the most negative eigenvector:
1. Try `coords + mode_follow_alpha × v_neg` → evaluate Hessian
2. Try `coords - mode_follow_alpha × v_neg` → evaluate Hessian
3. Pick direction where min_vib_eval improves most
4. Return `(best_coords, info, best_out)` — `best_out` is reused to avoid re-evaluation

**Activation conditions** (disjoint from stagnation escape):
```python
if (mode_follow_eval_threshold > 0.0
    and step >= mode_follow_after_steps     # e.g. 2000
    and n_neg > 0
    and abs(min_vib_eval) > mode_follow_eval_threshold  # e.g. 0.01
):
```

Note the conditions are disjoint: mode-following requires `|λ_min| > threshold` (large eigenvalue, true saddle), while stagnation escape requires `|λ_min| < 0.02` (small eigenvalue, near convergence). No overlap at typical parameter values.

On success: resets both trust radii to `max_atom_disp`, resets stagnation counter, increments `total_mode_follows`.

**Why `mode_follow_after_steps`**: The early optimization phase often has large negative eigenvalues that the standard NR step handles well. Mode-following is only valuable after NR has done its job and converged the "easy" modes, leaving only the truly stuck saddle-point modes.

---

## v4 Diagnostic Additions

### Per-Step Negative-Mode Diagnostics (unchanged from v3)

Logged in every trajectory step (when `n_neg > 0`) under key `"neg_mode_diag"`:

| Field | What it tells you |
|---|---|
| `neg_mode_grad_overlaps` | `\|g·v_i\|/\|g\|` for each negative mode. If near zero → NR step literally cannot address that mode. |
| `neg_mode_eigenvalues` | The negative eigenvalues themselves (sorted, most negative first). |
| `step_along_neg_frac` | Fraction of `\|\|dx\|\|²` in the negative-eigenvalue subspace. |
| `step_along_pos_frac` | Fraction in the positive-eigenvalue subspace. |
| `max_neg_grad_overlap` | Largest overlap (easiest mode to fix). |
| `min_neg_grad_overlap` | Smallest overlap (bottleneck mode). |

### v4 Per-Step Fields

| Field | Description |
|---|---|
| `neg_trust_radius` | Current neg-mode trust radius (None if split trust off) |
| `blind_correction` | Dict: `{n_blind_modes, blind_modes: [{eval, overlap, sign}, ...]}` |
| `escape_accepted` | True when v4 bidirectional escape was accepted |
| `escape_rejected` | True when v4 escape was rejected (eigenvalue worsened) |
| `escape_rejected_reason` | String explaining why escape was rejected |
| `escape_info` | Dict with bidirectional probe results |
| `mode_follow_triggered` | True on steps where mode-following activated |
| `mode_follow_info` | Dict: `{original_min_eval, best_min_eval, probes: {plus, minus}, improved}` |
| `trust_radius_reset` | Reason for trust radius reset (e.g. `"n_neg_decreased"`) |
| `trust_radius_50step_grow` | True when 50-step eigenvalue window triggered growth |

### v3 Per-Step Fields (unchanged)

| Field | Description |
|---|---|
| `stagnation_counter` | Consecutive steps with unchanged `n_neg` |
| `energy_plateau` | True if energy range over last 10 steps < 1e-8 |
| `effective_lm_mu` | Current μ value (shows when annealing kicks in) |
| `escape_triggered` | True on steps where stagnation escape fired |
| `line_search_info` | Dict with line search trial results (when enabled) |

### Trajectory-Level Fields

| Field | Description |
|---|---|
| `total_escapes` | Total stagnation escape events |
| `total_line_searches` | Total line search events |
| `total_mode_follows` | Total mode-following events (**v4 new**) |

---

## v4 Grid Search Design

### SLURM Template: `minimization_nr_grid_v4_run.slurm`

**Fixed config**: N_STEPS=10000 (env-overridable), MAX_SAMPLES=15, TRUST_RADIUS_FLOOR=0.01, walltime=9h

**Folder tag convention** for v4 params:
- `ntf<v>` = neg-trust-floor
- `bmt<v>` = blind-mode-threshold
- `bca<v>` = blind-correction-alpha
- `atr` = aggressive-trust-recovery flag
- `ebd` = escape-bidirectional flag
- `mft<v>` = mode-follow-eval-threshold
- `mfa<v>` = mode-follow-alpha

Full tag format:
```
mad<v>_se<v>[_ntf<v>][_bmt<v>_bca<v>][_atr][_sw<v>_ea<v>][_ebd][_ls][_mft<v>][_mfa<v>]_pg<bool>_ph<bool>
```

### Grid Phases (25 combinations total)

| Phase | Purpose | Sweep Axes | Fixed | Combos |
|-------|---------|------------|-------|--------|
| **A** | Extended SN baseline (same budget, narrow ε range around sweet spot) | ε ∈ {2e-4, 3e-4, 5e-4, 7e-4} | all v4 off | 4 |
| **B** | Neg-mode trust radius sweep | ε ∈ {3e-4, 5e-4} × ntf ∈ {0.03, 0.05, 0.1} | bmt=0 | 6 |
| **C** | Blind correction sweep (layered on ntf) | ε ∈ {3e-4, 5e-4} × bca ∈ {0.01, 0.02, 0.05} | ntf=0.05, bmt=0.05 | 6 |
| **D** | Aggressive trust recovery (layered on ntf+blind) | ε ∈ {3e-4, 5e-4} | ntf=0.05, bmt=0.05, bca=0.02, atr=on | 2 |
| **E** | Stagnation escape overhaul (v3 vs v4 escape) | ea ∈ {0.03, 0.05} × {ebd_off, ebd_on} | se=5e-4 + all above, sw=100 | 4 |
| **F** | Mode-following for true saddle points | mft ∈ {0.01, 0.05} | all above + ebd=on | 2 |
| **G** | Kitchen sink (all features at guessed-best params) | — | se=5e-4, ntf=0.05, bmt=0.05, bca=0.02, atr, sw=100, ea=0.05, ebd, mft=0.01, mfa=0.15 | 1 |

### Grid Design Rationale

The phases are **layered**: each phase adds one feature on top of the previous best. This allows isolating the marginal contribution of each feature:

- Phase A → baseline
- Phase B → A + neg trust radius → measures trust-radius decoupling alone
- Phase C → B + blind correction → measures gradient-independent exploration on top of trust fix
- Phase D → C + aggressive recovery → measures trust dynamics improvement
- Phase E → D + escape overhaul → measures smarter escape direction selection
- Phase F → E + mode following → measures true-saddle handling
- Phase G → all together → final check for synergistic/antagonistic interactions

---

## Updated Scripts & Analysis

### Analysis Script (`analyze_minimization_nr_grid.py`)

**New regex**: `COMBO_RE_SHIFTED_V4` — tried before the v3 `COMBO_RE_SHIFTED` regex (more specific pattern takes precedence):
```python
COMBO_RE_SHIFTED_V4 = re.compile(
    r"mad(?P<mad>[^_]+)_se(?P<se>[^_]+)"
    r"(?:_ntf(?P<ntf>[^_]+))?"
    r"(?:_bmt(?P<bmt>[^_]+)_bca(?P<bca>[^_]+))?"
    r"(?:_(?P<atr>atr))?"
    r"(?:_sw(?P<sw>[^_]+)_ea(?P<ea>[^_]+))?"
    r"(?:_(?P<ebd>ebd))?"
    r"(?:_(?P<ls>ls))?"
    r"(?:_mft(?P<mft>[^_]+))?"
    r"(?:_mfa(?P<mfa>[^_]+))?"
    r"_pg(?P<pg>true|false)_ph(?P<ph>true|false)$"
)
```

**Extended `ComboRecord`** with 7 new v4 fields:
- `neg_trust_floor` (float, default 0.0)
- `blind_mode_threshold` (float, default 0.0)
- `blind_correction_alpha` (float, default 0.02)
- `aggressive_trust_recovery` (bool, default False)
- `escape_bidirectional` (bool, default False)
- `mode_follow_eval_threshold` (float, default 0.0)
- `total_mode_follows` (int, default 0)

**New main effects**: `neg_trust_floor`, `blind_mode_threshold`, `aggressive_trust_recovery`, `escape_bidirectional`, `mode_follow_eval_threshold`

**CSV output**: includes all v4 columns + `total_mode_follows`

### Failure Autopsy (`analyze_nr_failure_autopsy.py`)

**New v4 diagnostics extracted per trajectory**:

| Field | Source |
|---|---|
| `total_mode_follows` | Trajectory-level counter |
| `n_blind_corrections` | From final step's `blind_correction.n_blind_modes` |
| `final_neg_trust_radius` | From final step's `neg_trust_radius` |
| `escape_accepted_count` | Count of steps with `escape_accepted == True` |
| `escape_rejected_count` | Count of steps with `escape_rejected == True` |
| `n_mode_follow_events` | Count of steps with `mode_follow_triggered == True` |

All added to CSV output.

### Failure Classification (unchanged from v3)

| Classification | Criteria |
|---|---|
| `almost_converged` | `n_neg ≤ 3` and `\|λ_min\| < 0.002` |
| `oscillating` | Eigenvalues bouncing up/down in last 100 steps |
| `energy_plateau` | Energy range < 1e-6 over last 100 steps |
| `genuinely_stuck` | `n_neg` unchanged for >50% of all steps |
| `slow_convergence` | Eigenvalues still improving but ran out of steps |
| `drifting` | None of the above |

---

## Files Changed

| File | Change |
|---|---|
| `src/noisy/v2_tests/baselines/minimization.py` | 4 new helpers + main loop: split trust, blind correction, v4 escape, mode-following, aggressive trust recovery |
| `src/noisy/v2_tests/runners/run_minimization_parallel.py` | 8 new CLI args, params dict, function call, `total_mode_follows` in trajectory/return |
| `src/noisy/v2_tests/scripts/analyze_minimization_nr_grid.py` | `COMBO_RE_SHIFTED_V4` regex, 7 new ComboRecord fields, v4 main effects, CSV columns |
| `src/noisy/v2_tests/scripts/analyze_nr_failure_autopsy.py` | v4 diagnostics extraction (6 new fields), CSV columns |
| `src/noisy/v2_tests/scripts/analyze_nr_trajectory_stats.py` | Broadened combo regex to accept all tag formats |
| `src/noisy/v2_tests/scripts/slurm_templates/minimization_nr_grid_v4_run.slurm` | **New** — v4 grid search (25 combos, 7 phases) |
| `src/noisy/v2_tests/scripts/slurm_templates/minimization_nr_grid_v3_run.slurm` | v3 grid search (19 combos, 5 phases) |
| `src/noisy/v2_tests/scripts/analyze_nr_failure_autopsy.py` | **New in v3** — failure classification and autopsy |

---

## How to Interpret Results

### Reading the Cascade Table

```
optimizer                        eval≥-0.0  eval≥-0.001  eval≥-0.005  strict
SN  ε=0.0005                       0.778      0.911        0.911      0.778
SN  ε=0.0005 +ntf+bmt+atr+ebd     0.933      0.933        0.933      0.933
```

- **Gap between `eval≥0.0` and `eval≥0.001`** → Population 1/2 size (tiny eigenvalues that the optimizer can't push to exactly zero)
- **v4 goal: gap → 0** → split trust + blind correction should push those last eigenvalues past zero
- **`eval≥0.01` column** → ceiling from Population 3 (true saddle); mode-following should raise this
- **`strict` = `eval≥0.0`** when all eigenvalues are truly non-negative

### Reading the Autopsy Report

For each failed sample:
- If `n_blind_corrections > 0`: blind correction is active but may need larger `bca`
- If `final_neg_trust_radius` ≈ `neg_trust_floor`: neg-mode trust also collapsed — may need higher floor
- If `escape_accepted_count` low and `escape_rejected_count` high: v4 escape is firing but geometry keeps worsening — deeper structural issue
- If `n_mode_follow_events > 0` but still failing: mode-following fired but couldn't fix the saddle — may need larger `mfa` or more steps
- If all v4 diagnostics are zero for a failed sample: v4 features aren't activating — check thresholds

### What v4 Success Looks Like

**Target**: all 15 samples converging with strict n_neg==0. The ideal v4 config should show:
- Cascade table: `eval≥0.0` column approaching 1.0
- Autopsy: no `oscillating` or `almost_converged` failures (these are the v3 failure modes v4 targets)
- `genuinely_stuck` count = 0 (maintained from v3)
- Escape accept rate > reject rate (bidirectional escape is making informed choices)
- Blind correction active on the hardest samples (non-zero `n_blind_corrections`)
- Mode-following events rare but impactful (sample_012-type saddles resolved)
- Trust radius NOT flat for thousands of steps (aggressive recovery breaking the pattern)

### Feature Interaction Matrix

| Feature A ↓ / Feature B → | Neg Trust | Blind Corr | Aggressive TR | Bidir Escape | Mode Follow |
|---|---|---|---|---|---|
| **Neg Trust** | — | **synergy**: blind correction survives trust collapse | complementary: recovery helps neg-trust grow | independent | independent |
| **Blind Correction** | **synergy** | — | complementary | complementary: blind handles daily noise, escape handles stuck | independent |
| **Aggressive TR** | complementary | complementary | — | synergy: trust recovery means escape starts from better position | independent |
| **Bidir Escape** | independent | complementary | synergy | — | disjoint conditions |
| **Mode Follow** | independent | independent | independent | disjoint conditions | — |

Mode-following and stagnation escape are **condition-disjoint**: mode-following requires `|λ_min| > threshold` (large), escape requires `|λ_min| < 0.02` (small). They cannot both trigger on the same step.
