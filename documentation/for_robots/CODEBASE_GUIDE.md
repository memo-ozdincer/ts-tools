# ts-tools codebase guide (refactored core + HPC runners)

This repo is organized around **transition-state (TS) search / saddle-point experiments** on the Transition1x dataset.

The codebase is intentionally split into two worlds:

- **Refactored “new stack”** (what you should run now):
  - `python -m src.runners.*`
  - algorithm core in `src/core_algos/`
  - backend glue + shared utilities in `src/dependencies/`
  - standardized plot + W&B import path in `src/logging/`

- **Legacy snapshot** (parked for archaeology / comparison only):
  - `legacy/src/` and `legacy/scripts/`
  - these are the old monolithic entrypoints and old cluster scripts

Two realities shape most of the design decisions:

1. **Two calculator backends**
  - **HIP** (Equiformer / ML potential): GPU-friendly, can provide **forces + Hessians**, and supports an **autograd path** in this repo.
  - **SCINE Sparrow**: CPU-only backend, can provide forces/Hessians, but is **not autograd-differentiable** here.

2. **Hessians contain rigid-body null modes**
  - Translation/rotation directions are ~0 curvature.
  - Any “smallest eigenvalue” logic must project them out, then compute the **vibrational spectrum**.

This guide documents the refactored layout, the key interfaces, the “weird dependencies”, and what is safe to delete.

---

## Directory structure (what lives where)

### `src/` (the refactored Python package)

At the top level, `src/` contains only packages. The old single-file scripts are now in `legacy/src/`.

- `src/core_algos/` (refactored “algorithm core”)
  - Backend-agnostic algorithms parameterized by a `predict_fn(coords, atomic_nums, do_hessian, require_grad)` callable.
  - `core_algos/gad.py`: core GAD vector computation + Euler step + RK45 integrator.
  - `core_algos/eigenproduct.py`: eigenvalue-product descent step (used when doing gradient-based eig-product optimization).
  - `core_algos/signenforcer.py`: sign-enforcer loss (pushes spectrum toward exactly 1 negative vibrational eigenvalue).
  - `core_algos/types.py`: minimal typing/protocols.

- `src/dependencies/` (backend adapters + shared utilities)
  This folder is intentionally the “support layer” for runners.

  Key modules:

  - `dependencies/common_utils.py`
    - Dataset loading (`Transition1xDataset`) + PyG `DataLoader` setup.
    - CLI arg helpers (`add_common_args`).
    - Starting geometry selection + noise injection (`parse_starting_geometry`).
    - Vibrational spectrum helper (`extract_vibrational_eigenvalues(...)`).
    - **Weird but intentional**: HIP checkpoint loading monkey-patch (lenient dataset paths).

  - `dependencies/experiment_logger.py`
    - `ExperimentLogger` + `RunResult` container.
    - JSON dumps (`all_runs.json`, `aggregate_stats.json`) and sampled plot saving.
    - W&B helpers: `init_wandb_run`, `log_sample`, `log_summary`, `finish_wandb`.

  - `dependencies/differentiable_projection.py`
    - Differentiable mass-weighting + Eckart projection helpers.
    - Used to remove translation/rotation directions from Hessians.

  - `dependencies/scine_calculator.py`
    - SCINE Sparrow wrapper used when `--calculator scine`.

  - `dependencies/calculators.py`:
    - `make_hip_predict_fn(calculator)`
    - `make_scine_predict_fn(scine_calculator)`
  - `dependencies/pyg_batch.py`: converts `(coords, atomic_nums)` into a PyG `Batch` in the layout HIP expects.
  - `dependencies/hessian.py`:
    - `project_hessian_remove_rigid_modes(...)`: mass-weight + Eckart projection.
    - `vibrational_eigvals(...)`: canonical “project then drop rigid modes” vibrational spectrum helper.

- `src/logging/` (standardized plotting + stable W&B import path)
  - `logging/trajectory_plots.py`: the **standard 3×2 trajectory plot** (`plot_gad_trajectory_3x2`).
  - `logging/wandb.py`: re-exports W&B helpers from `dependencies/experiment_logger.py`.

- `src/runners/` (cluster-facing entrypoints)
  - Thin CLI wrappers meant for `python -m src.runners.<...>`.
  - These coordinate:
    - dataset + calculator creation (`dependencies.common_utils.setup_experiment`)
    - backend adapters (`dependencies.calculators.*`)
    - algorithm core (`core_algos/*`)
    - standardized logging (`logging/*` + `dependencies.experiment_logger`)

### `legacy/` (legacy snapshot)

- `legacy/src/`
  - Old single-file scripts like `gad_gad_euler_rmsd.py`, `gad_rk45_search.py`, etc.
  - Kept only for reference.

- `legacy/scripts/`
  - Old SLURM scripts (e.g. CSLab variants).

### `scripts/` (cluster SLURM scripts)

- `scripts/killarney/`: **current** cluster scripts (repointed at `python -m src.runners.*`).
- `legacy/scripts/killarney/`: the older Killarney scripts (archived).

---

## The “predict_fn” contract (core/adapter boundary)

The refactor is based on a single interface:

- `predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False) -> dict`

Expected keys in the returned dict:

- `energy`: scalar
- `forces`: shape `(1,N,3)` or `(N,3)`
- `hessian`: shape `(3N,3N)` or another reshapeable form

### HIP vs SCINE behavior

- HIP (`make_hip_predict_fn`):
  - `require_grad=False` uses `calculator.predict(...)`.
  - `require_grad=True` uses `calculator.potential.forward(...)` to enable autograd w.r.t. `coords`.

- SCINE (`make_scine_predict_fn`):
  - Always CPU-only.
  - `require_grad=True` is **not supported** (raises `NotImplementedError`).

This is deliberate: it keeps `core_algos/*` pure, and the backend quirks live in `dependencies/`.

### Differentiability rule of thumb

- If a workflow needs gradients w.r.t. coordinates (autograd), it must use **HIP** and call `predict_fn(..., require_grad=True)`.
- SCINE predict functions are **non-differentiable** in this repo and only support `require_grad=False`.

---

## Hessian projection + rigid mode removal (critical)

Many TS heuristics use the smallest Hessian eigenvalues. Raw Hessians contain rigid-body modes:

- Nonlinear molecules: 6 rigid modes (3 translations + 3 rotations)
- Linear molecules: 5 rigid modes (3 translations + 2 rotations)

The canonical pipeline in this repo is:

1. **Mass-weight + Eckart project** the Hessian:
  - `src/dependencies/hessian.py:project_hessian_remove_rigid_modes(...)`
  - Uses `src/dependencies/differentiable_projection.py:differentiable_massweigh_and_eckartprojection_torch`.

2. **Remove remaining rigid eigenvalues** from the projected Hessian spectrum:
  - `src/dependencies/common_utils.py:extract_vibrational_eigenvalues(...)`
   - It detects linear vs nonlinear via coordinate rank and drops 5 or 6 eigenvalues (smallest by absolute value).

3. Use vibrational eigenvalues in all “order” / eig-product logic:
   - `src/dependencies/hessian.py:vibrational_eigvals(...)`

This is especially important for HIP Hessians.

---

## Logging and W&B

There are two logging layers:

1. **Local artifacts** (`ExperimentLogger`)
   - Writes per-run JSON (`all_runs.json`) and aggregate stats (`aggregate_stats.json`).
   - Saves plots into transition buckets like `0neg-to-1neg/`.

2. **W&B integration**
  - Centralized in `src/dependencies/experiment_logger.py`.
  - Re-exported via `src/logging/wandb.py` to keep runner imports stable.

### Standard trajectory plot

The standard plot is the 3×2 grid extracted from the legacy Euler script:

- `src/logging/trajectory_plots.py:plot_gad_trajectory_3x2(...)`

It expects a trajectory dict with keys like:

- `energy`, `force_mean`, `eig0`, `eig1`, `eig_product`, `disp_from_last`, `disp_from_start`

---

## Runners (what `scripts/killarney/` uses)

### `python -m src.runners.gad_euler_core`

- Implements “plain” Euler GAD stepping using `core_algos.gad.gad_euler_step`.
- Computes vibrational eigenvalues each step via `dependencies.hessian.vibrational_eigvals`.
- Writes standardized plots + JSON results.

Supports HIP and SCINE.

### `python -m src.runners.eigenvalue_descent_core`

- Implements gradient descent on either:
  - `eig_product` (minimize $\lambda_0\lambda_1$)
  - `sign_enforcer` (push spectrum toward exactly one negative eigenvalue)

Important limitation:

- This runner requires `require_grad=True` and therefore **HIP only**.

### `python -m src.runners.gad_rk45_core`

- Integrates GAD dynamics using a minimal RK45 solver (`core_algos.gad.gad_rk45_integrate`).
- No hybrid mode / no saddle-order switching logic.
- Computes trajectory metrics after the fact using the same Hessian projection + vibrational eigenvalues pipeline.

---

## SLURM scripts: active vs legacy

This repo now treats:

- `scripts/killarney/` as the **active** operational scripts (new-stack runners).
- `legacy/scripts/killarney/` as the archived “old killarney” snapshot.

### Updated in `scripts/killarney/`

- Euler GAD:
  - `gad_euler.slurm` → `python -m src.runners.gad_euler_core`
  - `run_gad_euler_noisy.slurm` → `python -m src.runners.gad_euler_core --start-from reactant_noise*`
  - `scine_gad_euler.slurm` → `python -m src.runners.gad_euler_core --calculator scine`
  - `scine_gad_euler_noisy.slurm` → `python -m src.runners.gad_euler_core --calculator scine --start-from reactant_noise*`

- RK45 GAD:
  - `run_gad_rk45.slurm` → `python -m src.runners.gad_rk45_core`
  - `run_gad_rk45_noisy.slurm` → `python -m src.runners.gad_rk45_core --start-from reactant_noise*`

- HIP eigenvalue descent:
  - `run_eigen_optimization.slurm` → `python -m src.runners.eigenvalue_descent_core`
  - Note: legacy-only flags like `--use-line-search` are intentionally removed.

### Not yet refactored (still calls legacy modules)

Some scripts may still call legacy utilities (e.g., frequency analysis, starting-geometry stats, LBFGS minimization) if you haven’t provided runner equivalents.

In general:

- If a SLURM script uses `python -m src.runners.*`, it is on the new stack.
- If it uses `python -m legacy.src.*` (or points into `legacy/`), it is legacy.

---

## “Weird dependencies” / gotchas

### HIP checkpoint loading patch

`src/dependencies/common_utils.py` monkey-patches HIP’s dataset path resolver:

- HIP’s `fix_dataset_path(...)` is overridden to be lenient.
- This is to allow loading checkpoints for inference even if the original training dataset paths aren’t present on the cluster.

If you ever see checkpoint-load failures mentioning dataset paths, this patch is the first place to look.

### Transition1x dataset layout

The dataset loader expects `transition1x.h5` and uses `transition1x.Dataloader` with `only_final=True`. Each sample includes:

- `pos_transition`
- `pos_reactant`
- `z` (atomic numbers)
- `formula` and `rxn` metadata

### SCINE is CPU only

Even if `--device cuda` is set, SCINE calculations run on CPU.

The refactored runners defensively force `device="cpu"` when `--calculator scine` so noisy-geometry runs don’t accidentally call `.to("cuda")`.

### Global `torch.set_grad_enabled(False)`

`dependencies.common_utils.setup_experiment(...)` disables grads globally.

- This is correct for the dynamics runners (`gad_euler_core`, `gad_rk45_core`) which do not rely on autograd.
- The eigenvalue-descent runner re-enables it (`torch.set_grad_enabled(True)`) because it needs autograd.

---

## Practical: running locally vs on the cluster

### Environment setup (from `CLAUDE.md`)

This repo is usually run on a cluster with SLURM and a UV-managed venv.

- Create/install:
  - `uv venv .venv`
  - `source .venv/bin/activate`
  - `uv pip install -e .`
  - `uv pip install -e ../HIP` (HIP is expected as a sibling repo)

The `pyproject.toml` dependency list is intentionally incomplete for the full research stack; in practice you also need:

- `hip` (editable install from `../HIP`)
- `transition1x` (dataset loader)
- `nets` (atom symbol mappings / utilities)
- Optional SCINE: `scine-utilities`, `scine-sparrow`

If an import fails in a runner, it is usually because one of these external research packages is missing from the environment.

### Data + checkpoint conventions

Common defaults used by the Killarney scripts (override via CLI args if needed):

- Dataset: `transition1x.h5`
- Model checkpoint: `hip_v2.ckpt`

Where these live depends on your cluster account layout; two common patterns are:

- Project-style layout (code + dependencies + durable artifacts), with scratch for high-I/O outputs.
- Allocation-style layout under `/project/<alloc>/...` with scratch outputs under `/scratch/<user>/...`.

The authoritative paths are whatever your SLURM scripts in `scripts/killarney/` set for `--h5-path`, `--checkpoint-path`, and `--out-dir`.

On Killarney, the `scripts/killarney*/` SLURM scripts are the source of truth.

If you run locally, you’ll need the same scientific stack (PyTorch, torch-geometric, HIP, transition1x, etc.). The repo is primarily HPC-oriented.

---

## What is safe to delete now?

You already moved the old code into `legacy/`. After that, the **new stack** is essentially:

- `src/core_algos/`
- `src/dependencies/`
- `src/logging/`
- `src/runners/`

Anything else should either live in `legacy/` (if you want to keep it around) or be removed.

Practical deletion rule:

- If nothing in `scripts/` calls an entrypoint, and nothing in `src/` imports it, it’s safe to delete.

If you want, we can add a short `scripts/killarney/README.md` section that lists which SLURM files are "new stack" vs "legacy utilities".

---

## SCINE Implementation (Clean and Self-Contained)

### Overview

SCINE Sparrow is a CPU-only semi-empirical calculator backend (DFTB0, PM6, AM1, etc.) that provides an alternative to HIP. The SCINE implementation is now **fully self-contained** and follows the clean code structure of this repo.

### Key Module: `src/dependencies/scine_masses.py`

**Purpose**: SCINE-specific mass-weighting and frequency analysis, **independent of HIP**.

**Previous Issue**: The old implementation relied on `hip.masses.MASS_DICT`, creating unwanted coupling.

**Current Solution**: This module provides:

1. **Complete element mapping**:
   - `SCINE_ELEMENT_MASSES`: Mass dictionary for all SCINE ElementType objects (H through Xe, Z=1-54)
   - `Z_TO_SCINE_ELEMENT`: Atomic number → SCINE ElementType mapping (extended from Z=20 to Z=54)

2. **Frequency Analysis**:
   - `ScineFrequencyAnalyzer`: NumPy/SciPy-based implementation using **SVD projection method**
   - `project_hessian()`: Mass-weights Hessian and applies Eckart projection to remove rigid modes
   - `analyze()`: Full pipeline returning vibrational frequencies (cm⁻¹) and eigenvalues

3. **PyTorch Interface** (for compatibility):
   - `scine_project_hessian_remove_rigid_modes()`: Returns PyTorch tensor
   - `scine_vibrational_eigvals()`: SCINE equivalent of `vibrational_eigvals()` from `hessian.py`

**Technical Details**:
- Uses **SVD-based projection**: Projects Hessian from 3N to (3N-k) dimensional space, where k is the number of rigid modes (5 for linear, 6 for non-linear)
- Numerically robust: No need to manually drop near-zero eigenvalues
- Units: Converts SCINE output (Hartree/Bohr) to eV/Å internally

### Updated Module: `src/dependencies/scine_calculator.py`

**Improvements**:
1. **Centralized element mapping**: Imports `Z_TO_SCINE_ELEMENT` from `scine_masses.py` instead of duplicating
2. **Element caching**: Stores `_last_elements` after each calculation
3. **Access method**: `get_last_elements()` provides downstream code access to SCINE ElementType list

**Why caching?**
- Avoids redundant atomic number → ElementType conversions
- Enables SCINE-specific mass-weighting in `hessian.py` helpers

### Unified Interface: `src/dependencies/hessian.py`

**Key Change**: `vibrational_eigvals()` now supports both HIP and SCINE:

```python
def vibrational_eigvals(
    hessian_raw: torch.Tensor,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    scine_elements: Optional[list] = None,  # <-- New parameter
) -> torch.Tensor:
```

**Behavior**:
- If `scine_elements` is `None`: Uses HIP mass-weighting (default, backward compatible)
- If `scine_elements` is provided: Uses SCINE mass-weighting from `scine_masses.py`

**Helper Function**:
- `get_scine_elements_from_predict_output(out)`: Extracts SCINE elements from predict output dict
  - Checks for `"_scine_calculator"` key in output
  - Calls `calculator.get_last_elements()` if present
  - Returns `None` for HIP calculations

### Predict Function Adapter: `src/dependencies/calculators.py`

**SCINE-specific change**:

`make_scine_predict_fn()` now attaches the calculator instance to the output dict:

```python
result["_scine_calculator"] = scine_calculator
```

This allows downstream code to:
1. Detect that SCINE is being used
2. Access `get_last_elements()` for mass-weighting

### Runner Updates: `gad_euler_core.py` and `gad_rk45_core.py`

**Pattern** (applied consistently in both runners):

```python
from ..dependencies.hessian import vibrational_eigvals, get_scine_elements_from_predict_output

# Inside trajectory loop:
out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
scine_elements = get_scine_elements_from_predict_output(out)  # None if HIP
vib = vibrational_eigvals(hessian, coords, atomic_nums, scine_elements=scine_elements)
```

**Benefits**:
- Automatic detection: No manual `if calculator_type == "scine"` checks
- Correct mass-weighting: Uses SCINE masses when SCINE is used, HIP masses otherwise
- Clean code: Single call site works for both backends

### Critical Design: GAD Dynamics vs. Vibrational Analysis

**Important distinction** (correctly implemented):

1. **GAD vector computation** (`src/core_algos/gad.py:compute_gad_vector`):
   - Uses **raw Cartesian Hessian** (no mass-weighting, no projection)
   - Formula: `gad_vec = forces + 2.0 * dot(-forces, v_min) * v_min`
   - `v_min` is the lowest eigenvector of the **geometric Hessian**
   - **Why?** GAD explores the potential energy surface (PES) geometry. Mass-weighting would change the surface shape.

2. **Vibrational eigenvalue extraction** (in runners, for logging/verification):
   - Uses **mass-weighted + Eckart-projected Hessian**
   - Removes translation/rotation modes
   - **Why?** Provides chemically meaningful saddle order (number of imaginary frequencies)

This distinction is critical for correct transition state search.

### Comparison: SCINE vs. HIP Mass-Weighting

| Feature | SCINE (`scine_masses.py`) | HIP (`differentiable_projection.py`) |
|---------|---------------------------|--------------------------------------|
| Implementation | NumPy/SciPy | PyTorch |
| Projection Method | SVD-based (projects to 3N-k space) | QR-based (keeps 3N space) |
| Output Hessian | (3N-k, 3N-k) | (3N, 3N) with k near-zeros |
| Eigenvalue Filtering | Automatic (via projection) | Manual via `extract_vibrational_eigenvalues()` |
| Differentiable | No | Yes |
| Device | CPU only | CPU or GPU |
| Numerical Equivalence | Yes (< 1e-6 difference) | Yes |

Both methods are **mathematically equivalent** and produce the same vibrational eigenvalues (within numerical precision).

### Usage Example

**Run GAD with SCINE**:

```bash
python -m src.runners.gad_euler_core \
    --calculator scine \
    --h5-path transition1x.h5 \
    --n-steps 150 \
    --dt 0.001 \
    --out-dir results/scine_gad
```

**Automatic behavior**:
1. `setup_experiment()` creates `ScineSparrowCalculator` (from `scine_calculator.py`)
2. `make_scine_predict_fn()` wraps it (from `calculators.py`)
3. Each `predict_fn()` call:
   - Caches elements in calculator
   - Attaches `"_scine_calculator"` to output
4. `vibrational_eigvals()` automatically uses SCINE mass-weighting

**No code changes needed in runners** - everything is handled via the adapter pattern.

### Environment Setup for SCINE

```bash
uv pip install scine-utilities scine-sparrow
```

SCINE is **optional**: If not installed, HIP-only workflows continue to work unchanged.

### Performance Considerations

**SCINE characteristics**:
- **CPU-only**: Ignores `--device cuda`
- **Single-threaded by default**: Sets `OMP_NUM_THREADS=1` to avoid conflicts
- **Best for**: Small to medium molecules (< 50 atoms) with semi-empirical methods

**Typical use cases**:
- ✅ Quick testing with DFTB0 before running expensive HIP calculations
- ✅ Benchmarking against semi-empirical reference methods
- ✅ CPU-only clusters where GPU access is limited
- ❌ Large-scale production runs (HIP is faster on GPU)
- ❌ Workflows requiring autograd (HIP only)

### File Organization

```
src/dependencies/
├── scine_masses.py              # New: SCINE-specific masses + frequency analysis
├── scine_calculator.py          # Updated: Element caching + complete mappings
├── calculators.py               # Updated: Attaches calculator to output
├── hessian.py                   # Updated: Accepts scine_elements parameter
├── differentiable_projection.py # Unchanged: HIP-specific (PyTorch)
└── common_utils.py              # Unchanged: Generic helpers

src/runners/
├── gad_euler_core.py            # Updated: Uses get_scine_elements_from_predict_output
└── gad_rk45_core.py             # Updated: Uses get_scine_elements_from_predict_output
```

### Troubleshooting

**Error: `RuntimeError: SCINE mass-weighting requested but scine_masses module not available`**

→ Install SCINE: `uv pip install scine-utilities scine-sparrow`

**Error: `ValueError: Unsupported element with Z=X`**

→ Add element to `Z_TO_SCINE_ELEMENT` and `SCINE_ELEMENT_MASSES` in `scine_masses.py`

**Small numerical differences between HIP and SCINE eigenvalues**

→ Expected. Different implementations (SVD vs. QR) and precision (float64 vs. float32). Differences should be < 1e-6.

### Summary

The SCINE implementation is now:

1. **Self-contained**: Zero HIP dependencies in SCINE-specific code
2. **Clean**: Follows repo structure (`core_algos/`, `dependencies/`, `runners/`)
3. **Complete**: Full element support (Z=1-54), robust SVD projection
4. **Correct**: Proper GAD dynamics with raw Hessian, mass-weighted verification
5. **Unified**: Single `vibrational_eigvals()` interface for both backends

SCINE + GAD is production-ready for CPU-based transition state searches.
