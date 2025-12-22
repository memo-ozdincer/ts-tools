# ts-tools codebase guide (HIP/SCINE + GAD/TS search)

This repo is organized around **transition-state (TS) search / saddle detection experiments** on the Transition1x dataset.

Two important realities shape the design:

1. **Two calculator backends**:
   - **HIP** (Equiformer / ML potential): fast, GPU, can provide **forces + Hessians**, and supports an **autograd path** (for some workflows).
   - **SCINE Sparrow**: CPU-only backend, can provide forces/Hessians, but is **not autograd-differentiable** in this repo.

2. **Hessians have rigid-body null modes**:
   - Raw Hessians contain translation/rotation directions with ~0 curvature.
   - Many “smallest-eigenvalue” workflows break unless we **project out** these null modes and then extract the **vibrational spectrum**.

This guide documents the current structure, the “weird dependencies”, and how the refactor is meant to be used on the cluster.

---

## Directory structure (what lives where)

### `src/` (Python package)

- `src/common_utils.py`
  - Dataset loading (`Transition1xDataset`) + PyG `DataLoader` setup.
  - CLI arg helpers (`add_common_args`).
  - Starting geometry selection + noise injection (`parse_starting_geometry`).
  - **Central** helper for removing rigid modes from a *projected* Hessian: `extract_vibrational_eigenvalues(...)`.
  - **Important**: also includes a monkey-patch that makes HIP checkpoint loading more lenient by overriding HIP’s dataset-path resolution.

- `src/experiment_logger.py`
  - Standard logging utilities:
    - Output directory structure.
    - `RunResult` schema.
    - Sampling and saving plots per transition bucket.
    - Aggregate stats + JSON dumps.
  - W&B helpers: `init_wandb_run`, `log_sample`, `log_summary`, `finish_wandb`.

- `src/core_algos/` (refactored “algorithm core”)
  - Backend-agnostic algorithms parameterized by a `predict_fn(coords, atomic_nums, do_hessian, require_grad)` callable.
  - `core_algos/gad.py`: core GAD vector computation + Euler step + RK45 integrator.
  - `core_algos/eigenproduct.py`: eigenvalue-product descent step (used when doing gradient-based eig-product optimization).
  - `core_algos/signenforcer.py`: sign-enforcer loss (pushes spectrum toward exactly 1 negative vibrational eigenvalue).
  - `core_algos/types.py`: minimal typing/protocols.

- `src/dependencies/` (backend adapters + Hessian utilities)
  - `dependencies/calculators.py`:
    - `make_hip_predict_fn(calculator)`
    - `make_scine_predict_fn(scine_calculator)`
  - `dependencies/pyg_batch.py`: converts `(coords, atomic_nums)` into a PyG `Batch` in the layout HIP expects.
  - `dependencies/hessian.py`:
    - `project_hessian_remove_rigid_modes(...)`: mass-weight + Eckart projection.
    - `vibrational_eigvals(...)`: canonical “project then drop rigid modes” vibrational spectrum helper.

- `src/logging/` (standardized plotting + W&B import path)
  - `logging/trajectory_plots.py`: the **standard 3×2 trajectory plot** (`plot_gad_trajectory_3x2`).
  - `logging/wandb.py`: re-exports W&B helpers from `experiment_logger.py` so “logging API” has a stable import path.

- `src/runners/` (cluster-facing entrypoints)
  - Thin CLI wrappers meant for `python -m src.runners.<...>`.
  - These coordinate:
    - dataset + calculator creation (`setup_experiment`)
    - algorithm core (`core_algos/*`)
    - standardized logging (`logging/*` + `experiment_logger.py`)

### `scripts/` (cluster SLURM scripts)

- `scripts/killarney/`: original cluster scripts that call legacy monolithic modules.
- `scripts/killarney2/`: “refactor-aware” scripts that call `src.runners.*` where available.

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

---

## Hessian projection + rigid mode removal (critical)

Many TS heuristics use the smallest Hessian eigenvalues. Raw Hessians contain rigid-body modes:

- Nonlinear molecules: 6 rigid modes (3 translations + 3 rotations)
- Linear molecules: 5 rigid modes (3 translations + 2 rotations)

The canonical pipeline in this repo is:

1. **Mass-weight + Eckart project** the Hessian:
   - `src/dependencies/hessian.py:project_hessian_remove_rigid_modes(...)`
   - Uses `src/differentiable_projection.py:differentiable_massweigh_and_eckartprojection_torch`.

2. **Remove remaining rigid eigenvalues** from the projected Hessian spectrum:
   - `src/common_utils.py:extract_vibrational_eigenvalues(...)`
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
   - Centralized in `src/experiment_logger.py`.
   - Imported via `src/logging/wandb.py` to keep runner imports stable.

### Standard trajectory plot

The standard plot is the 3×2 grid extracted from the legacy Euler script:

- `src/logging/trajectory_plots.py:plot_gad_trajectory_3x2(...)`

It expects a trajectory dict with keys like:

- `energy`, `force_mean`, `eig0`, `eig1`, `eig_product`, `disp_from_last`, `disp_from_start`

---

## Runners (what Killarney2 uses)

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

---

## SLURM scripts: `killarney` vs `killarney2`

### Updated in `scripts/killarney2/`

- Euler GAD:
  - `gad_euler.slurm` → `python -m src.runners.gad_euler_core`
  - `scine_gad_euler.slurm` → `python -m src.runners.gad_euler_core --calculator scine`

- HIP eigenvalue descent:
  - `run_eigen_optimization.slurm` → `python -m src.runners.eigenvalue_descent_core`
  - Note: legacy-only flags like `--use-line-search` are intentionally removed in `killarney2`.

### Not yet refactored (still calls legacy modules)

Some scripts contain features that have **not** been moved into `core_algos/` yet (e.g., hybrid RK45 switching, BFGS fallbacks, frequency analysis tooling). Those remain pointed at the legacy entrypoints for now.

---

## “Weird dependencies” / gotchas

### HIP checkpoint loading patch

`src/common_utils.py` monkey-patches HIP’s dataset path resolver:

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

Even if `--device cuda` is set, SCINE calculations run on CPU (and the setup code prints a note about this).

---

## Practical: running locally vs on the cluster

On Killarney, the `scripts/killarney2/*.slurm` scripts are the source of truth.

If you run locally, you’ll need the same scientific stack (PyTorch, torch-geometric, HIP, transition1x, etc.). The repo is primarily HPC-oriented.
