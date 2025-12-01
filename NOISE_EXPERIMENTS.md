# Noise Robustness Experiments

## Problem

TS search algorithms (eigenvalue descent, RK45-GAD, Euler-GAD) fail universally when starting from noisy geometries. The algorithms get stuck at higher-order saddle points (geometries with 2+ negative eigenvalues) instead of finding the true transition state (exactly 1 negative eigenvalue).

## Hypothesis

**Adaptive step sizing based on saddle order** (number of negative eigenvalues) will improve robustness to noisy starting geometries.

**Rationale**: Different saddle orders require different optimization strategies:
- **Higher-order saddles** (order-2, order-3+): Broad, flat regions where small steps wander aimlessly → need LARGE steps to "hop over" barriers to lower-order regions
- **Order-1 (TS candidate)**: Near the true transition state → need SMALL steps for precise convergence
- **Minima** (order-0): Local minimum → need NORMAL steps to continue searching

Current methods use fixed or geometry-agnostic step sizes, which are inappropriate across all regimes.

## Implementation

### Core Components

1. **Saddle Detection** (`src/saddle_detection.py`):
   - `classify_saddle_point()`: Classifies geometry based on vibrational eigenvalue spectrum
   - `compute_adaptive_step_scale()`: Returns step size multiplier based on saddle order

2. **Centralized Utilities** (`src/common_utils.py`):
   - `extract_vibrational_eigenvalues()`: Consistent eigenvalue extraction with rigid mode removal

3. **Adaptive Step Sizing Strategy**:
   - **Order-2 saddles**: 5× larger steps (default)
   - **Order-3 saddles**: 10× larger steps
   - **Order-4+ saddles**: 15×, 20×, etc. (scales linearly)
   - **Order-1 (TS)**: 0.5× smaller steps for refinement
   - **Order-0 (minimum)**: 1× normal steps

### Modified Files

#### 1. Eigenvalue Descent (`src/gad_eigenvalue_descent.py`) ✅ COMPLETE
- Added imports for saddle detection and centralized eigenvalue extraction
- New CLI flags:
  - `--adaptive-step-sizing`: Enable feature
  - `--higher-order-multiplier 5.0`: Multiplier for higher-order saddles
  - `--ts-multiplier 0.5`: Multiplier near TS
- Replaced duplicated eigenvalue extraction logic with `extract_vibrational_eigenvalues()`
- Added saddle classification after gradient computation
- Applied adaptive scaling to line search multipliers
- Tracking: `saddle_order`, `step_scale`, `classification` in history

#### 2. RK45-GAD (`src/gad_rk45_search.py`) ⏳ TODO
- Add imports and CLI flags
- Modify `RK45Solver` class to accept adaptive step sizing parameters
- Combine error-based step adjustment with saddle-based scaling
- Update saddle info during frequency analysis
- Track saddle order in trajectory

#### 3. Euler-GAD (`src/gad_gad_euler_rmsd.py`) ⏳ TODO
- Add imports and CLI flags
- Classify saddle order after frequency analysis
- Apply adaptive `dt` based on saddle order
- Track saddle order and step scale

#### 4. SLURM Scripts ⏳ TODO
- `scripts/run_eigen_optimization.slurm`
- `scripts/run_gad_rk45.slurm`
- `scripts/gad_euler.slurm`

Add `--adaptive-step-sizing` flag to enable the new behavior.

## Methods Being Tested

1. **Eigenvalue descent** with adaptive line search ✅
2. **RK45-GAD** with adaptive time stepping ⏳
3. **Euler-GAD** with adaptive dt ⏳

## Parameters

### Default Values
- `--higher-order-multiplier`: 5.0
- `--ts-multiplier`: 0.5

### Parameter Sweep (To Be Tested)
- `higher_order_multiplier`: [3.0, 5.0, 7.0, 10.0]
- `ts_multiplier`: [0.3, 0.5, 0.7]

### Test Noise Levels
- 0.5 Å RMS displacement
- 1.0 Å RMS displacement
- 2.0 Å RMS displacement
- 5.0 Å RMS displacement

### Starting Geometries
- `reactant_noise0.5A`
- `reactant_noise1A`
- `reactant_noise2A`
- `reactant_noise5A`
- (Also test: `midpoint_rt_noiseXA`, `three_quarter_rt_noiseXA`)

## Validation Criteria (No Cheating!)

Success measured **without** using RMSD to true TS (which wouldn't be available in real applications):

1. **Eigenvalue signature**:
   - λ₀ < -0.01 eV/Å² (clearly negative)
   - λ₁ > +0.01 eV/Å² (clearly positive)
   - Exactly 1 negative vibrational eigenvalue

2. **Force convergence**: ||F|| < 1e-3 eV/Å (stationary point)

3. **Stability test** (optional): Small perturbations (0.01 Å) should return to same geometry

## Expected Outcomes

### Baseline (Current, No Adaptive Sizing)
- **0.5-2Å noise**: ~0% success rate (stuck at higher-order saddles)
- **5Å+ noise**: ~0% success rate

### With Adaptive Step Sizing (Expected)
- **0.5-1Å noise**: ~50-70% success rate
- **2Å noise**: ~30-50% success rate
- **5Å noise**: ~10-20% success rate

## Testing Plan

### Phase 1: Single Molecule Validation ⏳
- Select 1-2 representative molecules
- Test noise levels: 0.5, 1.0, 2.0 Å
- Compare: baseline (no adaptive) vs. adaptive step sizing
- Metrics:
  - Success rate (found order-1 saddle)
  - Number of optimization steps
  - Saddle order evolution over trajectory
  - Final eigenvalue product

### Phase 2: Parameter Sweep ⏳
- Use best-performing molecules from Phase 1
- Test `higher_order_multiplier`: [3.0, 5.0, 7.0, 10.0]
- Test `ts_multiplier`: [0.3, 0.5, 0.7]
- Find optimal parameter settings

### Phase 3: Full Dataset Evaluation ⏳
- Run on 30 molecules with optimized parameters
- Test all noise levels: 0.5, 1.0, 2.0, 5.0 Å
- Measure:
  - Success rate vs. noise level
  - Average steps to convergence
  - Failure mode analysis (what saddle orders do failures get stuck at?)

## Results

[To be filled in after experiments are run]

## Conclusions

[To be filled in after analysis]

## Implementation Status

### ✅ Completed
- Created `noise-experiments` Git branch
- Created `src/saddle_detection.py` module
- Added `extract_vibrational_eigenvalues()` to `src/common_utils.py`
- Modified `src/gad_eigenvalue_descent.py` for adaptive step sizing
- Created this documentation file

### ⏳ In Progress / TODO
- Modify `src/gad_rk45_search.py` for adaptive step sizing
- Modify `src/gad_gad_euler_rmsd.py` for adaptive step sizing
- Update SLURM scripts with new CLI flags
- Run Phase 1 validation experiments
- Parameter sweep (Phase 2)
- Full dataset evaluation (Phase 3)

## Notes

### Why Not Eigenvector-Following?
Initial plan considered mode-following (moving along eigenvector directions to reduce saddle order). However, **PyTorch's eigendecomposition returns eigenvectors with inconsistent signs**, making eigenvector-following unreliable.

Adaptive step sizing based solely on eigenvalue count (saddle order) is:
- More robust (no sign ambiguity)
- Simpler to implement
- Still theoretically sound (larger steps help escape broad saddle regions)
- Leverages existing GAD directional intelligence

### Why Separate Branch?
This is an experimental feature being developed on the `noise-experiments` branch to allow:
- Controlled testing without affecting main workflow
- Easy comparison of baseline vs. adaptive methods
- Iterative refinement based on experimental results
- Clean merge to main only after validation
