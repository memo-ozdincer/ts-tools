# Starting Geometry Eigenvalue Statistics

## Overview

This tool generates and analyzes starting geometries with various noise levels to understand the distribution of negative eigenvalues (imaginary frequencies). The goal is to find noise levels that produce geometries with multiple negative eigenvalues (higher-order saddle points), which are useful for testing transition state search algorithms.

## Motivation

Most transition state search methods assume you're starting from a first-order saddle point (exactly 1 negative eigenvalue). However, in practice, random starting points may have:
- 0 negative eigenvalues (local minimum)
- 1 negative eigenvalue (first-order saddle / TS)
- 2+ negative eigenvalues (higher-order saddle)

Understanding this distribution helps design robust algorithms that can handle various starting conditions.

## What It Does

The script:
1. **Generates starting geometries** from:
   - Base geometries: reactant, TS, midpoint, quarter-point, three-quarter-point
   - Noisy versions: adds Gaussian noise with specified RMS displacement
2. **Computes Hessian eigenvalues** at each geometry
3. **Removes 6 rigid-body modes** (3 translations + 3 rotations) using Eckart projection
4. **Counts negative eigenvalues** in the vibrational subspace
5. **Generates statistics and plots**

## Usage

### On Compute Canada (SLURM)

```bash
sbatch scripts/run_starting_geom_stats.slurm
```

### Running Directly

```bash
python -m src.starting_geometry_stats \
    --max-samples 30 \
    --noise-levels 0.5 1.0 2.0 5.0 10.0 \
    --n-samples-per-noise 5
```

### Command-Line Arguments

- `--max-samples`: Number of molecules to analyze (default: 30)
- `--noise-levels`: List of RMS noise levels in Angstroms (default: [0.5, 1.0, 2.0, 5.0, 10.0])
- `--n-samples-per-noise`: Number of random samples per noise level per base geometry (default: 5)
- Standard arguments: `--h5-path`, `--checkpoint-path`, `--out-dir`, `--device`

## Output

### Files

All output is saved to `{out_dir}/starting_geometry_stats/`:

1. **`all_geometry_stats.json`**: Complete results including:
   - Configuration (noise levels, sample counts)
   - Per-geometry results (eigenvalue counts, success/failure)
   - Per-molecule summary statistics

2. **`starting_geom_stats_{idx:03d}_{formula}.png`**: Per-molecule plots showing:
   - Distribution of negative eigenvalue counts
   - Negative eigenvalues vs noise level

3. **`starting_geom_summary_stats.png`**: Summary across all molecules:
   - Overall distribution histogram
   - Mean negative eigenvalues vs noise
   - Violin plots by noise level
   - Percentage of higher-order saddle points vs noise

### Interpreting Results

#### Key Metrics

- **Negative eigenvalue count**: Number of imaginary frequencies (goal: >1 for higher-order saddles)
- **Mean ± Std by noise level**: Shows typical behavior at each noise level
- **% Higher-order**: Fraction of geometries with >1 negative eigenvalue
- **Distribution**: Shows full histogram of eigenvalue counts

#### Typical Distances

From the CLAUDE.md:
> Usually the distances between TS and reactant is a few Å, usually less honestly.

So expect:
- **0.5-2 Å noise**: Small perturbations around stable structures
- **2-5 Å noise**: Medium perturbations, may cross into different basins
- **5-10+ Å noise**: Large perturbations, highly distorted geometries

#### What to Look For

**Good noise level** for generating higher-order saddle points:
- Mean negative eigenvalue count: **>1.5**
- % Higher-order: **>50%**
- Distribution: Spread across multiple eigenvalue counts (not all 0 or 1)

**Too little noise**:
- Most geometries have 0-1 negative eigenvalues
- Stuck near local minima or first-order saddles

**Too much noise**:
- Calculator may fail (atoms too far apart)
- Unphysical geometries

## Example Workflow

1. **Initial scan** with default noise levels:
   ```bash
   python -m src.starting_geometry_stats \
       --max-samples 30 \
       --noise-levels 0.5 1.0 2.0 5.0 10.0 \
       --n-samples-per-noise 5
   ```

2. **Analyze results**: Look at summary plots and JSON
   - Which noise level gives most higher-order saddles?
   - What's the typical eigenvalue distribution?

3. **Refine search** around optimal noise level:
   ```bash
   python -m src.starting_geometry_stats \
       --max-samples 50 \
       --noise-levels 1.5 2.0 2.5 3.0 3.5 4.0 \
       --n-samples-per-noise 10
   ```

4. **Generate dataset** using optimal noise level for future experiments

## Implementation Details

### Noise Generation

Gaussian noise is added per-atom, per-coordinate:
```python
noise = torch.randn_like(coords) * noise_rms_angstrom
noisy_coords = coords + noise
```

This gives RMS displacement ≈ `noise_rms_angstrom`.

### Eigenvalue Analysis

1. Compute Hessian using Equiformer model
2. Mass-weight Hessian: `H_mw = M^(-1/2) H M^(-1/2)`
3. Project out rigid-body modes using Eckart projection: `H_proj = P H_mw P`
4. Diagonalize: `H_proj v = λ v`
5. Remove 6 smallest eigenvalues (by absolute value)
6. Count negatives in remaining vibrational eigenvalues

### Why Remove 6 Modes?

- 3 translations + 3 rotations are **not** molecular vibrations
- These modes have zero (or near-zero) frequencies
- Must be removed before analyzing vibrational spectrum
- For linear molecules, only 5 modes (3 trans + 2 rot)

## Related Files

- **`src/starting_geometry_stats.py`**: Main implementation
- **`src/gad_eigenvalue_descent.py`**: Uses similar starting geometries for optimization
- **`src/differentiable_projection.py`**: Eckart projection implementation
- **`src/common_utils.py`**: Shared dataset and setup utilities
- **`scripts/run_starting_geom_stats.slurm`**: SLURM job script

## Future Extensions

Potential improvements:
- Add support for generating geometries along reaction paths
- Include energy and force statistics
- Support for other noise distributions (uniform, biased towards TS)
- Parallel processing for faster computation
- Interactive visualization of eigenvalue distributions
