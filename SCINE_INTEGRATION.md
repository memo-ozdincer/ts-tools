# SCINE Sparrow Integration

This document describes the integration of SCINE Sparrow analytical forcefields as an alternative to the HIP machine-learned force field for transition state search algorithms.

## Overview

The code now supports two calculator backends:
- **HIP** (default): Machine-learned Equiformer model from the HIP package
- **SCINE**: Analytical semiempirical methods (DFTB0, PM6, AM1, etc.)

Both calculators provide the same interface, so all GAD algorithms work seamlessly with either backend.

## Installation

### Prerequisites

The following packages need to be installed in your virtual environment:

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install SCINE packages
uv pip install scine-utilities scine-sparrow

# Or with pip:
# pip install scine-utilities scine-sparrow
```

### Verification

To verify the installation:

```python
import scine_utilities
import scine_sparrow
print("SCINE installation successful!")
```

## Available Functionals

SCINE Sparrow supports the following semiempirical methods:

| Functional | Description | Speed | Accuracy |
|------------|-------------|-------|----------|
| **DFTB0** | Density Functional Tight Binding (zeroth order) | Fast | Moderate |
| **DFTB2** | DFTB with self-consistent charges | Medium | Good |
| **DFTB3** | DFTB with improved hydrogen bonding | Medium | Better |
| **PM6** | Parametric Method 6 | Fast | Good for organic molecules |
| **AM1** | Austin Model 1 | Fast | Good for organic molecules |
| **RM1** | Recife Model 1 | Fast | Good for organic molecules |
| **MNDO** | Modified Neglect of Diatomic Overlap | Fast | Moderate |

**Recommendation**: Start with **DFTB0** for speed, or **PM6** for better accuracy on organic molecules.

## Usage

### Command-Line Arguments

Two new arguments have been added to all GAD scripts:

```bash
--calculator scine           # Use SCINE instead of HIP (default: hip)
--scine-functional DFTB0     # Which SCINE method to use (default: DFTB0)
```

### Example 1: GAD Euler with SCINE DFTB0

```bash
python -m src.gad_gad_euler_rmsd \
    --calculator scine \
    --scine-functional DFTB0 \
    --h5-path /path/to/transition1x.h5 \
    --out-dir ./output \
    --max-samples 20 \
    --n-steps 150 \
    --start-from midpoint_rt \
    --stop-at-ts \
    --dt 0.001
```

### Example 2: Eigenvalue Descent with SCINE PM6

```bash
python -m src.gad_eigenvalue_descent \
    --calculator scine \
    --scine-functional PM6 \
    --h5-path /path/to/transition1x.h5 \
    --out-dir ./output \
    --max-samples 30 \
    --start-from reactant \
    --loss-type eig_product \
    --use-bfgs \
    --bfgs-maxiter 100
```

### Example 3: GAD Euler with HIP (default behavior)

```bash
# These two are equivalent:
python -m src.gad_gad_euler_rmsd --checkpoint-path hip_v2.ckpt ...
python -m src.gad_gad_euler_rmsd --calculator hip --checkpoint-path hip_v2.ckpt ...
```

## SLURM Scripts

Example SLURM scripts have been provided:

### Killarney Cluster

- `scripts/killarney/scine_gad_euler.slurm` - GAD Euler with SCINE
- `scripts/killarney/scine_eigen_optimization.slurm` - Eigenvalue descent with SCINE

To submit:

```bash
sbatch scripts/killarney/scine_gad_euler.slurm
sbatch scripts/killarney/scine_eigen_optimization.slurm
```

### Key Differences from HIP Scripts

1. **No GPU required**: SCINE runs on CPU only
   - Replace `--gres=gpu:l40s:1` with `--cpus-per-task=8`

2. **No checkpoint path needed**: SCINE functionals don't require model checkpoints
   - Remove `--checkpoint-path` argument

3. **Add calculator arguments**:
   - `--calculator scine`
   - `--scine-functional DFTB0` (or PM6, AM1, etc.)

## Performance Considerations

### Speed Comparison

- **HIP**: ~0.5-2 seconds per geometry (GPU-accelerated)
- **SCINE DFTB0**: ~0.1-0.5 seconds per geometry (CPU, depends on molecule size)
- **SCINE PM6**: ~0.2-1 second per geometry (CPU)

### Parallelization

SCINE calculations are single-threaded per geometry but can be parallelized across molecules in the dataset. The calculator automatically sets threading environment variables to avoid oversubscription.

### Resource Requirements

For typical transition state search runs:
- **Memory**: 16-32 GB (same as HIP)
- **CPUs**: 4-8 cores recommended
- **GPU**: Not needed
- **Time**: Similar wall-clock time to HIP for small-medium molecules

## Output and Logging

All output formats remain the same:
- **Plots**: Same trajectory plots as HIP runs (energy, forces, eigenvalues, etc.)
- **JSON results**: Same format for all runs
- **W&B logging**: Fully supported (use `--wandb` flag)

The only difference is the calculator metadata in the output files.

## Limitations

1. **CPU-only**: SCINE doesn't support GPU acceleration
2. **Element support**: Limited to elements 1-20 (H through Ca) by default
   - Can be extended by adding more elements to `Z_TO_ELEMENT_TYPE` in `src/scine_calculator.py`
3. **Accuracy**: Semiempirical methods are less accurate than DFT or high-quality ML models
4. **Hessian quality**: Analytical Hessians from SCINE may be noisier than ML-predicted Hessians

## Troubleshooting

### Error: "Calculator 'XXX' not found"

The requested functional is not available. Check the spelling and use one of:
`DFTB0`, `DFTB2`, `DFTB3`, `PM6`, `AM1`, `RM1`, `MNDO`

### Error: "Unsupported element with Z=XX"

The molecule contains an element not in the mapping. Add it to `Z_TO_ELEMENT_TYPE` in `src/scine_calculator.py`:

```python
Z_TO_ELEMENT_TYPE = {
    1: scine_utilities.ElementType.H,
    6: scine_utilities.ElementType.C,
    # Add more elements here...
    21: scine_utilities.ElementType.Sc,  # Example
}
```

### Module Not Found: scine_sparrow

Install SCINE packages:
```bash
source .venv/bin/activate
uv pip install scine-utilities scine-sparrow
```

## Comparison: HIP vs SCINE

| Feature | HIP | SCINE |
|---------|-----|-------|
| Backend | Machine learning (Equiformer) | Semiempirical quantum chemistry |
| Hardware | GPU (CUDA) | CPU only |
| Speed | Fast (GPU) | Fast (CPU, depends on method) |
| Accuracy | High (trained on DFT) | Moderate (analytical approximations) |
| Generalization | Good within training distribution | Good for organic chemistry |
| Hessian | Predicted (smooth) | Analytical (may be noisy) |
| Setup | Requires trained checkpoint | No training needed |
| Checkpoint | hip_v2.ckpt (~500 MB) | None needed |

## Use Cases

### When to use SCINE:
- ✅ Quick prototyping and testing
- ✅ Large-scale screening where speed matters
- ✅ Exploring new starting geometries or noise levels
- ✅ Comparing ML predictions to analytical baselines
- ✅ CPU-only clusters or debugging on local machines
- ✅ Molecules well-represented by semiempirical methods

### When to use HIP:
- ✅ High-accuracy TS searches
- ✅ Final production runs for publications
- ✅ Molecules outside typical organic chemistry
- ✅ When GPU resources are available
- ✅ Benchmarking against known DFT results

## Implementation Details

### Calculator Interface

Both calculators implement the same interface:

```python
calculator.predict(batch, do_hessian=True)
# Returns:
# {
#     "energy": torch.Tensor,      # (1,) in eV
#     "forces": torch.Tensor,      # (N, 3) in eV/Å
#     "hessian": torch.Tensor,     # (3N, 3N) in eV/Å² (if do_hessian=True)
# }
```

### Unit Conversions

SCINE internally uses Hartree/Bohr, but the wrapper automatically converts to eV/Å to match HIP:

- Energy: Hartree → eV (×27.211386245988)
- Forces: -Gradients in Hartree/Bohr → Forces in eV/Å
- Hessian: Hartree/Bohr² → eV/Å²

### Thread Safety

The wrapper sets single-threaded execution for BLAS/LAPACK to avoid conflicts with PyTorch's threading.

## Citation

If you use SCINE in your research, please cite:

```
@article{scine,
  title={SCINE—Software for chemical interaction networks},
  author={Bensberg, Moritz and Reiher, Markus},
  journal={The Journal of Chemical Physics},
  volume={155},
  number={10},
  pages={104104},
  year={2021},
  publisher={AIP Publishing LLC}
}
```

## Support

For issues specific to the SCINE integration:
- Check this documentation first
- Verify SCINE installation: `python -c "import scine_sparrow"`
- Test with the simple water molecule example in `test_scine_calculator.py`

For general SCINE questions:
- https://scine.ethz.ch/
- https://github.com/qcscine/
