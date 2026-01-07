#!/usr/bin/env python
"""
Diagnostic script to debug HIP + Sella issues.

Tests:
1. HIP Hessian consistency: Compare predicted Hessian with finite-difference Hessian from forces
2. Internal vs Cartesian coordinates: See if internal coordinate conversion is the issue
3. Trust ratio tracking: Monitor rho values during optimization
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from ase import Atoms
from dependencies.common_utils import load_calculator, Transition1xDataset
from dependencies.pyg_batch import coords_to_pyg_batch


def compute_numerical_hessian(calculator, atoms, delta=1e-4):
    """Compute Hessian via finite differences of forces."""
    n_atoms = len(atoms)
    n_dof = 3 * n_atoms
    hessian = np.zeros((n_dof, n_dof))

    positions = atoms.get_positions().copy()
    device = next(calculator.potential.parameters()).device
    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)

    for i in range(n_dof):
        atom_idx = i // 3
        coord_idx = i % 3

        # +delta
        pos_plus = positions.copy()
        pos_plus[atom_idx, coord_idx] += delta
        coords_plus = torch.tensor(pos_plus, dtype=torch.float32, device=device)
        batch_plus = coords_to_pyg_batch(coords_plus, z, device=device)
        with torch.no_grad():
            result_plus = calculator.predict(batch_plus, do_hessian=False)
        forces_plus = result_plus["forces"].detach().cpu().numpy().flatten()

        # -delta
        pos_minus = positions.copy()
        pos_minus[atom_idx, coord_idx] -= delta
        coords_minus = torch.tensor(pos_minus, dtype=torch.float32, device=device)
        batch_minus = coords_to_pyg_batch(coords_minus, z, device=device)
        with torch.no_grad():
            result_minus = calculator.predict(batch_minus, do_hessian=False)
        forces_minus = result_minus["forces"].detach().cpu().numpy().flatten()

        # Central difference: H_ij = -dF_j/dx_i
        hessian[i, :] = -(forces_plus - forces_minus) / (2 * delta)

    # Symmetrize
    hessian = 0.5 * (hessian + hessian.T)
    return hessian


def test_hessian_consistency(calculator, atoms, verbose=True):
    """Compare HIP's predicted Hessian with numerical Hessian from forces."""
    positions = atoms.get_positions()
    device = next(calculator.potential.parameters()).device

    coords = torch.tensor(positions, dtype=torch.float32, device=device)
    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    batch = coords_to_pyg_batch(coords, z, device=device)

    # Get predicted Hessian
    with torch.no_grad():
        result = calculator.predict(batch, do_hessian=True)

    energy = result["energy"].item()
    forces = result["forces"].detach().cpu().numpy()
    hessian_pred = result["hessian"].detach().cpu().numpy()

    n_atoms = len(atoms)
    hessian_pred = hessian_pred.reshape(3 * n_atoms, 3 * n_atoms)

    # Compute numerical Hessian
    print("Computing numerical Hessian (this may take a moment)...")
    hessian_num = compute_numerical_hessian(calculator, atoms, delta=1e-4)

    # Compare
    diff = hessian_pred - hessian_num
    max_diff = np.abs(diff).max()
    mean_diff = np.abs(diff).mean()
    frobenius_norm_diff = np.linalg.norm(diff, 'fro')
    frobenius_norm_pred = np.linalg.norm(hessian_pred, 'fro')
    relative_error = frobenius_norm_diff / frobenius_norm_pred

    # Eigenvalue comparison
    eigvals_pred, _ = np.linalg.eigh(hessian_pred)
    eigvals_num, _ = np.linalg.eigh(hessian_num)

    if verbose:
        print("\n" + "="*60)
        print("HIP HESSIAN CONSISTENCY CHECK")
        print("="*60)
        print(f"Energy: {energy:.6f} eV")
        print(f"Max force: {np.abs(forces).max():.6f} eV/Å")
        print(f"\nHessian comparison:")
        print(f"  Max absolute diff: {max_diff:.6e}")
        print(f"  Mean absolute diff: {mean_diff:.6e}")
        print(f"  Relative Frobenius error: {relative_error:.4%}")
        print(f"\nEigenvalue comparison (first 10):")
        print(f"  Predicted: {eigvals_pred[:10]}")
        print(f"  Numerical: {eigvals_num[:10]}")
        print(f"  Diff:      {eigvals_pred[:10] - eigvals_num[:10]}")
        print(f"\nNumber of negative eigenvalues:")
        print(f"  Predicted: {np.sum(eigvals_pred < -0.01)}")
        print(f"  Numerical: {np.sum(eigvals_num < -0.01)}")

    return {
        "energy": energy,
        "forces": forces,
        "hessian_pred": hessian_pred,
        "hessian_num": hessian_num,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "relative_error": relative_error,
        "eigvals_pred": eigvals_pred,
        "eigvals_num": eigvals_num,
    }


def test_sella_cartesian_vs_internal(calculator, atoms, max_steps=50):
    """Compare Sella optimization with internal=True vs internal=False."""
    from sella import Sella
    from experiments.Sella.ase_calculators import create_ase_calculator, create_hessian_function

    device = str(next(calculator.potential.parameters()).device)
    ase_calc = create_ase_calculator(calculator, "hip", device=device)
    hess_fn = create_hessian_function(calculator, "hip", device=device)

    results = {}

    for use_internal in [False, True]:
        label = "internal" if use_internal else "cartesian"
        print(f"\n{'='*60}")
        print(f"Testing Sella with {label} coordinates")
        print("="*60)

        test_atoms = atoms.copy()
        test_atoms.calc = ase_calc

        # Track values during optimization
        energies = []
        fmax_values = []
        rho_values = []

        try:
            opt = Sella(
                test_atoms,
                internal=use_internal,
                hessian_function=hess_fn,
                order=1,  # saddle point
                delta0=0.1,
                diag_every_n=1,
                trajectory=None,
                logfile="-",
            )

            # Monkey-patch to capture rho values
            original_step = opt.step
            def patched_step():
                original_step()
                energies.append(test_atoms.get_potential_energy())
                fmax_values.append(np.abs(test_atoms.get_forces()).max())
                rho_values.append(opt.rho)
            opt.step = patched_step

            opt.run(fmax=0.03, steps=max_steps)
            converged = opt.converged()

        except Exception as e:
            print(f"Optimization failed: {e}")
            converged = False

        results[label] = {
            "converged": converged,
            "energies": energies,
            "fmax_values": fmax_values,
            "rho_values": rho_values,
            "n_steps": len(energies),
        }

        print(f"\nResults ({label}):")
        print(f"  Converged: {converged}")
        print(f"  Steps: {len(energies)}")
        if len(energies) > 0:
            print(f"  Energy: {energies[0]:.4f} -> {energies[-1]:.4f}")
            print(f"  fmax: {fmax_values[0]:.4f} -> {fmax_values[-1]:.4f}")
            if len(rho_values) > 0 and rho_values[0] is not None:
                rho_arr = np.array([r for r in rho_values if r is not None])
                print(f"  rho range: {rho_arr.min():.4f} to {rho_arr.max():.4f}")
                print(f"  rho mean: {rho_arr.mean():.4f}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="HIP checkpoint path")
    parser.add_argument("--h5-path", type=str, default=None, help="Path to Transition1x HDF5")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to test")
    parser.add_argument("--test", type=str, default="all",
                       choices=["hessian", "sella", "all"], help="Which test to run")
    args = parser.parse_args()

    # Load calculator
    print("Loading HIP calculator...")
    calculator = load_calculator(args.checkpoint)

    # Load sample
    if args.h5_path:
        print(f"Loading sample {args.sample_idx} from {args.h5_path}...")
        dataset = Transition1xDataset(args.h5_path, split="test", max_samples=args.sample_idx + 1)
        sample = dataset[args.sample_idx]
        positions = sample["ts_positions"].numpy()
        atomic_numbers = sample["atomic_numbers"].numpy()
    else:
        # Use a simple test molecule
        print("Using test water molecule...")
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.587],
            [0.0, -0.757, 0.587],
        ])
        atomic_numbers = np.array([8, 1, 1])

    atoms = Atoms(numbers=atomic_numbers, positions=positions)
    print(f"Molecule: {atoms.get_chemical_formula()}, {len(atoms)} atoms")

    # Run tests
    if args.test in ["hessian", "all"]:
        results = test_hessian_consistency(calculator, atoms)

    if args.test in ["sella", "all"]:
        results = test_sella_cartesian_vs_internal(calculator, atoms)


if __name__ == "__main__":
    main()
