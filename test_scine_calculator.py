#!/usr/bin/env python
"""
Simple test script to verify SCINE Sparrow calculator integration.
"""
import torch
from torch_geometric.data import Data as TGData, Batch

from src.scine_calculator import create_scine_calculator


def test_scine_calculator():
    """Test SCINE calculator with a simple water molecule."""
    print("=" * 60)
    print("Testing SCINE Sparrow Calculator Integration")
    print("=" * 60)

    # Create a simple water molecule (H2O)
    # Positions in Angstroms (bent geometry)
    positions = torch.tensor([
        [0.0, 0.0, 0.0],      # O
        [0.96, 0.0, 0.0],     # H
        [-0.24, 0.93, 0.0],   # H
    ], dtype=torch.float32)

    atomic_nums = torch.tensor([8, 1, 1], dtype=torch.long)  # O, H, H

    # Create PyG Data object
    data = TGData(
        pos=positions,
        z=atomic_nums,
        charges=atomic_nums,
        natoms=torch.tensor([3], dtype=torch.long),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool),
    )

    # Create batch
    batch = Batch.from_data_list([data])

    # Test with different functionals
    functionals_to_test = ["DFTB0", "PM6"]

    for functional in functionals_to_test:
        print(f"\n--- Testing functional: {functional} ---")
        try:
            # Create calculator
            calculator = create_scine_calculator(functional=functional, device="cpu")

            # Test prediction WITH Hessian
            print("  Computing energy, forces, and Hessian...")
            results_with_hess = calculator.predict(batch, do_hessian=True)

            # Check results
            assert "energy" in results_with_hess, "Missing energy in results"
            assert "forces" in results_with_hess, "Missing forces in results"
            assert "hessian" in results_with_hess, "Missing hessian in results"

            energy = results_with_hess["energy"]
            forces = results_with_hess["forces"]
            hessian = results_with_hess["hessian"]

            print(f"    Energy shape: {energy.shape}, value: {energy.item():.6f} eV")
            print(f"    Forces shape: {forces.shape}")
            print(f"    Forces (eV/Å):")
            for i, force in enumerate(forces):
                print(f"      Atom {i}: [{force[0]:.6f}, {force[1]:.6f}, {force[2]:.6f}]")
            print(f"    Hessian shape: {hessian.shape}")

            # Check shapes
            assert energy.shape == (1,), f"Expected energy shape (1,), got {energy.shape}"
            assert forces.shape == (3, 3), f"Expected forces shape (3, 3), got {forces.shape}"
            assert hessian.shape == (9, 9), f"Expected hessian shape (9, 9), got {hessian.shape}"

            # Check types
            assert isinstance(energy, torch.Tensor), "Energy should be a torch.Tensor"
            assert isinstance(forces, torch.Tensor), "Forces should be a torch.Tensor"
            assert isinstance(hessian, torch.Tensor), "Hessian should be a torch.Tensor"

            # Test prediction WITHOUT Hessian
            print("  Computing energy and forces only...")
            results_no_hess = calculator.predict(batch, do_hessian=False)

            assert "energy" in results_no_hess, "Missing energy in results"
            assert "forces" in results_no_hess, "Missing forces in results"
            assert "hessian" not in results_no_hess, "Hessian should not be computed when do_hessian=False"

            print(f"    Energy: {results_no_hess['energy'].item():.6f} eV")
            print(f"    Forces shape: {results_no_hess['forces'].shape}")

            # Test that calculate() alias works
            print("  Testing calculate() alias...")
            results_calc = calculator.calculate(batch, do_hessian=True)
            assert "energy" in results_calc, "calculate() alias should work"

            # Test potential property
            print("  Testing potential property...")
            potential = calculator.potential
            assert hasattr(potential, 'device'), "Potential should have device attribute"
            print(f"    Potential device: {potential.device}")

            print(f"  ✓ {functional} tests passed!")

        except Exception as e:
            print(f"  ✗ {functional} test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_scine_calculator()
    exit(0 if success else 1)
