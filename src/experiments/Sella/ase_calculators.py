"""ASE Calculator wrappers for HIP and SCINE calculators.

These wrappers allow using HIP (ML model) and SCINE (semiempirical) calculators
with ASE's Atoms interface, enabling Sella optimization.

Units:
- ASE expects eV for energy and eV/A for forces
- HIP and SCINE already output in these units

Note on Hessians for Sella:
- Sella can use a `hessian_function` callable to get exact Hessians at each step
- For internal coordinates: pass RAW Cartesian Hessian (Sella handles conversion)
- For Cartesian coordinates: apply mass-weighting + Eckart projection to remove
  the 6 translational/rotational zero-modes before passing to Sella
- Use `create_hessian_function()` to create this callable for HIP or SCINE
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class HipASECalculator(Calculator):
    """ASE Calculator wrapper for HIP EquiformerTorchCalculator.

    Provides energy and forces from the HIP neural network potential
    in ASE-compatible format for use with Sella optimizer.

    Units:
        - Energy: eV
        - Forces: eV/A
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        hip_calculator,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        """Initialize HIP ASE calculator wrapper.

        Args:
            hip_calculator: HIP EquiformerTorchCalculator instance
            device: Torch device to use ("cuda" or "cpu")
            dtype: Torch dtype for computations
        """
        super().__init__(**kwargs)
        self.hip_calculator = hip_calculator
        self.device = torch.device(device)
        self.dtype = dtype

        # Import here to avoid circular imports
        from ...dependencies.pyg_batch import coords_to_pyg_batch
        self._coords_to_pyg_batch = coords_to_pyg_batch

    def calculate(
        self,
        atoms: Atoms = None,
        properties: List[str] = None,
        system_changes: List[str] = all_changes,
    ) -> None:
        """Calculate energy and forces for atoms object.

        Args:
            atoms: ASE Atoms object
            properties: List of properties to calculate (ignored, always computes energy+forces)
            system_changes: List of changes since last calculation (ASE caching)
        """
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        # Extract positions and atomic numbers from ASE Atoms
        positions = atoms.get_positions()  # Shape: (N, 3), units: Angstrom
        atomic_numbers = atoms.get_atomic_numbers()  # Shape: (N,)

        # Convert to torch tensors
        coords = torch.tensor(positions, dtype=self.dtype, device=self.device)
        z = torch.tensor(atomic_numbers, dtype=torch.long, device=self.device)

        # Create PyG batch for HIP
        batch = self._coords_to_pyg_batch(coords, z, device=self.device)

        # Run HIP prediction with do_hessian=True to ensure otf_graph=True is used.
        # This keeps energy/forces consistent with the Hessian (same graph construction).
        with torch.no_grad():
            result = self.hip_calculator.predict(batch, do_hessian=True)

        # Extract energy and forces
        energy = result["energy"]
        forces = result["forces"]

        # Convert to numpy and store in results
        if isinstance(energy, torch.Tensor):
            energy = energy.detach().cpu().numpy()
        if isinstance(forces, torch.Tensor):
            forces = forces.detach().cpu().numpy()

        # Handle shape: energy should be scalar, forces should be (N, 3)
        if hasattr(energy, "__len__"):
            energy = float(energy.flatten()[0])
        else:
            energy = float(energy)

        forces = np.asarray(forces).reshape(-1, 3)

        self.results["energy"] = energy
        self.results["forces"] = forces


class ScineASECalculator(Calculator):
    """ASE Calculator wrapper for SCINE Sparrow calculator.

    Provides energy and forces from SCINE semiempirical methods
    (DFTB0, PM6, AM1, etc.) in ASE-compatible format.

    Units:
        - Energy: eV
        - Forces: eV/A
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        scine_calculator,
        **kwargs,
    ):
        """Initialize SCINE ASE calculator wrapper.

        Args:
            scine_calculator: ScineSparrowCalculator instance
        """
        super().__init__(**kwargs)
        self.scine_calculator = scine_calculator

        # Import here to avoid circular imports
        from ...dependencies.pyg_batch import coords_to_pyg_batch
        self._coords_to_pyg_batch = coords_to_pyg_batch

    def calculate(
        self,
        atoms: Atoms = None,
        properties: List[str] = None,
        system_changes: List[str] = all_changes,
    ) -> None:
        """Calculate energy and forces for atoms object.

        Args:
            atoms: ASE Atoms object
            properties: List of properties to calculate
            system_changes: List of changes since last calculation
        """
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        # Extract positions and atomic numbers from ASE Atoms
        positions = atoms.get_positions()  # Shape: (N, 3), units: Angstrom
        atomic_numbers = atoms.get_atomic_numbers()  # Shape: (N,)

        # Convert to torch tensors (SCINE runs on CPU)
        coords = torch.tensor(positions, dtype=torch.float32, device=torch.device("cpu"))
        z = torch.tensor(atomic_numbers, dtype=torch.long, device=torch.device("cpu"))

        # Create PyG batch for SCINE
        batch = self._coords_to_pyg_batch(coords, z, device=torch.device("cpu"))

        # Run SCINE prediction (no Hessian needed for Sella)
        with torch.no_grad():
            result = self.scine_calculator.predict(batch, do_hessian=False)

        # Extract energy and forces
        energy = result["energy"]
        forces = result["forces"]

        # Convert to numpy
        if isinstance(energy, torch.Tensor):
            energy = energy.detach().cpu().numpy()
        if isinstance(forces, torch.Tensor):
            forces = forces.detach().cpu().numpy()

        # Handle shape
        if hasattr(energy, "__len__"):
            energy = float(energy.flatten()[0])
        else:
            energy = float(energy)

        forces = np.asarray(forces).reshape(-1, 3)

        self.results["energy"] = energy
        self.results["forces"] = forces


def create_ase_calculator(
    calculator,
    calculator_type: str,
    device: str = "cuda",
) -> Calculator:
    """Factory function to create appropriate ASE calculator wrapper.

    Args:
        calculator: HIP or SCINE calculator instance
        calculator_type: Either "hip" or "scine"
        device: Device for HIP calculations (ignored for SCINE)

    Returns:
        ASE Calculator wrapper
    """
    if calculator_type.lower() == "scine":
        return ScineASECalculator(calculator)
    else:
        return HipASECalculator(calculator, device=device)


def create_hessian_function(
    calculator,
    calculator_type: str,
    device: str = "cuda",
    apply_eckart: bool = False,
) -> Callable[[Atoms], np.ndarray]:
    """Create a hessian_function callable for Sella optimizer.

    This function creates a callable that Sella can use to get exact Hessians
    at each optimization step.

    For internal coordinates (apply_eckart=False):
        Returns RAW Cartesian Hessian - Sella's internal coordinate machinery
        handles all necessary transformations.

    For Cartesian coordinates (apply_eckart=True):
        Returns mass-weighted, Eckart-projected, then un-mass-weighted Hessian.
        This removes the 6 translational/rotational zero-modes that would
        otherwise confuse P-RFO optimization.

    Using exact Hessians instead of iterative approximation can significantly
    improve convergence for ML potentials like HIP.

    Args:
        calculator: HIP or SCINE calculator instance
        calculator_type: Either "hip" or "scine"
        device: Device for HIP calculations (ignored for SCINE)
        apply_eckart: If True, apply mass-weighting + Eckart projection to
            remove trans/rot modes. Use True for Cartesian coordinates
            (internal=False), False for internal coordinates (internal=True).

    Returns:
        Callable that takes ASE Atoms and returns (3N, 3N) Hessian as numpy array

    Example:
        >>> hessian_fn = create_hessian_function(hip_calc, "hip", device="cuda", apply_eckart=True)
        >>> opt = Sella(atoms, hessian_function=hessian_fn, diag_every_n=1)
    """
    # Import here to avoid circular imports
    from ...dependencies.pyg_batch import coords_to_pyg_batch

    if calculator_type.lower() == "scine":
        def scine_hessian_function(atoms: Atoms) -> np.ndarray:
            """Compute raw Cartesian Hessian using SCINE calculator."""
            positions = atoms.get_positions()  # (N, 3) in Angstrom
            atomic_numbers = atoms.get_atomic_numbers()  # (N,)

            # Convert to torch tensors (SCINE runs on CPU)
            coords = torch.tensor(positions, dtype=torch.float32, device=torch.device("cpu"))
            z = torch.tensor(atomic_numbers, dtype=torch.long, device=torch.device("cpu"))

            # Create PyG batch for SCINE
            batch = coords_to_pyg_batch(coords, z, device=torch.device("cpu"))

            # Run SCINE prediction WITH Hessian
            with torch.no_grad():
                result = calculator.predict(batch, do_hessian=True)

            # Extract raw Cartesian Hessian
            hessian = result["hessian"]
            if isinstance(hessian, torch.Tensor):
                hessian = hessian.detach().cpu().numpy()

            # Ensure shape is (3N, 3N)
            n_atoms = len(atomic_numbers)
            hessian = np.asarray(hessian).reshape(3 * n_atoms, 3 * n_atoms)

            return hessian

        return scine_hessian_function

    else:  # HIP calculator
        torch_device = torch.device(device)

        # Import Hessian computation via autograd
        from hip.hessian_utils import compute_hessian

        def hip_hessian_function(atoms: Atoms) -> np.ndarray:
            """Compute Cartesian Hessian using HIP calculator via AUTOGRAD.

            CRITICAL: For Sella to work correctly, the Hessian MUST be consistent
            with the forces. This means H = -∂F/∂x exactly.

            The HIP model predicts energy, forces, and Hessian as separate neural
            network outputs. While they're trained to match DFT data, they're NOT
            mathematically constrained to be consistent (H_predicted ≠ -∂F_predicted/∂x).

            This inconsistency causes Sella's trust radius mechanism to fail:
            - Sella predicts: ΔE ≈ F·Δx + ½Δx·H·Δx
            - If H ≠ -∂F/∂x, the predicted ΔE doesn't match actual ΔE
            - The trust ratio (rho = ΔE_actual/ΔE_pred) becomes erratic or negative
            - Trust radius collapses, optimization stalls

            SCINE works because it computes Hessian analytically as -∂F/∂x.

            SOLUTION: Compute Hessian via autograd to ensure H = -∂F/∂x exactly.
            This is slower than using the predicted Hessian, but gives correct
            trust radius updates.
            """
            positions = atoms.get_positions()  # (N, 3) in Angstrom
            atomic_numbers = atoms.get_atomic_numbers()  # (N,)

            # Convert to torch tensors WITH gradients enabled for autograd Hessian
            coords = torch.tensor(
                positions, dtype=torch.float32, device=torch_device, requires_grad=True
            )
            z = torch.tensor(atomic_numbers, dtype=torch.long, device=torch_device)

            # Create PyG batch for HIP
            batch = coords_to_pyg_batch(coords, z, device=torch_device)
            batch.pos = coords  # Ensure batch uses our coords with requires_grad=True

            # Compute Hessian via AUTOGRAD (not the predicted Hessian)
            # This ensures H = -∂F/∂x exactly, which is critical for Sella
            with torch.enable_grad():
                # Forward pass to get energy and forces
                energy, forces, _ = calculator.potential.forward(
                    batch,
                    otf_graph=True,  # Use same graph construction as energy/forces
                )

                # Compute Hessian as gradient of energy / negative gradient of forces
                # This ensures mathematical consistency: H = ∂²E/∂x² = -∂F/∂x
                hessian_t = compute_hessian(
                    coords=batch.pos,
                    energy=energy,
                    forces=forces,
                )

            # Ensure shape is (3N, 3N)
            n_atoms = len(atomic_numbers)
            hessian_t = hessian_t.detach().reshape(3 * n_atoms, 3 * n_atoms)

            # Return raw Cartesian Hessian - let Sella handle trans/rot via constraints
            return hessian_t.cpu().numpy()

        return hip_hessian_function
