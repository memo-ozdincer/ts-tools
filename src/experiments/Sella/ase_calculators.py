"""ASE Calculator wrappers for HIP and SCINE calculators.

These wrappers allow using HIP (ML model) and SCINE (semiempirical) calculators
with ASE's Atoms interface, enabling Sella optimization.

Units:
- ASE expects eV for energy and eV/A for forces
- HIP and SCINE already output in these units

Note on Hessians for Sella:
- Sella can use a `hessian_function` callable to get exact Hessians at each step
- The Hessian should be the RAW Cartesian Hessian (not mass-weighted, not Eckart-projected)
- Sella's internal coordinate machinery handles the conversion to internal coords
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
) -> Callable[[Atoms], np.ndarray]:
    """Create a hessian_function callable for Sella optimizer.

    This function creates a callable that Sella can use to get exact Hessians
    at each optimization step. The returned Hessian is the RAW Cartesian Hessian
    (not mass-weighted, not Eckart-projected) since Sella's internal coordinate
    machinery handles all necessary transformations.

    Using exact Hessians instead of iterative approximation can significantly
    improve convergence for ML potentials like HIP.

    Args:
        calculator: HIP or SCINE calculator instance
        calculator_type: Either "hip" or "scine"
        device: Device for HIP calculations (ignored for SCINE)

    Returns:
        Callable that takes ASE Atoms and returns (3N, 3N) Hessian as numpy array

    Example:
        >>> hessian_fn = create_hessian_function(hip_calc, "hip", device="cuda")
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

        def hip_hessian_function(atoms: Atoms) -> np.ndarray:
            """Compute raw Cartesian Hessian using HIP calculator.

            Note: HIP returns the Hessian directly without needing
            any gradient computation - it's a direct model output.
            """
            positions = atoms.get_positions()  # (N, 3) in Angstrom
            atomic_numbers = atoms.get_atomic_numbers()  # (N,)

            # Convert to torch tensors
            coords = torch.tensor(positions, dtype=torch.float32, device=torch_device)
            z = torch.tensor(atomic_numbers, dtype=torch.long, device=torch_device)

            # Create PyG batch for HIP
            batch = coords_to_pyg_batch(coords, z, device=torch_device)

            # Run HIP prediction WITH Hessian
            # HIP directly outputs Hessians - no autograd needed
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

        return hip_hessian_function
