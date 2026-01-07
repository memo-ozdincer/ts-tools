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

        # Import Eckart projection if needed
        if apply_eckart:
            from ...dependencies.differentiable_projection import (
                differentiable_massweigh_and_eckartprojection_torch,
            )
            from hip.ff_lmdb import Z_TO_ATOM_SYMBOL

        def hip_hessian_function(atoms: Atoms) -> np.ndarray:
            """Compute Cartesian Hessian using HIP calculator.

            Note: HIP returns the Hessian directly without needing
            any gradient computation - it's a direct model output.

            If apply_eckart=True, applies mass-weighting + Eckart projection
            to remove translational/rotational modes, then un-mass-weights.
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
                hessian_t = hessian.detach()
            else:
                hessian_t = torch.tensor(hessian, device=torch_device)

            # Ensure shape is (3N, 3N)
            n_atoms = len(atomic_numbers)
            hessian_t = hessian_t.reshape(3 * n_atoms, 3 * n_atoms)

            if apply_eckart:
                # Apply mass-weighting + Eckart projection to remove trans/rot modes
                # Keep in mass-weighted space - the 6 trans/rot eigenvalues are zero
                # only in this space, which is what P-RFO needs for correct mode analysis
                atomsymbols = [Z_TO_ATOM_SYMBOL[int(z_i)] for z_i in atomic_numbers]
                coords_3d = coords.reshape(-1, 3)

                # Get mass-weighted, Eckart-projected Hessian
                H_mw_proj = differentiable_massweigh_and_eckartprojection_torch(
                    hessian_t, coords_3d, atomsymbols
                )
                hessian = H_mw_proj.cpu().numpy()
            else:
                hessian = hessian_t.cpu().numpy()

            return hessian

        return hip_hessian_function
