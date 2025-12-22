# src/scine_calculator.py
"""
SCINE Sparrow calculator wrapper that provides the same interface as EquiformerTorchCalculator.
This allows using analytical forcefields (DFT0, PM6, AM1, etc.) as drop-in replacements for HIP.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager

import torch
import numpy as np
import scine_sparrow
import scine_utilities
from torch_geometric.data import Batch


# Backup threading environment variables before modifying them
_THREADING_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]
_THREADING_ENV_BACKUP = {var: os.environ.get(var) for var in _THREADING_ENV_VARS}


@contextmanager
def suppress_output():
    """Context manager to suppress stdout at file descriptor level, keeping stderr for exceptions."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)  # Save stdout file descriptor
    os.dup2(devnull_fd, 1)  # Redirect stdout to /dev/null
    try:
        yield
    finally:
        os.dup2(old_stdout_fd, 1)  # Restore stdout
        os.close(devnull_fd)
        os.close(old_stdout_fd)


# Import element mapping from scine_masses module (centralized)
from .scine_masses import Z_TO_SCINE_ELEMENT as Z_TO_ELEMENT_TYPE


class ScineSparrowCalculator:
    """
    Wrapper around SCINE Sparrow that mimics EquiformerTorchCalculator interface.

    Supports various semiempirical methods:
    - DFTB0, DFTB2, DFTB3
    - PM6, AM1, RM1, MNDO

    Args:
        functional: SCINE method name (e.g., "DFTB0", "PM6", "AM1")
        device: Device string (e.g., "cpu", "cuda"). Note: SCINE only runs on CPU.
    """

    def __init__(self, functional: str = "DFTB0", device: str = "cpu", **kwargs):
        self.functional = functional
        self.device_str = "cpu"  # SCINE always runs on CPU

        # Cache for element list (used for mass-weighting in vibrational analysis)
        self._last_elements = None

        # Force single-threaded execution to avoid conflicts with PyTorch
        for var in _THREADING_ENV_VARS:
            os.environ[var] = "1"

        # Initialize SCINE module manager
        self.manager = scine_utilities.core.ModuleManager.get_instance()
        sparrow_module = Path(scine_sparrow.__file__).parent / "sparrow.module.so"

        if not sparrow_module.exists():
            raise RuntimeError(f"SCINE Sparrow module not found at {sparrow_module}")

        self.manager.load(os.fspath(sparrow_module))

        # Get calculator (but don't initialize yet - we'll create fresh ones per call)
        test_calc = self.manager.get("calculator", functional)
        if test_calc is None:
            raise ValueError(
                f"Calculator '{functional}' not found. "
                f"Available methods: DFTB0, DFTB2, DFTB3, PM6, AM1, RM1, MNDO"
            )

        # Suppress SCINE output
        scine_utilities.core.Log.silent()

        print(f"Initialized SCINE Sparrow calculator: {functional}")

    @property
    def potential(self):
        """Mimics EquiformerTorchCalculator.potential for compatibility."""
        # Return a minimal object with device attribute
        class FakePotential:
            def __init__(self, device_str):
                self.device = torch.device(device_str)
        return FakePotential(self.device_str)

    def get_last_elements(self) -> list:
        """
        Get the cached element list from the last calculation.

        This is used by SCINE-specific mass-weighting helpers to avoid
        redundant atomic number -> ElementType conversions.

        Returns:
            List of scine_utilities.ElementType from last predict() call
        """
        if self._last_elements is None:
            raise RuntimeError(
                "No elements cached. Call predict() or calculate() first."
            )
        return self._last_elements

    def _batch_to_geometry(self, batch: Batch) -> tuple:
        """
        Convert PyG batch to SCINE-compatible format.

        Returns:
            (elements, positions_angstrom, atomic_nums_tensor)
        """
        # Extract atomic numbers and positions
        if hasattr(batch, 'z'):
            atomic_nums = batch.z.detach().cpu().numpy()
        elif hasattr(batch, 'charges'):
            atomic_nums = batch.charges.detach().cpu().numpy()
        else:
            raise ValueError("Batch must have 'z' or 'charges' attribute")

        positions = batch.pos.detach().cpu().numpy().reshape(-1, 3)

        # Convert atomic numbers to SCINE ElementType
        elements = []
        for z in atomic_nums:
            z_int = int(z)
            if z_int not in Z_TO_ELEMENT_TYPE:
                raise ValueError(f"Unsupported element with Z={z_int}")
            elements.append(Z_TO_ELEMENT_TYPE[z_int])

        # Cache elements for use in mass-weighting (accessed via get_last_elements())
        self._last_elements = elements

        return elements, positions, torch.tensor(atomic_nums, dtype=torch.long)

    def _compute_single_geometry(
        self,
        elements,
        positions_angstrom: np.ndarray,
        do_hessian: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute energy, forces, and optionally Hessian for a single geometry.

        Returns:
            Dictionary with torch tensors in HIP-compatible format:
            - energy: (1,) tensor in eV
            - forces: (N, 3) tensor in eV/Å (NOTE: forces, not gradients)
            - hessian: (3N, 3N) tensor in eV/Å² (if do_hessian=True)
        """
        # Get fresh calculator instance
        calculator = self.manager.get("calculator", self.functional)

        # Convert positions from Angstrom to Bohr
        positions_bohr = positions_angstrom * scine_utilities.BOHR_PER_ANGSTROM
        structure = scine_utilities.AtomCollection(elements, positions_bohr)

        # Set structure and required properties
        calculator.structure = structure
        required_props = [
            scine_utilities.Property.Energy,
            scine_utilities.Property.Gradients,
        ]
        if do_hessian:
            required_props.append(scine_utilities.Property.Hessian)

        calculator.set_required_properties(required_props)

        # Calculate with output suppression
        with suppress_output():
            results = calculator.calculate()

        # Extract results and convert units
        energy_hartree = results.energy
        gradients_hartree_bohr = results.gradients  # Shape: (N*3,) in Hartree/Bohr

        # Unit conversion constants
        hartree_to_ev = 27.211386245988
        bohr_to_ang = 0.529177210903

        # Convert energy: Hartree → eV
        energy_ev = energy_hartree * hartree_to_ev

        # Convert gradients to forces:
        # - Gradients are in Hartree/Bohr
        # - Forces = -gradients (because F = -dE/dx)
        # - Convert units: Hartree/Bohr → eV/Å
        # Conversion factor: Hartree/Bohr → eV/Å = hartree_to_ev / bohr_to_ang
        gradient_to_force_factor = -hartree_to_ev / bohr_to_ang
        forces_ev_ang = gradients_hartree_bohr * gradient_to_force_factor

        # Prepare output dictionary
        output = {
            "energy": torch.tensor([energy_ev], dtype=torch.float32),
            "forces": torch.tensor(
                forces_ev_ang.reshape(-1, 3),
                dtype=torch.float32
            ),
        }

        if do_hessian:
            hessian_hartree_bohr2 = results.hessian  # Shape: (3N, 3N) in Hartree/Bohr²
            # Convert Hessian: Hartree/Bohr² → eV/Å²
            hessian_ev_ang2 = hessian_hartree_bohr2 * (hartree_to_ev / (bohr_to_ang ** 2))
            output["hessian"] = torch.tensor(hessian_ev_ang2, dtype=torch.float32)

        return output

    def predict(
        self,
        batch: Batch,
        do_hessian: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Predict energy, forces, and optionally Hessian for a batch.

        Args:
            batch: PyG Batch containing molecular geometry
            do_hessian: Whether to compute Hessian (default: True)

        Returns:
            Dictionary with:
            - energy: (1,) tensor in eV
            - forces: (N, 3) tensor in eV/Å
            - hessian: (3N, 3N) tensor in eV/Å² (if do_hessian=True)
        """
        # Convert batch to SCINE format
        elements, positions_angstrom, atomic_nums = self._batch_to_geometry(batch)

        # Compute properties
        results = self._compute_single_geometry(
            elements,
            positions_angstrom,
            do_hessian=do_hessian
        )

        return results

    def calculate(
        self,
        batch: Batch,
        do_hessian: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Alias for predict() to support both naming conventions."""
        return self.predict(batch, do_hessian=do_hessian, **kwargs)


def create_scine_calculator(
    functional: str = "DFTB0",
    device: str = "cpu",
    **kwargs
) -> ScineSparrowCalculator:
    """
    Factory function to create SCINE Sparrow calculator.

    Args:
        functional: SCINE method name (e.g., "DFTB0", "PM6", "AM1")
        device: Device (ignored, SCINE always uses CPU)

    Returns:
        ScineSparrowCalculator instance
    """
    return ScineSparrowCalculator(functional=functional, device=device, **kwargs)
