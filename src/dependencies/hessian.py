from __future__ import annotations

from typing import List, Optional

import torch

from ..common_utils import extract_vibrational_eigenvalues
from ..differentiable_projection import (
    differentiable_massweigh_and_eckartprojection_torch as _massweigh_and_eckartprojection_torch,
)

# SCINE-specific imports (will be None if SCINE is not installed)
try:
    from .scine_masses import (
        scine_project_hessian_remove_rigid_modes as _scine_project_hessian,
        scine_vibrational_eigvals as _scine_vibrational_eigvals,
    )
    SCINE_AVAILABLE = True
except ImportError:
    SCINE_AVAILABLE = False
    _scine_project_hessian = None
    _scine_vibrational_eigvals = None


def atomic_nums_to_symbols(atomic_nums: torch.Tensor) -> List[str]:
    """Convert atomic numbers to element symbols.

    Uses HIP's `Z_TO_ATOM_SYMBOL` mapping (present in your environment).
    """

    from hip.ff_lmdb import Z_TO_ATOM_SYMBOL

    return [Z_TO_ATOM_SYMBOL[int(z)] for z in atomic_nums.detach().cpu().tolist()]


def prepare_hessian(hess: torch.Tensor, num_atoms: int) -> torch.Tensor:
    """Normalize Hessian shapes to (3N, 3N)."""

    if hess.dim() == 1:
        side = int(hess.numel() ** 0.5)
        return hess.view(side, side)
    if hess.dim() == 3 and hess.shape[0] == 1:
        hess = hess[0]
    if hess.dim() > 2:
        return hess.reshape(3 * num_atoms, 3 * num_atoms)
    return hess


def project_hessian_remove_rigid_modes(
    hessian_raw: torch.Tensor,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
) -> torch.Tensor:
    """Mass-weight + Eckart-project Hessian to remove translation/rotation null modes."""

    coords3d = coords.reshape(-1, 3)
    num_atoms = int(coords3d.shape[0])
    hess = prepare_hessian(hessian_raw, num_atoms)
    atomsymbols = atomic_nums_to_symbols(atomic_nums)
    return _massweigh_and_eckartprojection_torch(hess, coords3d, atomsymbols)


def get_scine_elements_from_predict_output(predict_output: dict) -> Optional[list]:
    """
    Extract SCINE elements from predict function output.

    If the predict output contains a "_scine_calculator" key (added by
    make_scine_predict_fn), retrieves the cached element list.

    Args:
        predict_output: Output dict from predict_fn

    Returns:
        List of scine_utilities.ElementType if SCINE was used, None otherwise
    """
    scine_calc = predict_output.get("_scine_calculator")
    if scine_calc is None:
        return None
    return scine_calc.get_last_elements()


def vibrational_eigvals(
    hessian_raw: torch.Tensor,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    scine_elements: Optional[list] = None,
) -> torch.Tensor:
    """Projected vibrational eigenvalues with rigid modes removed.

    This is the canonical helper to ensure we remove rotation/translation null
    directions for both HIP and SCINE calculators.

    Args:
        hessian_raw: Raw Hessian tensor
        coords: Atomic coordinates
        atomic_nums: Atomic numbers
        scine_elements: Optional list of scine_utilities.ElementType objects.
            If provided, uses SCINE-specific mass-weighting. Otherwise uses HIP.

    Returns:
        Vibrational eigenvalues with rigid modes removed
    """
    if scine_elements is not None:
        # Use SCINE-specific mass-weighting
        if not SCINE_AVAILABLE:
            raise RuntimeError(
                "SCINE mass-weighting requested but scine_masses module not available. "
                "Install SCINE or remove scine_elements argument."
            )
        return _scine_vibrational_eigvals(hessian_raw, coords, scine_elements)

    # Default: Use HIP mass-weighting
    hess_proj = project_hessian_remove_rigid_modes(hessian_raw, coords, atomic_nums)
    return extract_vibrational_eigenvalues(hess_proj, coords.reshape(-1, 3))
