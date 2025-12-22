from __future__ import annotations

from typing import List

import torch

from ..common_utils import extract_vibrational_eigenvalues
from ..differentiable_projection import (
    differentiable_massweigh_and_eckartprojection_torch as _massweigh_and_eckartprojection_torch,
)


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


def vibrational_eigvals(
    hessian_raw: torch.Tensor,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
) -> torch.Tensor:
    """Projected vibrational eigenvalues with rigid modes removed.

    This is the canonical helper to ensure we remove rotation/translation null
    directions for HIP (and for SCINE as well if you choose to use the same path).
    """

    hess_proj = project_hessian_remove_rigid_modes(hessian_raw, coords, atomic_nums)
    return extract_vibrational_eigenvalues(hess_proj, coords.reshape(-1, 3))
