"""k-HiSD (High-index Saddle Dynamics) implementation.

Per the iHiSD paper (Yin et al.), k-HiSD uses a reflection matrix to descend
from index-k saddles:

    ẋ = -R∇E    where R = I - 2∑ᵢ₌₁ᵏ vᵢvᵢᵀ

Key insight (Theorem 3.2): Index-k saddles are stable fixed points of k-HiSD.
To escape an index-k saddle, you need k-HiSD (reflect along all k negative modes).

**Adaptive k-HiSD**:
1. Compute Morse index k (count of negative eigenvalues)
2. Build reflection matrix R = I - 2∑ᵢ₌₁ᵏ vᵢvᵢᵀ
3. Take step: x += dt * R * (-∇E)
4. As k decreases (eigenvalue crossings), we approach index-1

This is mathematically equivalent to:
- Ascending along v₁, v₂, ..., vₖ (negative curvature modes)
- Descending along vₖ₊₁, vₖ₊₂, ... (positive curvature modes)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


def compute_reflection_matrix(
    evecs: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Compute k-HiSD reflection matrix R = I - 2∑ᵢ₌₁ᵏ vᵢvᵢᵀ.

    Args:
        evecs: Eigenvectors matrix (3N, 3N) with columns sorted by eigenvalue
        k: Number of modes to reflect (typically = Morse index)

    Returns:
        Reflection matrix R (3N, 3N)
    """
    n = evecs.shape[0]
    device = evecs.device
    dtype = evecs.dtype

    if k <= 0:
        # No reflection, just identity (gradient descent)
        return torch.eye(n, device=device, dtype=dtype)

    # Clamp k to available modes
    k = min(k, evecs.shape[1])

    # Extract first k eigenvectors
    V_k = evecs[:, :k]  # (3N, k)

    # R = I - 2 * V_k @ V_k.T
    R = torch.eye(n, device=device, dtype=dtype) - 2.0 * (V_k @ V_k.T)

    return R


def compute_hisd_direction(
    gradient: torch.Tensor,
    evecs: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Compute k-HiSD direction: -R∇E.

    This direction:
    - Ascends along the k lowest eigenvectors (negative curvature)
    - Descends along higher eigenvectors (positive curvature)

    At an index-k saddle:
    - k-HiSD with k reflections will descend (cross eigenvalue to index-(k-1))
    - (k-1)-HiSD would be stuck

    Args:
        gradient: ∇E vector (3N,) - NOT forces! (gradient = -forces)
        evecs: Eigenvectors (3N, 3N) sorted by eigenvalue
        k: Number of modes to reflect

    Returns:
        HiSD direction vector (3N,)
    """
    R = compute_reflection_matrix(evecs, k)
    # Direction = -R * ∇E
    direction = -R @ gradient
    return direction


def compute_adaptive_k(
    evals: torch.Tensor,
    tr_threshold: float = 1e-6,
) -> int:
    """Compute adaptive k = Morse index (count of negative vibrational eigenvalues).

    Args:
        evals: Eigenvalues sorted ascending
        tr_threshold: Threshold for filtering TR modes (near-zero)

    Returns:
        k = number of negative vibrational eigenvalues
    """
    # Filter out TR modes (near-zero eigenvalues)
    vib_mask = torch.abs(evals) > tr_threshold
    vib_evals = evals[vib_mask]

    if len(vib_evals) == 0:
        return 0

    # Count negative eigenvalues
    k = int((vib_evals < -tr_threshold).sum().item())
    return k


def adaptive_k_hisd_step(
    coords: torch.Tensor,
    forces: torch.Tensor,
    hessian_proj: torch.Tensor,
    dt: float,
    *,
    tr_threshold: float = 1e-6,
    min_k: int = 1,
    max_k: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Take one adaptive k-HiSD step.

    Automatically determines k from Morse index and takes a step using k-HiSD.

    Args:
        coords: Current coordinates (N, 3)
        forces: Current forces (N, 3) - note: forces = -∇E
        hessian_proj: Projected Hessian (3N, 3N)
        dt: Timestep
        tr_threshold: Threshold for TR mode filtering
        min_k: Minimum k to use (default 1 = standard GAD behavior)
        max_k: Maximum k to use (None = no limit)

    Returns:
        new_coords: Updated coordinates (N, 3)
        info: Dictionary with step info (k_used, morse_index, eigenvalues, etc.)
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    num_atoms = int(coords.shape[0])

    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    forces_flat = forces.reshape(-1)
    grad_flat = -forces_flat  # ∇E = -forces

    # Eigendecomposition
    evals, evecs = torch.linalg.eigh(hessian_proj)

    # Filter TR modes
    vib_mask = torch.abs(evals) > tr_threshold
    vib_indices = torch.where(vib_mask)[0]

    if len(vib_indices) == 0:
        # No vibrational modes, just do gradient descent
        new_coords = coords + dt * forces.reshape(-1, 3)
        return new_coords.detach(), {
            "k_used": 0,
            "morse_index": 0,
            "eig_spectrum": [],
            "direction_type": "gradient_descent",
        }

    vib_evals = evals[vib_indices]
    vib_evecs = evecs[:, vib_indices]

    # Compute Morse index
    morse_index = int((vib_evals < -tr_threshold).sum().item())

    # Determine k to use
    k = morse_index
    if k < min_k:
        k = min_k
    if max_k is not None and k > max_k:
        k = max_k

    # Compute k-HiSD direction
    direction = compute_hisd_direction(grad_flat, vib_evecs, k)
    direction_3d = direction.reshape(num_atoms, 3)

    # Take step
    new_coords = coords + dt * direction_3d

    # Extract eigenvalue spectrum for logging
    n_eigs = min(6, len(vib_evals))
    eig_spectrum = [float(vib_evals[i].item()) for i in range(n_eigs)]

    info = {
        "k_used": k,
        "morse_index": morse_index,
        "eig_spectrum": eig_spectrum,
        "eig_0": eig_spectrum[0] if len(eig_spectrum) > 0 else float("nan"),
        "eig_1": eig_spectrum[1] if len(eig_spectrum) > 1 else float("nan"),
        "direction_norm": float(direction.norm().item()),
        "grad_norm": float(grad_flat.norm().item()),
        "direction_type": f"{k}-HiSD" if k > 0 else "gradient_descent",
    }

    return new_coords.detach(), info


def compute_gad_vs_hisd_angle(
    forces: torch.Tensor,
    hessian_proj: torch.Tensor,
    k: int,
    tr_threshold: float = 1e-6,
) -> float:
    """Compute angle between GAD direction and k-HiSD direction.

    Useful for understanding when GAD and k-HiSD diverge.

    Args:
        forces: Forces (N, 3)
        hessian_proj: Projected Hessian (3N, 3N)
        k: Number of modes for HiSD
        tr_threshold: TR threshold

    Returns:
        Angle in degrees between GAD and k-HiSD directions
    """
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    forces_flat = forces.reshape(-1)
    grad_flat = -forces_flat

    # Eigendecomposition
    evals, evecs = torch.linalg.eigh(hessian_proj)

    # Filter TR
    vib_mask = torch.abs(evals) > tr_threshold
    vib_evecs = evecs[:, vib_mask]

    if vib_evecs.shape[1] < 1:
        return float("nan")

    # GAD direction (1-HiSD equivalent)
    v1 = vib_evecs[:, 0]
    v1 = v1 / (v1.norm() + 1e-12)
    gad_dir = forces_flat + 2.0 * torch.dot(grad_flat, v1) * v1

    # k-HiSD direction
    hisd_dir = compute_hisd_direction(grad_flat, vib_evecs, k)

    # Compute angle
    gad_norm = gad_dir.norm()
    hisd_norm = hisd_dir.norm()

    if gad_norm < 1e-12 or hisd_norm < 1e-12:
        return float("nan")

    cos_angle = torch.dot(gad_dir, hisd_dir) / (gad_norm * hisd_norm)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    return float(torch.acos(cos_angle).item() * 180.0 / np.pi)
