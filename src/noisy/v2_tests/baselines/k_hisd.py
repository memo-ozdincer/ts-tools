"""k-HiSD (High-index Saddle Dynamics) implementation.

Per the iHiSD paper (Yin et al.), k-HiSD uses a reflection matrix:

    ẋ = -R∇E    where R = I - 2∑ᵢ₌₁ᵏ vᵢvᵢᵀ

**Critical Theorem (3.2)**: Index-k saddles are STABLE fixed points of k-HiSD.

This means:
- 1-HiSD (k=1) = GAD: FINDS index-1 saddles (TS) - this is our target!
- k-HiSD with k = Morse index: STABILIZES current saddle (WRONG for escaping)
- k-HiSD with k < Morse index: Makes current saddle UNSTABLE (correct for escaping)

**Correct Adaptive k-HiSD Algorithm**:
1. Start with k=1 (target index-1 saddles, same as GAD)
2. Take k-HiSD steps normally
3. When STUCK (grad small but Morse index > k):
   - Increase k by 1 (or to current Morse index)
   - This makes the current saddle UNSTABLE under the new k-HiSD
   - We escape and continue
4. Eventually reach index-1 region where k=1 works

Note: 1-HiSD IS mathematically identical to GAD:
    1-HiSD: -R∇E = -(I - 2v₁v₁ᵀ)∇E = -∇E + 2(v₁ᵀ∇E)v₁
    GAD:    -∇E + 2(v₁ᵀ∇E)v₁
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def compute_reflection_matrix(
    evecs: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Compute k-HiSD reflection matrix R = I - 2∑ᵢ₌₁ᵏ vᵢvᵢᵀ.

    Args:
        evecs: Eigenvectors matrix (3N, m) with columns sorted by eigenvalue (ascending)
        k: Number of modes to reflect

    Returns:
        Reflection matrix R (3N, 3N)
    """
    n = evecs.shape[0]
    device = evecs.device
    dtype = evecs.dtype

    if k <= 0:
        # No reflection = gradient descent direction
        return torch.eye(n, device=device, dtype=dtype)

    # Clamp k to available modes
    k = min(k, evecs.shape[1])

    # Extract first k eigenvectors (lowest eigenvalues)
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

    For k=1, this is identical to GAD:
        -R∇E = -(I - 2v₁v₁ᵀ)∇E = -∇E + 2(v₁ᵀ∇E)v₁

    Args:
        gradient: ∇E vector (3N,) - NOT forces! (gradient = -forces)
        evecs: Eigenvectors (3N, m) sorted by eigenvalue ascending
        k: Number of modes to reflect

    Returns:
        HiSD direction vector (3N,)
    """
    R = compute_reflection_matrix(evecs, k)
    direction = -R @ gradient
    return direction


def compute_morse_index(
    evals: torch.Tensor,
    tr_threshold: float = 1e-6,
    neg_threshold: float = -1e-4,
) -> int:
    """Compute Morse index (count of negative vibrational eigenvalues).

    Args:
        evals: Eigenvalues sorted ascending
        tr_threshold: Threshold for filtering TR modes (|λ| < tr_threshold → TR mode)
        neg_threshold: Threshold for "negative" eigenvalue (λ < neg_threshold → negative)

    Returns:
        Number of negative vibrational eigenvalues
    """
    # Filter out TR modes (near-zero eigenvalues)
    vib_mask = torch.abs(evals) > tr_threshold
    vib_evals = evals[vib_mask]

    if len(vib_evals) == 0:
        return 0

    # Count eigenvalues below neg_threshold
    return int((vib_evals < neg_threshold).sum().item())


@dataclass
class AdaptiveKState:
    """State for adaptive k-HiSD algorithm.

    The key insight: k starts at 1 and only GROWS when stuck.
    This ensures we destabilize high-index saddles rather than stabilizing them.
    """
    k: int = 1                    # Current k (starts at 1 = GAD)
    k_target: int = 1             # Target index (always 1 for TS finding)
    stuck_counter: int = 0        # Steps with small gradient but wrong index
    stuck_threshold: int = 10     # Steps before increasing k
    last_morse_index: int = 0     # For tracking index changes


def adaptive_k_hisd_step(
    coords: torch.Tensor,
    forces: torch.Tensor,
    hessian_proj: torch.Tensor,
    dt: float,
    state: AdaptiveKState,
    *,
    tr_threshold: float = 1e-6,
    neg_threshold: float = -1e-4,
    grad_stuck_threshold: float = 1e-4,
) -> Tuple[torch.Tensor, Dict, AdaptiveKState]:
    """Take one adaptive k-HiSD step with CORRECT adaptive logic.

    Algorithm:
    1. Always use current k for HiSD direction
    2. Check if stuck: gradient small AND Morse index > k
    3. If stuck for stuck_threshold steps, increase k
    4. This makes current saddle UNSTABLE, allowing escape

    Args:
        coords: Current coordinates (N, 3)
        forces: Current forces (N, 3) - note: forces = -∇E
        hessian_proj: Projected Hessian (3N, 3N)
        dt: Timestep
        state: AdaptiveKState tracking k and stuck counter
        tr_threshold: Threshold for TR mode filtering
        neg_threshold: Threshold for "negative" eigenvalue
        grad_stuck_threshold: Gradient norm below which we're "stuck"

    Returns:
        new_coords: Updated coordinates (N, 3)
        info: Dictionary with step info
        new_state: Updated AdaptiveKState
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    num_atoms = int(coords.shape[0])

    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    forces_flat = forces.reshape(-1)
    grad_flat = -forces_flat  # ∇E = -forces

    grad_norm = float(grad_flat.norm().item())

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
            "grad_norm": grad_norm,
            "stuck": False,
            "k_increased": False,
        }, state

    vib_evals = evals[vib_indices]
    vib_evecs = evecs[:, vib_indices]

    # Compute Morse index
    morse_index = int((vib_evals < neg_threshold).sum().item())

    # === ADAPTIVE K LOGIC ===
    new_state = AdaptiveKState(
        k=state.k,
        k_target=state.k_target,
        stuck_counter=state.stuck_counter,
        stuck_threshold=state.stuck_threshold,
        last_morse_index=morse_index,
    )

    k_increased = False
    is_stuck = False

    # Check if stuck: gradient small but Morse index > current k
    if grad_norm < grad_stuck_threshold and morse_index > state.k:
        is_stuck = True
        new_state.stuck_counter += 1

        # If stuck for too long, increase k
        if new_state.stuck_counter >= state.stuck_threshold:
            # Increase k by 1, or jump to morse_index if much higher
            new_k = min(state.k + 1, morse_index)
            new_state.k = new_k
            new_state.stuck_counter = 0
            k_increased = True
    else:
        # Not stuck, reset counter
        new_state.stuck_counter = 0

        # If we've descended to target index, ensure k = target
        if morse_index == state.k_target:
            new_state.k = state.k_target

    # Use current k for this step (new k takes effect next step)
    k_used = state.k

    # Compute k-HiSD direction
    # For k=1, this is GAD: -∇E + 2(v₁ᵀ∇E)v₁
    direction = compute_hisd_direction(grad_flat, vib_evecs, k_used)
    direction_3d = direction.reshape(num_atoms, 3)

    # Take step
    new_coords = coords + dt * direction_3d

    # Extract eigenvalue spectrum for logging
    n_eigs = min(6, len(vib_evals))
    eig_spectrum = [float(vib_evals[i].item()) for i in range(n_eigs)]

    info = {
        "k_used": k_used,
        "k_next": new_state.k,
        "morse_index": morse_index,
        "eig_spectrum": eig_spectrum,
        "eig_0": eig_spectrum[0] if len(eig_spectrum) > 0 else float("nan"),
        "eig_1": eig_spectrum[1] if len(eig_spectrum) > 1 else float("nan"),
        "direction_norm": float(direction.norm().item()),
        "grad_norm": grad_norm,
        "direction_type": f"{k_used}-HiSD",
        "stuck": is_stuck,
        "stuck_counter": new_state.stuck_counter,
        "k_increased": k_increased,
    }

    return new_coords.detach(), info, new_state


def run_adaptive_k_hisd_trajectory(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    get_projected_hessian_fn,
    *,
    n_steps: int = 1000,
    dt: float = 0.005,
    dt_min: float = 1e-6,
    dt_max: float = 0.1,
    max_atom_disp: float = 0.3,
    tr_threshold: float = 1e-6,
    neg_threshold: float = -1e-4,
    grad_stuck_threshold: float = 1e-4,
    stuck_threshold_steps: int = 10,
    grad_converged_threshold: float = 1e-5,
    k_target: int = 1,
    scine_elements=None,
) -> Tuple[Dict, list]:
    """Run full adaptive k-HiSD trajectory.

    This is the correct implementation:
    - Starts with k=1 (GAD)
    - Only increases k when stuck at high-index saddles
    - k < morse_index makes saddles UNSTABLE (we escape)
    - Eventually converges to k_target (index-1 TS)

    Args:
        predict_fn: Function returning energy, forces, hessian
        coords0: Starting coordinates (N, 3)
        atomic_nums: Atomic numbers
        get_projected_hessian_fn: Function to project Hessian
        n_steps: Maximum steps
        dt: Initial timestep
        dt_min, dt_max: Timestep bounds
        max_atom_disp: Maximum per-atom displacement per step
        tr_threshold: TR mode filter threshold
        neg_threshold: Negative eigenvalue threshold
        grad_stuck_threshold: Gradient norm for "stuck" detection
        stuck_threshold_steps: Steps before increasing k
        grad_converged_threshold: Gradient norm for convergence
        k_target: Target Morse index (1 for TS)
        scine_elements: SCINE element types if using SCINE

    Returns:
        result: Dictionary with final state and summary
        trajectory: List of per-step info dicts
    """
    coords = coords0.detach().clone().to(torch.float32)
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)

    # Initialize adaptive state
    state = AdaptiveKState(
        k=1,  # START AT k=1 (GAD behavior)
        k_target=k_target,
        stuck_counter=0,
        stuck_threshold=stuck_threshold_steps,
    )

    trajectory = []
    converged = False
    converged_step = None
    dt_eff = dt

    for step in range(n_steps):
        # Get energy, forces, Hessian
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        energy = float(out["energy"].item()) if hasattr(out["energy"], "item") else float(out["energy"])
        forces = out["forces"]
        hessian = out["hessian"]

        # Get projected Hessian
        hess_proj = get_projected_hessian_fn(hessian, coords, atomic_nums, scine_elements)

        # Take adaptive k-HiSD step
        new_coords, info, new_state = adaptive_k_hisd_step(
            coords,
            forces,
            hess_proj,
            dt_eff,
            state,
            tr_threshold=tr_threshold,
            neg_threshold=neg_threshold,
            grad_stuck_threshold=grad_stuck_threshold,
        )

        # Apply max displacement cap
        step_vec = new_coords - coords
        max_disp = float(step_vec.norm(dim=1).max().item())
        if max_disp > max_atom_disp and max_disp > 0:
            scale = max_atom_disp / max_disp
            new_coords = coords + scale * step_vec
            dt_eff = max(dt_eff * 0.8, dt_min)
        else:
            dt_eff = min(dt_eff * 1.05, dt_max)

        info["step"] = step
        info["energy"] = energy
        info["dt_eff"] = dt_eff
        trajectory.append(info)

        # Check convergence: small gradient AND correct index
        if info["grad_norm"] < grad_converged_threshold and info["morse_index"] == k_target:
            converged = True
            converged_step = step
            break

        # Update state
        state = new_state
        coords = new_coords

    # Final analysis
    result = {
        "converged": converged,
        "converged_step": converged_step,
        "final_k": state.k,
        "final_morse_index": trajectory[-1]["morse_index"] if trajectory else -1,
        "final_grad_norm": trajectory[-1]["grad_norm"] if trajectory else float("nan"),
        "total_steps": len(trajectory),
        "k_increases": sum(1 for t in trajectory if t.get("k_increased", False)),
        "max_k_used": max(t["k_used"] for t in trajectory) if trajectory else 0,
        "final_coords": coords.detach().cpu(),
    }

    return result, trajectory


# Legacy function for backwards compatibility
def compute_adaptive_k(evals: torch.Tensor, tr_threshold: float = 1e-6) -> int:
    """Compute Morse index. DEPRECATED: Use compute_morse_index instead."""
    return compute_morse_index(evals, tr_threshold=tr_threshold, neg_threshold=-tr_threshold)
