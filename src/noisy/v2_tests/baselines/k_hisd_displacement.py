"""k-HiSD with DISPLACEMENT-based stuck detection.

The original k-HiSD uses gradient-norm-based stuck detection:
    if grad_norm < threshold:  # threshold = 1e-4

But this NEVER triggers because:
- At high-index saddles, grad_norm stays moderate (0.2-3 eV/Å)
- Displacement drops to ~1 μÅ (this is the true "stuck" signal)

This version fixes the issue by using displacement-based detection,
matching what works in the v2 kicking implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .k_hisd import compute_hisd_direction, compute_morse_index


@dataclass
class DisplacementKState:
    """State for displacement-based adaptive k-HiSD.

    Key difference from AdaptiveKState:
    - Uses displacement history instead of gradient norm for stuck detection
    - Matches the successful v2 kicking detection strategy
    """
    k: int = 1                        # Current k (starts at 1 = GAD)
    k_target: int = 1                 # Target index (always 1 for TS finding)
    disp_history: List[float] = field(default_factory=list)
    neg_vib_history: List[int] = field(default_factory=list)
    # Detection parameters
    window: int = 10                  # Window for displacement averaging
    disp_threshold: float = 1e-4     # Displacement threshold (Å)
    neg_vib_std_threshold: float = 0.5  # Std of neg_vib for "stable" detection
    stuck_steps: int = 0              # Consecutive stuck steps
    steps_before_increase: int = 5    # Steps stuck before increasing k


def _is_stuck_displacement_based(
    state: DisplacementKState,
    current_disp: float,
    current_neg_vib: int,
) -> bool:
    """Check if stuck using displacement-based detection.

    Triggers when:
    1. mean(disp[-window:]) < disp_threshold (tiny steps)
    2. std(neg_vib[-window:]) <= neg_vib_std_threshold (stable index)
    3. current_neg_vib > k (at higher-index saddle than we're reflecting)
    """
    if len(state.disp_history) < state.window:
        return False

    recent_disp = state.disp_history[-state.window:]
    recent_neg_vib = state.neg_vib_history[-state.window:]

    mean_disp = float(np.mean(recent_disp))
    std_neg_vib = float(np.std(recent_neg_vib))

    return (
        mean_disp < state.disp_threshold
        and std_neg_vib <= state.neg_vib_std_threshold
        and current_neg_vib > state.k
    )


def adaptive_k_hisd_displacement_step(
    coords: torch.Tensor,
    forces: torch.Tensor,
    hessian_proj: torch.Tensor,
    dt: float,
    state: DisplacementKState,
    coords_prev: Optional[torch.Tensor] = None,
    *,
    tr_threshold: float = 1e-6,
    neg_threshold: float = -1e-4,
) -> Tuple[torch.Tensor, Dict, DisplacementKState]:
    """Take one adaptive k-HiSD step with DISPLACEMENT-based stuck detection.

    This fixes the key bug in the original k-HiSD: using gradient norm for
    stuck detection doesn't work because grad_norm stays moderate at plateaus.

    Algorithm:
    1. Track displacement history (like v2 kicking does)
    2. When displacement plateau detected AND morse_index > k:
       - Increase k by 1
       - This destabilizes the current high-index saddle
    3. Take k-HiSD step with current k

    Args:
        coords: Current coordinates (N, 3)
        forces: Current forces (N, 3) - note: forces = -∇E
        hessian_proj: Projected Hessian (3N, 3N)
        dt: Timestep
        state: DisplacementKState tracking k and displacement history
        coords_prev: Previous coordinates for displacement computation
        tr_threshold: Threshold for TR mode filtering
        neg_threshold: Threshold for "negative" eigenvalue

    Returns:
        new_coords: Updated coordinates (N, 3)
        info: Dictionary with step info
        new_state: Updated DisplacementKState
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
            "mean_disp": 0.0,
        }, state

    vib_evals = evals[vib_indices]
    vib_evecs = evecs[:, vib_indices]

    # Compute Morse index
    morse_index = int((vib_evals < neg_threshold).sum().item())

    # Compute displacement from previous step
    if coords_prev is not None:
        if coords_prev.dim() == 3 and coords_prev.shape[0] == 1:
            coords_prev = coords_prev[0]
        disp = float((coords - coords_prev.reshape(-1, 3)).norm(dim=1).mean().item())
    else:
        disp = 0.0

    # === DISPLACEMENT-BASED ADAPTIVE K LOGIC ===
    new_state = DisplacementKState(
        k=state.k,
        k_target=state.k_target,
        disp_history=state.disp_history.copy(),
        neg_vib_history=state.neg_vib_history.copy(),
        window=state.window,
        disp_threshold=state.disp_threshold,
        neg_vib_std_threshold=state.neg_vib_std_threshold,
        stuck_steps=state.stuck_steps,
        steps_before_increase=state.steps_before_increase,
    )

    # Update histories
    new_state.disp_history.append(disp)
    new_state.neg_vib_history.append(morse_index)

    # Keep only recent history
    max_history = state.window * 2
    if len(new_state.disp_history) > max_history:
        new_state.disp_history = new_state.disp_history[-max_history:]
        new_state.neg_vib_history = new_state.neg_vib_history[-max_history:]

    k_increased = False
    is_stuck = _is_stuck_displacement_based(new_state, disp, morse_index)

    if is_stuck:
        new_state.stuck_steps += 1

        # If stuck for enough steps, increase k
        if new_state.stuck_steps >= state.steps_before_increase:
            # Increase k by 1 (or jump to morse_index if we want faster convergence)
            new_k = min(state.k + 1, morse_index)
            new_state.k = new_k
            new_state.stuck_steps = 0  # Reset counter
            k_increased = True
    else:
        new_state.stuck_steps = 0

        # If we've descended to target index, ensure k = target
        if morse_index == state.k_target:
            new_state.k = state.k_target

    # Use current k for this step
    k_used = state.k

    # Compute k-HiSD direction
    direction = compute_hisd_direction(grad_flat, vib_evecs, k_used)
    direction_3d = direction.reshape(num_atoms, 3)

    # Take step
    new_coords = coords + dt * direction_3d

    # Extract eigenvalue spectrum for logging
    n_eigs = min(6, len(vib_evals))
    eig_spectrum = [float(vib_evals[i].item()) for i in range(n_eigs)]

    # Compute mean displacement for logging
    mean_disp = float(np.mean(new_state.disp_history[-state.window:])) if len(new_state.disp_history) >= state.window else 0.0

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
        "stuck_steps": new_state.stuck_steps,
        "k_increased": k_increased,
        "disp_from_prev": disp,
        "mean_disp": mean_disp,
    }

    return new_coords.detach(), info, new_state


def run_k_hisd_displacement(
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
    # Displacement detection parameters
    window: int = 10,
    disp_threshold: float = 1e-4,
    neg_vib_std_threshold: float = 0.5,
    steps_before_increase: int = 5,
    # Other parameters
    tr_threshold: float = 1e-6,
    neg_threshold: float = -1e-4,
    grad_converged_threshold: float = 1e-5,
    k_target: int = 1,
    scine_elements=None,
) -> Tuple[Dict, list]:
    """Run k-HiSD with displacement-based stuck detection.

    This is the corrected version that actually works because it uses
    displacement-based detection (like v2 kicking) instead of gradient norm.
    """
    coords = coords0.detach().clone().to(torch.float32)
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)

    # Initialize state with displacement detection
    state = DisplacementKState(
        k=1,
        k_target=k_target,
        window=window,
        disp_threshold=disp_threshold,
        neg_vib_std_threshold=neg_vib_std_threshold,
        steps_before_increase=steps_before_increase,
    )

    trajectory = []
    converged = False
    converged_step = None
    dt_eff = dt
    coords_prev = None

    for step in range(n_steps):
        # Get energy, forces, Hessian
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        energy = float(out["energy"].item()) if hasattr(out["energy"], "item") else float(out["energy"])
        forces = out["forces"]
        hessian = out["hessian"]

        # Get projected Hessian
        hess_proj = get_projected_hessian_fn(hessian, coords, atomic_nums, scine_elements)

        # Take step with displacement-based detection
        new_coords, info, new_state = adaptive_k_hisd_displacement_step(
            coords,
            forces,
            hess_proj,
            dt_eff,
            state,
            coords_prev=coords_prev,
            tr_threshold=tr_threshold,
            neg_threshold=neg_threshold,
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
        coords_prev = coords.clone()
        state = new_state
        coords = new_coords

    # Summary
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
        "algorithm": "k_hisd_displacement",
    }

    return result, trajectory
