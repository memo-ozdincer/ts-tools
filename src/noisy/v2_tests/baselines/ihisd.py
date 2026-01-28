"""iHiSD (Improved High-index Saddle Dynamics) implementation.

Based on Algorithm 1 from the iHiSD paper, this uses a crossover parameter
theta that smoothly transitions from gradient flow (theta ≈ 0) to full
k-HiSD (theta → 1).

Key features:
1. Nonlocal convergence: Can start outside the region of attraction
2. Smooth transition: Avoids discontinuous jumps in dynamics
3. Guaranteed convergence: Theorem proves convergence to index-k saddles

The crossover direction is:
    d_m = (1 - s*theta) * grad + 2*theta * sum_i(v_i * <grad, v_i>)

Where s = ±1 determines upward (+1) or downward (-1) search direction.

For saddle finding from initial guess:
- s = +1: Search upward (toward higher potential, good for minima → saddle)
- s = -1: Search downward (toward lower potential, good for high-index → TS)

The step size formula from the paper:
    tau = 2 / (L * (2*theta - 1))   for theta > 0.5

where L is the Lipschitz constant of the gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class IHiSDConfig:
    """Configuration for iHiSD algorithm."""
    # Crossover parameter settings
    theta_0: float = 1e-11          # Initial theta (near gradient flow)
    theta_schedule: str = "sigmoid"  # "sigmoid", "linear", "exponential"
    theta_rate: float = 0.01        # Rate parameter for theta update

    # Search direction
    search_direction: int = 1       # +1 upward, -1 downward

    # Target index
    target_k: int = 1               # Target Morse index

    # Step size
    dt_base: float = 0.005          # Base step size
    use_adaptive_dt: bool = True    # Use theta-dependent step size
    lipschitz_estimate: float = 1.0  # Estimated Lipschitz constant

    # Convergence
    grad_threshold: float = 1e-5
    max_steps: int = 5000

    # Eigenvalue thresholds
    tr_threshold: float = 1e-6
    neg_threshold: float = -1e-4

    # Step size limits
    dt_min: float = 1e-6
    dt_max: float = 0.08
    max_atom_disp: float = 0.35


@dataclass
class IHiSDState:
    """State for iHiSD algorithm."""
    step: int = 0
    theta: float = 1e-11           # Current crossover parameter
    theta_0: float = 1e-11         # Initial theta


def compute_theta(
    step: int,
    theta_0: float,
    schedule: str = "sigmoid",
    rate: float = 0.01,
) -> float:
    """Compute crossover parameter theta at given step.

    The theta parameter transitions from ~0 (gradient flow) to 1 (full HiSD).

    Args:
        step: Current step number
        theta_0: Initial theta value
        schedule: "sigmoid", "linear", or "exponential"
        rate: Rate parameter (lambda in paper)

    Returns:
        theta value in (0, 1]
    """
    if schedule == "sigmoid":
        # theta = 2 / (1 + exp(-lambda * m)) - 1
        # Maps to (0, 1) as m goes from 0 to infinity
        # Adding theta_0 ensures we start at theta_0, not at ~0
        raw = 2.0 / (1.0 + np.exp(-rate * step)) - 1.0
        return max(theta_0, raw)

    elif schedule == "linear":
        # Linear ramp from theta_0 to 1
        # Reaches theta=1 at step = 1/rate
        theta = theta_0 + rate * step
        return min(1.0, theta)

    elif schedule == "exponential":
        # Exponential approach to 1
        # theta = 1 - (1 - theta_0) * exp(-rate * step)
        return 1.0 - (1.0 - theta_0) * np.exp(-rate * step)

    else:
        raise ValueError(f"Unknown theta schedule: {schedule}")


def compute_ihisd_step_size(
    theta: float,
    lipschitz: float = 1.0,
    dt_base: float = 0.005,
    dt_min: float = 1e-6,
    dt_max: float = 0.08,
) -> float:
    """Compute step size based on theta.

    From the paper: tau = 2 / (L * (2*theta - 1)) for theta > 0.5

    For theta <= 0.5, we use the base step size.

    Args:
        theta: Current crossover parameter
        lipschitz: Estimated Lipschitz constant L
        dt_base: Base step size for theta <= 0.5
        dt_min: Minimum step size
        dt_max: Maximum step size

    Returns:
        Step size tau
    """
    if theta <= 0.5:
        return dt_base

    # tau = 2 / (L * (2*theta - 1))
    denom = lipschitz * (2.0 * theta - 1.0)
    if abs(denom) < 1e-10:
        return dt_base

    tau = 2.0 / denom

    # Clamp to bounds
    return max(dt_min, min(dt_max, tau))


def ihisd_step(
    coords: torch.Tensor,
    forces: torch.Tensor,
    hessian_proj: torch.Tensor,
    state: IHiSDState,
    config: IHiSDConfig,
) -> Tuple[torch.Tensor, Dict, IHiSDState]:
    """Take one iHiSD step with crossover dynamics.

    The direction is:
        d = (1 - s*theta) * g + 2*theta * sum_i(v_i * <g, v_i>)

    where g = gradient, s = search direction, v_i = eigenvectors

    For s = +1 (upward): starts as gradient ascent, transitions to HiSD
    For s = -1 (downward): starts as gradient descent, transitions to HiSD

    Args:
        coords: Current coordinates (N, 3)
        forces: Current forces (N, 3)
        hessian_proj: Projected Hessian (3N, 3N)
        state: Current IHiSDState
        config: IHiSDConfig

    Returns:
        new_coords: Updated coordinates
        info: Step information
        new_state: Updated state
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    num_atoms = int(coords.shape[0])

    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    forces_flat = forces.reshape(-1)
    grad_flat = -forces_flat  # gradient = -forces

    grad_norm = float(grad_flat.norm().item())

    # Get current theta
    theta = compute_theta(
        state.step,
        state.theta_0,
        config.theta_schedule,
        config.theta_rate,
    )

    # Eigendecomposition
    evals, evecs = torch.linalg.eigh(hessian_proj)

    # Filter TR modes
    vib_mask = torch.abs(evals) > config.tr_threshold
    vib_indices = torch.where(vib_mask)[0]

    if len(vib_indices) == 0:
        # No vibrational modes, gradient-based step
        s = config.search_direction
        direction = (1 - s * theta) * grad_flat
        direction_3d = direction.reshape(num_atoms, 3)
        dt = config.dt_base
        new_coords = coords + dt * direction_3d

        new_state = IHiSDState(
            step=state.step + 1,
            theta=theta,
            theta_0=state.theta_0,
        )

        return new_coords.detach(), {
            "k_used": 0,
            "morse_index": 0,
            "theta": theta,
            "direction_type": "gradient",
            "grad_norm": grad_norm,
            "direction_norm": float(direction.norm().item()),
            "dt_used": dt,
            "eig_spectrum": [],
        }, new_state

    vib_evals = evals[vib_indices]
    vib_evecs = evecs[:, vib_indices]

    # Compute Morse index
    morse_index = int((vib_evals < config.neg_threshold).sum().item())

    # Number of modes to use (target k)
    k = min(config.target_k, len(vib_evals))

    # Compute crossover direction
    # d = (1 - s*theta) * g + 2*theta * R_term
    # where R_term = sum_i(v_i * <g, v_i>) for i = 1..k
    s = config.search_direction

    # Compute R_term = sum of projections onto first k eigenvectors
    V_k = vib_evecs[:, :k]  # (3N, k)
    projections = V_k.T @ grad_flat  # (k,) - projections of gradient onto each v_i
    R_term = V_k @ projections  # (3N,) - weighted sum of eigenvectors

    # iHiSD direction
    direction = (1 - s * theta) * grad_flat + 2 * theta * R_term
    direction_3d = direction.reshape(num_atoms, 3)

    # Compute step size
    if config.use_adaptive_dt:
        dt = compute_ihisd_step_size(
            theta,
            config.lipschitz_estimate,
            config.dt_base,
            config.dt_min,
            config.dt_max,
        )
    else:
        dt = config.dt_base

    # Take step
    new_coords = coords + dt * direction_3d

    # Update state
    new_state = IHiSDState(
        step=state.step + 1,
        theta=theta,
        theta_0=state.theta_0,
    )

    # Eigenvalue spectrum for logging
    n_eigs = min(6, len(vib_evals))
    eig_spectrum = [float(vib_evals[i].item()) for i in range(n_eigs)]

    info = {
        "k_used": k,
        "morse_index": morse_index,
        "theta": theta,
        "search_direction": s,
        "direction_type": f"iHiSD-{k}",
        "grad_norm": grad_norm,
        "direction_norm": float(direction.norm().item()),
        "dt_used": dt,
        "eig_spectrum": eig_spectrum,
        "R_term_norm": float(R_term.norm().item()),
        "gradient_component": (1 - s * theta),
        "hisd_component": 2 * theta,
    }

    return new_coords.detach(), info, new_state


def run_ihisd(
    predict_fn: Callable,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    get_projected_hessian_fn: Callable,
    *,
    config: Optional[IHiSDConfig] = None,
    scine_elements=None,
) -> Tuple[Dict, List[Dict]]:
    """Run full iHiSD algorithm.

    iHiSD starts as gradient flow and smoothly transitions to k-HiSD,
    providing nonlocal convergence guarantees.

    Args:
        predict_fn: Energy/force/Hessian prediction function
        coords0: Starting coordinates
        atomic_nums: Atomic numbers
        get_projected_hessian_fn: Hessian projection function
        config: IHiSDConfig (uses defaults if None)
        scine_elements: Optional SCINE element types

    Returns:
        result: Dictionary with final state
        trajectory: List of step info dicts
    """
    if config is None:
        config = IHiSDConfig()

    if coords0 is None:
        raise ValueError("coords0 cannot be None")

    if coords0.dim() == 3 and coords0.shape[0] == 1:
        coords0 = coords0[0]
    coords = coords0.reshape(-1, 3).detach().clone()

    state = IHiSDState(
        step=0,
        theta=config.theta_0,
        theta_0=config.theta_0,
    )

    trajectory = []
    converged = False
    converged_step = None

    for step in range(config.max_steps):
        # Get energy, forces, Hessian
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        forces = out["forces"]
        hessian = out["hessian"]
        energy = out.get("energy", 0)

        # Project Hessian
        hess_proj = get_projected_hessian_fn(hessian, coords, atomic_nums, scine_elements)

        # Take iHiSD step
        new_coords, info, new_state = ihisd_step(
            coords, forces, hess_proj, state, config
        )

        # Apply max displacement cap
        step_vec = new_coords - coords
        max_disp = float(step_vec.norm(dim=1).max().item())
        if max_disp > config.max_atom_disp and max_disp > 0:
            scale = config.max_atom_disp / max_disp
            new_coords = coords + scale * step_vec

        info["step"] = step
        info["energy"] = float(energy.item()) if hasattr(energy, "item") else float(energy)
        trajectory.append(info)

        # Check convergence
        if info["grad_norm"] < config.grad_threshold:
            if info["morse_index"] == config.target_k:
                converged = True
                converged_step = step
                break

        state = new_state
        coords = new_coords

    # Final analysis
    final_out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
    hess_proj = get_projected_hessian_fn(final_out["hessian"], coords, atomic_nums, scine_elements)
    evals, _ = torch.linalg.eigh(hess_proj)
    vib_mask = torch.abs(evals) > config.tr_threshold
    vib_evals = evals[vib_mask]
    final_index = int((vib_evals < config.neg_threshold).sum().item())

    # Collect theta trajectory
    theta_values = [t["theta"] for t in trajectory]

    result = {
        "converged": converged,
        "converged_step": converged_step,
        "final_index": final_index,
        "target_k": config.target_k,
        "total_steps": len(trajectory),
        "final_theta": state.theta,
        "theta_min": min(theta_values) if theta_values else config.theta_0,
        "theta_max": max(theta_values) if theta_values else config.theta_0,
        "search_direction": config.search_direction,
        "final_coords": coords.detach().cpu(),
        "algorithm": "ihisd",
    }

    return result, trajectory
