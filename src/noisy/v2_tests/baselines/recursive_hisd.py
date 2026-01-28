"""Recursive HiSD implementation.

Per the paper, Recursive HiSD starts from a high-index saddle point and
recursively descends to lower index saddles until reaching an index-1 TS.

Algorithm:
1. Start at index-n saddle point x_current
2. Base case: if n == 1, return (we found the TS)
3. Perturb along unstable direction to escape current saddle
4. Run (n-1)-HiSD to find index-(n-1) saddle
5. Recurse with the new saddle point

Key insight from Theorem 5.1: High-index saddles are connected through
a network of saddle-saddle connections. Recursive HiSD exploits this
by systematically descending the saddle hierarchy.

Unlike adaptive k-HiSD which increases k when stuck, Recursive HiSD
explicitly targets decreasing indices from the start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .k_hisd import compute_hisd_direction


@dataclass
class RecursiveHiSDState:
    """State for Recursive HiSD algorithm."""
    current_index: int              # Current target index (descending from initial)
    initial_index: int              # Starting Morse index
    recursion_depth: int = 0        # How many recursive descents we've done
    perturbed: bool = False         # Whether we've perturbed for current level
    converged_at_level: bool = False  # Whether converged at current level


@dataclass
class RecursiveHiSDConfig:
    """Configuration for Recursive HiSD."""
    # Perturbation settings
    perturb_magnitude: float = 0.01  # Magnitude of perturbation along unstable direction
    perturb_strategy: str = "unstable"  # "unstable", "random", "mixed"

    # Convergence at each level
    grad_threshold: float = 1e-5    # Gradient convergence threshold
    max_steps_per_level: int = 500  # Max steps at each k level

    # Step size
    dt: float = 0.005
    dt_min: float = 1e-6
    dt_max: float = 0.08
    max_atom_disp: float = 0.35

    # Eigenvalue thresholds
    tr_threshold: float = 1e-6
    neg_threshold: float = -1e-4


def perturb_along_unstable_direction(
    coords: torch.Tensor,
    evecs: torch.Tensor,
    evals: torch.Tensor,
    magnitude: float = 0.01,
    neg_threshold: float = -1e-4,
    strategy: str = "unstable",
) -> torch.Tensor:
    """Perturb coordinates along unstable (negative curvature) directions.

    Args:
        coords: Current coordinates (N, 3)
        evecs: Eigenvectors (3N, m) sorted by eigenvalue ascending
        evals: Eigenvalues sorted ascending
        magnitude: Perturbation magnitude in Angstrom
        neg_threshold: Threshold for negative eigenvalue
        strategy: "unstable" - along most negative direction
                  "random" - random direction
                  "mixed" - combination of unstable + random

    Returns:
        Perturbed coordinates (N, 3)
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    num_atoms = coords.shape[0]

    if strategy == "random":
        # Pure random perturbation
        perturb = torch.randn_like(coords) * magnitude
        return coords + perturb

    # Find negative eigenvalue modes
    neg_mask = evals < neg_threshold
    neg_indices = torch.where(neg_mask)[0]

    if len(neg_indices) == 0:
        # No negative modes, use random perturbation
        perturb = torch.randn_like(coords) * magnitude
        return coords + perturb

    if strategy == "unstable":
        # Perturb along the most negative (first) eigenvector
        v_unstable = evecs[:, neg_indices[0]].reshape(num_atoms, 3)
        # Random sign to escape in either direction
        sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
        perturb = sign * magnitude * v_unstable / v_unstable.norm()

    elif strategy == "mixed":
        # Combination: mainly unstable + small random component
        v_unstable = evecs[:, neg_indices[0]].reshape(num_atoms, 3)
        sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
        unstable_perturb = sign * 0.8 * magnitude * v_unstable / v_unstable.norm()
        random_perturb = 0.2 * magnitude * torch.randn_like(coords)
        perturb = unstable_perturb + random_perturb
    else:
        raise ValueError(f"Unknown perturbation strategy: {strategy}")

    return coords + perturb


def recursive_hisd_step(
    coords: torch.Tensor,
    forces: torch.Tensor,
    hessian_proj: torch.Tensor,
    dt: float,
    target_k: int,
    *,
    tr_threshold: float = 1e-6,
    neg_threshold: float = -1e-4,
) -> Tuple[torch.Tensor, Dict]:
    """Take one k-HiSD step targeting index-k saddles.

    In Recursive HiSD, target_k = n-1 where n is current Morse index.
    This makes the current index-n saddle UNSTABLE while index-(n-1)
    saddles are STABLE fixed points.

    Args:
        coords: Current coordinates (N, 3)
        forces: Current forces (N, 3)
        hessian_proj: Projected Hessian (3N, 3N)
        dt: Timestep
        target_k: Target index (k in k-HiSD)
        tr_threshold: TR mode filter threshold
        neg_threshold: Negative eigenvalue threshold

    Returns:
        new_coords: Updated coordinates (N, 3)
        info: Step information dictionary
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
            "direction_norm": float(forces_flat.norm().item()),
        }

    vib_evals = evals[vib_indices]
    vib_evecs = evecs[:, vib_indices]

    # Compute current Morse index
    morse_index = int((vib_evals < neg_threshold).sum().item())

    # Use target_k for HiSD direction
    k_used = max(1, min(target_k, len(vib_evals)))

    # Compute k-HiSD direction: -R * gradient
    direction = compute_hisd_direction(grad_flat, vib_evecs, k_used)
    direction_3d = direction.reshape(num_atoms, 3)

    # Take step
    new_coords = coords + dt * direction_3d

    # Extract eigenvalue spectrum
    n_eigs = min(6, len(vib_evals))
    eig_spectrum = [float(vib_evals[i].item()) for i in range(n_eigs)]

    info = {
        "k_used": k_used,
        "target_k": target_k,
        "morse_index": morse_index,
        "eig_spectrum": eig_spectrum,
        "direction_norm": float(direction.norm().item()),
        "grad_norm": grad_norm,
        "direction_type": f"{k_used}-HiSD",
    }

    return new_coords.detach(), info


def run_recursive_hisd_level(
    predict_fn: Callable,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    get_projected_hessian_fn: Callable,
    target_k: int,
    config: RecursiveHiSDConfig,
    scine_elements=None,
) -> Tuple[torch.Tensor, bool, List[Dict]]:
    """Run HiSD at a single level until convergence or max steps.

    This runs k-HiSD with k = target_k until either:
    1. Gradient converges (found index-k saddle)
    2. Max steps reached

    Args:
        predict_fn: Energy/force/Hessian prediction function
        coords: Starting coordinates
        atomic_nums: Atomic numbers
        get_projected_hessian_fn: Hessian projection function
        target_k: Target Morse index for this level
        config: RecursiveHiSDConfig
        scine_elements: Optional SCINE element types

    Returns:
        final_coords: Coordinates after convergence/max_steps
        converged: Whether gradient converged
        trajectory: List of step info dicts
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3).detach().clone()

    trajectory = []
    dt_eff = config.dt
    converged = False

    for step in range(config.max_steps_per_level):
        # Get energy, forces, Hessian
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        forces = out["forces"]
        hessian = out["hessian"]
        energy = out.get("energy", 0)

        # Project Hessian
        hess_proj = get_projected_hessian_fn(hessian, coords, atomic_nums, scine_elements)

        # Take k-HiSD step
        new_coords, info = recursive_hisd_step(
            coords,
            forces,
            hess_proj,
            dt_eff,
            target_k,
            tr_threshold=config.tr_threshold,
            neg_threshold=config.neg_threshold,
        )

        # Apply max displacement cap
        step_vec = new_coords - coords
        max_disp = float(step_vec.norm(dim=1).max().item())
        if max_disp > config.max_atom_disp and max_disp > 0:
            scale = config.max_atom_disp / max_disp
            new_coords = coords + scale * step_vec
            dt_eff = max(dt_eff * 0.8, config.dt_min)
        else:
            dt_eff = min(dt_eff * 1.05, config.dt_max)

        info["step"] = step
        info["energy"] = float(energy.item()) if hasattr(energy, "item") else float(energy)
        info["dt_eff"] = dt_eff
        trajectory.append(info)

        # Check convergence
        if info["grad_norm"] < config.grad_threshold:
            # Also verify we've reached the target index
            if info["morse_index"] == target_k:
                converged = True
                break

        coords = new_coords

    return coords, converged, trajectory


def run_recursive_hisd(
    predict_fn: Callable,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    get_projected_hessian_fn: Callable,
    *,
    config: Optional[RecursiveHiSDConfig] = None,
    scine_elements=None,
    max_recursion_depth: int = 10,
) -> Tuple[Dict, List[Dict]]:
    """Run full Recursive HiSD algorithm.

    Recursively descends from high-index saddle to index-1 TS:
    1. Detect current Morse index n
    2. If n == 1, done
    3. Perturb to escape current saddle
    4. Run (n-1)-HiSD to find index-(n-1) saddle
    5. Recurse

    Args:
        predict_fn: Energy/force/Hessian prediction function
        coords0: Starting coordinates (should be near a saddle)
        atomic_nums: Atomic numbers
        get_projected_hessian_fn: Hessian projection function
        config: RecursiveHiSDConfig (uses defaults if None)
        scine_elements: Optional SCINE element types
        max_recursion_depth: Maximum descent levels

    Returns:
        result: Dictionary with final state
        full_trajectory: Combined trajectory from all levels
    """
    if config is None:
        config = RecursiveHiSDConfig()

    if coords0.dim() == 3 and coords0.shape[0] == 1:
        coords0 = coords0[0]
    coords = coords0.reshape(-1, 3).detach().clone()

    full_trajectory = []
    level_info = []

    # Get initial Morse index
    out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
    hess_proj = get_projected_hessian_fn(out["hessian"], coords, atomic_nums, scine_elements)
    evals, evecs = torch.linalg.eigh(hess_proj)

    vib_mask = torch.abs(evals) > config.tr_threshold
    vib_evals = evals[vib_mask]
    vib_evecs = evecs[:, vib_mask]

    initial_index = int((vib_evals < config.neg_threshold).sum().item())
    current_index = initial_index

    state = RecursiveHiSDState(
        current_index=current_index,
        initial_index=initial_index,
    )

    # Recursive descent
    for depth in range(max_recursion_depth):
        state.recursion_depth = depth

        # Base case: reached index-1
        if current_index <= 1:
            # Run 1-HiSD (= GAD) to converge to TS
            target_k = 1
            final_coords, converged, traj = run_recursive_hisd_level(
                predict_fn, coords, atomic_nums, get_projected_hessian_fn,
                target_k, config, scine_elements,
            )

            for t in traj:
                t["recursion_depth"] = depth
                t["target_index"] = target_k
            full_trajectory.extend(traj)

            level_info.append({
                "depth": depth,
                "target_k": target_k,
                "steps": len(traj),
                "converged": converged,
            })

            coords = final_coords
            break

        # Target the next lower index
        target_k = current_index - 1

        # Perturb to escape current saddle
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        hess_proj = get_projected_hessian_fn(out["hessian"], coords, atomic_nums, scine_elements)
        evals, evecs = torch.linalg.eigh(hess_proj)

        vib_mask = torch.abs(evals) > config.tr_threshold
        vib_evals = evals[vib_mask]
        vib_evecs = evecs[:, vib_mask]

        coords = perturb_along_unstable_direction(
            coords,
            vib_evecs,
            vib_evals,
            magnitude=config.perturb_magnitude,
            neg_threshold=config.neg_threshold,
            strategy=config.perturb_strategy,
        )

        # Run k-HiSD at this level
        final_coords, converged, traj = run_recursive_hisd_level(
            predict_fn, coords, atomic_nums, get_projected_hessian_fn,
            target_k, config, scine_elements,
        )

        for t in traj:
            t["recursion_depth"] = depth
            t["target_index"] = target_k
        full_trajectory.extend(traj)

        level_info.append({
            "depth": depth,
            "target_k": target_k,
            "steps": len(traj),
            "converged": converged,
        })

        # Update for next level
        if traj:
            current_index = traj[-1]["morse_index"]
        coords = final_coords

    # Final analysis
    final_out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
    hess_proj = get_projected_hessian_fn(final_out["hessian"], coords, atomic_nums, scine_elements)
    evals, _ = torch.linalg.eigh(hess_proj)
    vib_mask = torch.abs(evals) > config.tr_threshold
    vib_evals = evals[vib_mask]
    final_index = int((vib_evals < config.neg_threshold).sum().item())

    result = {
        "initial_index": initial_index,
        "final_index": final_index,
        "converged": final_index == 1,
        "total_steps": len(full_trajectory),
        "recursion_depth": state.recursion_depth,
        "level_info": level_info,
        "final_coords": coords.detach().cpu(),
        "algorithm": "recursive_hisd",
    }

    return result, full_trajectory
