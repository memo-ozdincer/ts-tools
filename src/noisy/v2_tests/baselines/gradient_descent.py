"""Pure gradient descent baseline.

This is a sanity check baseline that finds energy MINIMA.
If this doesn't converge, something is wrong with the energy surface or model.

Also useful for:
1. Verifying the model works correctly
2. Finding minima as starting points for TS search
3. Testing the "gradient descent escape" strategy
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().reshape(-1)[0].item())
    return float(x)


def _force_mean(forces: torch.Tensor) -> float:
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    f = forces.reshape(-1, 3)
    return float(f.norm(dim=1).mean().item())


def _min_interatomic_distance(coords: torch.Tensor) -> float:
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    n = coords.shape[0]
    if n < 2:
        return float("inf")
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    dist = diff.norm(dim=2)
    dist = dist + torch.eye(n, device=coords.device, dtype=coords.dtype) * 1e10
    return float(dist.min().item())


def run_gradient_descent(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    n_steps: int = 1000,
    step_size: float = 0.01,
    step_size_min: float = 1e-6,
    step_size_max: float = 0.1,
    max_atom_disp: float = 0.3,
    force_converged: float = 1e-4,
    min_interatomic_dist: float = 0.5,
    adaptive_step: bool = True,
    armijo_c: float = 1e-4,  # Armijo line search parameter
    backtrack_factor: float = 0.5,
    max_backtrack: int = 10,
) -> Tuple[Dict[str, Any], list]:
    """Run gradient descent to find energy minimum.

    This is a baseline to verify the energy model works.

    Uses optional Armijo backtracking line search for robustness.

    Args:
        predict_fn: Energy/force prediction function
        coords0: Starting coordinates
        atomic_nums: Atomic numbers
        n_steps: Maximum steps
        step_size: Initial step size
        step_size_min: Minimum step size
        step_size_max: Maximum step size
        max_atom_disp: Maximum per-atom displacement
        force_converged: Force convergence threshold (eV/Å)
        min_interatomic_dist: Minimum allowed interatomic distance
        adaptive_step: Use adaptive step sizing
        armijo_c: Armijo condition parameter
        backtrack_factor: Step size reduction factor
        max_backtrack: Maximum backtracking iterations

    Returns:
        result: Summary dictionary
        trajectory: Per-step data
    """
    coords = coords0.detach().clone().to(torch.float32)
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)

    trajectory = []
    step_size_eff = step_size
    prev_energy = None

    for step in range(n_steps):
        out = predict_fn(coords, atomic_nums, do_hessian=False, require_grad=False)
        energy = _to_float(out["energy"])
        forces = out["forces"]
        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)

        force_norm = _force_mean(forces)

        # Log
        trajectory.append({
            "step": step,
            "energy": energy,
            "force_norm": force_norm,
            "step_size": step_size_eff,
            "min_dist": _min_interatomic_distance(coords),
        })

        # Check convergence
        if force_norm < force_converged:
            return {
                "converged": True,
                "converged_step": step,
                "final_energy": energy,
                "final_force_norm": force_norm,
                "final_coords": coords.detach().cpu(),
                "total_steps": step + 1,
            }, trajectory

        # Direction: forces = -∇E, so move along forces
        direction = forces / (forces.norm() + 1e-12)

        # Backtracking line search
        current_step = step_size_eff
        for bt in range(max_backtrack):
            new_coords = coords + current_step * forces

            # Check geometry validity
            if _min_interatomic_distance(new_coords) < min_interatomic_dist:
                current_step *= backtrack_factor
                continue

            # Check Armijo condition: E(new) <= E(old) - c * step * ||f||²
            try:
                new_out = predict_fn(new_coords, atomic_nums, do_hessian=False, require_grad=False)
                new_energy = _to_float(new_out["energy"])
                if not np.isfinite(new_energy):
                    current_step *= backtrack_factor
                    continue

                force_sq = float((forces ** 2).sum().item())
                if new_energy <= energy - armijo_c * current_step * force_sq:
                    # Accept step
                    coords = new_coords
                    break
            except Exception:
                current_step *= backtrack_factor
                continue

            current_step *= backtrack_factor
        else:
            # All backtracking failed, take small step anyway
            new_coords = coords + step_size_min * forces
            if _min_interatomic_distance(new_coords) >= min_interatomic_dist:
                coords = new_coords

        # Adaptive step sizing
        if adaptive_step and prev_energy is not None:
            if energy < prev_energy:
                step_size_eff = min(step_size_eff * 1.1, step_size_max)
            else:
                step_size_eff = max(step_size_eff * 0.5, step_size_min)

        prev_energy = energy

    # Did not converge
    return {
        "converged": False,
        "converged_step": None,
        "final_energy": trajectory[-1]["energy"] if trajectory else float("nan"),
        "final_force_norm": trajectory[-1]["force_norm"] if trajectory else float("nan"),
        "final_coords": coords.detach().cpu(),
        "total_steps": len(trajectory),
    }, trajectory


def run_steepest_descent_with_hessian(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    get_projected_hessian_fn,
    *,
    n_steps: int = 1000,
    max_atom_disp: float = 0.3,
    force_converged: float = 1e-4,
    min_interatomic_dist: float = 0.5,
    tr_threshold: float = 1e-6,
    scine_elements=None,
) -> Tuple[Dict[str, Any], list]:
    """Steepest descent with Hessian-based step sizing.

    Uses the minimum eigenvalue to scale the step size optimally.
    This is more robust for ill-conditioned surfaces.
    """
    coords = coords0.detach().clone().to(torch.float32)
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)

    trajectory = []

    for step in range(n_steps):
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        energy = _to_float(out["energy"])
        forces = out["forces"]
        hessian = out["hessian"]

        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)

        force_norm = _force_mean(forces)

        # Get projected Hessian
        hess_proj = get_projected_hessian_fn(hessian, coords, atomic_nums, scine_elements)
        evals, _ = torch.linalg.eigh(hess_proj)

        # Filter TR modes
        vib_mask = torch.abs(evals) > tr_threshold
        vib_evals = evals[vib_mask] if vib_mask.any() else evals

        # For descent, we want positive eigenvalues (curvature)
        # If all negative, we're at a saddle - use magnitude of smallest
        min_pos_eval = vib_evals[vib_evals > tr_threshold].min() if (vib_evals > tr_threshold).any() else torch.abs(vib_evals).min()
        min_pos_eval = float(min_pos_eval.item())

        # Optimal step size for steepest descent: 1 / λ_max (approximately)
        # But we use 1 / |λ_min| for robustness
        step_size = min(1.0 / max(abs(min_pos_eval), 1e-6), max_atom_disp)

        trajectory.append({
            "step": step,
            "energy": energy,
            "force_norm": force_norm,
            "step_size": step_size,
            "min_eval": min_pos_eval,
            "n_neg_evals": int((vib_evals < -tr_threshold).sum().item()),
        })

        # Check convergence
        if force_norm < force_converged:
            return {
                "converged": True,
                "converged_step": step,
                "final_energy": energy,
                "final_force_norm": force_norm,
                "final_coords": coords.detach().cpu(),
                "total_steps": step + 1,
            }, trajectory

        # Take step
        new_coords = coords + step_size * forces

        # Check geometry
        if _min_interatomic_distance(new_coords) >= min_interatomic_dist:
            coords = new_coords
        else:
            # Reduce step size
            for scale in [0.5, 0.25, 0.1, 0.05]:
                new_coords = coords + step_size * scale * forces
                if _min_interatomic_distance(new_coords) >= min_interatomic_dist:
                    coords = new_coords
                    break

    return {
        "converged": False,
        "converged_step": None,
        "final_energy": trajectory[-1]["energy"] if trajectory else float("nan"),
        "final_force_norm": trajectory[-1]["force_norm"] if trajectory else float("nan"),
        "final_coords": coords.detach().cpu(),
        "total_steps": len(trajectory),
    }, trajectory
