"""Minimization baselines: fixed-step gradient descent and Newton-Raphson.

Both methods find energy MINIMA on the potential energy surface.

1. Fixed-step gradient descent:
   x_{k+1} = x_k - alpha * grad E(x_k)
   = x_k + alpha * forces(x_k)

2. Newton-Raphson:
   x_{k+1} = x_k - H(x_k)^{-1} * grad E(x_k)
   = x_k + H(x_k)^{-1} * forces(x_k)

   The inverse Hessian is computed via pseudoinverse in the vibrational
   subspace to avoid singularities from translation/rotation modes.

Both support optional Eckart projection of the gradient to prevent
translation/rotation drift.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from src.dependencies.differentiable_projection import (
    project_vector_to_vibrational_torch,
)


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


def _cap_displacement(step_disp: torch.Tensor, max_atom_disp: float) -> torch.Tensor:
    """Cap per-atom displacement to max_atom_disp."""
    disp_3d = step_disp.reshape(-1, 3)
    max_disp = float(disp_3d.norm(dim=1).max().item())
    if max_disp > max_atom_disp and max_disp > 0:
        disp_3d = disp_3d * (max_atom_disp / max_disp)
    return disp_3d.reshape(step_disp.shape)


def run_fixed_step_gd(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    get_projected_hessian_fn,
    *,
    n_steps: int = 5000,
    step_size: float = 0.01,
    max_atom_disp: float = 0.3,
    force_converged: float = 1e-4,
    min_interatomic_dist: float = 0.5,
    tr_threshold: float = 1e-6,
    project_gradient_and_v: bool = False,
    atomsymbols: Optional[list] = None,
    scine_elements=None,
    purify_hessian: bool = False,
) -> Tuple[Dict[str, Any], list]:
    """Fixed-step gradient descent to find energy minimum.

    Update rule: x_{k+1} = x_k + alpha * forces(x_k)

    No line search, no adaptive step sizing. Pure fixed-step descent.
    Convergence: force norm < threshold AND zero negative vibrational eigenvalues.

    Args:
        predict_fn: Energy/force prediction function.
        coords0: Starting coordinates.
        atomic_nums: Atomic numbers.
        get_projected_hessian_fn: Function to compute projected Hessian.
        n_steps: Maximum number of steps.
        step_size: Fixed step size alpha.
        max_atom_disp: Maximum per-atom displacement per step.
        force_converged: Force convergence threshold (eV/A).
        min_interatomic_dist: Minimum allowed interatomic distance.
        tr_threshold: Threshold for filtering translation/rotation modes.
        project_gradient_and_v: If True, Eckart-project the gradient before stepping.
        atomsymbols: Atom symbols (required if project_gradient_and_v=True).
        scine_elements: SCINE element list (passed to get_projected_hessian_fn).
        purify_hessian: If True, enforce translational sum rules on Hessian.

    Returns:
        result: Summary dictionary.
        trajectory: Per-step data.
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

        # Compute Hessian eigenvalues to check for minimum (n_neg == 0)
        hess_proj = get_projected_hessian_fn(
            hessian, coords, atomic_nums, scine_elements,
            purify_hessian=purify_hessian,
        )
        evals, _ = torch.linalg.eigh(hess_proj)
        vib_mask = torch.abs(evals) > tr_threshold
        vib_evals = evals[vib_mask] if vib_mask.any() else evals
        n_neg = int((vib_evals < -tr_threshold).sum().item())

        trajectory.append({
            "step": step,
            "energy": energy,
            "force_norm": force_norm,
            "step_size": step_size,
            "n_neg_evals": n_neg,
            "min_dist": _min_interatomic_distance(coords),
        })

        # Converged: small forces AND zero negative eigenvalues (true minimum)
        if force_norm < force_converged and n_neg == 0:
            return {
                "converged": True,
                "converged_step": step,
                "final_energy": energy,
                "final_force_norm": force_norm,
                "final_coords": coords.detach().cpu(),
                "total_steps": step + 1,
                "final_n_neg_evals": n_neg,
            }, trajectory

        # Optionally project gradient to remove TR components
        forces_flat = forces.reshape(-1)
        if project_gradient_and_v and atomsymbols is not None:
            forces_flat = project_vector_to_vibrational_torch(
                forces_flat, coords, atomsymbols,
            )

        # Fixed-step update: x_{k+1} = x_k + alpha * forces
        step_disp = step_size * forces_flat.reshape(-1, 3)
        step_disp = _cap_displacement(step_disp, max_atom_disp)

        new_coords = coords + step_disp
        if _min_interatomic_distance(new_coords) < min_interatomic_dist:
            # Halve displacement until geometry is valid
            for scale in [0.5, 0.25, 0.1, 0.05]:
                new_coords = coords + step_disp * scale
                if _min_interatomic_distance(new_coords) >= min_interatomic_dist:
                    break
            else:
                continue  # skip step entirely if nothing works

        coords = new_coords.detach()

    return {
        "converged": False,
        "converged_step": None,
        "final_energy": trajectory[-1]["energy"] if trajectory else float("nan"),
        "final_force_norm": trajectory[-1]["force_norm"] if trajectory else float("nan"),
        "final_coords": coords.detach().cpu(),
        "total_steps": len(trajectory),
        "final_n_neg_evals": trajectory[-1]["n_neg_evals"] if trajectory else -1,
    }, trajectory


def run_newton_raphson(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    get_projected_hessian_fn,
    *,
    n_steps: int = 5000,
    max_atom_disp: float = 0.3,
    force_converged: float = 1e-4,
    min_interatomic_dist: float = 0.5,
    tr_threshold: float = 1e-6,
    project_gradient_and_v: bool = True,
    atomsymbols: Optional[list] = None,
    scine_elements=None,
    purify_hessian: bool = False,
    known_ts_coords: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, Any], list]:
    """Newton-Raphson optimization to find energy minimum with Trust Region.

    Update rule: x_{k+1} = x_k - H(x_k)^{-1} * grad E(x_k)

    The inverse Hessian step is computed via pseudoinverse in the
    vibrational subspace, using absolute values of eigenvalues to ensure
    it is always a descent direction.

    Adaptive step scaling (Trust Region):
    Predicted energy change dE_pred = g^T dx + 0.5 dx^T H dx
    Actual energy change dE_actual = E_new - E_old
    rho = dE_actual / dE_pred
    If rho > 0.75, increase trust radius.
    If rho < 0.25, decrease trust radius.
    Steps with dE_actual > 0 are rejected and retried with a smaller radius.

    Convergence criteria: ONLY Zero negative vibrational eigenvalues (n_neg == 0).
    """
    coords = coords0.detach().clone().to(torch.float32)
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)

    coords0_reshaped = coords.clone()

    if known_ts_coords is not None:
        known_ts_coords = known_ts_coords.detach().clone().to(torch.float32).to(coords.device)
        if known_ts_coords.dim() == 3 and known_ts_coords.shape[0] == 1:
            known_ts_coords = known_ts_coords[0]
        known_ts_coords = known_ts_coords.reshape(-1, 3)

    trajectory = []
    
    current_trust_radius = max_atom_disp
    
    # Evaluate initial state
    out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
    
    step = 0
    while step < n_steps:
        energy = _to_float(out["energy"])
        forces = out["forces"]
        hessian = out["hessian"]
        
        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)

        force_norm = _force_mean(forces)

        hess_proj = get_projected_hessian_fn(
            hessian, coords, atomic_nums, scine_elements,
            purify_hessian=purify_hessian,
        )

        evals, evecs = torch.linalg.eigh(hess_proj)
        vib_mask = torch.abs(evals) > tr_threshold
        vib_evals = evals[vib_mask] if vib_mask.any() else evals
        n_neg = int((vib_evals < -tr_threshold).sum().item())

        vib_pos = vib_evals[vib_evals > tr_threshold]
        eff_step = float(1.0 / vib_pos.min().item()) if vib_pos.numel() > 0 else float("nan")

        if vib_evals.numel() > 0:
            min_abs_vib = float(torch.abs(vib_evals).min().item())
            max_abs_vib = float(torch.abs(vib_evals).max().item())
            cond_num = max_abs_vib / min_abs_vib if min_abs_vib > 0 else float("inf")
        else:
            cond_num = float("nan")

        disp_from_start_max = float((coords - coords0_reshaped).norm(dim=1).max().item())
        dist_to_ts_max = float((coords - known_ts_coords).norm(dim=1).max().item()) if known_ts_coords is not None else None

        trajectory.append({
            "step": step,
            "energy": energy,
            "force_norm": force_norm,
            "n_neg_evals": n_neg,
            "min_vib_eval": float(vib_evals.min().item()) if vib_evals.numel() > 0 else float("nan"),
            "max_vib_eval": float(vib_evals.max().item()) if vib_evals.numel() > 0 else float("nan"),
            "cond_num": cond_num,
            "eff_step_size": eff_step,
            "min_dist": _min_interatomic_distance(coords),
            "trust_radius": current_trust_radius,
            "disp_from_start_max": disp_from_start_max,
            "dist_to_ts_max": dist_to_ts_max,
        })

        # Convergence criteria: ONLY ZERO NEGATIVE EIGENVALUES
        if n_neg == 0:
            return {
                "converged": True,
                "converged_step": step,
                "final_energy": energy,
                "final_force_norm": force_norm,
                "final_coords": coords.detach().cpu(),
                "total_steps": step + 1,
                "final_n_neg_evals": n_neg,
            }, trajectory

        # ALWAYS use project_gradient_and_v for NR
        grad = -forces.reshape(-1)
        if atomsymbols is not None:
            grad = -project_vector_to_vibrational_torch(
                forces.reshape(-1), coords, atomsymbols,
            )

        # Newton-Raphson step explicitly using the Hessian eigenvectors/values
        vib_indices = torch.where(vib_mask)[0]
        if vib_indices.numel() > 0:
            V_vib = evecs[:, vib_indices]  
            lam_vib = evals[vib_indices]   
            coeffs = V_vib.T @ grad.to(V_vib.dtype)  
            
            # Use absolute value of eigenvalues to ensure descent direction!
            inv_lam_min = 1.0 / torch.abs(lam_vib)
            
            nr_step = V_vib @ (inv_lam_min * coeffs)
            delta_x = -nr_step 
        else:
            delta_x = forces.reshape(-1) * 0.001

        step_disp = delta_x.to(coords.dtype).reshape(-1, 3)
        
        # Adaptive step sizing (Trust Region + Line search)
        accepted = False
        max_retries = 10
        retries = 0
        
        while not accepted and retries < max_retries:
            radius_used_for_step = current_trust_radius
            capped_disp = _cap_displacement(step_disp, radius_used_for_step)
            
            # Predict energy change using explicit Hessian: dE = g^T dx + 0.5 dx^T H dx
            dx_flat = capped_disp.reshape(-1)
            pred_dE = float((grad.dot(dx_flat) + 0.5 * dx_flat.dot(hess_proj @ dx_flat)).item())
            
            new_coords = coords + capped_disp
            
            if _min_interatomic_distance(new_coords) < min_interatomic_dist:
                current_trust_radius *= 0.5
                retries += 1
                continue
                
            # Evaluate new energy
            out_new = predict_fn(new_coords, atomic_nums, do_hessian=True, require_grad=False)
            energy_new = _to_float(out_new["energy"])
            
            actual_dE = energy_new - energy
            
            # Check acceptance (allowing tiny numerical noise increase)
            if actual_dE <= 1e-5:
                accepted = True
                
                # Update trust radius based on rho
                rho = actual_dE / pred_dE if pred_dE < -1e-8 else 0.0
                
                # Dynamic Trust Region growth and shrinkage
                if rho > 0.75:
                    current_trust_radius = min(current_trust_radius * 1.5, max_atom_disp)
                elif rho < 0.25:
                    current_trust_radius = max(current_trust_radius * 0.5, 0.001)
                elif rho < 0.0:
                    current_trust_radius = max(current_trust_radius * 0.25, 0.001)
                    
                coords = new_coords.detach()
                out = out_new
            else:
                # Reject step, shrink trust radius sharply, and retry
                current_trust_radius *= 0.25
                retries += 1
                
        if not accepted:
            # If we failed to find a good step after retries, just take the smallest step and continue
            coords = new_coords.detach()
            out = out_new

        actual_step_disp = float(capped_disp.reshape(-1, 3).norm(dim=1).max().item())
        trajectory[-1]["actual_step_disp"] = actual_step_disp
        trajectory[-1]["hit_trust_radius"] = bool(actual_step_disp >= radius_used_for_step * 0.99)
        trajectory[-1]["retries"] = retries

        step += 1

    return {
        "converged": False,
        "converged_step": None,
        "final_energy": trajectory[-1]["energy"] if trajectory else float("nan"),
        "final_force_norm": trajectory[-1]["force_norm"] if trajectory else float("nan"),
        "final_coords": coords.detach().cpu(),
        "total_steps": len(trajectory),
        "final_n_neg_evals": trajectory[-1]["n_neg_evals"] if trajectory else -1,
    }, trajectory
