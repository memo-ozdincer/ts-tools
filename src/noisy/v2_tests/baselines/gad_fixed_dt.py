"""GAD with FIXED dt (no adaptive control).

**CRITICAL HYPOTHESIS**:
The adaptive dt controller might be the root cause of GAD "stalling",
NOT a fundamental limitation of the GAD dynamics.

The typical failure pattern:
1. Morse index not improving → adaptive dt shrinks
2. Small dt → tiny displacements
3. Plateau detected → v₂ kick applied
4. v₂ kick RESETS dt to original value
5. GAD resumes with normal step sizes

If this hypothesis is correct:
- v₂ kicking works because it resets dt, not because v₂ is special
- Fixed dt GAD should work just as well (or better!) without kicks

This experiment tests:
1. Does GAD with fixed dt converge more reliably?
2. Does it avoid the "stalling" behavior entirely?
3. What is the effect of different fixed dt values?

If GAD with fixed dt works well, the solution is trivial:
**Just remove the adaptive dt controller.**
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.core_algos.gad import pick_tracked_mode
from src.noisy.multi_mode_eckartmw import (
    get_projected_hessian,
    _to_float,
    _mean_atom_norm,
    _min_interatomic_distance,
)
from src.dependencies.hessian import (
    get_scine_elements_from_predict_output,
    prepare_hessian,
)
from src.noisy.v2_tests.logging import TrajectoryLogger


def _vib_mask_from_evals(evals: torch.Tensor, tr_threshold: float) -> torch.Tensor:
    """Mask out translation/rotation (near-zero) modes."""
    return evals.abs() > float(tr_threshold)


def _step_metrics(
    forces: torch.Tensor,
    hessian_proj: torch.Tensor,
    tr_threshold: float,
    v_prev: Optional[torch.Tensor],
    track_mode: bool,
    k_track: int = 8,
) -> Tuple[torch.Tensor, float, float, int, Optional[torch.Tensor], float, int]:
    """Compute GAD vector and metrics from projected Hessian."""
    forces = forces[0] if forces.dim() == 3 and forces.shape[0] == 1 else forces
    forces = forces.reshape(-1, 3)
    num_atoms = int(forces.shape[0])

    hess = hessian_proj
    if hess.dim() != 2 or hess.shape[0] != 3 * num_atoms:
        hess = prepare_hessian(hess, num_atoms)

    evals, evecs = torch.linalg.eigh(hess)

    vib_mask = _vib_mask_from_evals(evals, tr_threshold)
    vib_indices = torch.where(vib_mask)[0]

    if len(vib_indices) == 0:
        evals_vib = evals
        candidate_indices = torch.arange(min(k_track, evecs.shape[1]), device=evecs.device)
    else:
        evals_vib = evals[vib_mask]
        candidate_indices = vib_indices[:min(k_track, len(vib_indices))]

    V = evecs[:, candidate_indices].to(device=forces.device, dtype=forces.dtype)
    if track_mode:
        v_prev_local = v_prev.to(device=forces.device, dtype=forces.dtype).reshape(-1) if v_prev is not None else None
    else:
        v_prev_local = None
    v_new, j, overlap = pick_tracked_mode(V, v_prev_local, k=int(V.shape[1]))
    v = v_new

    f_flat = forces.reshape(-1)
    gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
    gad_vec = gad_flat.view(num_atoms, 3)

    eig0 = float(evals_vib[0].item()) if len(evals_vib) >= 1 else float("nan")
    eig1 = float(evals_vib[1].item()) if len(evals_vib) >= 2 else float("nan")
    neg_vib = int((evals_vib < -tr_threshold).sum().item()) if len(evals_vib) > 0 else -1

    v_next = v.detach().clone().reshape(-1) if track_mode else None
    return gad_vec, eig0, eig1, neg_vib, v_next, overlap, int(j)


def run_gad_fixed_dt(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    n_steps: int = 5000,
    dt: float = 0.01,  # Fixed timestep - no adaptation
    max_atom_disp: float = 0.35,  # Only cap extreme displacements
    ts_eps: float = 1e-5,
    stop_at_ts: bool = True,
    min_interatomic_dist: float = 0.5,
    tr_threshold: float = 1e-6,
    track_mode: bool = True,
    scine_elements=None,
    log_dir: Optional[str] = None,
    sample_id: str = "unknown",
    formula: str = "",
) -> Tuple[Dict[str, Any], List[Dict], Optional[TrajectoryLogger]]:
    """Run GAD with FIXED dt - no adaptive control.

    This tests the hypothesis that adaptive dt is the root cause of stalling.

    If GAD with fixed dt converges reliably, then:
    1. The problem is the adaptive dt controller, not GAD
    2. The solution is to remove/fix the adaptive controller
    3. v₂ kicking is only needed because it resets dt

    Args:
        predict_fn: Energy/force/Hessian prediction function
        coords0: Starting coordinates
        atomic_nums: Atomic numbers
        n_steps: Maximum steps
        dt: FIXED timestep (no adaptation!)
        max_atom_disp: Maximum per-atom displacement (safety cap only)
        ts_eps: TS convergence threshold (eig0 * eig1 < -ts_eps)
        stop_at_ts: Stop when TS found
        min_interatomic_dist: Minimum interatomic distance
        tr_threshold: TR mode threshold
        scine_elements: SCINE element types

    Returns:
        result: Summary dictionary
        trajectory: Per-step data
    """
    coords = coords0.detach().clone().to(torch.float32)
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)

    trajectory = []
    logger = TrajectoryLogger(sample_id=sample_id, formula=formula) if log_dir else None
    v_prev = None
    converged = False
    converged_step = None
    start_pos = coords.clone()
    prev_pos = coords.clone()
    disp_history: List[float] = []

    for step in range(n_steps):
        # Get energy, forces, Hessian
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        energy = _to_float(out["energy"])
        forces = out["forces"]
        hessian = out["hessian"]

        if scine_elements is None:
            scine_elements = get_scine_elements_from_predict_output(out)

        # Get projected Hessian
        hess_proj = get_projected_hessian(hessian, coords, atomic_nums, scine_elements)

        # Compute GAD vector
        gad_vec, eig0, eig1, neg_vib, v_next, overlap, mode_index = _step_metrics(
            forces, hess_proj, tr_threshold, v_prev, track_mode
        )
        v_prev = v_next if track_mode else None

        eig_product = eig0 * eig1 if np.isfinite(eig0) and np.isfinite(eig1) else float("inf")
        gad_norm = _mean_atom_norm(gad_vec)
        disp_from_last = float((coords - prev_pos).norm(dim=1).mean().item()) if step > 0 else 0.0
        disp_from_start = float((coords - start_pos).norm(dim=1).mean().item())

        if step > 0:
            disp_history.append(disp_from_last)
        x_disp_window = float(np.mean(disp_history[-10:])) if disp_history else float("nan")

        trajectory.append({
            "step": step,
            "energy": energy,
            "eig0": eig0,
            "eig1": eig1,
            "eig_product": eig_product,
            "neg_vib": neg_vib,
            "gad_norm": gad_norm,
            "dt": dt,  # Always the same - FIXED
            "disp_from_last": disp_from_last,
            "disp_from_start": disp_from_start,
            "mode_overlap": overlap,
        })

        if logger is not None:
            logger.log_step(
                step=step,
                coords=coords,
                energy=energy,
                forces=forces,
                hessian_proj=hess_proj,
                gad_vec=gad_vec,
                dt_eff=dt,
                coords_prev=prev_pos if step > 0 else None,
                energy_prev=trajectory[-2]["energy"] if step > 0 else None,
                mode_index=mode_index,
                x_disp_window=x_disp_window,
            )

        # Check for TS (index = 1)
        if stop_at_ts and np.isfinite(eig_product) and eig_product < -abs(ts_eps):
            converged = True
            converged_step = step
            break

        # Take GAD step with FIXED dt
        step_disp = dt * gad_vec

        # Only cap if exceeds max_atom_disp (safety, not adaptation)
        max_disp = float(step_disp.norm(dim=1).max().item())
        if max_disp > max_atom_disp and max_disp > 0:
            scale = max_atom_disp / max_disp
            step_disp = scale * step_disp

        new_coords = coords + step_disp

        # Check geometry validity
        if _min_interatomic_distance(new_coords) < min_interatomic_dist:
            # Only scale down if geometry invalid
            step_disp = step_disp * 0.5
            new_coords = coords + step_disp

        prev_pos = coords.clone()
        coords = new_coords.detach()

    # Final analysis
    final_neg_vib = trajectory[-1]["neg_vib"] if trajectory else -1

    result = {
        "converged": converged,
        "converged_step": converged_step,
        "final_morse_index": final_neg_vib,
        "total_steps": len(trajectory),
        "final_energy": trajectory[-1]["energy"] if trajectory else float("nan"),
        "final_eig_product": trajectory[-1]["eig_product"] if trajectory else float("nan"),
        "algorithm": "gad_fixed_dt",
        "dt_used": dt,
        "success": converged or final_neg_vib == 1,
        "final_coords": coords.detach().cpu(),
    }

    if logger is not None:
        logger.finalize(
            final_coords=coords,
            final_morse_index=final_neg_vib,
            converged_to_ts=converged or final_neg_vib == 1,
        )
        logger.save(log_dir)
    return result, trajectory, logger


def run_gad_fixed_dt_sweep(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    dt_values: List[float] = None,
    n_steps: int = 5000,
    **kwargs,
) -> Dict[float, Tuple[Dict, List]]:
    """Sweep over different fixed dt values.

    This helps find the optimal fixed dt for a given system.

    Args:
        dt_values: List of dt values to try
        Other args passed to run_gad_fixed_dt

    Returns:
        Dictionary mapping dt → (result, trajectory)
    """
    if dt_values is None:
        dt_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    results = {}
    for dt in dt_values:
        print(f"  Testing dt={dt}...")
        result, trajectory, _logger = run_gad_fixed_dt(
            predict_fn,
            coords0.clone(),
            atomic_nums,
            n_steps=n_steps,
            dt=dt,
            **kwargs,
        )
        results[dt] = (result, trajectory)
        print(f"    → Success: {result['success']}, Steps: {result['total_steps']}, Final index: {result['final_morse_index']}")

    return results


def compare_fixed_vs_adaptive_dt(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    run_adaptive_fn,  # The current multi_mode_eckartmw function
    *,
    fixed_dt: float = 0.01,
    n_steps: int = 5000,
    **kwargs,
) -> Dict[str, Any]:
    """Direct comparison: fixed dt vs adaptive dt.

    This is the critical experiment to test whether adaptive dt is the problem.

    Returns:
        Comparison summary
    """
    print("Running fixed dt...")
    fixed_result, fixed_traj, _logger = run_gad_fixed_dt(
        predict_fn,
        coords0.clone(),
        atomic_nums,
        n_steps=n_steps,
        dt=fixed_dt,
        **kwargs,
    )

    print("Running adaptive dt (with v2 kicks)...")
    adaptive_result, adaptive_traj = run_adaptive_fn(
        predict_fn,
        coords0.clone(),
        atomic_nums,
        n_steps=n_steps,
        **kwargs,
    )

    return {
        "fixed_dt": {
            "success": fixed_result["success"],
            "steps": fixed_result["total_steps"],
            "final_index": fixed_result["final_morse_index"],
            "dt": fixed_dt,
        },
        "adaptive_dt": {
            "success": adaptive_result.get("success", adaptive_result.get("steps_to_ts") is not None),
            "steps": adaptive_result.get("steps_taken", adaptive_result.get("total_steps", 0)),
            "escape_cycles": adaptive_result.get("escape_cycles", 0),
            "final_index": adaptive_result.get("final_neg_vibrational", -1),
        },
        "hypothesis_supported": fixed_result["success"] and not adaptive_result.get("success", False),
    }
