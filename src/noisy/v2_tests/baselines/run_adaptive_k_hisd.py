"""Adaptive k-HiSD runner with diagnostic logging.

**CORRECTED IMPLEMENTATION**

The key insight from iHiSD Theorem 3.2:
    "Index-k saddles are STABLE fixed points of k-HiSD"

This means:
- k-HiSD with k = Morse index STABILIZES the current saddle (WRONG!)
- k-HiSD with k < Morse index makes the current saddle UNSTABLE (CORRECT!)

**Correct Algorithm**:
1. Start with k=1 (same as GAD, targets index-1 saddles)
2. Take k-HiSD steps normally
3. When STUCK (grad small but Morse index > k):
   - Increase k by 1
   - This makes the current high-index saddle UNSTABLE
   - We escape and continue
4. Eventually reach index-1 where k=1 is stable

Note: 1-HiSD IS mathematically identical to GAD:
    1-HiSD: -R∇E = -(I - 2v₁v₁ᵀ)∇E = -∇E + 2(v₁ᵀ∇E)v₁ = GAD direction
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Import existing components
from src.dependencies.hessian import (
    vibrational_eigvals,
    get_scine_elements_from_predict_output,
)

# Import from existing multi_mode_eckartmw
from src.noisy.multi_mode_eckartmw import (
    get_projected_hessian,
    _force_mean,
    _to_float,
)

# Import k-HiSD with corrected logic
from .k_hisd import adaptive_k_hisd_step, AdaptiveKState

# Import logging infrastructure
from ..logging import TrajectoryLogger


def run_adaptive_k_hisd(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    n_steps: int,
    dt: float,
    stop_at_ts: bool,
    ts_eps: float,
    dt_min: float,
    dt_max: float,
    max_atom_disp: Optional[float],
    tr_threshold: float = 1e-6,
    neg_threshold: float = -1e-4,
    grad_stuck_threshold: float = 1e-4,
    stuck_threshold_steps: int = 10,
    # Adaptive dt control
    dt_control: str = "adaptive",
    dt_grow_factor: float = 1.1,
    dt_shrink_factor: float = 0.5,
    # Logging
    sample_id: str = "unknown",
    formula: str = "",
    known_ts_coords: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, Any], TrajectoryLogger]:
    """Run CORRECTED adaptive k-HiSD with diagnostic logging.

    Key difference from old implementation:
    - OLD: k = Morse index always → STABILIZES current saddle (wrong!)
    - NEW: k starts at 1, increases when stuck → DESTABILIZES current saddle

    Args:
        predict_fn: Energy/force/Hessian prediction function
        coords0: Starting coordinates (N, 3)
        atomic_nums: Atomic numbers
        n_steps: Maximum number of steps
        dt: Initial timestep
        stop_at_ts: Stop when index-1 TS found
        ts_eps: Threshold for TS detection (eig₀ * eig₁ < -ts_eps)
        dt_min: Minimum timestep
        dt_max: Maximum timestep
        max_atom_disp: Maximum per-atom displacement per step
        tr_threshold: Threshold for TR mode filtering
        neg_threshold: Threshold for "negative" eigenvalue
        grad_stuck_threshold: Gradient norm threshold for "stuck" detection
        stuck_threshold_steps: Steps stuck before increasing k
        dt_control: "fixed" or "adaptive"
        dt_grow_factor: Factor to grow dt when making progress
        dt_shrink_factor: Factor to shrink dt when regressing
        sample_id: Sample identifier for logging
        formula: Chemical formula for logging
        known_ts_coords: Known TS for validation

    Returns:
        final_out_dict: Results dictionary
        trajectory_logger: Full diagnostic data
    """
    coords = coords0.detach().clone().to(torch.float32)
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)

    # Initialize trajectory logger
    logger = TrajectoryLogger(sample_id=sample_id, formula=formula)
    logger.known_ts_coords = known_ts_coords

    # Initialize ADAPTIVE STATE - starts at k=1
    state = AdaptiveKState(
        k=1,  # START AT k=1 (GAD behavior)
        k_target=1,  # We want index-1 saddles
        stuck_counter=0,
        stuck_threshold=stuck_threshold_steps,
    )

    # Trajectory storage
    trajectory = {k: [] for k in [
        "energy", "force_mean", "eig0", "eig1", "eig_product", "neg_vib",
        "disp_from_last", "disp_from_start", "dt_eff", "gad_norm",
        "k_used", "k_next", "morse_index", "direction_type",
        "stuck", "stuck_counter", "k_increased",
        # Extended eigenvalue spectrum
        "eig_0", "eig_1", "eig_2", "eig_3", "eig_4", "eig_5",
    ]}

    start_pos = coords.clone()
    prev_pos = coords.clone()
    prev_energy = None
    prev_morse_index = None
    disp_history: list[float] = []

    steps_to_ts: Optional[int] = None
    total_steps = 0
    dt_eff = float(dt)

    while total_steps < n_steps:
        # Get energy, forces, Hessian
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        energy = out.get("energy")
        forces = out.get("forces")
        hessian = out.get("hessian")

        energy_value = _to_float(energy)
        force_mean = _force_mean(forces)

        scine_elements = get_scine_elements_from_predict_output(out)

        # Get projected Hessian
        hess_proj = get_projected_hessian(hessian, coords, atomic_nums, scine_elements=scine_elements)

        # Take adaptive k-HiSD step with STATE TRACKING
        new_coords, step_info, new_state = adaptive_k_hisd_step(
            coords,
            forces,
            hess_proj,
            dt_eff,
            state,  # Pass current state
            tr_threshold=tr_threshold,
            neg_threshold=neg_threshold,
            grad_stuck_threshold=grad_stuck_threshold,
        )

        k_used = step_info["k_used"]
        morse_index = step_info["morse_index"]
        eig_spectrum = step_info["eig_spectrum"]

        # Eigenvalue products for TS detection
        eig0 = eig_spectrum[0] if len(eig_spectrum) > 0 else float("nan")
        eig1 = eig_spectrum[1] if len(eig_spectrum) > 1 else float("nan")
        eig_prod = eig0 * eig1 if np.isfinite(eig0) and np.isfinite(eig1) else float("inf")

        # Displacements
        disp_from_last = float((coords - prev_pos).norm(dim=1).mean().item()) if total_steps > 0 else 0.0
        disp_from_start = float((coords - start_pos).norm(dim=1).mean().item()) if total_steps > 0 else 0.0

        gad_norm = step_info["direction_norm"]

        # Log to trajectory
        trajectory["energy"].append(energy_value)
        trajectory["force_mean"].append(force_mean)
        trajectory["eig0"].append(eig0)
        trajectory["eig1"].append(eig1)
        trajectory["eig_product"].append(eig_prod)
        trajectory["neg_vib"].append(morse_index)
        trajectory["disp_from_last"].append(disp_from_last)
        trajectory["disp_from_start"].append(disp_from_start)
        trajectory["dt_eff"].append(dt_eff)
        trajectory["gad_norm"].append(gad_norm)
        trajectory["k_used"].append(k_used)
        trajectory["k_next"].append(step_info["k_next"])
        trajectory["morse_index"].append(morse_index)
        trajectory["direction_type"].append(step_info["direction_type"])
        trajectory["stuck"].append(step_info["stuck"])
        trajectory["stuck_counter"].append(step_info["stuck_counter"])
        trajectory["k_increased"].append(step_info["k_increased"])

        # Extended spectrum
        for i in range(6):
            key = f"eig_{i}"
            trajectory[key].append(eig_spectrum[i] if i < len(eig_spectrum) else float("nan"))

        if total_steps > 0:
            disp_history.append(disp_from_last)
        x_disp_window = float(np.mean(disp_history[-10:])) if disp_history else float("nan")

        # Log to TrajectoryLogger
        logger.log_step(
            step=total_steps,
            coords=coords,
            energy=energy_value,
            forces=forces,
            hessian_proj=hess_proj,
            gad_vec=forces,  # Placeholder
            dt_eff=dt_eff,
            coords_prev=prev_pos if total_steps > 0 else None,
            energy_prev=prev_energy,
            x_disp_window=x_disp_window,
        )

        # Check for TS (index = 1 and product condition)
        if stop_at_ts and steps_to_ts is None and morse_index == 1:
            if np.isfinite(eig_prod) and eig_prod < -abs(ts_eps):
                steps_to_ts = total_steps
                break

        # Adaptive dt control
        if dt_control == "adaptive" and prev_morse_index is not None:
            if morse_index < prev_morse_index:
                # Index decreased - good progress, grow dt
                dt_eff = min(dt_eff * dt_grow_factor, dt_max)
            elif morse_index > prev_morse_index:
                # Index increased - shrink dt
                dt_eff = max(dt_eff * dt_shrink_factor, dt_min)

        # Apply max atom displacement cap
        if max_atom_disp is not None and max_atom_disp > 0:
            step_disp = (new_coords - coords).reshape(-1, 3)
            max_disp = float(step_disp.norm(dim=1).max().item())
            if np.isfinite(max_disp) and max_disp > float(max_atom_disp) and max_disp > 0:
                scale = float(max_atom_disp) / max_disp
                new_coords = coords + scale * (new_coords - coords)

        # Update state
        prev_pos = coords.clone()
        prev_energy = energy_value
        prev_morse_index = morse_index
        state = new_state  # Update adaptive state
        coords = new_coords.detach()
        total_steps += 1

    # Final vibrational analysis
    final_neg_vib = -1
    try:
        final_out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        scine_elements = get_scine_elements_from_predict_output(final_out)
        final_vib_eigvals = vibrational_eigvals(final_out["hessian"], coords, atomic_nums, scine_elements=scine_elements)
        final_neg_vib = int((final_vib_eigvals < 0).sum().item())
    except Exception:
        pass

    # Finalize logger
    logger.finalize(
        final_coords=coords,
        final_morse_index=final_neg_vib,
        converged_to_ts=(steps_to_ts is not None) or (final_neg_vib == 1),
    )

    # Count k increases
    k_increases = sum(1 for k_inc in trajectory["k_increased"] if k_inc)
    max_k_used = max(trajectory["k_used"]) if trajectory["k_used"] else 1

    final_out_dict = {
        "final_coords": coords.detach().cpu(),
        "trajectory": trajectory,
        "steps_taken": total_steps,
        "steps_to_ts": steps_to_ts,
        "final_eig0": trajectory["eig0"][-1] if trajectory["eig0"] else None,
        "final_eig1": trajectory["eig1"][-1] if trajectory["eig1"] else None,
        "final_eig_product": trajectory["eig_product"][-1] if trajectory["eig_product"] else None,
        "final_neg_vibrational": final_neg_vib,
        "total_steps": total_steps,
        "final_k": state.k,
        "k_increases": k_increases,
        "max_k_used": max_k_used,
        "algorithm": "adaptive_k_hisd_corrected",
    }

    return final_out_dict, logger


def run_single_sample_adaptive_k_hisd(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    params: Dict[str, Any],
    n_steps: int,
    sample_id: str,
    formula: str,
    out_dir: str,
) -> Dict[str, Any]:
    """Run adaptive k-HiSD on a single sample with diagnostics."""
    t0 = time.time()

    diag_dir = Path(out_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    try:
        out_dict, traj_logger = run_adaptive_k_hisd(
            predict_fn,
            coords,
            atomic_nums,
            n_steps=n_steps,
            dt=params.get("dt", 0.005),
            stop_at_ts=params.get("stop_at_ts", True),
            ts_eps=params.get("ts_eps", 1e-5),
            dt_min=params.get("dt_min", 1e-6),
            dt_max=params.get("dt_max", 0.08),
            max_atom_disp=params.get("max_atom_disp", 0.35),
            tr_threshold=params.get("tr_threshold", 1e-6),
            neg_threshold=params.get("neg_threshold", -1e-4),
            grad_stuck_threshold=params.get("grad_stuck_threshold", 1e-4),
            stuck_threshold_steps=params.get("stuck_threshold_steps", 10),
            dt_control=params.get("dt_control", "adaptive"),
            dt_grow_factor=params.get("dt_grow_factor", 1.1),
            dt_shrink_factor=params.get("dt_shrink_factor", 0.5),
            sample_id=sample_id,
            formula=formula,
        )

        traj_logger.save(diag_dir)

        wall_time = time.time() - t0

        final_neg_vib = out_dict.get("final_neg_vibrational", -1)
        steps_taken = out_dict.get("steps_taken", n_steps)
        steps_to_ts = out_dict.get("steps_to_ts")

        return {
            "final_neg_vib": final_neg_vib,
            "steps_taken": steps_taken,
            "steps_to_ts": steps_to_ts,
            "success": final_neg_vib == 1,
            "wall_time": wall_time,
            "error": None,
            "algorithm": "adaptive_k_hisd_corrected",
            "k_increases": out_dict.get("k_increases", 0),
            "max_k_used": out_dict.get("max_k_used", 1),
            "final_k": out_dict.get("final_k", 1),
        }

    except Exception as e:
        wall_time = time.time() - t0
        return {
            "final_neg_vib": -1,
            "steps_taken": 0,
            "steps_to_ts": None,
            "success": False,
            "wall_time": wall_time,
            "error": str(e),
            "algorithm": "adaptive_k_hisd_corrected",
            "k_increases": 0,
            "max_k_used": 0,
            "final_k": 1,
        }
