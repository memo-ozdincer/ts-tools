"""Adaptive k-HiSD runner with diagnostic logging.

Unlike standard GAD + v₂ kicking, this uses the theoretically-justified
adaptive k-HiSD algorithm from the iHiSD paper:

1. At each step, compute Morse index k (count of negative eigenvalues)
2. Use k-HiSD direction: -R∇E where R = I - 2∑ᵢ₌₁ᵏ vᵢvᵢᵀ
3. This makes index-k saddles unstable, allowing descent to index-(k-1)
4. Eventually converge to index-1 transition state

Key difference from v₂ kicking:
- v₂ kick: only perturbs along 2nd mode (works for index-2, not index-5)
- Adaptive k-HiSD: reflects along ALL k negative modes continuously

Per iHiSD Theorem 3.2:
- k-HiSD is stable at index-k saddles
- To descend from index-k, you NEED k reflections
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Import existing components
from src.core_algos.gad import pick_tracked_mode
from src.dependencies.hessian import (
    vibrational_eigvals,
    get_scine_elements_from_predict_output,
)

# Import from existing multi_mode_eckartmw
from src.noisy.multi_mode_eckartmw import (
    get_projected_hessian,
    _force_mean,
    _mean_atom_norm,
    _max_atom_norm,
    _to_float,
)

# Import k-HiSD
from .k_hisd import adaptive_k_hisd_step, compute_adaptive_k

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
    min_k: int = 1,
    max_k: Optional[int] = None,
    # Adaptive dt control
    dt_control: str = "adaptive",
    dt_grow_factor: float = 1.1,
    dt_shrink_factor: float = 0.5,
    # Logging
    sample_id: str = "unknown",
    formula: str = "",
    known_ts_coords: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, Any], TrajectoryLogger]:
    """Run adaptive k-HiSD with comprehensive diagnostic logging.

    Unlike GAD + v₂ kicking:
    - No explicit "escape" mechanism needed
    - Continuously uses k-HiSD where k = Morse index
    - Naturally descends from high-index saddles

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
        min_k: Minimum k to use (1 = standard GAD behavior at index-1)
        max_k: Maximum k to use (None = no limit)
        dt_control: "fixed" or "adaptive"
        dt_grow_factor: Factor to grow dt when index decreases
        dt_shrink_factor: Factor to shrink dt when index increases
        sample_id: Sample identifier for logging
        formula: Chemical formula for logging
        known_ts_coords: Known TS for validation

    Returns:
        final_out_dict: Results dictionary
        trajectory_logger: Full diagnostic data
    """
    coords = coords0.detach().clone().to(torch.float32)

    # Initialize trajectory logger
    logger = TrajectoryLogger(sample_id=sample_id, formula=formula)
    logger.known_ts_coords = known_ts_coords

    # Trajectory storage
    trajectory = {k: [] for k in [
        "energy", "force_mean", "eig0", "eig1", "eig_product", "neg_vib",
        "disp_from_last", "disp_from_start", "dt_eff", "gad_norm",
        "k_used", "morse_index", "direction_type",
        # Extended eigenvalue spectrum
        "eig_0", "eig_1", "eig_2", "eig_3", "eig_4", "eig_5",
    ]}

    start_pos = coords.clone()
    prev_pos = coords.clone()
    prev_energy = None
    prev_k = None

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

        # Take adaptive k-HiSD step
        new_coords, step_info = adaptive_k_hisd_step(
            coords,
            forces,
            hess_proj,
            dt_eff,
            tr_threshold=tr_threshold,
            min_k=min_k,
            max_k=max_k,
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
        trajectory["morse_index"].append(morse_index)
        trajectory["direction_type"].append(step_info["direction_type"])

        # Extended spectrum
        for i in range(6):
            key = f"eig_{i}"
            trajectory[key].append(eig_spectrum[i] if i < len(eig_spectrum) else float("nan"))

        # Log to TrajectoryLogger for extended metrics
        # (Simplified - full logging would compute all ExtendedMetrics)
        logger.log_step(
            step=total_steps,
            coords=coords,
            energy=energy_value,
            forces=forces,
            hessian_proj=hess_proj,
            gad_vec=forces,  # Placeholder - actual direction is k-HiSD
            dt_eff=dt_eff,
            coords_prev=prev_pos if total_steps > 0 else None,
            energy_prev=prev_energy,
        )

        # Check for TS (index = 1 and product condition)
        if stop_at_ts and steps_to_ts is None and morse_index == 1:
            if np.isfinite(eig_prod) and eig_prod < -abs(ts_eps):
                steps_to_ts = total_steps
                break

        # Adaptive dt control
        if dt_control == "adaptive" and prev_k is not None:
            if morse_index < prev_k:
                # Index decreased - good progress, grow dt
                dt_eff = min(dt_eff * dt_grow_factor, dt_max)
            elif morse_index > prev_k:
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
        prev_k = morse_index
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
        "algorithm": "adaptive_k_hisd",
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
    """Run adaptive k-HiSD on a single sample with diagnostics.

    Args:
        predict_fn: Prediction function
        coords: Starting coordinates
        atomic_nums: Atomic numbers
        params: Algorithm parameters
        n_steps: Max steps
        sample_id: Sample identifier
        formula: Chemical formula
        out_dir: Output directory

    Returns:
        Result dictionary
    """
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
            dt_max=params.get("dt_max", 0.05),
            max_atom_disp=params.get("max_atom_disp", 0.25),
            tr_threshold=params.get("tr_threshold", 1e-6),
            min_k=params.get("min_k", 1),
            max_k=params.get("max_k", None),
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
            "algorithm": "adaptive_k_hisd",
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
            "algorithm": "adaptive_k_hisd",
        }
