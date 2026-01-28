"""iHiSD runner with diagnostic logging.

iHiSD (Improved High-index Saddle Dynamics) uses a crossover parameter
theta that smoothly transitions from gradient flow to full k-HiSD.

Key features:
- Nonlocal convergence: Can start outside the region of attraction
- Smooth transition: Avoids discontinuous jumps in dynamics
- Guaranteed convergence: Theorem proves convergence to index-k saddles

The crossover direction is:
    d = (1 - s*theta) * grad + 2*theta * sum_i(v_i * <grad, v_i>)

Where s = ±1 determines search direction (upward/downward).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.dependencies.hessian import (
    vibrational_eigvals,
    get_scine_elements_from_predict_output,
)
from src.noisy.multi_mode_eckartmw import (
    get_projected_hessian,
    _force_mean,
    _to_float,
)

from .ihisd import (
    IHiSDConfig,
    IHiSDState,
    ihisd_step,
)
from ..logging import TrajectoryLogger


def run_ihisd_with_logging(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    config: Optional[IHiSDConfig] = None,
    sample_id: str = "unknown",
    formula: str = "",
    known_ts_coords: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, Any], TrajectoryLogger]:
    """Run iHiSD with full diagnostic logging.

    This version computes the FULL Hessian and forces at each step for proper logging.

    Args:
        predict_fn: Energy/force/Hessian prediction function
        coords0: Starting coordinates (N, 3)
        atomic_nums: Atomic numbers
        config: IHiSDConfig (uses defaults if None)
        sample_id: Sample identifier for logging
        formula: Chemical formula for logging
        known_ts_coords: Known TS for validation

    Returns:
        final_out_dict: Results dictionary
        trajectory_logger: Full diagnostic data
    """
    if config is None:
        config = IHiSDConfig()

    if coords0 is None:
        raise ValueError("coords0 cannot be None")

    coords = coords0.detach().clone().to(torch.float32)
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)

    # Initialize trajectory logger
    logger = TrajectoryLogger(sample_id=sample_id, formula=formula)
    logger.known_ts_coords = known_ts_coords

    # Get SCINE elements and compute initial Morse index
    out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
    scine_elements = get_scine_elements_from_predict_output(out)

    # Compute initial Morse index for tracking
    hess_proj = get_projected_hessian(out["hessian"], coords, atomic_nums, scine_elements=scine_elements)
    evals, _ = torch.linalg.eigh(hess_proj)
    vib_mask = torch.abs(evals) > config.tr_threshold
    vib_evals = evals[vib_mask]
    initial_morse_index = int((vib_evals < config.neg_threshold).sum().item())

    # Initialize state
    state = IHiSDState(
        step=0,
        theta=config.theta_0,
        theta_0=config.theta_0,
    )

    start_pos = coords.clone()
    prev_pos = coords.clone()
    prev_energy = None
    disp_history: list[float] = []
    total_steps = 0
    steps_to_ts: Optional[int] = None
    converged = False

    for step in range(config.max_steps):
        # Get energy, forces, Hessian - ALWAYS COMPUTE FULL HESSIAN
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        energy = out.get("energy")
        forces = out.get("forces")
        hessian = out.get("hessian")

        energy_value = _to_float(energy)
        force_mean = _force_mean(forces)

        # Project Hessian
        hess_proj = get_projected_hessian(hessian, coords, atomic_nums, scine_elements=scine_elements)

        # Take iHiSD step
        new_coords, step_info, new_state = ihisd_step(
            coords, forces, hess_proj, state, config
        )

        # Apply max displacement cap
        step_vec = new_coords - coords
        max_disp = float(step_vec.norm(dim=1).max().item())
        if max_disp > config.max_atom_disp and max_disp > 0:
            scale = config.max_atom_disp / max_disp
            new_coords = coords + scale * step_vec

        # Displacements
        disp_from_last = float((new_coords - prev_pos).norm(dim=1).mean().item())
        disp_from_start = float((new_coords - start_pos).norm(dim=1).mean().item())

        if total_steps > 0:
            disp_history.append(disp_from_last)
        x_disp_window = float(np.mean(disp_history[-10:])) if disp_history else float("nan")

        # Log to TrajectoryLogger with FULL HESSIAN
        logger.log_step(
            step=total_steps,
            coords=coords,
            energy=energy_value,
            forces=forces,
            hessian_proj=hess_proj,
            gad_vec=forces,
            dt_eff=step_info.get("dt_used", config.dt_base),
            coords_prev=prev_pos if total_steps > 0 else None,
            energy_prev=prev_energy,
            x_disp_window=x_disp_window,
        )

        # Check convergence
        if step_info["grad_norm"] < config.grad_threshold:
            if step_info["morse_index"] == config.target_k:
                converged = True
                steps_to_ts = total_steps
                break

        prev_pos = coords.clone()
        prev_energy = energy_value
        state = new_state
        coords = new_coords
        total_steps += 1

    # Final vibrational analysis
    final_neg_vib = -1
    try:
        final_out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        scine_elements = get_scine_elements_from_predict_output(final_out)
        final_vib_eigvals = vibrational_eigvals(
            final_out["hessian"], coords, atomic_nums, scine_elements=scine_elements
        )
        final_neg_vib = int((final_vib_eigvals < 0).sum().item())
    except Exception:
        pass

    # Finalize logger
    logger.finalize(
        final_coords=coords,
        final_morse_index=final_neg_vib,
        converged_to_ts=converged or (final_neg_vib == 1),
    )

    final_out_dict = {
        "final_coords": coords.detach().cpu(),
        "steps_taken": total_steps,
        "steps_to_ts": steps_to_ts,
        "final_neg_vibrational": final_neg_vib,
        "initial_index": initial_morse_index,
        "final_index": final_neg_vib,
        "final_theta": state.theta,
        "theta_min": config.theta_0,
        "theta_max": state.theta,
        "search_direction": config.search_direction,
        "algorithm": "ihisd",
    }

    return final_out_dict, logger


def run_single_sample_ihisd(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    params: Dict[str, Any],
    n_steps: int,
    sample_id: str,
    formula: str,
    out_dir: str,
) -> Dict[str, Any]:
    """Run iHiSD on a single sample with diagnostics."""
    t0 = time.time()

    diag_dir = Path(out_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    if coords is None:
        return {
            "final_neg_vib": -1,
            "steps_taken": 0,
            "steps_to_ts": None,
            "success": False,
            "wall_time": 0,
            "error": "coords is None",
            "algorithm": "ihisd",
            "initial_index": -1,
            "final_index": -1,
            "final_theta": 0,
            "theta_max": 0,
        }

    try:
        # Build config from params
        config = IHiSDConfig(
            theta_0=params.get("theta_0", 1e-11),
            theta_schedule=params.get("theta_schedule", "sigmoid"),
            theta_rate=params.get("theta_rate", 0.01),
            search_direction=params.get("search_direction", 1),
            target_k=params.get("target_k", 1),
            dt_base=params.get("dt", 0.005),
            use_adaptive_dt=params.get("use_adaptive_dt", True),
            lipschitz_estimate=params.get("lipschitz_estimate", 1.0),
            grad_threshold=params.get("grad_threshold", 1e-5),
            max_steps=n_steps,
            tr_threshold=params.get("tr_threshold", 1e-6),
            neg_threshold=params.get("neg_threshold", -1e-4),
            dt_min=params.get("dt_min", 1e-6),
            dt_max=params.get("dt_max", 0.08),
            max_atom_disp=params.get("max_atom_disp", 0.35),
        )

        out_dict, traj_logger = run_ihisd_with_logging(
            predict_fn,
            coords,
            atomic_nums,
            config=config,
            sample_id=sample_id,
            formula=formula,
        )

        traj_logger.save(diag_dir)

        wall_time = time.time() - t0

        final_neg_vib = out_dict.get("final_neg_vibrational", -1)
        steps_taken = out_dict.get("steps_taken", 0)

        return {
            "final_neg_vib": final_neg_vib,
            "steps_taken": steps_taken,
            "steps_to_ts": out_dict.get("steps_to_ts"),
            "success": final_neg_vib == 1,
            "wall_time": wall_time,
            "error": None,
            "algorithm": "ihisd",
            "initial_index": out_dict.get("initial_index", -1),
            "final_index": out_dict.get("final_index", -1),
            "final_theta": out_dict.get("final_theta", 0),
            "theta_max": out_dict.get("theta_max", 0),
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
            "algorithm": "ihisd",
            "initial_index": -1,
            "final_index": -1,
            "final_theta": 0,
            "theta_max": 0,
        }
