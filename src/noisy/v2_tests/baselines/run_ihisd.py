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
from src.noisy.multi_mode_eckartmw import get_projected_hessian

from .ihisd import (
    IHiSDConfig,
    run_ihisd,
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
    initial_morse_index = -1
    try:
        initial_vib_eigvals = vibrational_eigvals(
            out["hessian"], coords, atomic_nums, scine_elements=scine_elements
        )
        initial_morse_index = int((initial_vib_eigvals < 0).sum().item())
    except Exception:
        pass

    def get_proj_hessian_fn(hessian, coords, atomic_nums, scine_elem):
        return get_projected_hessian(hessian, coords, atomic_nums, scine_elements=scine_elem)

    # Run iHiSD
    result, trajectory = run_ihisd(
        predict_fn,
        coords,
        atomic_nums,
        get_proj_hessian_fn,
        config=config,
        scine_elements=scine_elements,
    )

    # Log trajectory to TrajectoryLogger
    disp_history = []
    prev_coords = coords.clone()
    prev_energy = None
    default_dt = config.dt_base if config else 0.005

    for i, step_info in enumerate(trajectory):
        step_energy = step_info.get("energy", float("nan"))

        disp = step_info.get("direction_norm", 0) * step_info.get("dt_used", default_dt)
        disp_history.append(disp)
        x_disp_window = float(np.mean(disp_history[-10:])) if disp_history else float("nan")

        logger.log_step(
            step=i,
            coords=prev_coords,
            energy=step_energy,
            forces=None,
            hessian_proj=None,
            gad_vec=None,
            dt_eff=step_info.get("dt_used", default_dt),
            coords_prev=prev_coords if i > 0 else None,
            energy_prev=prev_energy,
            x_disp_window=x_disp_window,
        )
        prev_energy = step_energy

    # Final vibrational analysis
    final_coords = result["final_coords"]
    if isinstance(final_coords, torch.Tensor):
        final_coords = final_coords.to(torch.float32)
    else:
        final_coords = torch.tensor(final_coords, dtype=torch.float32)

    final_neg_vib = -1
    try:
        final_out = predict_fn(final_coords, atomic_nums, do_hessian=True, require_grad=False)
        scine_elements = get_scine_elements_from_predict_output(final_out)
        final_vib_eigvals = vibrational_eigvals(
            final_out["hessian"], final_coords, atomic_nums, scine_elements=scine_elements
        )
        final_neg_vib = int((final_vib_eigvals < 0).sum().item())
    except Exception:
        pass

    # Finalize logger
    logger.finalize(
        final_coords=final_coords,
        final_morse_index=final_neg_vib,
        converged_to_ts=result.get("converged", False) or (final_neg_vib == 1),
    )

    # Collect theta trajectory
    theta_trajectory = [t.get("theta", 0) for t in trajectory]

    final_out_dict = {
        "final_coords": final_coords.detach().cpu(),
        "trajectory": trajectory,
        "steps_taken": result["total_steps"],
        "steps_to_ts": result.get("converged_step"),
        "final_neg_vibrational": final_neg_vib,
        "initial_index": initial_morse_index,
        "final_index": result.get("final_index", -1),
        "final_theta": result.get("final_theta", 0),
        "theta_min": result.get("theta_min", 0),
        "theta_max": result.get("theta_max", 0),
        "theta_trajectory": theta_trajectory,
        "search_direction": config.search_direction if config else 1,
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
