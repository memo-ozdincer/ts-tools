"""Recursive HiSD runner with diagnostic logging.

Recursive HiSD starts from a (potentially high-index) saddle point and
recursively descends to lower index saddles until reaching an index-1 TS.

Algorithm:
1. Detect current Morse index n
2. If n == 1, we're done (found TS)
3. Perturb along unstable direction to escape current saddle
4. Run (n-1)-HiSD to find index-(n-1) saddle
5. Recurse to step 1

Key difference from adaptive k-HiSD:
- Adaptive k-HiSD: starts with k=1, increases k when stuck
- Recursive HiSD: explicitly targets each index level in sequence
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

from .recursive_hisd import (
    RecursiveHiSDConfig,
    run_recursive_hisd,
)
from ..logging import TrajectoryLogger


def run_recursive_hisd_with_logging(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    config: Optional[RecursiveHiSDConfig] = None,
    sample_id: str = "unknown",
    formula: str = "",
    known_ts_coords: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, Any], TrajectoryLogger]:
    """Run Recursive HiSD with full diagnostic logging.

    Args:
        predict_fn: Energy/force/Hessian prediction function
        coords0: Starting coordinates (N, 3)
        atomic_nums: Atomic numbers
        config: RecursiveHiSDConfig (uses defaults if None)
        sample_id: Sample identifier for logging
        formula: Chemical formula for logging
        known_ts_coords: Known TS for validation

    Returns:
        final_out_dict: Results dictionary
        trajectory_logger: Full diagnostic data
    """
    if config is None:
        config = RecursiveHiSDConfig()

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

    # Run Recursive HiSD
    result, trajectory = run_recursive_hisd(
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

    for i, step_info in enumerate(trajectory):
        # Reconstruct approximate coords from step info
        step_energy = step_info.get("energy", float("nan"))

        # Compute displacement from step direction/norm (approximate)
        disp = step_info.get("direction_norm", 0) * step_info.get("dt_eff", config.dt)
        disp_history.append(disp)
        x_disp_window = float(np.mean(disp_history[-10:])) if disp_history else float("nan")

        logger.log_step(
            step=i,
            coords=prev_coords,  # Approximate
            energy=step_energy,
            forces=None,
            hessian_proj=None,
            gad_vec=None,
            dt_eff=step_info.get("dt_eff", config.dt),
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

    # Use computed initial_morse_index if result doesn't have it
    result_initial_index = result.get("initial_index", initial_morse_index)
    if result_initial_index == -1:
        result_initial_index = initial_morse_index

    final_out_dict = {
        "final_coords": final_coords.detach().cpu(),
        "trajectory": trajectory,
        "steps_taken": result["total_steps"],
        "steps_to_ts": result.get("converged_step"),
        "final_neg_vibrational": final_neg_vib,
        "initial_index": result_initial_index,
        "final_index": result.get("final_index", -1),
        "recursion_depth": result.get("recursion_depth", 0),
        "level_info": result.get("level_info", []),
        "algorithm": "recursive_hisd",
    }

    return final_out_dict, logger


def run_single_sample_recursive_hisd(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    params: Dict[str, Any],
    n_steps: int,
    sample_id: str,
    formula: str,
    out_dir: str,
) -> Dict[str, Any]:
    """Run Recursive HiSD on a single sample with diagnostics."""
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
            "algorithm": "recursive_hisd",
            "initial_index": -1,
            "final_index": -1,
            "recursion_depth": 0,
        }

    try:
        # Build config from params
        config = RecursiveHiSDConfig(
            perturb_magnitude=params.get("perturb_magnitude", 0.01),
            perturb_strategy=params.get("perturb_strategy", "unstable"),
            grad_threshold=params.get("grad_threshold", 1e-5),
            max_steps_per_level=params.get("max_steps_per_level", n_steps // 5),
            dt=params.get("dt", 0.005),
            dt_min=params.get("dt_min", 1e-6),
            dt_max=params.get("dt_max", 0.08),
            max_atom_disp=params.get("max_atom_disp", 0.35),
            tr_threshold=params.get("tr_threshold", 1e-6),
            neg_threshold=params.get("neg_threshold", -1e-4),
        )

        out_dict, traj_logger = run_recursive_hisd_with_logging(
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
            "algorithm": "recursive_hisd",
            "initial_index": out_dict.get("initial_index", -1),
            "final_index": out_dict.get("final_index", -1),
            "recursion_depth": out_dict.get("recursion_depth", 0),
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
            "algorithm": "recursive_hisd",
            "initial_index": -1,
            "final_index": -1,
            "recursion_depth": 0,
        }
