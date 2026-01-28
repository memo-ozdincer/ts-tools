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
from src.noisy.multi_mode_eckartmw import (
    get_projected_hessian,
    _force_mean,
    _to_float,
)

from .recursive_hisd import (
    RecursiveHiSDConfig,
    RecursiveHiSDState,
    recursive_hisd_step,
    perturb_along_unstable_direction,
)
from ..logging import TrajectoryLogger


def _require_full_hessian_output(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    out: Dict[str, Any],
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
    """Ensure full forces + Hessian are present (no approximations)."""
    forces = out.get("forces")
    hessian = out.get("hessian")
    if forces is None or hessian is None:
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        forces = out.get("forces")
        hessian = out.get("hessian")
    if forces is None or hessian is None:
        raise RuntimeError(
            "Recursive HiSD requires full forces and full Hessian at every step; received None. "
            "Disable lightweight mode and enable do_hessian=True in the calculator."
        )
    return out, forces, hessian


def run_recursive_hisd_with_logging(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    config: Optional[RecursiveHiSDConfig] = None,
    sample_id: str = "unknown",
    formula: str = "",
    known_ts_coords: Optional[torch.Tensor] = None,
    max_recursion_depth: int = 10,
) -> Tuple[Dict[str, Any], TrajectoryLogger]:
    """Run Recursive HiSD with full diagnostic logging.

    This version computes the FULL Hessian and forces at each step for proper logging.

    Args:
        predict_fn: Energy/force/Hessian prediction function
        coords0: Starting coordinates (N, 3)
        atomic_nums: Atomic numbers
        config: RecursiveHiSDConfig (uses defaults if None)
        sample_id: Sample identifier for logging
        formula: Chemical formula for logging
        known_ts_coords: Known TS for validation
        max_recursion_depth: Maximum recursion levels

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

    # Get SCINE elements
    out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
    out, _, hessian0 = _require_full_hessian_output(predict_fn, coords, atomic_nums, out)
    scine_elements = get_scine_elements_from_predict_output(out)

    # Compute initial Morse index
    hess_proj = get_projected_hessian(hessian0, coords, atomic_nums, scine_elements=scine_elements)
    evals, evecs = torch.linalg.eigh(hess_proj)
    vib_mask = torch.abs(evals) > config.tr_threshold
    vib_evals = evals[vib_mask]
    initial_morse_index = int((vib_evals < config.neg_threshold).sum().item())
    current_index = initial_morse_index

    # State tracking
    state = RecursiveHiSDState(
        current_index=current_index,
        initial_index=initial_morse_index,
    )

    start_pos = coords.clone()
    prev_pos = coords.clone()
    prev_energy = None
    disp_history: list[float] = []
    total_steps = 0
    steps_to_ts: Optional[int] = None
    level_info = []

    # Recursive descent
    for depth in range(max_recursion_depth):
        state.recursion_depth = depth

        # Get current Morse index
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        out, _, hessian = _require_full_hessian_output(predict_fn, coords, atomic_nums, out)
        hess_proj = get_projected_hessian(hessian, coords, atomic_nums, scine_elements=scine_elements)
        evals, evecs = torch.linalg.eigh(hess_proj)
        vib_mask = torch.abs(evals) > config.tr_threshold
        vib_evals = evals[vib_mask]
        vib_evecs = evecs[:, vib_mask]
        current_index = int((vib_evals < config.neg_threshold).sum().item())

        # Base case: reached index-1
        if current_index <= 1:
            target_k = 1
            # Run 1-HiSD (= GAD) to converge to TS
            for level_step in range(config.max_steps_per_level):
                # Get energy, forces, Hessian - ALWAYS COMPUTE FULL HESSIAN
                out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
                out, forces, hessian = _require_full_hessian_output(predict_fn, coords, atomic_nums, out)
                energy = out.get("energy")

                energy_value = _to_float(energy)
                force_mean = _force_mean(forces)

                # Project Hessian
                hess_proj = get_projected_hessian(hessian, coords, atomic_nums, scine_elements=scine_elements)

                # Take step
                new_coords, step_info = recursive_hisd_step(
                    coords, forces, hess_proj, config.dt, target_k,
                    tr_threshold=config.tr_threshold,
                    neg_threshold=config.neg_threshold,
                )

                # Apply max displacement cap
                step_vec = new_coords - coords
                max_disp = float(step_vec.norm(dim=1).max().item())
                dt_eff = config.dt
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
                    gad_vec=forces,  # Use forces as GAD direction approximation
                    dt_eff=dt_eff,
                    coords_prev=prev_pos if total_steps > 0 else None,
                    energy_prev=prev_energy,
                    x_disp_window=x_disp_window,
                )

                # Check convergence
                if step_info["grad_norm"] < config.grad_threshold and step_info["morse_index"] == 1:
                    steps_to_ts = total_steps
                    level_info.append({
                        "depth": depth,
                        "target_k": target_k,
                        "steps": level_step + 1,
                        "converged": True,
                    })
                    coords = new_coords
                    break

                prev_pos = coords.clone()
                prev_energy = energy_value
                coords = new_coords
                total_steps += 1

            else:
                # Max steps reached
                level_info.append({
                    "depth": depth,
                    "target_k": target_k,
                    "steps": config.max_steps_per_level,
                    "converged": False,
                })

            # Done - reached index-1 level
            break

        # Target the next lower index
        target_k = current_index - 1

        # Perturb to escape current saddle
        coords = perturb_along_unstable_direction(
            coords,
            vib_evecs,
            vib_evals,
            magnitude=config.perturb_magnitude,
            neg_threshold=config.neg_threshold,
            strategy=config.perturb_strategy,
        )

        # Run k-HiSD at this level
        level_converged = False
        for level_step in range(config.max_steps_per_level):
            # Get energy, forces, Hessian - ALWAYS COMPUTE FULL HESSIAN
            out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
            out, forces, hessian = _require_full_hessian_output(predict_fn, coords, atomic_nums, out)
            energy = out.get("energy")

            energy_value = _to_float(energy)
            force_mean = _force_mean(forces)

            # Project Hessian
            hess_proj = get_projected_hessian(hessian, coords, atomic_nums, scine_elements=scine_elements)

            # Take step
            new_coords, step_info = recursive_hisd_step(
                coords, forces, hess_proj, config.dt, target_k,
                tr_threshold=config.tr_threshold,
                neg_threshold=config.neg_threshold,
            )

            # Apply max displacement cap
            step_vec = new_coords - coords
            max_disp = float(step_vec.norm(dim=1).max().item())
            dt_eff = config.dt
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
                dt_eff=dt_eff,
                coords_prev=prev_pos if total_steps > 0 else None,
                energy_prev=prev_energy,
                x_disp_window=x_disp_window,
            )

            # Check convergence
            if step_info["grad_norm"] < config.grad_threshold:
                if step_info["morse_index"] == target_k:
                    level_converged = True
                    break

            prev_pos = coords.clone()
            prev_energy = energy_value
            coords = new_coords
            total_steps += 1

        level_info.append({
            "depth": depth,
            "target_k": target_k,
            "steps": level_step + 1 if level_converged else config.max_steps_per_level,
            "converged": level_converged,
        })

        # Update current index for next iteration
        current_index = step_info["morse_index"]

    # Final vibrational analysis
    final_neg_vib = -1
    try:
        final_out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        final_out, _, final_hessian = _require_full_hessian_output(predict_fn, coords, atomic_nums, final_out)
        scine_elements = get_scine_elements_from_predict_output(final_out)
        final_vib_eigvals = vibrational_eigvals(
            final_hessian, coords, atomic_nums, scine_elements=scine_elements
        )
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
        "steps_taken": total_steps,
        "steps_to_ts": steps_to_ts,
        "final_neg_vibrational": final_neg_vib,
        "initial_index": initial_morse_index,
        "final_index": final_neg_vib,
        "recursion_depth": state.recursion_depth,
        "level_info": level_info,
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
