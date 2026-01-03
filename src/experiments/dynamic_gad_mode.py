from __future__ import annotations

"""Dynamic GAD Mode Switching Experiment.

This is a fundamentally different approach from multi_mode_eckartmw.py:

Multi-mode (kicks): GAD always follows v1, with discrete "kicks" in v2/v3/etc direction
Dynamic mode: GAD direction itself switches to v2/v3/etc when stuck (pure GAD mode switching)

Algorithm:
1. Start with GAD following v1 (lowest eigenvector)
2. When stuck (displacement < threshold, neg_vib stable), escalate to v2
3. When stuck again, escalate to v3, etc.
4. If displacement goes back above threshold, de-escalate (v3 -> v2 -> v1)
5. Repeat until index = 1 (TS found) or max steps reached

Key insight: Instead of perturbing/kicking, we smoothly change which eigenvector GAD follows.
This allows continuous dynamics without discontinuous jumps.
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..core_algos.gad import pick_tracked_mode
from ..dependencies.common_utils import add_common_args, parse_starting_geometry, setup_experiment
from ..dependencies.experiment_logger import ExperimentLogger, RunResult, build_loss_type_flags
from ..dependencies.hessian import (
    vibrational_eigvals,
    get_scine_elements_from_predict_output,
    project_hessian_remove_rigid_modes,
    prepare_hessian,
)
from ..logging import finish_wandb, init_wandb_run, log_sample, log_summary
from ..logging.plotly_utils import plot_gad_trajectory_interactive
from ..runners._predict import make_predict_fn_from_calculator

# Import common functions from multi_mode_eckartmw
from .multi_mode_eckartmw import (
    get_projected_hessian,
    _step_metrics_from_projected_hessian,
    _to_float,
    _max_atom_norm,
    _sanitize_wandb_name,
    _auto_wandb_name,
    _save_trajectory_json,
)

# SCINE projection (may not be available)
try:
    from ..dependencies.scine_masses import (
        ScineFrequencyAnalyzer,
        get_scine_masses,
    )
    SCINE_PROJECTION_AVAILABLE = True
except ImportError:
    SCINE_PROJECTION_AVAILABLE = False


def compute_gad_vector_mode_n(
    forces: torch.Tensor,
    hessian_proj: torch.Tensor,
    v_prev: torch.Tensor | None,
    *,
    gad_mode: int = 1,
    k_track: int = 8,
    beta: float = 1.0,
    tr_threshold: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """GAD direction using the Nth eigenvector (mode) of projected Hessian.

    This is the key difference from standard GAD:
    - Standard GAD always uses v1 (lowest eigenvector)
    - This uses v_N where N = gad_mode (1=v1, 2=v2, 3=v3, etc.)

    Args:
        forces: Force tensor
        hessian_proj: Projected Hessian (3N x 3N)
        v_prev: Previous eigenvector for mode tracking
        gad_mode: Which vibrational mode to use (1=v1, 2=v2, etc.)
        k_track: Number of eigenvectors to search for mode tracking
        beta: Smoothing parameter
        tr_threshold: Threshold for translation/rotation modes

    Returns:
        gad_vec: GAD direction vector (N, 3)
        v_next: Tracked eigenvector for next step
        info: Dict with mode_overlap, mode_index, etc.
    """
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    forces = forces.reshape(-1, 3)
    num_atoms = int(forces.shape[0])

    hess = hessian_proj
    if hess.dim() != 2 or hess.shape[0] != 3 * num_atoms:
        hess = prepare_hessian(hess, num_atoms)

    evals, evecs = torch.linalg.eigh(hess)

    # Skip translation/rotation modes (near-zero eigenvalues)
    vib_mask = torch.abs(evals) > float(tr_threshold)
    if not vib_mask.any():
        vib_indices = torch.arange(evecs.shape[1], device=evecs.device)
    else:
        vib_indices = torch.where(vib_mask)[0]

    # Select the Nth vibrational mode (0-indexed: gad_mode=1 -> index 0)
    mode_idx = gad_mode - 1

    # Clamp to available modes
    if mode_idx >= len(vib_indices):
        mode_idx = len(vib_indices) - 1
    if mode_idx < 0:
        mode_idx = 0

    # Get the candidate eigenvectors around the target mode for tracking
    # We want to track within a window around our target mode
    start_idx = max(0, mode_idx - k_track // 2)
    end_idx = min(len(vib_indices), start_idx + k_track)
    candidate_indices = vib_indices[start_idx:end_idx]

    if len(candidate_indices) == 0:
        candidate_indices = vib_indices[:min(k_track, len(vib_indices))]

    V = evecs[:, candidate_indices].to(device=forces.device, dtype=forces.dtype)
    v_prev_local = v_prev.to(device=forces.device, dtype=forces.dtype).reshape(-1) if v_prev is not None else None

    # Track the mode across steps
    if v_prev_local is not None:
        v_new, j, overlap = pick_tracked_mode(V, v_prev_local, k=int(V.shape[1]))
    else:
        # First step: use the target mode directly
        target_in_candidates = mode_idx - start_idx
        if 0 <= target_in_candidates < V.shape[1]:
            v_new = V[:, target_in_candidates]
        else:
            v_new = V[:, 0]
        v_new = v_new / (v_new.norm() + 1e-12)
        j = mode_idx
        overlap = 1.0

    # Apply smoothing if beta < 1
    if v_prev_local is not None and float(beta) < 1.0:
        v = (1.0 - float(beta)) * v_prev_local + float(beta) * v_new
        v = v / (v.norm() + 1e-12)
    else:
        v = v_new

    # Compute GAD vector: f + 2(f·v)v
    f_flat = forces.reshape(-1)
    gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
    gad_vec = gad_flat.view(num_atoms, 3)
    v_next = v.detach().clone().reshape(-1)

    return gad_vec, v_next, {"mode_overlap": float(overlap), "mode_index": float(j), "gad_mode": gad_mode}


def gad_euler_step_mode_n(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    dt: float,
    gad_mode: int = 1,
    out: Optional[Dict[str, Any]] = None,
    scine_elements: Optional[list] = None,
    v_prev: torch.Tensor | None = None,
    k_track: int = 8,
    beta: float = 1.0,
) -> Dict[str, Any]:
    """GAD Euler step using the Nth eigenvector (gad_mode) of projected Hessian."""
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)

    if out is None:
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)

    forces = out["forces"]
    hessian = out["hessian"]

    # Get projected Hessian
    hessian_proj = get_projected_hessian(hessian, coords, atomic_nums, scine_elements)

    # Compute GAD vector using Nth eigenvector (with mode tracking)
    gad_vec, v_next, info = compute_gad_vector_mode_n(
        forces,
        hessian_proj,
        v_prev,
        gad_mode=gad_mode,
        k_track=k_track,
        beta=beta,
    )
    new_coords = coords + dt * gad_vec

    return {
        "new_coords": new_coords,
        "gad_vec": gad_vec,
        "out": out,
        "hessian_proj": hessian_proj,
        "v_next": v_next,
        **info,
    }


def _check_should_escalate(
    disp_history: list[float],
    neg_vib_history: list[int],
    *,
    window: int,
    disp_threshold: float,
    neg_vib_std_threshold: float,
) -> bool:
    """Check if we should escalate GAD mode (switch to higher eigenvector).

    Triggers when:
    1. mean(disp[-window:]) < disp_threshold (tiny steps = stuck)
    2. std(neg_vib[-window:]) <= neg_vib_std_threshold (stable saddle index)
    """
    if len(disp_history) < window:
        return False

    recent_disp = disp_history[-window:]
    recent_neg_vib = neg_vib_history[-window:]

    mean_disp = float(np.mean(recent_disp))
    std_neg_vib = float(np.std(recent_neg_vib))

    return mean_disp < disp_threshold and std_neg_vib <= neg_vib_std_threshold


def _check_should_deescalate(
    disp_history: list[float],
    *,
    window: int,
    disp_threshold: float,
    deescalate_factor: float = 2.0,
) -> bool:
    """Check if we should de-escalate GAD mode (switch to lower eigenvector).

    Triggers when displacement goes back above threshold, indicating
    the current mode is making progress and we might be able to go back
    to a lower mode.
    """
    if len(disp_history) < window:
        return False

    recent_disp = disp_history[-window:]
    mean_disp = float(np.mean(recent_disp))

    # De-escalate if displacement is significantly above the escalation threshold
    return mean_disp > disp_threshold * deescalate_factor


def run_dynamic_gad_mode(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    n_steps: int,
    dt: float,
    stop_at_ts: bool,
    ts_eps: float,
    dt_control: str,
    dt_min: float,
    dt_max: float,
    max_atom_disp: Optional[float],
    plateau_patience: int,
    plateau_boost: float,
    plateau_shrink: float,
    # Mode switching parameters
    mode_switch_window: int,
    mode_switch_disp_threshold: float,
    mode_switch_neg_vib_std: float,
    max_gad_mode: int = 6,
    deescalate_factor: float = 2.0,
    min_steps_per_mode: int = 50,
    # HIP parameters
    hip_vib_mode: str = "projected",
    hip_rigid_tol: float = 1e-6,
    hip_eigh_device: str = "auto",
    profile_every: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run GAD with dynamic mode switching.

    Instead of kicks/perturbations, this dynamically changes which eigenvector
    GAD follows based on displacement:
    - Escalate (v1 -> v2 -> v3) when stuck (tiny displacement)
    - De-escalate (v3 -> v2 -> v1) when progress resumes (displacement increases)
    """
    coords = coords0.detach().clone().to(torch.float32)

    trajectory = {k: [] for k in [
        "energy",
        "force_mean",
        "eig0",
        "eig1",
        "eig_product",
        "neg_vib",
        "disp_from_last",
        "disp_from_start",
        "dt_eff",
        "gad_norm",
        "gad_mode",
    ]}

    start_pos = coords.clone()
    prev_pos = coords.clone()

    steps_to_ts: Optional[int] = None
    total_steps = 0

    # Dynamic mode state
    current_gad_mode = 1  # Start with v1
    steps_since_mode_change = 0
    escalation_count = 0
    deescalation_count = 0

    # Rolling history for mode switching detection
    disp_history: list[float] = []
    neg_vib_history: list[int] = []

    # Stateful dt controller variables
    dt_eff_state = float(dt)
    best_neg_vib: Optional[int] = None
    no_improve = 0

    # Mode tracking
    v_prev: torch.Tensor | None = None

    # Detect SCINE
    scine_elements: Optional[list] = None

    while total_steps < n_steps:
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        scine_elements = get_scine_elements_from_predict_output(out)

        hessian = out["hessian"]
        forces = out["forces"]
        energy = out["energy"]

        # Get metrics from projected Hessian
        (
            gad_vec, v_next, mode_overlap, mode_index,
            hessian_proj, eig0, eig1, eig_prod, neg_vib
        ) = _step_metrics_from_projected_hessian(
            forces,
            hessian,
            coords,
            atomic_nums,
            v_prev,
            scine_elements,
            k_track=8,
            beta=1.0,
        )

        # BUT: compute GAD using the current mode (not always v1)
        gad_vec, v_next, gad_info = compute_gad_vector_mode_n(
            forces,
            hessian_proj,
            v_prev,
            gad_mode=current_gad_mode,
            k_track=8,
            beta=1.0,
        )
        v_prev = v_next

        gad_norm = float(gad_vec.norm().item())
        energy_value = _to_float(energy)
        force_mean = float(forces.norm(dim=-1).mean().item()) if forces.dim() > 1 else float(forces.norm().item())

        disp_from_last = float((coords - prev_pos).norm().item())
        disp_from_start = float((coords - start_pos).norm().item())

        disp_history.append(disp_from_last)
        neg_vib_history.append(int(neg_vib))

        # Record trajectory
        trajectory["energy"].append(energy_value)
        trajectory["force_mean"].append(force_mean)
        trajectory["eig0"].append(eig0)
        trajectory["eig1"].append(eig1)
        trajectory["eig_product"].append(eig_prod)
        trajectory["neg_vib"].append(int(neg_vib))
        trajectory["disp_from_last"].append(disp_from_last)
        trajectory["disp_from_start"].append(disp_from_start)
        trajectory["gad_norm"].append(gad_norm)
        trajectory["gad_mode"].append(current_gad_mode)

        trajectory.setdefault("mode_overlap", []).append(float(mode_overlap))
        trajectory.setdefault("mode_index", []).append(int(mode_index))

        # Check for TS (index = 1)
        if stop_at_ts and steps_to_ts is None and np.isfinite(eig_prod) and eig_prod < -abs(ts_eps):
            steps_to_ts = total_steps
            trajectory["dt_eff"].append(float("nan"))
            break

        steps_since_mode_change += 1

        # Check for mode switching (only if we've been in current mode long enough)
        if steps_since_mode_change >= min_steps_per_mode:
            # Check for escalation (stuck -> switch to higher mode)
            if current_gad_mode < max_gad_mode and neg_vib > 1:
                should_escalate = _check_should_escalate(
                    disp_history,
                    neg_vib_history,
                    window=mode_switch_window,
                    disp_threshold=mode_switch_disp_threshold,
                    neg_vib_std_threshold=mode_switch_neg_vib_std,
                )
                if should_escalate:
                    current_gad_mode += 1
                    steps_since_mode_change = 0
                    escalation_count += 1
                    disp_history.clear()
                    neg_vib_history.clear()
                    v_prev = None  # Reset mode tracking for new mode

            # Check for de-escalation (making progress -> can try lower mode)
            elif current_gad_mode > 1:
                should_deescalate = _check_should_deescalate(
                    disp_history,
                    window=mode_switch_window,
                    disp_threshold=mode_switch_disp_threshold,
                    deescalate_factor=deescalate_factor,
                )
                if should_deescalate:
                    current_gad_mode -= 1
                    steps_since_mode_change = 0
                    deescalation_count += 1
                    disp_history.clear()
                    neg_vib_history.clear()
                    v_prev = None  # Reset mode tracking for new mode

        # Compute dt_eff using plateau controller
        if dt_control == "neg_eig_plateau":
            if best_neg_vib is None:
                best_neg_vib = int(neg_vib)
                no_improve = 0
            else:
                if int(neg_vib) < int(best_neg_vib):
                    best_neg_vib = int(neg_vib)
                    no_improve = 0
                    dt_eff_state = min(float(dt_eff_state), float(dt))
                elif int(neg_vib) > int(best_neg_vib):
                    dt_eff_state = max(float(dt_eff_state) * float(plateau_shrink), float(dt_min))
                    no_improve = 0
                else:
                    no_improve += 1

            if no_improve >= int(max(1, plateau_patience)):
                dt_eff_state = min(float(dt_eff_state) * float(plateau_boost), float(dt_max))
                no_improve = 0

            dt_eff = float(np.clip(dt_eff_state, float(dt_min), float(dt_max)))
        else:
            dt_eff = float(dt)

        # Apply max atom displacement cap
        if max_atom_disp is not None and max_atom_disp > 0:
            step = dt_eff * gad_vec
            max_disp = _max_atom_norm(step)
            if np.isfinite(max_disp) and max_disp > float(max_atom_disp) and max_disp > 0:
                dt_eff = dt_eff * (float(max_atom_disp) / float(max_disp))

        trajectory["dt_eff"].append(float(dt_eff))

        # Take GAD step
        prev_pos = coords.clone()
        coords = (coords + dt_eff * gad_vec).detach()
        total_steps += 1

    # Pad trajectories to same length
    while len(trajectory["dt_eff"]) < len(trajectory["energy"]):
        trajectory["dt_eff"].append(float("nan"))
    while len(trajectory["gad_norm"]) < len(trajectory["energy"]):
        trajectory["gad_norm"].append(float("nan"))
    while len(trajectory["gad_mode"]) < len(trajectory["energy"]):
        trajectory["gad_mode"].append(current_gad_mode)

    # Final vibrational analysis
    final_neg_vib = -1
    try:
        final_out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        scine_elements = get_scine_elements_from_predict_output(final_out)
        final_vib_eigvals = vibrational_eigvals(final_out["hessian"], coords, atomic_nums, scine_elements=scine_elements)
        final_neg_vib = int((final_vib_eigvals < 0).sum().item())
    except Exception:
        pass

    aux = {
        "steps_to_ts": steps_to_ts,
        "total_steps": total_steps,
        "escalation_count": escalation_count,
        "deescalation_count": deescalation_count,
        "final_gad_mode": current_gad_mode,
    }

    final_out_dict = {
        "final_coords": coords.detach().cpu(),
        "trajectory": trajectory,
        "steps_taken": total_steps,
        "steps_to_ts": steps_to_ts,
        "final_eig0": trajectory["eig0"][-1] if trajectory["eig0"] else None,
        "final_eig1": trajectory["eig1"][-1] if trajectory["eig1"] else None,
        "final_eig_product": trajectory["eig_product"][-1] if trajectory["eig_product"] else None,
        "final_neg_vibrational": final_neg_vib,
    }

    return final_out_dict, aux


def main(
    argv: Optional[list[str]] = None,
    *,
    default_calculator: Optional[str] = None,
    enforce_calculator: bool = False,
    script_name_prefix: str = "exp-dynamic-gad-mode",
) -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic GAD Mode Switching: switch eigenvector based on displacement"
    )
    add_common_args(parser)

    # GAD parameters
    parser.add_argument("--method", type=str, default="euler", choices=["euler"])
    parser.add_argument("--n-steps", type=int, default=15000)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--start-from", type=str, default="midpoint_rt")
    parser.add_argument("--stop-at-ts", action="store_true", default=True)
    parser.add_argument("--ts-eps", type=float, default=1e-5)

    # dt control
    parser.add_argument("--dt-control", type=str, default="neg_eig_plateau")
    parser.add_argument("--dt-min", type=float, default=1e-6)
    parser.add_argument("--dt-max", type=float, default=0.05)
    parser.add_argument("--max-atom-disp", type=float, default=0.25)
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--plateau-boost", type=float, default=1.5)
    parser.add_argument("--plateau-shrink", type=float, default=0.5)

    # Mode switching parameters
    parser.add_argument(
        "--mode-switch-window",
        type=int,
        default=20,
        help="Number of steps to average for mode switching detection",
    )
    parser.add_argument(
        "--mode-switch-disp-threshold",
        type=float,
        default=5e-4,
        help="Displacement threshold for escalation (Angstrom)",
    )
    parser.add_argument(
        "--mode-switch-neg-vib-std",
        type=float,
        default=0.5,
        help="Max std deviation of neg_vib count for stable detection",
    )
    parser.add_argument(
        "--max-gad-mode",
        type=int,
        default=6,
        help="Maximum GAD mode (1=v1, 6=v6)",
    )
    parser.add_argument(
        "--deescalate-factor",
        type=float,
        default=2.0,
        help="Factor above escalation threshold to trigger de-escalation",
    )
    parser.add_argument(
        "--min-steps-per-mode",
        type=int,
        default=50,
        help="Minimum steps before considering mode switch",
    )

    # HIP parameters
    parser.add_argument("--hip-vib-mode", type=str, default="projected")
    parser.add_argument("--hip-rigid-tol", type=float, default=1e-6)
    parser.add_argument("--hip-eigh-device", type=str, default="auto")

    # Profiling
    parser.add_argument("--profile-every", type=int, default=0)

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gad-ts-search")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)

    args = parser.parse_args(argv)

    if enforce_calculator and default_calculator is not None:
        args.calculator = default_calculator

    # Setup
    device = torch.device(args.device if hasattr(args, "device") and args.device else "cpu")
    predict_fn, samples, logger = setup_experiment(args)

    loss_type_flags = build_loss_type_flags(args)
    script_name = "dynamic_gad_mode"

    # W&B init
    if args.wandb:
        wandb_name = args.wandb_name or _auto_wandb_name(
            script=script_name, loss_type_flags=loss_type_flags, args=args
        )
        init_wandb_run(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_name,
            config=vars(args),
        )

    # Run samples
    results = []
    summary = defaultdict(list)

    for i, sample in enumerate(samples):
        if hasattr(args, "start_idx") and i < args.start_idx:
            continue
        if hasattr(args, "end_idx") and args.end_idx is not None and i >= args.end_idx:
            break

        formula = sample.get("formula", f"sample_{i}")
        transition_key = sample.get("transition_key", formula)

        start_coords, start_from_label = parse_starting_geometry(sample, args)
        start_coords = start_coords.to(device).to(torch.float32)
        atomic_nums = torch.tensor(sample["atomic_nums"], device=device, dtype=torch.long)

        # Get initial neg_vib
        try:
            init_out = predict_fn(start_coords, atomic_nums, do_hessian=True, require_grad=False)
            scine_elements = get_scine_elements_from_predict_output(init_out)
            init_vib_eigvals = vibrational_eigvals(
                init_out["hessian"], start_coords, atomic_nums, scine_elements=scine_elements
            )
            initial_neg = int((init_vib_eigvals < 0).sum().item())
        except Exception:
            initial_neg = -1

        t0 = time.time()
        try:
            out_dict, aux = run_dynamic_gad_mode(
                predict_fn,
                start_coords,
                atomic_nums,
                n_steps=int(args.n_steps),
                dt=float(args.dt),
                stop_at_ts=bool(args.stop_at_ts),
                ts_eps=float(args.ts_eps),
                dt_control=str(args.dt_control),
                dt_min=float(args.dt_min),
                dt_max=float(args.dt_max),
                max_atom_disp=float(args.max_atom_disp) if args.max_atom_disp is not None else None,
                plateau_patience=int(args.plateau_patience),
                plateau_boost=float(args.plateau_boost),
                plateau_shrink=float(args.plateau_shrink),
                mode_switch_window=int(args.mode_switch_window),
                mode_switch_disp_threshold=float(args.mode_switch_disp_threshold),
                mode_switch_neg_vib_std=float(args.mode_switch_neg_vib_std),
                max_gad_mode=int(args.max_gad_mode),
                deescalate_factor=float(args.deescalate_factor),
                min_steps_per_mode=int(args.min_steps_per_mode),
                hip_vib_mode=str(args.hip_vib_mode),
                hip_rigid_tol=float(args.hip_rigid_tol),
                hip_eigh_device=str(args.hip_eigh_device),
                profile_every=int(args.profile_every),
            )
            wall = time.time() - t0
        except Exception as e:
            wall = time.time() - t0
            stop_reason = f"{type(e).__name__}: {e}"
            print(f"[WARN] Sample {i} failed during run: {stop_reason}")

            result = RunResult(
                sample_index=i,
                formula=str(formula),
                transition_key=str(transition_key),
                initial_neg_vibrational=initial_neg,
                final_neg_vibrational=-1,
                steps_taken=0,
                steps_to_ts=None,
                stop_reason=stop_reason,
                final_eig0=None,
                final_eig1=None,
                final_eig_product=None,
                wall_time=wall,
            )
            results.append(result)
            continue

        # Process results
        steps_to_ts = out_dict.get("steps_to_ts")
        final_neg = out_dict.get("final_neg_vibrational", -1)
        final_eig0 = out_dict.get("final_eig0")
        final_eig1 = out_dict.get("final_eig1")
        final_eig_prod = out_dict.get("final_eig_product")

        if steps_to_ts is not None:
            stop_reason = "TS_FOUND"
        elif out_dict.get("steps_taken", 0) >= args.n_steps:
            stop_reason = "MAX_STEPS"
        else:
            stop_reason = "COMPLETED"

        result = RunResult(
            sample_index=i,
            formula=str(formula),
            transition_key=str(transition_key),
            initial_neg_vibrational=initial_neg,
            final_neg_vibrational=final_neg,
            steps_taken=out_dict.get("steps_taken", 0),
            steps_to_ts=steps_to_ts,
            stop_reason=stop_reason,
            final_eig0=final_eig0,
            final_eig1=final_eig1,
            final_eig_product=final_eig_prod,
            wall_time=wall,
        )
        results.append(result)

        # Log
        logger.log_run(result)

        # Save trajectory
        trajectory = out_dict.get("trajectory", {})
        _save_trajectory_json(logger, result, trajectory, [])

        # Update summary
        summary["steps_to_ts"].append(steps_to_ts if steps_to_ts else float("nan"))
        summary["final_neg_vib"].append(final_neg)
        summary["escalations"].append(aux.get("escalation_count", 0))
        summary["deescalations"].append(aux.get("deescalation_count", 0))
        summary["final_mode"].append(aux.get("final_gad_mode", 1))

        # W&B logging
        if args.wandb:
            log_sample(
                sample_index=i,
                formula=str(formula),
                result=result,
                trajectory=trajectory,
            )

        print(
            f"[{i}] {formula}: steps={result.steps_taken}, "
            f"TS@{steps_to_ts}, neg_vib={initial_neg}->{final_neg}, "
            f"escalations={aux.get('escalation_count', 0)}, "
            f"final_mode=v{aux.get('final_gad_mode', 1)}"
        )

    # Final summary
    n_samples = len(results)
    n_converged = sum(1 for r in results if r.steps_to_ts is not None)
    print(f"\n=== Summary ===")
    print(f"Samples: {n_samples}")
    print(f"Converged to TS: {n_converged} ({100*n_converged/n_samples:.1f}%)")

    if summary["escalations"]:
        print(f"Avg escalations: {np.mean(summary['escalations']):.1f}")
        print(f"Avg de-escalations: {np.mean(summary['deescalations']):.1f}")

    if args.wandb:
        log_summary(summary)
        finish_wandb()


if __name__ == "__main__":
    main()
