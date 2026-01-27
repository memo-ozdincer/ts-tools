"""Enhanced GAD runner with comprehensive diagnostic logging.

This runner wraps the existing multi_mode_eckartmw.py functionality but adds
the extended logging infrastructure from v2_tests/logging/ to capture:

1. Full eigenvalue spectrum at each step
2. Eigenvalue gaps for singularity detection
3. Morse index tracking
4. Detailed escape event logging
5. Failure mode analysis

Use this runner to diagnose WHY GAD fails and understand when v₂ kicking helps.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Import existing components
from src.core_algos.gad import pick_tracked_mode
from src.dependencies.common_utils import add_common_args, parse_starting_geometry, setup_experiment
from src.dependencies.experiment_logger import ExperimentLogger, RunResult, build_loss_type_flags
from src.dependencies.hessian import (
    vibrational_eigvals,
    get_scine_elements_from_predict_output,
    prepare_hessian,
)
from src.runners._predict import make_predict_fn_from_calculator

# Import our new logging infrastructure
from ..logging import TrajectoryLogger, create_escape_event

# Import from existing multi_mode_eckartmw
from src.noisy.multi_mode_eckartmw import (
    get_projected_hessian,
    perform_escape_perturbation,
    _check_plateau_convergence,
    _force_mean,
    _mean_atom_norm,
    _max_atom_norm,
    _to_float,
)

# SCINE projection (may not be available)
try:
    from src.dependencies.scine_masses import ScineFrequencyAnalyzer, get_scine_masses
    SCINE_PROJECTION_AVAILABLE = True
except ImportError:
    SCINE_PROJECTION_AVAILABLE = False


def _step_metrics_from_projected_hessian(
    *,
    forces: torch.Tensor,
    hessian_proj: torch.Tensor,
    tr_threshold: float,
    v_prev: torch.Tensor | None,
    k_track: int = 8,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Compute GAD vec + full eigenvalue info from projected Hessian.

    Returns:
        gad_vec: (N, 3) GAD direction
        v_next: (3N,) tracked mode for next step
        info: dict with eigenvalue spectrum, morse_index, gaps, etc.
    """
    forces = forces[0] if forces.dim() == 3 and forces.shape[0] == 1 else forces
    forces = forces.reshape(-1, 3)
    num_atoms = int(forces.shape[0])

    hess = hessian_proj
    if hess.dim() != 2 or hess.shape[0] != 3 * num_atoms:
        hess = prepare_hessian(hess, num_atoms)

    evals, evecs = torch.linalg.eigh(hess)

    # Filter TR modes
    vib_mask = torch.abs(evals) > tr_threshold
    vib_indices = torch.where(vib_mask)[0]

    if int(vib_indices.numel()) == 0:
        vib_evals = evals
        candidate_indices = torch.arange(min(int(k_track), int(evecs.shape[1])), device=evecs.device)
    else:
        vib_evals = evals[vib_indices]
        candidate_indices = vib_indices[: int(min(int(k_track), int(vib_indices.numel())))]

    V = evecs[:, candidate_indices].to(device=forces.device, dtype=forces.dtype)
    v_prev_local = v_prev.to(device=forces.device, dtype=forces.dtype).reshape(-1) if v_prev is not None else None
    v_new, j, overlap = pick_tracked_mode(V, v_prev_local, k=int(V.shape[1]))

    if v_prev_local is not None and float(beta) < 1.0:
        v = (1.0 - float(beta)) * v_prev_local + float(beta) * v_new
        v = v / (v.norm() + 1e-12)
    else:
        v = v_new

    f_flat = forces.reshape(-1)
    gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
    gad_vec = gad_flat.view(num_atoms, 3)

    # Compute eigenvalue spectrum (first 6 vibrational modes)
    n_eigs = min(6, len(vib_evals))
    eig_spectrum = [float("nan")] * 6
    for i in range(n_eigs):
        eig_spectrum[i] = float(vib_evals[i].item())

    # Compute eigenvalue gaps
    if n_eigs >= 2:
        eig_gap_01 = abs(eig_spectrum[1] - eig_spectrum[0])
        eig_gap_01_rel = (eig_spectrum[1] - eig_spectrum[0]) / abs(eig_spectrum[0]) if abs(eig_spectrum[0]) > 1e-12 else float("nan")
    else:
        eig_gap_01, eig_gap_01_rel = float("nan"), float("nan")

    if n_eigs >= 3:
        eig_gap_12 = abs(eig_spectrum[2] - eig_spectrum[1])
    else:
        eig_gap_12 = float("nan")

    # Morse index
    neg_vib = int((vib_evals < -tr_threshold).sum().item())
    neg_eig_sum = float(vib_evals[vib_evals < -tr_threshold].sum().item()) if neg_vib > 0 else 0.0

    # Singularity metric
    if n_eigs >= 2:
        gaps = torch.abs(vib_evals[1:n_eigs] - vib_evals[:n_eigs-1])
        singularity_metric = float(gaps.min().item()) if len(gaps) > 0 else float("nan")
    else:
        singularity_metric = float("nan")

    # Gradient projections
    grad_flat = -f_flat
    grad_norm = float(grad_flat.norm().item())

    v1 = V[:, 0] / (V[:, 0].norm() + 1e-12)
    v2 = V[:, 1] / (V[:, 1].norm() + 1e-12) if V.shape[1] > 1 else torch.zeros_like(v1)

    if grad_norm > 1e-12:
        grad_proj_v1 = float(torch.abs(torch.dot(grad_flat, v1)).item() / grad_norm)
        grad_proj_v2 = float(torch.abs(torch.dot(grad_flat, v2)).item() / grad_norm)
    else:
        grad_proj_v1 = grad_proj_v2 = float("nan")

    # GAD-gradient angle
    gad_flat_norm = gad_flat / (gad_flat.norm() + 1e-12)
    neg_grad_norm = -grad_flat / (grad_norm + 1e-12)
    cos_angle = torch.dot(gad_flat_norm, neg_grad_norm).clamp(-1, 1)
    gad_grad_angle = float(torch.acos(cos_angle).item() * 180.0 / np.pi)

    # Rayleigh quotient
    rayleigh_v1 = float(torch.dot(v1, hess @ v1).item())

    v_next = v.detach().clone().reshape(-1)

    info = {
        "mode_overlap": float(overlap),
        "mode_index": int(j),
        "eig_0": eig_spectrum[0],
        "eig_1": eig_spectrum[1],
        "eig_2": eig_spectrum[2],
        "eig_3": eig_spectrum[3],
        "eig_4": eig_spectrum[4],
        "eig_5": eig_spectrum[5],
        "eig_gap_01": eig_gap_01,
        "eig_gap_01_rel": eig_gap_01_rel,
        "eig_gap_12": eig_gap_12,
        "neg_vib": neg_vib,
        "neg_eig_sum": neg_eig_sum,
        "singularity_metric": singularity_metric,
        "grad_norm": grad_norm,
        "grad_proj_v1": grad_proj_v1,
        "grad_proj_v2": grad_proj_v2,
        "gad_grad_angle": gad_grad_angle,
        "rayleigh_v1": rayleigh_v1,
    }

    return gad_vec, v_next, info


def run_multi_mode_with_diagnostics(
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
    escape_disp_threshold: float,
    escape_window: int,
    tr_threshold: float = 1e-6,
    escape_neg_vib_std: float,
    escape_delta: float,
    adaptive_delta: bool,
    min_interatomic_dist: float,
    max_escape_cycles: int,
    sample_id: str = "unknown",
    formula: str = "",
    known_ts_coords: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, Any], TrajectoryLogger]:
    """Run GAD with multi-mode escape and comprehensive diagnostic logging.

    This is the enhanced version of run_multi_mode_escape that uses the
    TrajectoryLogger to capture extended metrics for post-hoc analysis.

    Returns:
        final_out_dict: Standard output dictionary
        trajectory_logger: TrajectoryLogger with all diagnostic data
    """
    coords = coords0.detach().clone().to(torch.float32)

    # Initialize trajectory logger
    logger = TrajectoryLogger(sample_id=sample_id, formula=formula)
    logger.known_ts_coords = known_ts_coords

    # Standard trajectory dict for backward compatibility
    trajectory = {k: [] for k in [
        "energy", "force_mean", "eig0", "eig1", "eig_product", "neg_vib",
        "disp_from_last", "disp_from_start", "dt_eff", "gad_norm", "escape_cycle",
        # Extended metrics
        "eig_0", "eig_1", "eig_2", "eig_3", "eig_4", "eig_5",
        "eig_gap_01", "eig_gap_01_rel", "eig_gap_12",
        "morse_index", "neg_eig_sum", "singularity_metric",
        "grad_norm", "grad_proj_v1", "grad_proj_v2", "gad_grad_angle", "rayleigh_v1",
        "mode_overlap", "mode_index",
    ]}

    start_pos = coords.clone()
    prev_pos = coords.clone()
    prev_energy = None

    steps_to_ts: Optional[int] = None
    total_steps = 0
    escape_cycle = 0
    escape_events: list[Dict[str, Any]] = []
    early_stopped = False

    disp_history: list[float] = []
    neg_vib_history: list[int] = []

    dt_eff_state = float(dt)
    best_neg_vib: Optional[int] = None
    no_improve = 0

    v_prev: torch.Tensor | None = None

    while escape_cycle < max_escape_cycles and total_steps < n_steps:
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        energy = out.get("energy")
        forces = out.get("forces")
        hessian = out.get("hessian")

        energy_value = _to_float(energy)
        force_mean = _force_mean(forces)

        scine_elements = get_scine_elements_from_predict_output(out)

        # Get projected Hessian
        hess_proj = get_projected_hessian(hessian, coords, atomic_nums, scine_elements=scine_elements)

        # Compute GAD vector with full eigenvalue info
        gad_vec, v_next, step_info = _step_metrics_from_projected_hessian(
            forces=forces,
            hessian_proj=hess_proj,
            tr_threshold=tr_threshold,
            v_prev=v_prev,
            k_track=8,
            beta=1.0,
        )
        v_prev = v_next

        neg_vib = step_info["neg_vib"]
        eig0 = step_info["eig_0"]
        eig1 = step_info["eig_1"]
        eig_prod = eig0 * eig1 if np.isfinite(eig0) and np.isfinite(eig1) else float("inf")

        disp_from_last = float((coords - prev_pos).norm(dim=1).mean().item()) if total_steps > 0 else 0.0
        disp_from_start = float((coords - start_pos).norm(dim=1).mean().item()) if total_steps > 0 else 0.0
        gad_norm = _mean_atom_norm(gad_vec)

        # Log to trajectory logger (extended logging)
        logger.log_step(
            step=total_steps,
            coords=coords,
            energy=energy_value,
            forces=forces,
            hessian_proj=hess_proj,
            gad_vec=gad_vec,
            dt_eff=dt_eff_state,
            coords_prev=prev_pos if total_steps > 0 else None,
            energy_prev=prev_energy,
        )

        # Update rolling history
        if total_steps > 0:
            disp_history.append(disp_from_last)
            neg_vib_history.append(neg_vib)

        # Update trajectory dict (backward compat)
        trajectory["energy"].append(energy_value)
        trajectory["force_mean"].append(force_mean)
        trajectory["eig0"].append(eig0)
        trajectory["eig1"].append(eig1)
        trajectory["eig_product"].append(eig_prod)
        trajectory["neg_vib"].append(int(neg_vib))
        trajectory["disp_from_last"].append(disp_from_last)
        trajectory["disp_from_start"].append(disp_from_start)
        trajectory["gad_norm"].append(gad_norm)
        trajectory["escape_cycle"].append(escape_cycle)
        trajectory["mode_overlap"].append(step_info["mode_overlap"])
        trajectory["mode_index"].append(step_info["mode_index"])

        # Extended metrics
        for key in ["eig_0", "eig_1", "eig_2", "eig_3", "eig_4", "eig_5",
                    "eig_gap_01", "eig_gap_01_rel", "eig_gap_12",
                    "neg_eig_sum", "singularity_metric",
                    "grad_norm", "grad_proj_v1", "grad_proj_v2", "gad_grad_angle", "rayleigh_v1"]:
            trajectory[key].append(step_info.get(key, float("nan")))
        trajectory["morse_index"].append(neg_vib)

        # Check for TS (index = 1)
        if stop_at_ts and steps_to_ts is None and np.isfinite(eig_prod) and eig_prod < -abs(ts_eps):
            steps_to_ts = total_steps
            trajectory["dt_eff"].append(float("nan"))
            break

        # Check for plateau convergence
        is_plateau = _check_plateau_convergence(
            disp_history,
            neg_vib_history,
            neg_vib,
            window=escape_window,
            disp_threshold=escape_disp_threshold,
            neg_vib_std_threshold=escape_neg_vib_std,
        )

        if is_plateau:
            trajectory["dt_eff"].append(float("nan"))

            # Store pre-escape state for logging
            pre_hess_proj = hess_proj.clone()
            pre_forces = forces.clone()
            pre_energy = energy_value

            # Perform escape
            new_coords, escape_info = perform_escape_perturbation(
                predict_fn,
                coords,
                atomic_nums,
                hessian,
                escape_delta=escape_delta,
                adaptive_delta=adaptive_delta,
                min_interatomic_dist=min_interatomic_dist,
                scine_elements=scine_elements,
            )

            # Compute post-escape state for logging
            post_out = predict_fn(new_coords, atomic_nums, do_hessian=True, require_grad=False)
            post_hess_proj = get_projected_hessian(post_out["hessian"], new_coords, atomic_nums, scine_elements=scine_elements)
            post_forces = post_out["forces"]
            post_energy = _to_float(post_out["energy"])

            # Create detailed escape event
            escape_event = create_escape_event(
                step=total_steps,
                escape_cycle=escape_cycle,
                trigger_reason="displacement_plateau",
                pre_hessian_proj=pre_hess_proj,
                pre_forces=pre_forces,
                pre_energy=pre_energy,
                kick_mode=2,  # v2
                kick_direction=f"+v2" if escape_info.get("direction", 1) > 0 else "-v2",
                kick_delta_base=escape_info.get("delta_base", escape_delta),
                kick_delta_effective=escape_info.get("delta_used", 0.0),
                kick_lambda=escape_info.get("lambda2", float("nan")),
                post_hessian_proj=post_hess_proj,
                post_forces=post_forces,
                post_energy=post_energy,
                accepted=escape_info.get("escape_success", False),
                rejection_reason=None if escape_info.get("escape_success", False) else "geometry_invalid",
                displacement_magnitude=escape_info.get("disp_per_atom", 0.0),
                min_dist_after=escape_info.get("min_dist_after", float("inf")),
                mean_disp_at_trigger=float(np.mean(disp_history[-escape_window:])) if len(disp_history) >= escape_window else 0.0,
                neg_vib_std_at_trigger=float(np.std(neg_vib_history[-escape_window:])) if len(neg_vib_history) >= escape_window else 0.0,
                tr_threshold=tr_threshold,
            )

            # Log to trajectory logger
            logger.log_escape(escape_event)

            # Also keep backward-compatible escape_events list
            escape_info["step"] = total_steps
            escape_info["neg_vib_before"] = neg_vib
            escape_info["gad_norm"] = gad_norm
            escape_info["mean_disp_at_trigger"] = float(np.mean(disp_history[-escape_window:])) if len(disp_history) >= escape_window else 0.0
            escape_events.append(escape_info)

            coords = new_coords
            disp_history.clear()
            neg_vib_history.clear()
            best_neg_vib = None
            no_improve = 0
            dt_eff_state = float(dt)
            prev_pos = coords.clone()
            v_prev = None

            escape_cycle += 1
            total_steps += 1
            prev_energy = post_energy
            continue

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
        prev_energy = energy_value
        coords = (coords + dt_eff * gad_vec).detach()
        total_steps += 1

    # Pad trajectories
    while len(trajectory["dt_eff"]) < len(trajectory["energy"]):
        trajectory["dt_eff"].append(float("nan"))

    # Final vibrational analysis
    final_neg_vib = -1
    try:
        final_out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        scine_elements = get_scine_elements_from_predict_output(final_out)
        final_vib_eigvals = vibrational_eigvals(final_out["hessian"], coords, atomic_nums, scine_elements=scine_elements)
        final_neg_vib = int((final_vib_eigvals < 0).sum().item())
    except Exception:
        pass

    # Finalize trajectory logger
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
        "escape_cycles_used": escape_cycle,
        "escape_events": escape_events,
        "total_steps": total_steps,
        "early_stopped": early_stopped,
    }

    return final_out_dict, logger


def main(argv: Optional[list[str]] = None) -> None:
    """Main entry point for diagnostic runner."""
    parser = argparse.ArgumentParser(
        description="GAD with comprehensive diagnostic logging for failure analysis."
    )
    parser = add_common_args(parser)

    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--start-from", type=str, default="midpoint_rt")
    parser.add_argument("--stop-at-ts", action="store_true")
    parser.add_argument("--ts-eps", type=float, default=1e-5)
    parser.add_argument("--dt-control", type=str, default="neg_eig_plateau")
    parser.add_argument("--dt-min", type=float, default=1e-6)
    parser.add_argument("--dt-max", type=float, default=0.05)
    parser.add_argument("--max-atom-disp", type=float, default=0.25)
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--plateau-boost", type=float, default=1.5)
    parser.add_argument("--plateau-shrink", type=float, default=0.5)
    parser.add_argument("--escape-disp-threshold", type=float, default=5e-4)
    parser.add_argument("--escape-window", type=int, default=20)
    parser.add_argument("--escape-neg-vib-std", type=float, default=0.5)
    parser.add_argument("--escape-delta", type=float, default=0.1)
    parser.add_argument("--adaptive-delta", action="store_true", default=True)
    parser.add_argument("--no-adaptive-delta", action="store_false", dest="adaptive_delta")
    parser.add_argument("--min-interatomic-dist", type=float, default=0.5)
    parser.add_argument("--max-escape-cycles", type=int, default=1000)
    parser.add_argument("--tr-threshold", type=float, default=1e-6)

    args = parser.parse_args(argv)

    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    calculator_type = getattr(args, "calculator", "scine").lower()
    if calculator_type == "scine":
        device = "cpu"

    predict_fn = make_predict_fn_from_calculator(calculator, calculator_type)

    # Create output directory for diagnostics
    diag_dir = Path(out_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break

        batch = batch.to(device)
        atomic_nums = batch.z.detach().cpu().to(device)
        formula = getattr(batch, "formula", f"sample_{i:03d}")

        start_coords = parse_starting_geometry(
            args.start_from,
            batch,
            noise_seed=getattr(args, "noise_seed", None),
            sample_index=i,
        ).detach().to(device)

        print(f"\n[Sample {i}] {formula}")
        t0 = time.time()

        try:
            out_dict, traj_logger = run_multi_mode_with_diagnostics(
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
                max_atom_disp=float(args.max_atom_disp),
                plateau_patience=int(args.plateau_patience),
                plateau_boost=float(args.plateau_boost),
                plateau_shrink=float(args.plateau_shrink),
                escape_disp_threshold=float(args.escape_disp_threshold),
                escape_window=int(args.escape_window),
                tr_threshold=float(args.tr_threshold),
                escape_neg_vib_std=float(args.escape_neg_vib_std),
                escape_delta=float(args.escape_delta),
                adaptive_delta=bool(args.adaptive_delta),
                min_interatomic_dist=float(args.min_interatomic_dist),
                max_escape_cycles=int(args.max_escape_cycles),
                sample_id=f"sample_{i:03d}",
                formula=str(formula),
            )
            wall = time.time() - t0

            # Print summary
            traj_logger.print_summary()

            # Save diagnostic files
            paths = traj_logger.save(diag_dir)
            print(f"  Saved diagnostics: {list(paths.values())}")

            # Collect summary for batch analysis
            summary = traj_logger.get_full_summary()
            summary["wallclock_s"] = wall
            all_summaries.append(summary)

        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")

    # Save batch summary
    batch_summary_path = diag_dir / "batch_summary.json"
    with open(batch_summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"\nBatch summary saved to: {batch_summary_path}")


if __name__ == "__main__":
    main()
