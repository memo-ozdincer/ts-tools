from __future__ import annotations

"""Multi-mode escape experiment for GAD.

This experiment implements the "Perturbation & Saddle Escape" algorithm to handle
high-index saddle points. Instead of using a step-size floor (which can be unstable),
we detect when GAD converges to a high-index saddle (index > 1) and explicitly
perturb along the second-lowest eigenvector (v2) to escape toward index-1.

Detection uses DISPLACEMENT-based criteria (not GAD norm), since:
- GAD norm ~ force_mean (0.2-3 eV/Å) stays moderate even at plateaus
- disp_from_last dropping to ~1µÅ is the true indicator (dt shrinks/caps)
- We trigger escape when: mean(disp[-window:]) < threshold AND neg_vib stable AND >1

Algorithm:
1. Run GAD, tracking recent displacement history
2. Detect plateau: mean displacement < threshold over window, stable neg_vib, index > 1
3. If index = 1: Success - found TS
4. If plateau at index > 1: Apply perturbation along v2, pick direction that lowers energy
5. Resume GAD from perturbed geometry
6. Repeat until index = 1 or max escape cycles reached
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

from ..core_algos.gad import gad_euler_step
from ..dependencies.common_utils import add_common_args, parse_starting_geometry, setup_experiment
from ..dependencies.experiment_logger import ExperimentLogger, RunResult, build_loss_type_flags
from ..dependencies.hessian import vibrational_eigvals, get_scine_elements_from_predict_output
from ..logging import finish_wandb, init_wandb_run, log_sample, log_summary
from ..logging.plotly_utils import plot_gad_trajectory_interactive
from ..runners._predict import make_predict_fn_from_calculator


def _sanitize_wandb_name(s: str) -> str:
    s = str(s)
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s[:128] if len(s) > 128 else s


def _auto_wandb_name(*, script: str, loss_type_flags: str, args: argparse.Namespace) -> str:
    calculator = getattr(args, "calculator", "hip")
    start_from = getattr(args, "start_from", "unknown")
    method = getattr(args, "method", "euler")
    escape_delta = getattr(args, "escape_delta", None)
    n_steps = getattr(args, "n_steps", None)
    noise_seed = getattr(args, "noise_seed", None)
    job_id = os.environ.get("SLURM_JOB_ID")

    parts = [
        script,
        str(calculator),
        str(start_from),
        str(method),
        f"delta{escape_delta}" if escape_delta is not None else None,
        f"steps{n_steps}" if n_steps is not None else None,
        f"seed{noise_seed}" if noise_seed is not None else None,
        f"job{job_id}" if job_id else None,
        str(loss_type_flags),
    ]
    parts = [p for p in parts if p]
    return _sanitize_wandb_name("__".join(parts))


def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().reshape(-1)[0].item())
    return float(x)


def _force_mean(forces: torch.Tensor) -> float:
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    f = forces.reshape(-1, 3)
    return float(f.norm(dim=1).mean().item())


def _mean_atom_norm(x: torch.Tensor) -> float:
    if x.dim() == 3 and x.shape[0] == 1:
        x = x[0]
    return float(x.reshape(-1, 3).norm(dim=1).mean().item())


def _max_atom_norm(x: torch.Tensor) -> float:
    if x.dim() == 3 and x.shape[0] == 1:
        x = x[0]
    return float(x.reshape(-1, 3).norm(dim=1).max().item())


def _prepare_hessian(hess: torch.Tensor, num_atoms: int) -> torch.Tensor:
    """Reshape hessian to (3*N, 3*N) matrix."""
    if hess.dim() == 1:
        side = int(hess.numel() ** 0.5)
        return hess.view(side, side)
    if hess.dim() == 3 and hess.shape[0] == 1:
        hess = hess[0]
    if hess.dim() > 2:
        return hess.reshape(3 * num_atoms, 3 * num_atoms)
    return hess


def perform_escape_perturbation(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    hessian: torch.Tensor,
    *,
    escape_delta: float,
    adaptive_delta: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Perturb geometry along v2 (second-lowest eigenvector) to escape high-index saddle.

    Tries both +delta and -delta directions, picks the one with lower energy.

    Args:
        predict_fn: Energy/forces prediction function
        coords: Current coordinates (N, 3)
        atomic_nums: Atomic numbers
        hessian: Hessian matrix
        escape_delta: Base displacement magnitude in Angstrom
        adaptive_delta: If True, scale delta by 1/sqrt(|lambda2|)

    Returns:
        new_coords: Perturbed coordinates
        info: Dict with escape details (delta_used, direction, energy_change, etc.)
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    num_atoms = int(coords.shape[0])

    hess = _prepare_hessian(hessian, num_atoms)

    # Get eigenvalues and eigenvectors
    evals, evecs = torch.linalg.eigh(hess)
    v2 = evecs[:, 1]  # Second-lowest eigenvector
    v2 = v2 / (v2.norm() + 1e-12)  # Normalize

    lambda2 = float(evals[1].item())

    # Adaptive delta scaling based on curvature
    delta = float(escape_delta)
    if adaptive_delta and lambda2 < -0.01:
        # Scale larger for strong negative curvature
        delta = float(escape_delta) / np.sqrt(abs(lambda2))
        delta = min(delta, 1.0)  # Cap at 1 Angstrom to avoid steric clashes

    # Reshape v2 to (N, 3)
    v2_3d = v2.reshape(num_atoms, 3)

    # Compute displacement magnitude per atom (for logging)
    disp_per_atom = float(v2_3d.norm(dim=1).mean().item()) * delta

    # Try both directions
    coords_plus = coords + delta * v2_3d
    coords_minus = coords - delta * v2_3d

    # Evaluate energies (no hessian needed here, just energy)
    out_plus = predict_fn(coords_plus, atomic_nums, do_hessian=False, require_grad=False)
    out_minus = predict_fn(coords_minus, atomic_nums, do_hessian=False, require_grad=False)

    E_current = predict_fn(coords, atomic_nums, do_hessian=False, require_grad=False)["energy"]
    E_plus = out_plus["energy"]
    E_minus = out_minus["energy"]

    E_current = _to_float(E_current)
    E_plus = _to_float(E_plus)
    E_minus = _to_float(E_minus)

    # Pick lower energy direction
    if E_plus < E_minus:
        new_coords = coords_plus
        direction = +1
        energy_after = E_plus
    else:
        new_coords = coords_minus
        direction = -1
        energy_after = E_minus

    info = {
        "delta_used": delta,
        "direction": direction,
        "lambda2": lambda2,
        "energy_before": E_current,
        "energy_after": energy_after,
        "energy_change": energy_after - E_current,
        "disp_per_atom": disp_per_atom,
    }

    return new_coords.detach(), info


def _check_plateau_convergence(
    disp_history: list[float],
    neg_vib_history: list[int],
    current_neg_vib: int,
    *,
    window: int,
    disp_threshold: float,
    neg_vib_std_threshold: float,
) -> bool:
    """Check if we've converged to a plateau based on displacement history.

    Triggers when:
    1. mean(disp[-window:]) < disp_threshold (tiny steps)
    2. std(neg_vib[-window:]) <= neg_vib_std_threshold (stable saddle index)
    3. current_neg_vib > 1 (high-index saddle)
    """
    if len(disp_history) < window:
        return False

    recent_disp = disp_history[-window:]
    recent_neg_vib = neg_vib_history[-window:]

    mean_disp = float(np.mean(recent_disp))
    std_neg_vib = float(np.std(recent_neg_vib))

    return (
        mean_disp < disp_threshold
        and std_neg_vib <= neg_vib_std_threshold
        and current_neg_vib > 1
    )


def run_multi_mode_escape(
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
    # Multi-mode escape parameters (displacement-based detection)
    escape_disp_threshold: float,
    escape_window: int,
    escape_neg_vib_std: float,
    escape_delta: float,
    adaptive_delta: bool,
    max_escape_cycles: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run GAD with multi-mode escape mechanism.

    Uses displacement-based plateau detection:
    - Triggers escape when mean(disp[-window:]) < threshold AND neg_vib stable AND >1
    - This is more robust than GAD norm, which stays ~0.1-1 eV/Å even at plateaus

    The algorithm:
    1. Run GAD, accumulating displacement history
    2. Check for plateau: tiny displacements + stable neg_vib + index > 1
    3. If plateau at index > 1: perturb along v2 and restart GAD
    4. Repeat until index = 1 or max_escape_cycles reached
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
        "escape_cycle",  # Track which escape cycle we're in
    ]}

    start_pos = coords.clone()
    prev_pos = coords.clone()

    steps_to_ts: Optional[int] = None
    total_steps = 0
    escape_cycle = 0
    escape_events: list[Dict[str, Any]] = []

    # Rolling history for displacement-based plateau detection
    disp_history: list[float] = []
    neg_vib_history: list[int] = []

    # Stateful dt controller variables
    dt_eff_state = float(dt)
    best_neg_vib: Optional[int] = None
    no_improve = 0

    while escape_cycle < max_escape_cycles and total_steps < n_steps:
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        energy = out.get("energy")
        forces = out.get("forces")
        hessian = out.get("hessian")

        energy_value = _to_float(energy)
        force_mean = _force_mean(forces)

        scine_elements = get_scine_elements_from_predict_output(out)
        vib = vibrational_eigvals(hessian, coords, atomic_nums, scine_elements=scine_elements)
        eig0 = float(vib[0].item()) if vib.numel() >= 1 else float("nan")
        eig1 = float(vib[1].item()) if vib.numel() >= 2 else float("nan")
        eig_prod = float((vib[0] * vib[1]).item()) if vib.numel() >= 2 else float("inf")
        neg_vib = int((vib < 0).sum().item()) if vib.numel() > 0 else -1

        disp_from_last = float((coords - prev_pos).norm(dim=1).mean().item()) if total_steps > 0 else 0.0
        disp_from_start = float((coords - start_pos).norm(dim=1).mean().item()) if total_steps > 0 else 0.0

        # Compute GAD vector and its norm
        step_out = gad_euler_step(predict_fn, coords, atomic_nums, dt=0.0, out=out)
        gad_vec = step_out["gad_vec"]
        gad_norm = _mean_atom_norm(gad_vec)

        # Update rolling history
        if total_steps > 0:  # Only add after first step (disp_from_last=0 at step 0)
            disp_history.append(disp_from_last)
            neg_vib_history.append(neg_vib)

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

        # Check for TS (index = 1)
        if stop_at_ts and steps_to_ts is None and np.isfinite(eig_prod) and eig_prod < -abs(ts_eps):
            steps_to_ts = total_steps
            trajectory["dt_eff"].append(float("nan"))
            break

        # Check for plateau convergence (displacement-based)
        is_plateau = _check_plateau_convergence(
            disp_history,
            neg_vib_history,
            neg_vib,
            window=escape_window,
            disp_threshold=escape_disp_threshold,
            neg_vib_std_threshold=escape_neg_vib_std,
        )

        if is_plateau:
            # Converged to high-index saddle, perform escape
            trajectory["dt_eff"].append(float("nan"))

            new_coords, escape_info = perform_escape_perturbation(
                predict_fn,
                coords,
                atomic_nums,
                hessian,
                escape_delta=escape_delta,
                adaptive_delta=adaptive_delta,
            )
            coords = new_coords
            escape_info["step"] = total_steps
            escape_info["neg_vib_before"] = neg_vib
            escape_info["gad_norm"] = gad_norm
            escape_info["mean_disp_at_trigger"] = float(np.mean(disp_history[-escape_window:]))
            escape_events.append(escape_info)

            # Reset state after escape
            disp_history.clear()
            neg_vib_history.clear()
            best_neg_vib = None
            no_improve = 0
            dt_eff_state = float(dt)
            prev_pos = coords.clone()

            escape_cycle += 1
            total_steps += 1
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
        coords = (coords + dt_eff * gad_vec).detach()
        total_steps += 1

    # Pad trajectories to same length
    while len(trajectory["dt_eff"]) < len(trajectory["energy"]):
        trajectory["dt_eff"].append(float("nan"))
    while len(trajectory["gad_norm"]) < len(trajectory["energy"]):
        trajectory["gad_norm"].append(float("nan"))
    while len(trajectory["escape_cycle"]) < len(trajectory["energy"]):
        trajectory["escape_cycle"].append(escape_cycle)

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
        "escape_cycles_used": escape_cycle,
        "escape_events": escape_events,
        "total_steps": total_steps,
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


def _save_trajectory_json(logger: ExperimentLogger, result: RunResult, trajectory: Dict[str, Any], escape_events: list) -> Optional[str]:
    transition_dir = logger.run_dir / result.transition_key
    transition_dir.mkdir(parents=True, exist_ok=True)
    path = transition_dir / f"trajectory_{result.sample_index:03d}.json"
    try:
        data = {
            "trajectory": trajectory,
            "escape_events": escape_events,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return str(path)
    except Exception:
        return None


def main(
    argv: Optional[list[str]] = None,
    *,
    default_calculator: Optional[str] = None,
    enforce_calculator: bool = False,
    script_name_prefix: str = "exp-multi-mode",
) -> None:
    parser = argparse.ArgumentParser(
        description="Experiment: Multi-mode escape for GAD (escape high-index saddles via v2 perturbation)."
    )
    parser = add_common_args(parser)

    if default_calculator is not None:
        parser.set_defaults(calculator=str(default_calculator))

    parser.add_argument("--method", type=str, default="euler", choices=["euler"])

    parser.add_argument("--n-steps", type=int, default=1500, help="Total max GAD steps across all cycles")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--start-from", type=str, default="midpoint_rt")
    parser.add_argument("--stop-at-ts", action="store_true")
    parser.add_argument("--ts-eps", type=float, default=1e-5)

    # dt control (plateau-based, but NO floor)
    parser.add_argument(
        "--dt-control",
        type=str,
        default="neg_eig_plateau",
        choices=["fixed", "neg_eig_plateau"],
        help="How to choose dt each step.",
    )
    parser.add_argument("--dt-min", type=float, default=1e-6)
    parser.add_argument("--dt-max", type=float, default=0.05)
    parser.add_argument(
        "--max-atom-disp",
        type=float,
        default=0.25,
        help="Safety cap: max per-atom displacement (A) per step.",
    )

    # Plateau controller knobs
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--plateau-boost", type=float, default=1.5)
    parser.add_argument("--plateau-shrink", type=float, default=0.5)

    # Multi-mode escape parameters (displacement-based detection)
    parser.add_argument(
        "--escape-disp-threshold",
        type=float,
        default=5e-4,
        help="Mean displacement threshold (A) for plateau detection. Trigger escape when "
             "mean(disp[-window:]) < this value.",
    )
    parser.add_argument(
        "--escape-window",
        type=int,
        default=20,
        help="Number of recent steps to consider for plateau detection.",
    )
    parser.add_argument(
        "--escape-neg-vib-std",
        type=float,
        default=0.5,
        help="Max std(neg_vib) over window for plateau detection (stable saddle index).",
    )
    parser.add_argument(
        "--escape-delta",
        type=float,
        default=0.2,
        help="Base displacement magnitude (A) for v2 perturbation.",
    )
    parser.add_argument(
        "--adaptive-delta",
        action="store_true",
        default=True,
        help="Scale delta by 1/sqrt(|lambda2|) for adaptive perturbation.",
    )
    parser.add_argument(
        "--no-adaptive-delta",
        action="store_false",
        dest="adaptive_delta",
        help="Disable adaptive delta scaling.",
    )
    parser.add_argument(
        "--max-escape-cycles",
        type=int,
        default=10,
        help="Maximum number of escape attempts.",
    )

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gad-ts-search")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Optional W&B run name. If omitted, an informative name is auto-generated.",
    )

    args = parser.parse_args(argv)

    if enforce_calculator and default_calculator is not None:
        if str(getattr(args, "calculator", "")).lower() != str(default_calculator).lower():
            raise ValueError(
                f"This entrypoint enforces --calculator={default_calculator}. "
                f"Got --calculator={getattr(args, 'calculator', None)}."
            )

    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    calculator_type = getattr(args, "calculator", "hip").lower()
    if calculator_type == "scine":
        device = "cpu"

    predict_fn = make_predict_fn_from_calculator(calculator, calculator_type)

    loss_type_flags = build_loss_type_flags(args)
    script_name = f"{script_name_prefix}-{args.method}"
    logger = ExperimentLogger(
        base_dir=out_dir,
        script_name=script_name,
        loss_type_flags=loss_type_flags,
        max_graphs_per_transition=10,
        random_seed=42,
    )

    if args.wandb:
        wandb_config = {
            "script": script_name,
            "method": args.method,
            "start_from": args.start_from,
            "stop_at_ts": bool(args.stop_at_ts),
            "calculator": getattr(args, "calculator", "hip"),
            "dt": args.dt,
            "n_steps": args.n_steps,
            "dt_control": args.dt_control,
            "dt_min": args.dt_min,
            "dt_max": args.dt_max,
            "max_atom_disp": args.max_atom_disp,
            "plateau_patience": args.plateau_patience,
            "plateau_boost": args.plateau_boost,
            "plateau_shrink": args.plateau_shrink,
            "escape_disp_threshold": args.escape_disp_threshold,
            "escape_window": args.escape_window,
            "escape_neg_vib_std": args.escape_neg_vib_std,
            "escape_delta": args.escape_delta,
            "adaptive_delta": args.adaptive_delta,
            "max_escape_cycles": args.max_escape_cycles,
        }
        wandb_name = args.wandb_name
        if not wandb_name:
            wandb_name = _auto_wandb_name(script=script_name, loss_type_flags=loss_type_flags, args=args)

        init_wandb_run(
            project=args.wandb_project,
            name=str(wandb_name),
            config=wandb_config,
            entity=args.wandb_entity,
            tags=[args.start_from, args.method, "multi-mode", str(args.dt_control)],
            run_dir=out_dir,
        )

    all_metrics = defaultdict(list)

    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break

        batch = batch.to(device)
        atomic_nums = batch.z.detach().cpu().to(device)
        formula = getattr(batch, "formula", "sample")

        start_coords = parse_starting_geometry(
            args.start_from,
            batch,
            noise_seed=getattr(args, "noise_seed", None),
            sample_index=i,
        ).detach().to(device)

        # Initial vibrational order for transition bucketing
        try:
            init_out = predict_fn(start_coords, atomic_nums, do_hessian=True, require_grad=False)
            init_scine_elements = get_scine_elements_from_predict_output(init_out)
            init_vib = vibrational_eigvals(init_out["hessian"], start_coords, atomic_nums, scine_elements=init_scine_elements)
            initial_neg = int((init_vib < 0).sum().item())
        except Exception:
            initial_neg = -1

        t0 = time.time()
        out_dict, aux = run_multi_mode_escape(
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
            escape_disp_threshold=float(args.escape_disp_threshold),
            escape_window=int(args.escape_window),
            escape_neg_vib_std=float(args.escape_neg_vib_std),
            escape_delta=float(args.escape_delta),
            adaptive_delta=bool(args.adaptive_delta),
            max_escape_cycles=int(args.max_escape_cycles),
        )
        wall = time.time() - t0

        final_neg = out_dict.get("final_neg_vibrational", -1)

        result = RunResult(
            sample_index=i,
            formula=str(formula),
            initial_neg_eigvals=initial_neg,
            final_neg_eigvals=int(final_neg) if final_neg is not None else -1,
            initial_neg_vibrational=None,
            final_neg_vibrational=int(final_neg) if final_neg is not None else None,
            steps_taken=int(out_dict["steps_taken"]),
            steps_to_ts=aux.get("steps_to_ts"),
            final_time=float(wall),
            final_eig0=out_dict.get("final_eig0"),
            final_eig1=out_dict.get("final_eig1"),
            final_eig_product=out_dict.get("final_eig_product"),
            final_loss=None,
            rmsd_to_known_ts=None,
            stop_reason=None,
            plot_path=None,
            extra_data={
                "method": str(args.method),
                "dt_control": str(args.dt_control),
                "escape_cycles_used": aux.get("escape_cycles_used"),
                "escape_events": aux.get("escape_events"),
                "escape_disp_threshold": float(args.escape_disp_threshold),
                "escape_window": int(args.escape_window),
                "escape_neg_vib_std": float(args.escape_neg_vib_std),
                "escape_delta": float(args.escape_delta),
                "adaptive_delta": bool(args.adaptive_delta),
                "max_escape_cycles": int(args.max_escape_cycles),
            },
        )

        logger.add_result(result)

        # Generate interactive figure
        fig_interactive = plot_gad_trajectory_interactive(
            out_dict["trajectory"],
            sample_index=i,
            formula=str(formula),
            start_from=args.start_from,
            initial_neg_num=initial_neg,
            final_neg_num=int(final_neg) if final_neg is not None else -1,
            steps_to_ts=aux.get("steps_to_ts"),
        )

        # Save HTML plot
        html_path = Path(out_dir) / f"traj_{i:03d}.html"
        fig_interactive.write_html(str(html_path))
        result.plot_path = str(html_path)

        # Save trajectory JSON with escape events
        _save_trajectory_json(logger, result, out_dict["trajectory"], aux.get("escape_events", []))

        metrics = {
            "steps_taken": result.steps_taken,
            "steps_to_ts": result.steps_to_ts,
            "final_eig0": result.final_eig0,
            "final_eig1": result.final_eig1,
            "final_eig_product": result.final_eig_product,
            "final_neg_vibrational": result.final_neg_vibrational,
            "wallclock_s": result.final_time,
            "escape_cycles_used": aux.get("escape_cycles_used"),
        }

        for k, v in metrics.items():
            if v is not None:
                all_metrics[k].append(v)

        if args.wandb:
            log_sample(i, metrics, fig=fig_interactive, plot_name="trajectory_interactive")

    all_runs_path, aggregate_stats_path = logger.save_all_results()
    summary = logger.compute_aggregate_stats()
    logger.print_summary()

    if args.wandb:
        log_summary(summary)
        finish_wandb()

    print(f"Saved results: {all_runs_path}")
    print(f"Saved stats:   {aggregate_stats_path}")


if __name__ == "__main__":
    # Default to SCINE for this module
    main(default_calculator="scine", enforce_calculator=True, script_name_prefix="exp-scine-multi-mode")
