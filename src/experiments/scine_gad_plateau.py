from __future__ import annotations

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

from ..core_algos.gad import compute_gad_vector
from ..dependencies.common_utils import add_common_args, parse_starting_geometry, setup_experiment
from ..dependencies.experiment_logger import ExperimentLogger, RunResult, build_loss_type_flags
from ..dependencies.hessian import vibrational_eigvals, get_scine_elements_from_predict_output
from ..logging import finish_wandb, init_wandb_run, log_sample, log_summary
from ..logging.trajectory_plots import plot_gad_trajectory_3x2
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
    method = getattr(args, "method", "unknown")
    dt_control = getattr(args, "dt_control", None)
    dt = getattr(args, "dt", None)
    n_steps = getattr(args, "n_steps", None)
    noise_seed = getattr(args, "noise_seed", None)
    job_id = os.environ.get("SLURM_JOB_ID")

    parts = [
        script,
        str(calculator),
        str(start_from),
        str(method),
        str(dt_control) if dt_control else None,
        f"dt{dt}" if dt is not None else None,
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


def _compute_dt_eff(
    *,
    base_dt: float,
    gad_vec: torch.Tensor,
    dt_control: str,
    dt_min: float,
    dt_max: float,
    target_mean_disp: Optional[float],
    min_mean_disp: Optional[float],
    max_atom_disp: Optional[float],
) -> float:
    if dt_control == "fixed":
        dt_eff = float(base_dt)
    else:
        gad_mean = _mean_atom_norm(gad_vec)
        if not np.isfinite(gad_mean) or gad_mean <= 0:
            dt_eff = float(base_dt)
        else:
            dt_eff = float(base_dt)
            if dt_control == "target_mean_disp":
                if target_mean_disp is None:
                    raise ValueError("dt_control=target_mean_disp requires --target-mean-disp")
                dt_eff = float(target_mean_disp) / float(gad_mean)
            elif dt_control == "min_mean_disp":
                if min_mean_disp is None:
                    raise ValueError("dt_control=min_mean_disp requires --min-mean-disp")
                dt_eff = max(float(base_dt), float(min_mean_disp) / float(gad_mean))
            else:
                raise ValueError(f"Unknown dt_control: {dt_control}")

    dt_eff = float(np.clip(dt_eff, float(dt_min), float(dt_max)))

    if max_atom_disp is not None and max_atom_disp > 0:
        step = dt_eff * gad_vec
        max_disp = _max_atom_norm(step)
        if np.isfinite(max_disp) and max_disp > float(max_atom_disp) and max_disp > 0:
            dt_eff = dt_eff * (float(max_atom_disp) / float(max_disp))

    return float(dt_eff)


def _apply_max_atom_disp_cap(dt_eff: float, gad_vec: torch.Tensor, max_atom_disp: Optional[float]) -> float:
    if max_atom_disp is None or max_atom_disp <= 0:
        return float(dt_eff)
    step = float(dt_eff) * gad_vec
    max_disp = _max_atom_norm(step)
    if np.isfinite(max_disp) and max_disp > float(max_atom_disp) and max_disp > 0:
        return float(dt_eff) * (float(max_atom_disp) / float(max_disp))
    return float(dt_eff)


def run_single_euler(
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
    target_mean_disp: Optional[float],
    min_mean_disp: Optional[float],
    max_atom_disp: Optional[float],
    plateau_patience: int,
    plateau_boost: float,
    plateau_shrink: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        # experiment diagnostics
        "dt_eff",
        "gad_mean_norm",
        "step_mean_disp",
        "step_max_disp",
    ]}

    start_pos = coords.clone()
    prev_pos = coords.clone()

    steps_to_ts: Optional[int] = None

    dt_eff_stats: list[float] = []
    step_mean_stats: list[float] = []

    # Stateful controller variables (used by dt_control=neg_eig_plateau)
    dt_eff_state = float(dt)
    best_neg_vib: Optional[int] = None
    no_improve = 0

    for step in range(n_steps + 1):
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

        disp_from_last = float((coords - prev_pos).norm(dim=1).mean().item()) if step > 0 else 0.0
        disp_from_start = float((coords - start_pos).norm(dim=1).mean().item()) if step > 0 else 0.0

        trajectory["energy"].append(energy_value)
        trajectory["force_mean"].append(force_mean)
        trajectory["eig0"].append(eig0)
        trajectory["eig1"].append(eig1)
        trajectory["eig_product"].append(eig_prod)
        trajectory["neg_vib"].append(int(neg_vib))
        trajectory["disp_from_last"].append(disp_from_last)
        trajectory["disp_from_start"].append(disp_from_start)

        # stop condition: eigenproduct negative
        if stop_at_ts and steps_to_ts is None and np.isfinite(eig_prod) and eig_prod < -abs(ts_eps):
            steps_to_ts = step
            break

        if step == n_steps:
            break

        gad_vec = compute_gad_vector(forces, hessian)
        gad_mean_norm = _mean_atom_norm(gad_vec)

        if dt_control == "neg_eig_plateau":
            if best_neg_vib is None:
                best_neg_vib = int(neg_vib)
                no_improve = 0
            else:
                if int(neg_vib) < int(best_neg_vib):
                    best_neg_vib = int(neg_vib)
                    no_improve = 0
                    # After an improvement, be conservative again.
                    dt_eff_state = min(float(dt_eff_state), float(dt))
                elif int(neg_vib) > int(best_neg_vib):
                    # Got worse -> shrink dt aggressively.
                    dt_eff_state = max(float(dt_eff_state) * float(plateau_shrink), float(dt_min))
                    no_improve = 0
                else:
                    no_improve += 1

            if no_improve >= int(max(1, plateau_patience)):
                dt_eff_state = min(float(dt_eff_state) * float(plateau_boost), float(dt_max))
                no_improve = 0

            dt_eff = float(np.clip(dt_eff_state, float(dt_min), float(dt_max)))
            dt_eff = _apply_max_atom_disp_cap(dt_eff, gad_vec, max_atom_disp)
        else:
            dt_eff = _compute_dt_eff(
                base_dt=float(dt),
                gad_vec=gad_vec,
                dt_control=dt_control,
                dt_min=float(dt_min),
                dt_max=float(dt_max),
                target_mean_disp=target_mean_disp,
                min_mean_disp=min_mean_disp,
                max_atom_disp=max_atom_disp,
            )

        step_vec = dt_eff * gad_vec
        step_mean_disp = _mean_atom_norm(step_vec)
        step_max_disp = _max_atom_norm(step_vec)

        trajectory["dt_eff"].append(float(dt_eff))
        trajectory["gad_mean_norm"].append(float(gad_mean_norm))
        trajectory["step_mean_disp"].append(float(step_mean_disp))
        trajectory["step_max_disp"].append(float(step_max_disp))

        dt_eff_stats.append(float(dt_eff))
        step_mean_stats.append(float(step_mean_disp))

        prev_pos = coords.clone()
        coords = (coords + step_vec).detach()

    # Pad diagnostics to match length if we broke early at step 0 etc.
    # (plotter doesn’t use them; this keeps JSON tidy.)
    while len(trajectory["dt_eff"]) < len(trajectory["energy"]) - 1:
        trajectory["dt_eff"].append(float("nan"))
        trajectory["gad_mean_norm"].append(float("nan"))
        trajectory["step_mean_disp"].append(float("nan"))
        trajectory["step_max_disp"].append(float("nan"))

    # Final vibrational analysis
    final_neg_vib = -1
    if isinstance(out.get("hessian"), torch.Tensor):
        scine_elements = get_scine_elements_from_predict_output(out)
        final_vib_eigvals = vibrational_eigvals(out["hessian"], coords, atomic_nums, scine_elements=scine_elements)
        final_neg_vib = int((final_vib_eigvals < 0).sum().item())

    aux = {
        "steps_to_ts": steps_to_ts,
        "dt_eff_min": float(np.min(dt_eff_stats)) if dt_eff_stats else None,
        "dt_eff_median": float(np.median(dt_eff_stats)) if dt_eff_stats else None,
        "dt_eff_max": float(np.max(dt_eff_stats)) if dt_eff_stats else None,
        "step_mean_min": float(np.min(step_mean_stats)) if step_mean_stats else None,
        "step_mean_median": float(np.median(step_mean_stats)) if step_mean_stats else None,
        "step_mean_max": float(np.max(step_mean_stats)) if step_mean_stats else None,
    }

    final_out = {
        "final_coords": coords.detach().cpu(),
        "trajectory": trajectory,
        "steps_taken": len(trajectory["energy"]) - 1,
        "steps_to_ts": steps_to_ts,
        "final_eig0": trajectory["eig0"][-1] if trajectory["eig0"] else None,
        "final_eig1": trajectory["eig1"][-1] if trajectory["eig1"] else None,
        "final_eig_product": trajectory["eig_product"][-1] if trajectory["eig_product"] else None,
        "final_neg_vibrational": final_neg_vib,
    }

    return final_out, aux


def _save_trajectory_json(logger: ExperimentLogger, result: RunResult, trajectory: Dict[str, Any]) -> Optional[str]:
    transition_dir = logger.run_dir / result.transition_key
    transition_dir.mkdir(parents=True, exist_ok=True)
    path = transition_dir / f"trajectory_{result.sample_index:03d}.json"
    try:
        with open(path, "w") as f:
            json.dump(trajectory, f, indent=2)
        return str(path)
    except Exception:
        return None


def main(
    argv: Optional[list[str]] = None,
    *,
    default_calculator: Optional[str] = None,
    enforce_calculator: bool = False,
    script_name_prefix: str = "exp-gad-plateau",
) -> None:
    parser = argparse.ArgumentParser(
        description="Experiment: diagnose/fix Euler GAD plateaus on SCINE (dt control + RK45 comparison)."
    )
    parser = add_common_args(parser)

    if default_calculator is not None:
        parser.set_defaults(calculator=str(default_calculator))

    parser.add_argument("--method", type=str, default="euler", choices=["euler", "rk45"])

    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--start-from", type=str, default="midpoint_rt")
    parser.add_argument("--stop-at-ts", action="store_true")
    parser.add_argument("--ts-eps", type=float, default=1e-5)

    # Euler dt control knobs (opt-in)
    parser.add_argument(
        "--dt-control",
        type=str,
        default="fixed",
        choices=["fixed", "target_mean_disp", "min_mean_disp", "neg_eig_plateau"],
        help="How to choose dt each step based on |GAD| magnitude.",
    )
    parser.add_argument("--dt-min", type=float, default=1e-6)
    parser.add_argument("--dt-max", type=float, default=0.05)
    parser.add_argument(
        "--target-mean-disp",
        type=float,
        default=None,
        help="Target mean per-atom displacement (Å) for dt_control=target_mean_disp.",
    )
    parser.add_argument(
        "--min-mean-disp",
        type=float,
        default=None,
        help="Minimum mean per-atom displacement (Å) for dt_control=min_mean_disp.",
    )
    parser.add_argument(
        "--max-atom-disp",
        type=float,
        default=0.25,
        help="Safety cap: max per-atom displacement (Å) per step (applies after dt control).",
    )

    # Plateau controller (neg eigenvalue count based)
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=10,
        help="For dt_control=neg_eig_plateau: steps without neg-eig improvement before boosting dt.",
    )
    parser.add_argument(
        "--plateau-boost",
        type=float,
        default=1.5,
        help="For dt_control=neg_eig_plateau: multiplicative dt increase when plateau detected.",
    )
    parser.add_argument(
        "--plateau-shrink",
        type=float,
        default=0.5,
        help="For dt_control=neg_eig_plateau: multiplicative dt decrease if neg-eig count worsens.",
    )

    # RK45 knobs (only used if --method=rk45)
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument(
        "--min-eig-product-threshold",
        type=float,
        default=-1e-4,
        help="For reporting steps_to_ts when running RK45.",
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
            "target_mean_disp": args.target_mean_disp,
            "min_mean_disp": args.min_mean_disp,
            "max_atom_disp": args.max_atom_disp,
            "plateau_patience": args.plateau_patience,
            "plateau_boost": args.plateau_boost,
            "plateau_shrink": args.plateau_shrink,
            "t_end": args.t_end,
            "rtol": args.rtol,
            "atol": args.atol,
            "max_steps": args.max_steps,
        }
        wandb_name = args.wandb_name
        if not wandb_name:
            wandb_name = _auto_wandb_name(script=script_name, loss_type_flags=loss_type_flags, args=args)

        init_wandb_run(
            project=args.wandb_project,
            name=str(wandb_name),
            config=wandb_config,
            entity=args.wandb_entity,
            tags=[args.start_from, args.method, "plateau", str(args.dt_control)],
            run_dir=out_dir,
        )

    all_metrics = defaultdict(list)

    # Import lazily to avoid mixing runner modules unless needed
    from ..core_algos.gad import gad_rk45_integrate

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

        # initial vibrational order for transition bucketing
        try:
            init_out = predict_fn(start_coords, atomic_nums, do_hessian=True, require_grad=False)
            init_scine_elements = get_scine_elements_from_predict_output(init_out)
            init_vib = vibrational_eigvals(init_out["hessian"], start_coords, atomic_nums, scine_elements=init_scine_elements)
            initial_neg = int((init_vib < 0).sum().item())
        except Exception:
            initial_neg = -1

        t0 = time.time()
        if args.method == "rk45":
            rk = gad_rk45_integrate(
                predict_fn,
                start_coords,
                atomic_nums,
                t1=float(args.t_end),
                rtol=float(args.rtol),
                atol=float(args.atol),
                max_steps=int(args.max_steps),
            )
            coords_traj = rk.get("trajectory", [])
            if not coords_traj:
                coords_traj = [start_coords.detach().cpu()]
            coords_traj = [c.to(device=device) for c in coords_traj]

            # Recompute metrics for trajectory the same way as the runner does
            trajectory = {k: [] for k in [
                "energy",
                "force_mean",
                "eig0",
                "eig1",
                "eig_product",
                "disp_from_last",
                "disp_from_start",
            ]}
            start_pos = coords_traj[0].detach().clone()
            prev_pos = start_pos
            steps_to_ts: Optional[int] = None
            for step_idx, coords in enumerate(coords_traj):
                out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
                energy_value = _to_float(out.get("energy"))
                forces = out.get("forces")
                hessian = out.get("hessian")
                force_mean = _force_mean(forces)

                scine_elements = get_scine_elements_from_predict_output(out)
                vib = vibrational_eigvals(hessian, coords, atomic_nums, scine_elements=scine_elements)
                eig0 = float(vib[0].item()) if vib.numel() >= 1 else float("nan")
                eig1 = float(vib[1].item()) if vib.numel() >= 2 else float("nan")
                eig_prod = float((vib[0] * vib[1]).item()) if vib.numel() >= 2 else float("inf")

                disp_from_last = float((coords - prev_pos).norm(dim=1).mean().item()) if step_idx > 0 else 0.0
                disp_from_start = float((coords - start_pos).norm(dim=1).mean().item()) if step_idx > 0 else 0.0

                trajectory["energy"].append(energy_value)
                trajectory["force_mean"].append(force_mean)
                trajectory["eig0"].append(eig0)
                trajectory["eig1"].append(eig1)
                trajectory["eig_product"].append(eig_prod)
                trajectory["disp_from_last"].append(disp_from_last)
                trajectory["disp_from_start"].append(disp_from_start)

                if steps_to_ts is None and np.isfinite(eig_prod) and eig_prod < float(args.min_eig_product_threshold):
                    steps_to_ts = step_idx

                prev_pos = coords

            out_dict = {
                "final_coords": coords_traj[-1].detach().cpu(),
                "trajectory": trajectory,
                "steps_taken": len(coords_traj) - 1,
                "steps_to_ts": steps_to_ts,
                "final_eig0": trajectory["eig0"][-1] if trajectory["eig0"] else None,
                "final_eig1": trajectory["eig1"][-1] if trajectory["eig1"] else None,
                "final_eig_product": trajectory["eig_product"][-1] if trajectory["eig_product"] else None,
            }
            aux = {"steps_to_ts": steps_to_ts}
        else:
            out_dict, aux = run_single_euler(
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
                target_mean_disp=args.target_mean_disp,
                min_mean_disp=args.min_mean_disp,
                max_atom_disp=float(args.max_atom_disp) if args.max_atom_disp is not None else None,
                plateau_patience=int(args.plateau_patience),
                plateau_boost=float(args.plateau_boost),
                plateau_shrink=float(args.plateau_shrink),
            )

        wall = time.time() - t0

        final_neg = out_dict.get("final_neg_vibrational")
        if final_neg is None:
            # fallback if not computed in RK45 branch
            try:
                final_out = predict_fn(out_dict["final_coords"].to(device), atomic_nums, do_hessian=True, require_grad=False)
                final_scine_elements = get_scine_elements_from_predict_output(final_out)
                final_vib = vibrational_eigvals(final_out["hessian"], out_dict["final_coords"].to(device), atomic_nums, scine_elements=final_scine_elements)
                final_neg = int((final_vib < 0).sum().item())
            except Exception:
                final_neg = -1

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
                "dt_control": str(args.dt_control) if args.method == "euler" else None,
                "dt": float(args.dt) if args.method == "euler" else None,
                "dt_min": float(args.dt_min) if args.method == "euler" else None,
                "dt_max": float(args.dt_max) if args.method == "euler" else None,
                "target_mean_disp": float(args.target_mean_disp) if args.target_mean_disp is not None else None,
                "min_mean_disp": float(args.min_mean_disp) if args.min_mean_disp is not None else None,
                "max_atom_disp": float(args.max_atom_disp) if args.max_atom_disp is not None else None,
                "plateau_patience": int(args.plateau_patience) if args.method == "euler" and args.dt_control == "neg_eig_plateau" else None,
                "plateau_boost": float(args.plateau_boost) if args.method == "euler" and args.dt_control == "neg_eig_plateau" else None,
                "plateau_shrink": float(args.plateau_shrink) if args.method == "euler" and args.dt_control == "neg_eig_plateau" else None,
                "dt_eff_min": aux.get("dt_eff_min"),
                "dt_eff_median": aux.get("dt_eff_median"),
                "dt_eff_max": aux.get("dt_eff_max"),
                "step_mean_min": aux.get("step_mean_min"),
                "step_mean_median": aux.get("step_mean_median"),
                "step_mean_max": aux.get("step_mean_max"),
            },
        )

        logger.add_result(result)

        fig, filename = plot_gad_trajectory_3x2(
            out_dict["trajectory"],
            sample_index=i,
            formula=str(formula),
            start_from=args.start_from,
            initial_neg_num=initial_neg,
            final_neg_num=int(final_neg) if final_neg is not None else -1,
            steps_to_ts=aux.get("steps_to_ts"),
        )
        plot_path = logger.save_graph(result, fig, filename)
        if plot_path:
            result.plot_path = plot_path

        # Always write trajectory JSON (useful for dt_eff diagnostics)
        _save_trajectory_json(logger, result, out_dict["trajectory"])

        metrics = {
            "steps_taken": result.steps_taken,
            "steps_to_ts": result.steps_to_ts,
            "final_eig0": result.final_eig0,
            "final_eig1": result.final_eig1,
            "final_eig_product": result.final_eig_product,
            "final_neg_vibrational": result.final_neg_vibrational,
            "wallclock_s": result.final_time,
            "dt_eff_median": result.extra_data.get("dt_eff_median"),
            "step_mean_median": result.extra_data.get("step_mean_median"),
        }

        for k, v in metrics.items():
            if v is not None:
                all_metrics[k].append(v)

        if args.wandb:
            log_sample(i, metrics, fig=fig if plot_path else None, plot_name="trajectory")

        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass

    all_runs_path, aggregate_stats_path = logger.save_all_results()
    summary = logger.compute_aggregate_stats()
    logger.print_summary()

    if args.wandb:
        log_summary(summary)
        finish_wandb()

    print(f"Saved results: {all_runs_path}")
    print(f"Saved stats:   {aggregate_stats_path}")


if __name__ == "__main__":
    # Keep this module SCINE-only going forward to avoid mixing HIP/SCINE runs.
    main(default_calculator="scine", enforce_calculator=True, script_name_prefix="exp-scine-gad-plateau")
