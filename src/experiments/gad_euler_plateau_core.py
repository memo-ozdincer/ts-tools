from __future__ import annotations

import argparse
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..core_algos.gad import gad_euler_step
from ..dependencies.common_utils import add_common_args, parse_starting_geometry, setup_experiment
from ..dependencies.experiment_logger import ExperimentLogger, RunResult, build_loss_type_flags
from ..dependencies.hessian import get_scine_elements_from_predict_output, vibrational_eigvals
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
    dt = getattr(args, "dt", None)
    n_steps = getattr(args, "n_steps", None)
    patience = getattr(args, "plateau_patience", None)
    boost = getattr(args, "plateau_boost", None)
    job_id = os.environ.get("SLURM_JOB_ID")

    parts = [
        script,
        str(calculator),
        str(start_from),
        f"dt{dt}" if dt is not None else None,
        f"steps{n_steps}" if n_steps is not None else None,
        f"pat{patience}" if patience is not None else None,
        f"boost{boost}" if boost is not None else None,
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


def run_single_plateau(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    n_steps: int,
    dt: float,
    dt_min: float,
    dt_max: float,
    plateau_patience: int,
    plateau_boost: float,
    plateau_shrink: float,
    stop_at_ts: bool,
    ts_eps: float,
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
        "dt_eff",
    ]}

    start_pos = coords.clone()
    prev_pos = coords.clone()

    best_neg_vib: Optional[int] = None
    no_improve = 0
    dt_eff_state = float(dt)
    steps_to_ts: Optional[int] = None

    for step in range(int(n_steps) + 1):
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        energy_value = _to_float(out.get("energy"))
        force_mean = _force_mean(out.get("forces"))

        scine_elements = get_scine_elements_from_predict_output(out)
        vib = vibrational_eigvals(out["hessian"], coords, atomic_nums, scine_elements=scine_elements)
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

        if stop_at_ts and steps_to_ts is None and np.isfinite(eig_prod) and eig_prod < -abs(float(ts_eps)):
            steps_to_ts = step
            break

        if step == int(n_steps):
            break

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
        trajectory["dt_eff"].append(dt_eff)

        # Use shared Euler step implementation, but reuse the already-computed `out`.
        step_out = gad_euler_step(predict_fn, coords, atomic_nums, dt=dt_eff, out=out)

        prev_pos = coords.clone()
        coords = step_out["new_coords"].detach()

    while len(trajectory["dt_eff"]) < len(trajectory["energy"]) - 1:
        trajectory["dt_eff"].append(float("nan"))

    final_neg_vib = -1
    try:
        final_out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        scine_elements = get_scine_elements_from_predict_output(final_out)
        final_vib = vibrational_eigvals(final_out["hessian"], coords, atomic_nums, scine_elements=scine_elements)
        final_neg_vib = int((final_vib < 0).sum().item())
    except Exception:
        pass

    final_out_dict = {
        "final_coords": coords.detach().cpu(),
        "trajectory": trajectory,
        "steps_taken": len(trajectory["energy"]) - 1,
        "steps_to_ts": steps_to_ts,
        "final_eig0": trajectory["eig0"][-1] if trajectory["eig0"] else None,
        "final_eig1": trajectory["eig1"][-1] if trajectory["eig1"] else None,
        "final_eig_product": trajectory["eig_product"][-1] if trajectory["eig_product"] else None,
        "final_neg_vibrational": final_neg_vib,
    }

    aux = {
        "steps_to_ts": steps_to_ts,
    }

    return final_out_dict, aux


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment: GAD Euler with neg-eig plateau dt boosting (runner-faithful).")
    parser = add_common_args(parser)

    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--dt-min", type=float, default=1e-6)
    parser.add_argument("--dt-max", type=float, default=0.05)

    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--plateau-boost", type=float, default=1.5)
    parser.add_argument("--plateau-shrink", type=float, default=0.5)

    parser.add_argument("--start-from", type=str, default="midpoint_rt")
    parser.add_argument("--stop-at-ts", action="store_true")
    parser.add_argument("--ts-eps", type=float, default=1e-5, help="Stop when eig_product < -eps")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gad-ts-search")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Optional W&B run name. If omitted, an informative name is auto-generated.",
    )

    args = parser.parse_args()

    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    calculator_type = getattr(args, "calculator", "hip").lower()
    if calculator_type == "scine":
        device = "cpu"
    predict_fn = make_predict_fn_from_calculator(calculator, calculator_type)

    loss_type_flags = build_loss_type_flags(args)
    logger = ExperimentLogger(
        base_dir=out_dir,
        script_name="exp-gad-euler-plateau-core",
        loss_type_flags=loss_type_flags,
        max_graphs_per_transition=10,
        random_seed=42,
    )

    if args.wandb:
        wandb_config = {
            "script": "exp-gad-euler-plateau-core",
            "start_from": args.start_from,
            "n_steps": args.n_steps,
            "dt": args.dt,
            "dt_min": args.dt_min,
            "dt_max": args.dt_max,
            "plateau_patience": args.plateau_patience,
            "plateau_boost": args.plateau_boost,
            "plateau_shrink": args.plateau_shrink,
            "stop_at_ts": bool(args.stop_at_ts),
            "calculator": getattr(args, "calculator", "hip"),
        }

        wandb_name = args.wandb_name
        if not wandb_name:
            wandb_name = _auto_wandb_name(script="exp-gad-euler-plateau-core", loss_type_flags=loss_type_flags, args=args)

        init_wandb_run(
            project=args.wandb_project,
            name=str(wandb_name),
            config=wandb_config,
            entity=args.wandb_entity,
            tags=[args.start_from, "euler", "plateau"],
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

        # initial vibrational order for transition bucketing
        try:
            init_out = predict_fn(start_coords, atomic_nums, do_hessian=True, require_grad=False)
            init_scine_elements = get_scine_elements_from_predict_output(init_out)
            init_vib = vibrational_eigvals(init_out["hessian"], start_coords, atomic_nums, scine_elements=init_scine_elements)
            initial_neg = int((init_vib < 0).sum().item())
        except Exception:
            initial_neg = -1

        t0 = time.time()
        out, aux = run_single_plateau(
            predict_fn,
            start_coords,
            atomic_nums,
            n_steps=int(args.n_steps),
            dt=float(args.dt),
            dt_min=float(args.dt_min),
            dt_max=float(args.dt_max),
            plateau_patience=int(args.plateau_patience),
            plateau_boost=float(args.plateau_boost),
            plateau_shrink=float(args.plateau_shrink),
            stop_at_ts=bool(args.stop_at_ts),
            ts_eps=float(args.ts_eps),
        )
        wall = time.time() - t0

        final_neg = out.get("final_neg_vibrational", -1)

        result = RunResult(
            sample_index=i,
            formula=str(formula),
            initial_neg_eigvals=initial_neg,
            final_neg_eigvals=int(final_neg) if final_neg is not None else -1,
            initial_neg_vibrational=None,
            final_neg_vibrational=int(final_neg) if final_neg is not None else None,
            steps_taken=int(out["steps_taken"]),
            steps_to_ts=aux.get("steps_to_ts"),
            final_time=float(wall),
            final_eig0=out.get("final_eig0"),
            final_eig1=out.get("final_eig1"),
            final_eig_product=out.get("final_eig_product"),
            final_loss=None,
            rmsd_to_known_ts=None,
            stop_reason=None,
            plot_path=None,
            extra_data={
                "dt": float(args.dt),
                "dt_min": float(args.dt_min),
                "dt_max": float(args.dt_max),
                "plateau_patience": int(args.plateau_patience),
                "plateau_boost": float(args.plateau_boost),
                "plateau_shrink": float(args.plateau_shrink),
            },
        )

        logger.add_result(result)

        # 1. Generate the interactive figure
        fig_interactive = plot_gad_trajectory_interactive(
            out["trajectory"],
            sample_index=i,
            formula=str(formula),
            start_from=args.start_from,
            initial_neg_num=initial_neg,
            final_neg_num=int(final_neg) if final_neg is not None else -1,
            steps_to_ts=aux.get("steps_to_ts"),
        )

        # 2. Save result logic (optional: Plotly can save to HTML if you want local copies)
        # To save Plotly locally as HTML:
        html_path = out_dir / f"traj_{i:03d}.html"
        fig_interactive.write_html(str(html_path))
        result.plot_path = str(html_path)

        metrics = {
            "steps_taken": result.steps_taken,
            "steps_to_ts": result.steps_to_ts,
            "final_eig0": result.final_eig0,
            "final_eig1": result.final_eig1,
            "final_eig_product": result.final_eig_product,
            "final_neg_vibrational": result.final_neg_vibrational,
            "wallclock_s": result.final_time,
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
    main()
