from __future__ import annotations

import argparse
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..dependencies.common_utils import add_common_args, parse_starting_geometry, setup_experiment
from ..dependencies.experiment_logger import ExperimentLogger, RunResult, build_loss_type_flags
from ..logging import (
    finish_wandb,
    init_wandb_run,
    log_sample,
    log_summary,
)
from ..logging.plotly_utils import plot_gad_trajectory_interactive
from src.core_algos.gad import gad_euler_step
from ..dependencies.hessian import vibrational_eigvals, get_scine_elements_from_predict_output
from ._predict import make_predict_fn_from_calculator


def _sanitize_wandb_name(s: str) -> str:
    s = str(s)
    s = s.strip().replace(" ", "_")
    # Keep only a conservative set of chars to avoid W&B/UI issues
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s[:128] if len(s) > 128 else s


def _auto_wandb_name(*, script: str, loss_type_flags: str, args: argparse.Namespace) -> str:
    calculator = getattr(args, "calculator", "hip")
    start_from = getattr(args, "start_from", "unknown")
    dt = getattr(args, "dt", None)
    n_steps = getattr(args, "n_steps", None)
    noise_seed = getattr(args, "noise_seed", None)
    job_id = os.environ.get("SLURM_JOB_ID")

    parts = [
        script,
        str(calculator),
        str(start_from),
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


def run_single(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    n_steps: int,
    dt: float,
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
        "disp_from_last",
        "disp_from_start",
    ]}

    start_pos = coords.clone()
    prev_pos = coords.clone()

    steps_to_ts: Optional[int] = None

    for step in range(n_steps + 1):
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        energy = out.get("energy")
        forces = out.get("forces")
        hessian = out.get("hessian")

        energy_value = _to_float(energy)
        force_mean = _force_mean(forces)

        # Extract SCINE elements if using SCINE calculator
        scine_elements = get_scine_elements_from_predict_output(out)
        vib = vibrational_eigvals(hessian, coords, atomic_nums, scine_elements=scine_elements)
        eig0 = float(vib[0].item()) if vib.numel() >= 1 else float("nan")
        eig1 = float(vib[1].item()) if vib.numel() >= 2 else float("nan")
        eig_prod = float((vib[0] * vib[1]).item()) if vib.numel() >= 2 else float("inf")

        disp_from_last = float((coords - prev_pos).norm(dim=1).mean().item()) if step > 0 else 0.0
        disp_from_start = float((coords - start_pos).norm(dim=1).mean().item()) if step > 0 else 0.0

        trajectory["energy"].append(energy_value)
        trajectory["force_mean"].append(force_mean)
        trajectory["eig0"].append(eig0)
        trajectory["eig1"].append(eig1)
        trajectory["eig_product"].append(eig_prod)
        trajectory["disp_from_last"].append(disp_from_last)
        trajectory["disp_from_start"].append(disp_from_start)

        # stop condition: eigenproduct negative
        if stop_at_ts and steps_to_ts is None and np.isfinite(eig_prod) and eig_prod < -abs(ts_eps):
            steps_to_ts = step
            break

        if step == n_steps:
            break

        prev_pos = coords.clone()
        step_out = gad_euler_step(predict_fn, coords, atomic_nums, dt=dt, out=out)
        coords = step_out["new_coords"].detach()

    # Get final vibrational analysis (use SCINE elements if applicable)
    final_neg_vib = -1
    if isinstance(out.get("hessian"), torch.Tensor):
        scine_elements = get_scine_elements_from_predict_output(out)
        final_vib_eigvals = vibrational_eigvals(out["hessian"], coords, atomic_nums, scine_elements=scine_elements)
        final_neg_vib = int((final_vib_eigvals < 0).sum().item())

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

    aux = {
        "steps_to_ts": steps_to_ts,
    }

    return final_out, aux


def main() -> None:
    parser = argparse.ArgumentParser(description="Core GAD Euler runner (refactored entrypoint).")
    parser = add_common_args(parser)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--dt", type=float, default=0.001)
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
        script_name="gad-euler-core",
        loss_type_flags=loss_type_flags,
        max_graphs_per_transition=10,
        random_seed=42,
    )

    if args.wandb:
        wandb_config = {
            "script": "gad_euler_core",
            "start_from": args.start_from,
            "n_steps": args.n_steps,
            "dt": args.dt,
            "stop_at_ts": bool(args.stop_at_ts),
            "calculator": getattr(args, "calculator", "hip"),
        }

        wandb_name = args.wandb_name
        if not wandb_name:
            wandb_name = _auto_wandb_name(script="gad-euler-core", loss_type_flags=loss_type_flags, args=args)

        init_wandb_run(
            project=args.wandb_project,
            name=str(wandb_name),
            config=wandb_config,
            entity=args.wandb_entity,
            tags=[args.start_from, "euler", "core"],
            run_dir=out_dir,
        )

    all_metrics = defaultdict(list)

    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break

        batch = batch.to(device)
        atomic_nums = batch.z.detach().cpu()
        formula = getattr(batch, "formula", "sample")

        start_coords = parse_starting_geometry(args.start_from, batch, noise_seed=getattr(args, "noise_seed", None), sample_index=i)
        start_coords = start_coords.detach().to(device)

        # Compute initial saddle order (vibrational) for proper transition bucketing.
        try:
            init_out = predict_fn(start_coords, atomic_nums.to(device), do_hessian=True, require_grad=False)
            init_scine_elements = get_scine_elements_from_predict_output(init_out)
            init_vib = vibrational_eigvals(
                init_out["hessian"],
                start_coords,
                atomic_nums,
                scine_elements=init_scine_elements,
            )
            initial_neg = int((init_vib < 0).sum().item())
        except Exception:
            initial_neg = -1

        t0 = time.time()
        try:
            out, aux = run_single(
                predict_fn,
                start_coords,
                atomic_nums.to(device),
                n_steps=args.n_steps,
                dt=args.dt,
                stop_at_ts=bool(args.stop_at_ts),
                ts_eps=float(args.ts_eps),
            )
            wall = time.time() - t0
        except Exception as e:
            wall = time.time() - t0
            stop_reason = f"{type(e).__name__}: {e}"
            print(f"[WARN] Sample {i} failed during run: {stop_reason}")

            result = RunResult(
                sample_index=i,
                formula=str(formula),
                initial_neg_eigvals=int(initial_neg),
                final_neg_eigvals=-1,
                initial_neg_vibrational=None,
                final_neg_vibrational=None,
                steps_taken=0,
                steps_to_ts=None,
                final_time=float(wall),
                final_eig0=None,
                final_eig1=None,
                final_eig_product=None,
                final_loss=None,
                rmsd_to_known_ts=None,
                stop_reason=stop_reason,
                plot_path=None,
            )

            logger.add_result(result)

            if args.wandb:
                log_sample(
                    i,
                    {
                        "steps_taken": result.steps_taken,
                        "wallclock_s": result.final_time,
                        "stop_reason": result.stop_reason,
                        "failed": 1,
                    },
                )
            continue

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
        html_path = Path(out_dir) / f"traj_{i:03d}.html"
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
