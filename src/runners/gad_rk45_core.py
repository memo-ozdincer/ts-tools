from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..dependencies.common_utils import add_common_args, parse_starting_geometry, setup_experiment
from ..core_algos.gad import gad_rk45_integrate
from ..dependencies.hessian import vibrational_eigvals, get_scine_elements_from_predict_output
from ..dependencies.experiment_logger import ExperimentLogger, RunResult, build_loss_type_flags
from ..logging import finish_wandb, init_wandb_run, log_sample, log_summary
from ..logging.trajectory_plots import plot_gad_trajectory_3x2
from ._predict import make_predict_fn_from_calculator


def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().reshape(-1)[0].item())
    return float(x)


def _force_mean(forces: torch.Tensor) -> float:
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    f = forces.reshape(-1, 3)
    return float(f.norm(dim=1).mean().item())


def _trajectory_metrics(
    predict_fn,
    coords_list: list[torch.Tensor],
    atomic_nums: torch.Tensor,
    *,
    min_eig_product_threshold: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    trajectory = {k: [] for k in [
        "energy",
        "force_mean",
        "eig0",
        "eig1",
        "eig_product",
        "disp_from_last",
        "disp_from_start",
    ]}

    start_pos = coords_list[0].detach().clone()
    prev_pos = start_pos

    steps_to_ts: Optional[int] = None

    for step, coords in enumerate(coords_list):
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        energy_value = _to_float(out.get("energy"))
        forces = out.get("forces")
        hessian = out.get("hessian")

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

        if steps_to_ts is None and np.isfinite(eig_prod) and eig_prod < float(min_eig_product_threshold):
            steps_to_ts = step

        prev_pos = coords

    # Get final vibrational analysis (use SCINE elements if applicable)
    final_coords = coords_list[-1].detach().cpu()
    final_out_data = predict_fn(coords_list[-1], atomic_nums, do_hessian=True, require_grad=False)
    final_scine_elements = get_scine_elements_from_predict_output(final_out_data)
    final_vib_eigvals = vibrational_eigvals(final_out_data["hessian"], coords_list[-1], atomic_nums, scine_elements=final_scine_elements)
    final_neg_vibrational = int((final_vib_eigvals < 0).sum().item())

    final_out = {
        "final_coords": final_coords,
        "trajectory": trajectory,
        "steps_taken": len(coords_list) - 1,
        "steps_to_ts": steps_to_ts,
        "final_eig0": trajectory["eig0"][-1] if trajectory["eig0"] else None,
        "final_eig1": trajectory["eig1"][-1] if trajectory["eig1"] else None,
        "final_eig_product": trajectory["eig_product"][-1] if trajectory["eig_product"] else None,
        "final_neg_vibrational": final_neg_vibrational,
    }

    aux = {
        "steps_to_ts": steps_to_ts,
    }

    return final_out, aux


def main() -> None:
    parser = argparse.ArgumentParser(description="Core GAD RK45 runner (refactored entrypoint; no hybrid mode).")
    parser = add_common_args(parser)

    parser.add_argument("--start-from", type=str, default="midpoint_rt")
    parser.add_argument("--stop-at-ts", action="store_true")

    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument(
        "--min-eig-product-threshold",
        type=float,
        default=-1e-4,
        help="Used only for reporting steps_to_ts (first step where λ0*λ1 < threshold).",
    )

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gad-ts-search")
    parser.add_argument("--wandb-entity", type=str, default=None)

    args = parser.parse_args()

    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    calculator_type = getattr(args, "calculator", "hip").lower()
    if calculator_type == "scine":
        device = "cpu"
    predict_fn = make_predict_fn_from_calculator(calculator, calculator_type)

    loss_type_flags = build_loss_type_flags(args)
    logger = ExperimentLogger(
        base_dir=out_dir,
        script_name="gad-rk45-core",
        loss_type_flags=loss_type_flags,
        max_graphs_per_transition=10,
        random_seed=42,
    )

    if args.wandb:
        wandb_config = {
            "script": "gad_rk45_core",
            "start_from": args.start_from,
            "t_end": args.t_end,
            "rtol": args.rtol,
            "atol": args.atol,
            "max_steps": args.max_steps,
            "min_eig_product_threshold": args.min_eig_product_threshold,
            "calculator": getattr(args, "calculator", "hip"),
        }
        init_wandb_run(
            project=args.wandb_project,
            name=f"gad-rk45-core_{loss_type_flags}",
            config=wandb_config,
            entity=args.wandb_entity,
            tags=[args.start_from, "rk45", "core"],
            run_dir=out_dir,
        )

    all_metrics = defaultdict(list)

    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break

        batch = batch.to(device)
        atomic_nums = batch.z.detach().cpu().to(device)
        formula = getattr(batch, "formula", "sample")

        start_coords = parse_starting_geometry(args.start_from, batch, noise_seed=getattr(args, "noise_seed", None), sample_index=i)
        start_coords = start_coords.detach().to(device)

        # initial vibrational order for transition bucketing
        try:
            init_out = predict_fn(start_coords, atomic_nums, do_hessian=True, require_grad=False)
            init_scine_elements = get_scine_elements_from_predict_output(init_out)
            init_vib = vibrational_eigvals(init_out["hessian"], start_coords, atomic_nums, scine_elements=init_scine_elements)
            initial_neg = int((init_vib < 0).sum().item())
        except Exception:
            initial_neg = -1

        t0 = time.time()
        rk = gad_rk45_integrate(
            predict_fn,
            start_coords,
            atomic_nums,
            t1=float(args.t_end),
            rtol=float(args.rtol),
            atol=float(args.atol),
            max_steps=int(args.max_steps),
        )
        wall = time.time() - t0

        coords_traj = rk.get("trajectory", [])
        if not coords_traj:
            coords_traj = [start_coords.detach().cpu()]

        coords_traj = [c.to(device=device) for c in coords_traj]

        out, aux = _trajectory_metrics(
            predict_fn,
            coords_traj,
            atomic_nums,
            min_eig_product_threshold=float(args.min_eig_product_threshold),
        )

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

        fig, filename = plot_gad_trajectory_3x2(
            out["trajectory"],
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

        metrics = {
            "steps_taken": result.steps_taken,
            "steps_to_ts": result.steps_to_ts,
            "final_eig0": result.final_eig0,
            "final_eig1": result.final_eig1,
            "final_eig_product": result.final_eig_product,
            "final_neg_vibrational": result.final_neg_vibrational,
            "wallclock_s": result.final_time,
            "rk45_steps": int(rk.get("steps", -1)),
            "rk45_t_final": float(rk.get("t_final", float("nan"))),
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
    main()
