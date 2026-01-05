"""Sella-based transition state search experiment.

This module implements the main experiment loop for Sella TS refinement,
following the same patterns as the multi-mode experiments but using Sella's
RS-P-RFO optimizer instead of GAD.

Key features:
- Uses Sella's internal coordinate optimization (robust for TS searches)
- RS-P-RFO trust radius management (prevents oscillation/divergence)
- Post-optimization eigenvalue validation using your existing Hessian pipeline
- W&B logging with trajectory artifacts
"""
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

from ...dependencies.common_utils import (
    add_common_args,
    parse_starting_geometry,
    setup_experiment,
)
from ...dependencies.experiment_logger import (
    ExperimentLogger,
    RunResult,
    build_loss_type_flags,
)
from ...dependencies.hessian import (
    vibrational_eigvals,
    get_scine_elements_from_predict_output,
)
from ...logging import finish_wandb, init_wandb_run, log_sample, log_summary
from ...runners._predict import make_predict_fn_from_calculator
from .sella_ts import run_sella_ts, validate_ts_eigenvalues


def _sanitize_wandb_name(s: str) -> str:
    """Sanitize string for W&B run names."""
    s = str(s)
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s[:128] if len(s) > 128 else s


def _auto_wandb_name(*, script: str, loss_type_flags: str, args: argparse.Namespace) -> str:
    """Generate automatic W&B run name from arguments."""
    calculator = getattr(args, "calculator", "hip")
    start_from = getattr(args, "start_from", "unknown")
    fmax = getattr(args, "fmax", None)
    max_steps = getattr(args, "max_steps", None)
    noise_seed = getattr(args, "noise_seed", None)
    job_id = os.environ.get("SLURM_JOB_ID")

    parts = [
        script,
        str(calculator),
        str(start_from),
        f"fmax{fmax}" if fmax is not None else None,
        f"steps{max_steps}" if max_steps is not None else None,
        f"seed{noise_seed}" if noise_seed is not None else None,
        f"job{job_id}" if job_id else None,
        str(loss_type_flags),
    ]
    parts = [p for p in parts if p]
    return _sanitize_wandb_name("__".join(parts))


def _save_trajectory_json(
    logger: ExperimentLogger,
    result: RunResult,
    trajectory: Dict[str, Any],
    aux: Dict[str, Any],
) -> Optional[str]:
    """Save trajectory data as JSON."""
    transition_dir = logger.run_dir / result.transition_key
    transition_dir.mkdir(parents=True, exist_ok=True)
    path = transition_dir / f"trajectory_{result.sample_index:03d}.json"
    try:
        data = {
            "trajectory": trajectory,
            "aux": {k: v for k, v in aux.items() if k != "trajectory_path"},
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
    script_name_prefix: str = "exp-sella",
) -> None:
    """Main entry point for Sella TS experiment.

    This follows the same pattern as your multi-mode experiments:
    1. Parse arguments
    2. Set up calculator and dataloader
    3. Loop over samples, running Sella optimization
    4. Validate final geometries with eigenvalue analysis
    5. Log results to W&B
    """
    parser = argparse.ArgumentParser(
        description="Experiment: Sella RS-P-RFO transition state refinement."
    )
    parser = add_common_args(parser)

    if default_calculator is not None:
        parser.set_defaults(calculator=str(default_calculator))

    # Sella-specific arguments
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.03,
        help="Force convergence threshold (eV/A). Default: 0.03",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum Sella optimization steps. Default: 200",
    )
    parser.add_argument(
        "--no-internal",
        action="store_true",
        help="Disable internal coordinates (not recommended for TS searches).",
    )
    parser.add_argument(
        "--delta0",
        type=float,
        default=0.1,
        help="Initial trust radius for Sella. Default: 0.1",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help="Saddle point order (1 for transition states). Default: 1",
    )

    # Starting geometry arguments
    parser.add_argument(
        "--start-from",
        type=str,
        default="midpoint_rt",
        help="Starting geometry: reactant, ts, midpoint_rt, three_quarter_rt, or *_noiseXA variants",
    )

    # W&B arguments
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gad-ts-search")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Optional W&B run name. If omitted, an informative name is auto-generated.",
    )

    # Logging
    parser.add_argument(
        "--sella-logfile",
        type=str,
        default=None,
        help="Path to Sella log file. Use '-' for stdout, None for no logging.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress messages.",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_false",
        dest="verbose",
        help="Disable progress messages.",
    )

    args = parser.parse_args(argv)

    # Enforce calculator if specified
    if enforce_calculator and default_calculator is not None:
        if str(getattr(args, "calculator", "")).lower() != str(default_calculator).lower():
            raise ValueError(
                f"This entrypoint enforces --calculator={default_calculator}. "
                f"Got --calculator={getattr(args, 'calculator', None)}."
            )

    # Set up experiment
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    calculator_type = getattr(args, "calculator", "hip").lower()
    if calculator_type == "scine":
        device = "cpu"  # SCINE only runs on CPU

    # Create predict function for eigenvalue validation
    predict_fn = make_predict_fn_from_calculator(calculator, calculator_type)

    # Set up logging
    loss_type_flags = build_loss_type_flags(args)
    script_name = f"{script_name_prefix}"
    logger = ExperimentLogger(
        base_dir=out_dir,
        script_name=script_name,
        loss_type_flags=loss_type_flags,
        max_graphs_per_transition=10,
        random_seed=42,
    )

    # Initialize W&B if requested
    if args.wandb:
        wandb_config = {
            "script": script_name,
            "calculator": calculator_type,
            "start_from": args.start_from,
            "fmax": args.fmax,
            "max_steps": args.max_steps,
            "internal": not args.no_internal,
            "delta0": args.delta0,
            "order": args.order,
            "noise_seed": args.noise_seed,
        }
        wandb_name = args.wandb_name
        if not wandb_name:
            wandb_name = _auto_wandb_name(
                script=script_name,
                loss_type_flags=loss_type_flags,
                args=args,
            )

        init_wandb_run(
            project=args.wandb_project,
            name=str(wandb_name),
            config=wandb_config,
            entity=args.wandb_entity,
            tags=[args.start_from, "sella", calculator_type],
            run_dir=out_dir,
        )

    # Trajectory directory
    traj_dir = os.path.join(out_dir, "sella_trajectories")

    all_metrics = defaultdict(list)

    # Main experiment loop
    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break

        batch = batch.to(device)
        atomic_nums = batch.z.detach().cpu().to(device)
        formula = getattr(batch, "formula", "sample")

        # Get starting geometry
        start_coords = parse_starting_geometry(
            args.start_from,
            batch,
            noise_seed=getattr(args, "noise_seed", None),
            sample_index=i,
        ).detach().to(device)

        # Initial eigenvalue analysis for transition bucketing
        try:
            init_out = predict_fn(start_coords, atomic_nums, do_hessian=True, require_grad=False)
            init_scine_elements = get_scine_elements_from_predict_output(init_out)
            init_vib = vibrational_eigvals(
                init_out["hessian"], start_coords, atomic_nums, scine_elements=init_scine_elements
            )
            initial_neg = int((init_vib < 0).sum().item())
        except Exception as e:
            if args.verbose:
                print(f"[WARN] Sample {i}: Initial eigenvalue analysis failed: {e}")
            initial_neg = -1

        t0 = time.time()

        # Run Sella optimization
        try:
            out_dict, aux = run_sella_ts(
                calculator,
                calculator_type,
                start_coords,
                atomic_nums,
                fmax=float(args.fmax),
                max_steps=int(args.max_steps),
                internal=not args.no_internal,
                delta0=float(args.delta0),
                order=int(args.order),
                device=device,
                save_trajectory=True,
                trajectory_dir=traj_dir,
                sample_index=i,
                logfile=args.sella_logfile,
                verbose=args.verbose,
            )
            wall_time = time.time() - t0

        except Exception as e:
            wall_time = time.time() - t0
            stop_reason = f"{type(e).__name__}: {e}"
            if args.verbose:
                print(f"[WARN] Sample {i} failed during Sella run: {stop_reason}")

            result = RunResult(
                sample_index=i,
                formula=str(formula),
                initial_neg_eigvals=int(initial_neg),
                final_neg_eigvals=-1,
                initial_neg_vibrational=None,
                final_neg_vibrational=None,
                steps_taken=0,
                steps_to_ts=None,
                final_time=float(wall_time),
                final_eig0=None,
                final_eig1=None,
                final_eig_product=None,
                final_loss=None,
                rmsd_to_known_ts=None,
                stop_reason=stop_reason,
                plot_path=None,
                extra_data={
                    "calculator": calculator_type,
                    "start_from": args.start_from,
                    "fmax": args.fmax,
                    "max_steps": args.max_steps,
                },
            )

            logger.add_result(result)

            if args.wandb:
                log_sample(
                    i,
                    {
                        "steps_taken": 0,
                        "wallclock_s": wall_time,
                        "stop_reason": stop_reason,
                        "failed": 1,
                    },
                )
            continue

        # Post-optimization eigenvalue validation
        final_coords = out_dict["final_coords"].to(device)
        try:
            final_out = predict_fn(final_coords, atomic_nums, do_hessian=True, require_grad=False)
            final_scine_elements = get_scine_elements_from_predict_output(final_out)
            final_vib = vibrational_eigvals(
                final_out["hessian"], final_coords, atomic_nums, scine_elements=final_scine_elements
            )
            final_neg = int((final_vib < 0).sum().item())
            final_eig0 = float(final_vib[0].item()) if final_vib.numel() >= 1 else None
            final_eig1 = float(final_vib[1].item()) if final_vib.numel() >= 2 else None
            final_eig_product = float((final_vib[0] * final_vib[1]).item()) if final_vib.numel() >= 2 else None
        except Exception as e:
            if args.verbose:
                print(f"[WARN] Sample {i}: Final eigenvalue analysis failed: {e}")
            final_neg = -1
            final_eig0 = None
            final_eig1 = None
            final_eig_product = None

        # Determine if we found a TS
        is_ts = final_neg == 1
        steps_to_ts = out_dict["steps_taken"] if is_ts else None

        result = RunResult(
            sample_index=i,
            formula=str(formula),
            initial_neg_eigvals=initial_neg,
            final_neg_eigvals=int(final_neg) if final_neg is not None else -1,
            initial_neg_vibrational=None,
            final_neg_vibrational=int(final_neg) if final_neg is not None else None,
            steps_taken=int(out_dict["steps_taken"]),
            steps_to_ts=steps_to_ts,
            final_time=float(wall_time),
            final_eig0=final_eig0,
            final_eig1=final_eig1,
            final_eig_product=final_eig_product,
            final_loss=None,
            rmsd_to_known_ts=None,
            stop_reason=None,
            plot_path=aux.get("trajectory_path"),
            extra_data={
                "calculator": calculator_type,
                "start_from": args.start_from,
                "fmax": args.fmax,
                "max_steps": args.max_steps,
                "converged": out_dict.get("converged", False),
                "final_energy": out_dict.get("final_energy"),
                "final_fmax": out_dict.get("final_fmax"),
                "final_force_mean": out_dict.get("final_force_mean"),
            },
        )

        logger.add_result(result)

        # Save trajectory JSON
        _save_trajectory_json(logger, result, out_dict.get("trajectory", {}), aux)

        # Collect metrics for summary
        metrics = {
            "steps_taken": result.steps_taken,
            "steps_to_ts": result.steps_to_ts,
            "final_eig0": result.final_eig0,
            "final_eig1": result.final_eig1,
            "final_eig_product": result.final_eig_product,
            "final_neg_vibrational": result.final_neg_vibrational,
            "wallclock_s": result.final_time,
            "converged": out_dict.get("converged", False),
            "is_ts": is_ts,
            "final_energy": out_dict.get("final_energy"),
            "final_fmax": out_dict.get("final_fmax"),
        }

        for k, v in metrics.items():
            if v is not None:
                all_metrics[k].append(v)

        if args.wandb:
            log_sample(i, metrics)

        if args.verbose:
            ts_status = "TS found!" if is_ts else f"neg_vib={final_neg}"
            print(f"[{i}] {formula}: {result.steps_taken} steps, {ts_status}, "
                  f"eig_product={final_eig_product:.2e if final_eig_product else 'N/A'}")

    # Save results and compute summary
    all_runs_path, aggregate_stats_path = logger.save_all_results()
    summary = logger.compute_aggregate_stats()
    logger.print_summary()

    # Compute Sella-specific summary stats
    sella_summary = {}
    if all_metrics.get("converged"):
        sella_summary["convergence_rate"] = sum(all_metrics["converged"]) / len(all_metrics["converged"])
    if all_metrics.get("is_ts"):
        sella_summary["ts_rate"] = sum(all_metrics["is_ts"]) / len(all_metrics["is_ts"])
    if all_metrics.get("final_fmax"):
        sella_summary["avg_final_fmax"] = float(np.mean(all_metrics["final_fmax"]))

    # Print Sella-specific summary
    print("\n" + "=" * 80)
    print("SELLA SUMMARY")
    print("=" * 80)
    print(f"Force convergence rate: {sella_summary.get('convergence_rate', 0) * 100:.1f}%")
    print(f"TS success rate (neg_vib=1): {sella_summary.get('ts_rate', 0) * 100:.1f}%")
    print(f"Average final fmax: {sella_summary.get('avg_final_fmax', 0):.4f} eV/A")
    print("=" * 80)

    if args.wandb:
        summary.update(sella_summary)
        log_summary(summary)
        finish_wandb()

    print(f"Saved results: {all_runs_path}")
    print(f"Saved stats:   {aggregate_stats_path}")


if __name__ == "__main__":
    main()
