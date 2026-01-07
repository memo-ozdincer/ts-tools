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
from typing import Any, Dict, List, Optional, Tuple

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
from ...logging.plotly_utils import plot_sella_trajectory_interactive
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
        # Filter out positions (numpy arrays) for JSON serialization
        traj_for_json = {k: v for k, v in trajectory.items() if k != "positions"}
        data = {
            "trajectory": traj_for_json,
            "aux": {k: v for k, v in aux.items() if k != "trajectory_path"},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return str(path)
    except Exception:
        return None


def _to_json_serializable(value: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _compute_eigenvalues_at_intervals(
    predict_fn,
    trajectory: Dict[str, Any],
    atomic_nums: torch.Tensor,
    device: str,
    eig_interval: int,
    verbose: bool = False,
) -> Dict[str, List[Optional[float]]]:
    """Compute eigenvalues at specified intervals from trajectory positions.

    Args:
        predict_fn: Prediction function that returns {energy, forces, hessian}
        trajectory: Trajectory dict containing 'positions' list
        atomic_nums: Atomic numbers tensor
        device: Device for computation
        eig_interval: Interval for eigenvalue computation:
            - positive int (1,2,5,10,...): compute every N steps
            - 0 or -1: only compute at first and last step
        verbose: Print progress

    Returns:
        Dictionary with eigenvalue data at each step (None for skipped steps):
        - eig0: Lowest eigenvalue
        - eig1: Second eigenvalue
        - eig_product: Product of eig0 * eig1
        - neg_vib: Number of negative eigenvalues
    """
    positions_list = trajectory.get("positions", [])
    n_steps = len(positions_list)

    eig_data: Dict[str, List[Optional[float]]] = {
        "eig0": [None] * n_steps,
        "eig1": [None] * n_steps,
        "eig_product": [None] * n_steps,
        "neg_vib": [None] * n_steps,
    }

    if n_steps == 0:
        return eig_data

    # Determine which steps to compute eigenvalues for
    if eig_interval <= 0:
        # Only first and last
        steps_to_compute = [0]
        if n_steps > 1:
            steps_to_compute.append(n_steps - 1)
    else:
        # Every eig_interval steps, always including first and last
        steps_to_compute = list(range(0, n_steps, eig_interval))
        if (n_steps - 1) not in steps_to_compute:
            steps_to_compute.append(n_steps - 1)

    if verbose:
        print(f"[Eigenvalues] Computing for {len(steps_to_compute)}/{n_steps} steps (interval={eig_interval})")

    for step_idx in steps_to_compute:
        try:
            pos = positions_list[step_idx]
            coords = torch.tensor(pos, dtype=torch.float32, device=device).reshape(-1, 3)

            out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
            scine_elements = get_scine_elements_from_predict_output(out)
            vib = vibrational_eigvals(
                out["hessian"], coords, atomic_nums, scine_elements=scine_elements
            )

            eig_data["eig0"][step_idx] = float(vib[0].item()) if vib.numel() >= 1 else None
            eig_data["eig1"][step_idx] = float(vib[1].item()) if vib.numel() >= 2 else None
            eig_data["eig_product"][step_idx] = float((vib[0] * vib[1]).item()) if vib.numel() >= 2 else None
            eig_data["neg_vib"][step_idx] = int((vib < 0).sum().item())

        except Exception as e:
            if verbose:
                print(f"[WARN] Eigenvalue computation failed at step {step_idx}: {e}")

    return eig_data


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

    # Eigenvalue tracking
    parser.add_argument(
        "--eig-interval",
        type=int,
        default=0,
        help="Interval for computing eigenvalues during optimization. "
             "Positive int (1,2,5,10,...): compute every N steps. "
             "0 or -1: only compute at first and last step (default: 0).",
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
            "eig_interval": args.eig_interval,
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

            # Compute eigenvalues at intervals from trajectory
            trajectory = out_dict.get("trajectory", {})
            if trajectory.get("positions"):
                eig_data = _compute_eigenvalues_at_intervals(
                    predict_fn,
                    trajectory,
                    atomic_nums,
                    device,
                    eig_interval=args.eig_interval,
                    verbose=args.verbose,
                )
                # Merge eigenvalue data into trajectory
                trajectory.update(eig_data)
                out_dict["trajectory"] = trajectory

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
                "converged": _to_json_serializable(out_dict.get("converged", False)),
                "final_energy": _to_json_serializable(out_dict.get("final_energy")),
                "final_fmax": _to_json_serializable(out_dict.get("final_fmax")),
                "final_force_mean": _to_json_serializable(out_dict.get("final_force_mean")),
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

        # Generate interactive Plotly figure
        fig_interactive = plot_sella_trajectory_interactive(
            out_dict.get("trajectory", {}) if isinstance(out_dict, dict) else {},
            sample_index=i,
            formula=str(formula),
            start_from=args.start_from,
            initial_neg_num=initial_neg,
            final_neg_num=int(final_neg) if final_neg is not None else -1,
            converged=out_dict.get("converged", False),
            final_eig0=final_eig0,
            final_eig1=final_eig1,
            final_eig_product=final_eig_product,
        )

        # Save HTML plot
        html_path = Path(out_dir) / f"sella_traj_{i:03d}.html"
        fig_interactive.write_html(str(html_path))
        result.plot_path = str(html_path)

        if args.wandb:
            log_sample(i, metrics, fig=fig_interactive, plot_name="sella_trajectory")

        if args.verbose:
            ts_status = "TS found!" if is_ts else f"neg_vib={final_neg}"
            eig_str = f"{final_eig_product:.2e}" if final_eig_product is not None else "N/A"
            print(f"[{i}] {formula}: {result.steps_taken} steps, {ts_status}, eig_product={eig_str}")

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

    # Detailed convergence analysis: Sella vs Eigenvalue criteria
    converged_list = all_metrics.get("converged", [])
    is_ts_list = all_metrics.get("is_ts", [])
    final_neg_vib_list = all_metrics.get("final_neg_vibrational", [])

    n_total = len(converged_list)

    # Count different categories
    n_sella_converged = sum(converged_list)
    n_eigenvalue_ts = sum(is_ts_list)

    # Both criteria
    n_both = sum(c and t for c, t in zip(converged_list, is_ts_list))

    # Sella converged but NOT eigenvalue TS
    n_sella_only = sum(c and not t for c, t in zip(converged_list, is_ts_list))

    # Eigenvalue TS but NOT Sella converged
    n_ts_only = sum(not c and t for c, t in zip(converged_list, is_ts_list))

    # Neither
    n_neither = sum(not c and not t for c, t in zip(converged_list, is_ts_list))

    # For Sella-converged non-TS samples, count negative eigenvalue distribution
    sella_converged_neg_vib_counts = defaultdict(int)
    for converged, is_ts, neg_vib in zip(converged_list, is_ts_list, final_neg_vib_list):
        if converged and not is_ts:
            if neg_vib is not None and neg_vib >= 0:
                sella_converged_neg_vib_counts[neg_vib] += 1

    # Add to summary dict for W&B
    sella_summary.update({
        "n_total_samples": n_total,
        "n_sella_converged": n_sella_converged,
        "n_eigenvalue_ts": n_eigenvalue_ts,
        "n_both_criteria": n_both,
        "n_sella_only": n_sella_only,
        "n_ts_only": n_ts_only,
        "n_neither": n_neither,
    })

    # Add neg_vib distribution for Sella-converged non-TS samples
    for neg_vib, count in sella_converged_neg_vib_counts.items():
        sella_summary[f"sella_converged_non_ts_neg{neg_vib}"] = count

    # Print Sella-specific summary
    print("\n" + "=" * 80)
    print("SELLA CONVERGENCE ANALYSIS")
    print("=" * 80)
    print(f"Total samples: {n_total}")
    print()
    print("Convergence by criteria:")
    print(f"  Sella converged (fmax reached):     {n_sella_converged:3d} ({n_sella_converged/n_total*100:5.1f}%)")
    print(f"  Eigenvalue TS (1 neg eigenvalue):   {n_eigenvalue_ts:3d} ({n_eigenvalue_ts/n_total*100:5.1f}%)")
    print()
    print("Overlap analysis:")
    print(f"  Both criteria satisfied:            {n_both:3d} ({n_both/n_total*100:5.1f}%)")
    print(f"  Sella only (NOT eigenvalue TS):     {n_sella_only:3d} ({n_sella_only/n_total*100:5.1f}%)")
    print(f"  Eigenvalue TS only (NOT Sella):     {n_ts_only:3d} ({n_ts_only/n_total*100:5.1f}%)")
    print(f"  Neither criterion satisfied:        {n_neither:3d} ({n_neither/n_total*100:5.1f}%)")
    print()

    if n_sella_only > 0:
        print("Negative eigenvalue distribution for Sella-converged non-TS samples:")
        sorted_counts = sorted(sella_converged_neg_vib_counts.items())
        for neg_vib, count in sorted_counts:
            print(f"  {neg_vib} negative eigenvalues: {count:3d} samples ({count/n_sella_only*100:5.1f}% of Sella-only)")

    print()
    print(f"Average final fmax: {sella_summary.get('avg_final_fmax', 0):.4f} eV/Å")
    print("=" * 80)

    if args.wandb:
        summary.update(sella_summary)
        log_summary(summary)
        finish_wandb()

    print(f"Saved results: {all_runs_path}")
    print(f"Saved stats:   {aggregate_stats_path}")


if __name__ == "__main__":
    main()
