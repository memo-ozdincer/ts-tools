#!/usr/bin/env python
"""Parallel SCINE runner for minimization baselines.

Methods:
- fixed_step_gd: Fixed-step gradient descent (no line search)
- newton_raphson: Newton-Raphson with Hessian-preconditioned steps

New in v2:
- --lm-mu         Levenberg-Marquardt damping coefficient (0 = hard filter, default)
- --anneal-force-threshold
                  Force norm at which two-phase annealing kicks in (0 = off)
- --cleanup-nr-threshold
                  NR threshold used in cleanup phase (0 = full pseudoinverse)
- --cleanup-max-steps
                  Maximum extra steps in cleanup phase (default 50)
- --log-spectrum-k
                  Number of bottom vibrational eigenvalues to log per step (default 10)

Cascade evaluation:
  Every trajectory step now contains "n_neg_at_<threshold>" fields for
  thresholds [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2].
  The summary JSON includes a "cascade_table" whose rows are keyed by
  nr_threshold and columns by eval_threshold, with values = convergence rate.
  This lets the analysis script build the 2D diagnostic plot without
  needing to re-read the full trajectories.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.dependencies.common_utils import Transition1xDataset, UsePos, parse_starting_geometry
from src.noisy.multi_mode_eckartmw import get_vib_evals_evecs, _atomic_nums_to_symbols
from src.noisy.v2_tests.baselines.minimization import (
    CASCADE_THRESHOLDS,
    run_fixed_step_gd,
    run_newton_raphson,
)
from src.parallel.scine_parallel import ParallelSCINEProcessor
from src.parallel.utils import run_batch_parallel


def create_dataloader(h5_path: str, split: str, max_samples: int):
    dataset = Transition1xDataset(
        h5_path=h5_path,
        split=split,
        max_samples=max_samples,
        transform=UsePos("pos_transition"),
    )
    if len(dataset) == 0:
        raise RuntimeError("No Transition1x samples loaded. Check h5 path and split.")
    return DataLoader(dataset, batch_size=1, shuffle=False)


def run_single_sample(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    params: Dict[str, Any],
    n_steps: int,
    *,
    sample_id: str,
    formula: str,
    known_ts_coords: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    method = params["method"]
    project_gradient_and_v = params.get("project_gradient_and_v", False)
    # Atom symbols are always needed: reduced-basis eigendecomposition requires them.
    all_atomsymbols = _atomic_nums_to_symbols(atomic_nums)

    def _count_neg_vib(coords_local: torch.Tensor) -> Optional[int]:
        try:
            out = predict_fn(coords_local, atomic_nums, do_hessian=True, require_grad=False)
            evals_vib, _, _ = get_vib_evals_evecs(out["hessian"], coords_local, all_atomsymbols)
            return int((evals_vib < 0.0).sum().item())
        except Exception:
            return None

    def _cascade_neg_vib(coords_local: torch.Tensor) -> Optional[Dict[str, int]]:
        """Count n_neg at every cascade threshold for the final geometry."""
        try:
            out = predict_fn(coords_local, atomic_nums, do_hessian=True, require_grad=False)
            evals_vib, _, _ = get_vib_evals_evecs(out["hessian"], coords_local, all_atomsymbols)
            result: Dict[str, int] = {}
            for thr in CASCADE_THRESHOLDS:
                result[f"n_neg_at_{thr}"] = int((evals_vib < -thr).sum().item())
            result["min_vib_eval"] = float(evals_vib.min().item()) if evals_vib.numel() > 0 else float("nan")
            return result
        except Exception:
            return None

    t0 = time.time()
    try:
        start_neg_vib = _count_neg_vib(coords)

        if method == "fixed_step_gd":
            result, trajectory = run_fixed_step_gd(
                predict_fn,
                coords,
                atomic_nums,
                all_atomsymbols,
                n_steps=n_steps,
                step_size=params["step_size"],
                max_atom_disp=params["max_atom_disp"],
                force_converged=params["force_converged"],
                min_interatomic_dist=params["min_interatomic_dist"],
                project_gradient_and_v=project_gradient_and_v,
                purify_hessian=params.get("purify_hessian", False),
                log_spectrum_k=params.get("log_spectrum_k", 10),
            )
        elif method == "newton_raphson":
            result, trajectory = run_newton_raphson(
                predict_fn,
                coords,
                atomic_nums,
                all_atomsymbols,
                n_steps=n_steps,
                max_atom_disp=params["max_atom_disp"],
                force_converged=params["force_converged"],
                min_interatomic_dist=params["min_interatomic_dist"],
                nr_threshold=params.get("nr_threshold", 8e-3),
                project_gradient_and_v=project_gradient_and_v,
                purify_hessian=params.get("purify_hessian", False),
                known_ts_coords=known_ts_coords,
                lm_mu=params.get("lm_mu", 0.0),
                anneal_force_threshold=params.get("anneal_force_threshold", 0.0),
                cleanup_nr_threshold=params.get("cleanup_nr_threshold", 0.0),
                cleanup_max_steps=params.get("cleanup_max_steps", 50),
                log_spectrum_k=params.get("log_spectrum_k", 10),
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        final_coords = result.get("final_coords")
        final_cascade = _cascade_neg_vib(
            final_coords.to(coords.device) if isinstance(final_coords, torch.Tensor) else coords
        )
        final_neg_vib = final_cascade.get("n_neg_at_0.0") if final_cascade else None

        if params.get("log_dir"):
            log_dir = Path(params["log_dir"])
            log_dir.mkdir(parents=True, exist_ok=True)
            traj_path = log_dir / f"{sample_id}_{method}_trajectory.json"
            with open(traj_path, "w") as f:
                json.dump(
                    {
                        "sample_id": sample_id,
                        "formula": formula,
                        "method": method,
                        "start_neg_vib": start_neg_vib,
                        "final_neg_vib": final_neg_vib,
                        "final_cascade": final_cascade,
                        "cascade_at_convergence": result.get("cascade_at_convergence", {}),
                        "bottom_spectrum_at_convergence": result.get("bottom_spectrum_at_convergence", []),
                        "final_min_vib_eval": result.get("final_min_vib_eval"),
                        "cleanup_steps_taken": result.get("cleanup_steps_taken", 0),
                        "trajectory": trajectory,
                    },
                    f,
                    indent=2,
                )

        wall_time = time.time() - t0
        return {
            "converged": bool(result.get("converged")),
            "converged_step": result.get("converged_step"),
            "final_energy": result.get("final_energy"),
            "final_force_norm": result.get("final_force_norm"),
            "total_steps": result.get("total_steps", n_steps),
            "start_neg_vib": start_neg_vib,
            "final_neg_vib": final_neg_vib,
            "final_cascade": final_cascade,
            "final_min_vib_eval": result.get("final_min_vib_eval"),
            "cascade_at_convergence": result.get("cascade_at_convergence", {}),
            "bottom_spectrum_at_convergence": result.get("bottom_spectrum_at_convergence", []),
            "cleanup_steps_taken": result.get("cleanup_steps_taken", 0),
            "wall_time": wall_time,
            "error": None,
        }
    except Exception as e:
        wall_time = time.time() - t0
        return {
            "converged": False,
            "converged_step": None,
            "final_energy": None,
            "final_force_norm": None,
            "total_steps": 0,
            "start_neg_vib": None,
            "final_neg_vib": None,
            "final_cascade": None,
            "final_min_vib_eval": None,
            "cascade_at_convergence": {},
            "bottom_spectrum_at_convergence": [],
            "cleanup_steps_taken": 0,
            "wall_time": wall_time,
            "error": str(e),
        }


def scine_worker_sample(predict_fn, payload) -> Dict[str, Any]:
    sample_idx, batch, params, n_steps, start_from, noise_seed = payload
    batch = batch.to("cpu")
    atomic_nums = batch.z.detach().to("cpu")
    start_coords = parse_starting_geometry(
        start_from,
        batch,
        noise_seed=noise_seed,
        sample_index=sample_idx,
    ).detach().to("cpu")

    known_ts_coords = getattr(batch, "pos_transition", None)
    if known_ts_coords is None:
        known_ts_coords = getattr(batch, "pos", None)
    if known_ts_coords is not None:
        known_ts_coords = known_ts_coords.detach().to("cpu")

    formula = getattr(batch, "formula", "")
    result = run_single_sample(
        predict_fn,
        start_coords,
        atomic_nums,
        params,
        n_steps,
        sample_id=f"sample_{sample_idx:03d}",
        formula=str(formula),
        known_ts_coords=known_ts_coords,
    )
    result["sample_idx"] = sample_idx
    result["formula"] = getattr(batch, "formula", "")
    return result


def _build_cascade_table(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a 2D convergence-rate table over cascade evaluation thresholds.

    For each eval_threshold in CASCADE_THRESHOLDS, a sample is "converged at
    that eval threshold" if its final_cascade["n_neg_at_<thresh>"] == 0.

    This lets the analysis script answer: "If we relaxed the evaluation to
    accept eigenvalues down to -<thresh>, what would our convergence rate be?"
    WITHOUT changing the optimizer at all. The optimizer's nr_threshold is
    the row key (passed in as a param label); columns are eval thresholds.

    Returns a dict:
      {
        "eval_thresholds": [0.0, 1e-4, ...],
        "n_samples": <int>,
        "n_converged_at_thr": {"0.0": <int>, "0.0001": <int>, ...},
        "rate_at_thr":        {"0.0": <float>, ...},
      }
    """
    n = len(results)
    n_converged_at: Dict[str, int] = {}
    rate_at: Dict[str, float] = {}
    for thr in CASCADE_THRESHOLDS:
        key = f"n_neg_at_{thr}"
        count = sum(
            1 for r in results
            if r.get("final_cascade") and r["final_cascade"].get(key, 1) == 0
        )
        n_converged_at[str(thr)] = count
        rate_at[str(thr)] = count / max(n, 1)
    return {
        "eval_thresholds": CASCADE_THRESHOLDS,
        "n_samples": n,
        "n_converged_at_thr": n_converged_at,
        "rate_at_thr": rate_at,
    }


def run_batch(
    processor: ParallelSCINEProcessor,
    dataloader,
    params: Dict[str, Any],
    n_steps: int,
    max_samples: int,
    start_from: str,
    noise_seed: Optional[int],
) -> Dict[str, Any]:
    samples = []
    for i, batch in enumerate(dataloader):
        if i >= max_samples:
            break
        payload = (i, batch, params, n_steps, start_from, noise_seed)
        samples.append((i, payload))

    results = run_batch_parallel(samples, processor)

    n_samples = len(results)
    n_converged = sum(1 for r in results if r.get("converged"))
    n_errors = sum(1 for r in results if r.get("error") is not None)

    steps_when_success = [r["converged_step"] for r in results if r.get("converged_step") is not None]
    wall_times = [r["wall_time"] for r in results]

    cascade_table = _build_cascade_table(results)

    return {
        "n_samples": n_samples,
        "n_converged": n_converged,
        "n_errors": n_errors,
        "convergence_rate": n_converged / max(n_samples, 1),
        "mean_steps_when_converged": float(np.mean(steps_when_success)) if steps_when_success else float("nan"),
        "mean_wall_time": float(np.mean(wall_times)) if wall_times else float("nan"),
        "total_wall_time": float(sum(wall_times)),
        "cascade_table": cascade_table,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimization baselines (parallel, SCINE)")
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--scine-functional", type=str, default="DFTB0")
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--start-from", type=str, default="midpoint_rt_noise1.0A")
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=4)

    # Method selection
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["fixed_step_gd", "newton_raphson"],
        help="Minimization method: 'fixed_step_gd' or 'newton_raphson'",
    )

    # Shared parameters
    parser.add_argument("--max-atom-disp", type=float, default=1.3)
    parser.add_argument("--force-converged", type=float, default=1e-4)
    parser.add_argument("--min-interatomic-dist", type=float, default=0.5)
    parser.add_argument(
        "--nr-threshold", type=float, default=8e-3,
        help="Eigenvalue cutoff for NR step filtering (|λ| < threshold → excluded from "
             "pseudoinverse).",
    )

    # --- New v2 flags ---
    parser.add_argument(
        "--lm-mu", type=float, default=0.0,
        help="Levenberg-Marquardt damping coefficient μ. 0 (default) = hard filter mode. "
             ">0 = LM mode: step_i = (g·v_i)*|λ_i|/(λ_i²+μ²). "
             "Sweep this with the same grid as --nr-threshold.",
    )
    parser.add_argument(
        "--anneal-force-threshold", type=float, default=0.0,
        help="Two-phase annealing: once force_norm < this value, switch from nr-threshold "
             "to cleanup-nr-threshold for cleanup-max-steps steps. 0 = off (default).",
    )
    parser.add_argument(
        "--cleanup-nr-threshold", type=float, default=0.0,
        help="NR threshold used in the cleanup phase of two-phase annealing. "
             "0 (default) = full pseudoinverse (no filtering). "
             "Only used when --anneal-force-threshold > 0.",
    )
    parser.add_argument(
        "--cleanup-max-steps", type=int, default=50,
        help="Maximum extra steps in cleanup phase (default 50).",
    )
    parser.add_argument(
        "--log-spectrum-k", type=int, default=10,
        help="Number of bottom vibrational eigenvalues to log per step (default 10, 0 = none).",
    )

    # Gradient descent parameters
    parser.add_argument("--step-size", type=float, default=0.01,
                        help="Fixed step size for gradient descent")

    # Projection options
    parser.add_argument(
        "--project-gradient-and-v",
        action="store_true",
        default=True,
        help="Eckart-project the gradient to prevent TR drift",
    )
    parser.add_argument(
        "--purify-hessian",
        action="store_true",
        default=False,
        help="Enforce translational sum rules on the Hessian (Newton-Raphson only)",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    diag_dir = Path(args.out_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "method": args.method,
        "step_size": args.step_size,
        "max_atom_disp": args.max_atom_disp,
        "force_converged": args.force_converged,
        "min_interatomic_dist": args.min_interatomic_dist,
        "nr_threshold": args.nr_threshold,
        "project_gradient_and_v": args.project_gradient_and_v,
        "purify_hessian": args.purify_hessian,
        "log_dir": str(diag_dir),
        # v2 additions
        "lm_mu": args.lm_mu,
        "anneal_force_threshold": args.anneal_force_threshold,
        "cleanup_nr_threshold": args.cleanup_nr_threshold,
        "cleanup_max_steps": args.cleanup_max_steps,
        "log_spectrum_k": args.log_spectrum_k,
    }

    processor = ParallelSCINEProcessor(
        functional=args.scine_functional,
        threads_per_worker=args.threads_per_worker,
        n_workers=args.n_workers,
        worker_fn=scine_worker_sample,
    )
    processor.start()

    try:
        dataloader = create_dataloader(args.h5_path, args.split, args.max_samples)
        metrics = run_batch(
            processor,
            dataloader,
            params,
            args.n_steps,
            args.max_samples,
            args.start_from,
            args.noise_seed,
        )

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        results_path = Path(args.out_dir) / f"minimization_{args.method}_{job_id}_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "method": args.method,
                    "params": params,
                    "config": {
                        "n_steps": args.n_steps,
                        "max_samples": args.max_samples,
                        "start_from": args.start_from,
                        "noise_seed": args.noise_seed,
                        "scine_functional": args.scine_functional,
                        "n_workers": args.n_workers,
                        "threads_per_worker": args.threads_per_worker,
                        "split": args.split,
                    },
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        print(f"Results saved to: {results_path}")

        # Print cascade table to stdout for quick inspection
        ct = metrics.get("cascade_table", {})
        print("\n--- Cascade Evaluation Table ---")
        print(f"{'eval_threshold':<20} {'n_converged':<15} {'rate':<10}")
        print("-" * 46)
        for thr in CASCADE_THRESHOLDS:
            key = str(thr)
            n_conv = ct.get("n_converged_at_thr", {}).get(key, "?")
            rate = ct.get("rate_at_thr", {}).get(key, float("nan"))
            rate_str = f"{rate:.3f}" if isinstance(rate, float) else str(rate)
            print(f"  {thr:<18} {n_conv:<15} {rate_str}")
        print("")

    finally:
        processor.close()


if __name__ == "__main__":
    main()
