#!/usr/bin/env python
"""Parallel SCINE runner for minimization baselines.

Methods:
- fixed_step_gd: Fixed-step gradient descent (no line search)
- newton_raphson: Newton-Raphson with Hessian-preconditioned steps
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.dependencies.common_utils import Transition1xDataset, UsePos, parse_starting_geometry
from src.dependencies.hessian import get_scine_elements_from_predict_output
from src.noisy.multi_mode_eckartmw import get_projected_hessian, _atomic_nums_to_symbols
from src.noisy.v2_tests.baselines.minimization import run_fixed_step_gd, run_newton_raphson
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
) -> Dict[str, Any]:
    method = params["method"]

    def _count_neg_vib(coords_local: torch.Tensor) -> Optional[int]:
        try:
            out = predict_fn(coords_local, atomic_nums, do_hessian=True, require_grad=False)
            scine_elements = get_scine_elements_from_predict_output(out)
            hess_proj = get_projected_hessian(out["hessian"], coords_local, atomic_nums, scine_elements)
            evals, _ = torch.linalg.eigh(hess_proj)
            tr_threshold = float(params.get("tr_threshold", 1e-6))
            vib_mask = torch.abs(evals) > tr_threshold
            vib_evals = evals[vib_mask]
            return int((vib_evals < -tr_threshold).sum().item())
        except Exception:
            return None

    # Pre-compute atom symbols for projection
    project_gradient_and_v = params.get("project_gradient_and_v", False)
    atomsymbols = _atomic_nums_to_symbols(atomic_nums) if project_gradient_and_v else None

    t0 = time.time()
    try:
        start_neg_vib = _count_neg_vib(coords)

        if method == "fixed_step_gd":
            result, trajectory = run_fixed_step_gd(
                predict_fn,
                coords,
                atomic_nums,
                n_steps=n_steps,
                step_size=params["step_size"],
                max_atom_disp=params["max_atom_disp"],
                force_converged=params["force_converged"],
                min_interatomic_dist=params["min_interatomic_dist"],
                project_gradient_and_v=project_gradient_and_v,
                atomsymbols=atomsymbols,
            )
        elif method == "newton_raphson":
            # For NR we need scine_elements at each step, but we pass the
            # get_projected_hessian function which handles it internally.
            # Pre-fetch scine_elements from an initial prediction.
            init_out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
            scine_elements = get_scine_elements_from_predict_output(init_out)
            nr_atomsymbols = _atomic_nums_to_symbols(atomic_nums)

            result, trajectory = run_newton_raphson(
                predict_fn,
                coords,
                atomic_nums,
                get_projected_hessian,
                n_steps=n_steps,
                max_atom_disp=params["max_atom_disp"],
                force_converged=params["force_converged"],
                min_interatomic_dist=params["min_interatomic_dist"],
                tr_threshold=params["tr_threshold"],
                project_gradient_and_v=project_gradient_and_v,
                atomsymbols=nr_atomsymbols if project_gradient_and_v else None,
                scine_elements=scine_elements,
                purify_hessian=params.get("purify_hessian", False),
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        final_coords = result.get("final_coords")
        final_neg_vib = _count_neg_vib(
            final_coords.to(coords.device) if isinstance(final_coords, torch.Tensor) else coords
        )

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
    formula = getattr(batch, "formula", "")
    result = run_single_sample(
        predict_fn,
        start_coords,
        atomic_nums,
        params,
        n_steps,
        sample_id=f"sample_{sample_idx:03d}",
        formula=str(formula),
    )
    result["sample_idx"] = sample_idx
    result["formula"] = getattr(batch, "formula", "")
    return result


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

    return {
        "n_samples": n_samples,
        "n_converged": n_converged,
        "n_errors": n_errors,
        "convergence_rate": n_converged / max(n_samples, 1),
        "mean_steps_when_converged": float(np.mean(steps_when_success)) if steps_when_success else float("nan"),
        "mean_wall_time": float(np.mean(wall_times)) if wall_times else float("nan"),
        "total_wall_time": float(sum(wall_times)),
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
    parser.add_argument("--max-atom-disp", type=float, default=0.3)
    parser.add_argument("--force-converged", type=float, default=1e-4)
    parser.add_argument("--min-interatomic-dist", type=float, default=0.5)
    parser.add_argument("--tr-threshold", type=float, default=1e-6)

    # Gradient descent parameters
    parser.add_argument("--step-size", type=float, default=0.01,
                        help="Fixed step size for gradient descent")

    # Projection options
    parser.add_argument(
        "--project-gradient-and-v",
        action="store_true",
        default=False,
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
        "tr_threshold": args.tr_threshold,
        "project_gradient_and_v": args.project_gradient_and_v,
        "purify_hessian": args.purify_hessian,
        "log_dir": str(diag_dir),
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
    finally:
        processor.close()


if __name__ == "__main__":
    main()
