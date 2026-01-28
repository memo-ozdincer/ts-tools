#!/usr/bin/env python
"""Parallel Recursive HiSD Runner with Diagnostic Logging.

Recursive HiSD starts from a (potentially high-index) saddle point and
recursively descends to lower index saddles until reaching an index-1 TS.

Key difference from adaptive k-HiSD:
- Adaptive k-HiSD: starts with k=1, increases k when stuck
- Recursive HiSD: explicitly targets each index level in sequence,
  perturbing along unstable directions to escape each saddle
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from torch_geometric.loader import DataLoader

warnings.filterwarnings("ignore")

from src.noisy.v2_tests.baselines.run_recursive_hisd import run_single_sample_recursive_hisd
from src.dependencies.common_utils import Transition1xDataset, UsePos, parse_starting_geometry
from src.parallel.scine_parallel import ParallelSCINEProcessor
from src.parallel.utils import run_batch_parallel


class SuppressStdout:
    """Context manager to suppress stdout."""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self
    def __exit__(self, *args):
        sys.stdout = self._stdout


BASELINE_PARAMS = {
    # Recursive HiSD parameters
    "perturb_magnitude": 0.01,
    "perturb_strategy": "unstable",  # "unstable", "random", "mixed"
    "grad_threshold": 1e-5,
    "max_steps_per_level": 500,
    "dt": 0.005,
    "dt_min": 1e-6,
    "dt_max": 0.08,
    "max_atom_disp": 0.35,
    "tr_threshold": 1e-6,
    "neg_threshold": -1e-4,
}


def scine_worker_recursive_hisd(predict_fn, payload) -> Dict[str, Any]:
    """Worker function for parallel Recursive HiSD."""
    sample_idx, batch, params, n_steps, start_from, noise_seed, out_dir = payload
    batch = batch.to("cpu")
    atomic_nums = batch.z.detach().to("cpu")
    formula = getattr(batch, "formula", f"sample_{sample_idx:03d}")
    sample_id = f"sample_{sample_idx:03d}"

    start_coords = parse_starting_geometry(
        start_from,
        batch,
        noise_seed=noise_seed,
        sample_index=sample_idx,
    ).detach().to("cpu")

    with SuppressStdout():
        result = run_single_sample_recursive_hisd(
            predict_fn,
            start_coords,
            atomic_nums,
            params,
            n_steps,
            sample_id=sample_id,
            formula=str(formula),
            out_dir=out_dir,
        )

    result["sample_idx"] = sample_idx
    return result


def run_batch(
    processor: ParallelSCINEProcessor,
    dataloader,
    params: Dict[str, Any],
    n_steps: int,
    max_samples: int,
    start_from: str,
    noise_seed: Optional[int],
    out_dir: str,
) -> Dict[str, Any]:
    """Run batch of samples through Recursive HiSD."""
    samples = []

    for i, batch in enumerate(dataloader):
        if i >= max_samples:
            break
        if i % 10 == 0:
            print(f"    [Sample {i}/{max_samples}]", file=sys.stderr, end="\r")

        payload = (i, batch, params, n_steps, start_from, noise_seed, out_dir)
        samples.append((i, payload))

    results = run_batch_parallel(samples, processor)

    n_samples = len(results)
    n_success = sum(1 for r in results if r["success"])
    n_errors = sum(1 for r in results if r["error"] is not None)

    steps_when_success = [r["steps_taken"] for r in results if r["success"]]
    wall_times = [r["wall_time"] for r in results]

    final_neg_vibs = [r["final_neg_vib"] for r in results if r["error"] is None]
    neg_vib_counts = {}
    for v in final_neg_vibs:
        neg_vib_counts[v] = neg_vib_counts.get(v, 0) + 1

    # Recursive HiSD specific stats
    recursion_depths = [r["recursion_depth"] for r in results if r["error"] is None]
    initial_indices = [r["initial_index"] for r in results if r["error"] is None and r["initial_index"] >= 0]

    # Initial index distribution
    initial_index_counts = {}
    for idx in initial_indices:
        initial_index_counts[idx] = initial_index_counts.get(idx, 0) + 1

    return {
        "n_samples": n_samples,
        "n_success": n_success,
        "n_errors": n_errors,
        "success_rate": n_success / max(n_samples, 1),
        "mean_steps_when_success": np.mean(steps_when_success) if steps_when_success else float("nan"),
        "mean_wall_time": np.mean(wall_times) if wall_times else float("nan"),
        "total_wall_time": sum(wall_times),
        "neg_vib_counts": neg_vib_counts,
        "mean_recursion_depth": np.mean(recursion_depths) if recursion_depths else float("nan"),
        "mean_initial_index": np.mean(initial_indices) if initial_indices else float("nan"),
        "initial_index_counts": initial_index_counts,
        "results": results,
    }


def create_dataloader(h5_path: str, split: str, max_samples: int):
    """Create dataloader for Transition1x dataset."""
    dataset = Transition1xDataset(
        h5_path=h5_path,
        split=split,
        max_samples=max_samples,
        transform=UsePos("pos_transition"),
    )
    if len(dataset) == 0:
        raise RuntimeError("No Transition1x samples loaded. Check h5 path and split.")
    return DataLoader(dataset, batch_size=1, shuffle=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel Recursive HiSD for Transition State Finding"
    )

    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--scine-functional", type=str, default="DFTB0")
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--start-from", type=str, default="midpoint_rt")
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--threads-per-worker", type=int, default=None)

    # Recursive HiSD specific params
    parser.add_argument("--perturb-magnitude", type=float, default=BASELINE_PARAMS["perturb_magnitude"])
    parser.add_argument("--perturb-strategy", type=str, default=BASELINE_PARAMS["perturb_strategy"],
                        choices=["unstable", "random", "mixed"])
    parser.add_argument("--grad-threshold", type=float, default=BASELINE_PARAMS["grad_threshold"])
    parser.add_argument("--max-steps-per-level", type=int, default=BASELINE_PARAMS["max_steps_per_level"])
    parser.add_argument("--dt", type=float, default=BASELINE_PARAMS["dt"])
    parser.add_argument("--dt-min", type=float, default=BASELINE_PARAMS["dt_min"])
    parser.add_argument("--dt-max", type=float, default=BASELINE_PARAMS["dt_max"])
    parser.add_argument("--max-atom-disp", type=float, default=BASELINE_PARAMS["max_atom_disp"])
    parser.add_argument("--tr-threshold", type=float, default=BASELINE_PARAMS["tr_threshold"])

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    diag_dir = Path(args.out_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    total_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1))
    if args.threads_per_worker is None:
        threads_per_worker = max(1, total_cpus // max(args.n_workers, 1))
    else:
        threads_per_worker = args.threads_per_worker

    params = BASELINE_PARAMS.copy()
    params.update({
        "perturb_magnitude": args.perturb_magnitude,
        "perturb_strategy": args.perturb_strategy,
        "grad_threshold": args.grad_threshold,
        "max_steps_per_level": args.max_steps_per_level,
        "dt": args.dt,
        "dt_min": args.dt_min,
        "dt_max": args.dt_max,
        "max_atom_disp": args.max_atom_disp,
        "tr_threshold": args.tr_threshold,
    })

    print("=" * 70)
    print("Recursive HiSD Runner")
    print("=" * 70)
    print(f"  Algorithm: Recursive HiSD (descend from high-index to TS)")
    print(f"  Workers: {args.n_workers} (Threads/worker: {threads_per_worker})")
    print(f"  Output: {args.out_dir}")
    print(f"  Perturb: {args.perturb_strategy} (magnitude: {args.perturb_magnitude})")
    print(f"  Max steps/level: {args.max_steps_per_level}")
    print("=" * 70)

    processor = ParallelSCINEProcessor(
        functional=args.scine_functional,
        threads_per_worker=threads_per_worker,
        n_workers=args.n_workers,
        worker_fn=scine_worker_recursive_hisd,
    )
    processor.start()

    try:
        dataloader = create_dataloader(
            args.h5_path,
            args.split,
            args.max_samples,
        )

        metrics = run_batch(
            processor,
            dataloader,
            params,
            args.n_steps,
            args.max_samples,
            args.start_from,
            args.noise_seed,
            args.out_dir,
        )

        print("\n" + "=" * 70)
        print("Recursive HiSD Completed")
        print("=" * 70)
        print(f"  Samples: {metrics['n_samples']}")
        print(f"  Success: {metrics['n_success']} ({metrics['success_rate']*100:.1f}%)")
        print(f"  Mean steps (when success): {metrics['mean_steps_when_success']:.1f}")
        print(f"  Mean recursion depth: {metrics['mean_recursion_depth']:.1f}")
        print(f"  Mean initial index: {metrics['mean_initial_index']:.1f}")
        print(f"  Initial index distribution: {metrics['initial_index_counts']}")
        print(f"  Final index distribution: {metrics['neg_vib_counts']}")

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        results_path = Path(args.out_dir) / f"recursive_hisd_{job_id}_summary.json"

        with open(results_path, "w") as f:
            json.dump({
                "job_id": job_id,
                "algorithm": "recursive_hisd",
                "params": params,
                "metrics": metrics
            }, f, indent=2, default=str)

        print(f"Summary saved to: {results_path}")

    finally:
        processor.close()


if __name__ == "__main__":
    main()
