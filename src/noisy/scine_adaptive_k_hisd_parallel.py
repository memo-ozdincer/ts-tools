#!/usr/bin/env python
"""Parallel Adaptive k-HiSD Runner with Diagnostic Logging.

This implements the theoretically-justified adaptive k-HiSD algorithm from
the iHiSD paper, in contrast to the empirical v₂ kicking approach.

Key difference:
- v₂ kicking: discrete perturbations that "unstick" GAD but don't systematically
  reduce the Morse index. Works empirically but through brute-force exploration.

- Adaptive k-HiSD: continuous k-reflection where k = Morse index. Theoretically
  guaranteed to make index-k saddles unstable, systematically descending toward
  index-1.

Comparison questions:
1. Does adaptive k-HiSD converge faster (fewer steps)?
2. Is the index trajectory cleaner (monotonic decrease)?
3. Does it avoid the dt collapse issue entirely?
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

warnings.filterwarnings("ignore")

# Import adaptive k-HiSD runner
from src.noisy.v2_tests.baselines.run_adaptive_k_hisd import run_single_sample_adaptive_k_hisd

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
    # Adaptive k-HiSD parameters
    "dt": 0.005,
    "dt_control": "adaptive",
    "dt_min": 1e-6,
    "dt_max": 0.08,
    "max_atom_disp": 0.35,
    "dt_grow_factor": 1.1,
    "dt_shrink_factor": 0.5,
    "stop_at_ts": True,
    "ts_eps": 1e-5,
    "tr_threshold": 1e-6,
    "min_k": 1,  # At index-1, behaves like standard GAD
    "max_k": None,  # No upper limit on k
}


def scine_worker_adaptive_k_hisd(predict_fn, payload) -> Dict[str, Any]:
    """Worker function for parallel adaptive k-HiSD."""
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
        result = run_single_sample_adaptive_k_hisd(
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
    """Run batch of samples through adaptive k-HiSD."""
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

    steps_when_success = [r["steps_to_ts"] for r in results if r["steps_to_ts"] is not None]
    wall_times = [r["wall_time"] for r in results]

    final_neg_vibs = [r["final_neg_vib"] for r in results if r["error"] is None]
    neg_vib_counts = {}
    for v in final_neg_vibs:
        neg_vib_counts[v] = neg_vib_counts.get(v, 0) + 1

    return {
        "n_samples": n_samples,
        "n_success": n_success,
        "n_errors": n_errors,
        "success_rate": n_success / max(n_samples, 1),
        "mean_steps_when_success": np.mean(steps_when_success) if steps_when_success else float("nan"),
        "mean_wall_time": np.mean(wall_times) if wall_times else float("nan"),
        "total_wall_time": sum(wall_times),
        "neg_vib_counts": neg_vib_counts,
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
        description="Parallel Adaptive k-HiSD for Transition State Finding"
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

    # Adaptive k-HiSD specific params
    parser.add_argument("--dt", type=float, default=BASELINE_PARAMS["dt"])
    parser.add_argument("--dt-control", type=str, default=BASELINE_PARAMS["dt_control"],
                        choices=["fixed", "adaptive"])
    parser.add_argument("--dt-min", type=float, default=BASELINE_PARAMS["dt_min"])
    parser.add_argument("--dt-max", type=float, default=BASELINE_PARAMS["dt_max"])
    parser.add_argument("--max-atom-disp", type=float, default=BASELINE_PARAMS["max_atom_disp"])
    parser.add_argument("--dt-grow-factor", type=float, default=BASELINE_PARAMS["dt_grow_factor"])
    parser.add_argument("--dt-shrink-factor", type=float, default=BASELINE_PARAMS["dt_shrink_factor"])
    parser.add_argument("--min-k", type=int, default=BASELINE_PARAMS["min_k"])
    parser.add_argument("--max-k", type=int, default=None)
    parser.add_argument("--tr-threshold", type=float, default=BASELINE_PARAMS["tr_threshold"])

    parser.add_argument("--stop-at-ts", dest="stop_at_ts", action="store_true")
    parser.add_argument("--no-stop-at-ts", dest="stop_at_ts", action="store_false")
    parser.set_defaults(stop_at_ts=BASELINE_PARAMS["stop_at_ts"])

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
        "dt": args.dt,
        "dt_control": args.dt_control,
        "dt_min": args.dt_min,
        "dt_max": args.dt_max,
        "max_atom_disp": args.max_atom_disp,
        "dt_grow_factor": args.dt_grow_factor,
        "dt_shrink_factor": args.dt_shrink_factor,
        "min_k": args.min_k,
        "max_k": args.max_k,
        "stop_at_ts": args.stop_at_ts,
        "tr_threshold": args.tr_threshold,
    })

    print("=" * 70)
    print("Adaptive k-HiSD Runner")
    print("=" * 70)
    print(f"  Algorithm: Adaptive k-HiSD (k = Morse index)")
    print(f"  Workers: {args.n_workers} (Threads/worker: {threads_per_worker})")
    print(f"  Output: {args.out_dir}")
    print(f"  dt: {args.dt} (control: {args.dt_control})")
    print(f"  k range: [{args.min_k}, {args.max_k or 'unlimited'}]")
    print("=" * 70)

    processor = ParallelSCINEProcessor(
        functional=args.scine_functional,
        threads_per_worker=threads_per_worker,
        n_workers=args.n_workers,
        worker_fn=scine_worker_adaptive_k_hisd,
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
        print("Adaptive k-HiSD Completed")
        print("=" * 70)
        print(f"  Samples: {metrics['n_samples']}")
        print(f"  Success: {metrics['n_success']} ({metrics['success_rate']*100:.1f}%)")
        print(f"  Mean steps (when success): {metrics['mean_steps_when_success']:.1f}")
        print(f"  Final index distribution: {metrics['neg_vib_counts']}")

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        results_path = Path(args.out_dir) / f"adaptive_k_hisd_{job_id}_summary.json"

        with open(results_path, "w") as f:
            json.dump({
                "job_id": job_id,
                "algorithm": "adaptive_k_hisd",
                "params": params,
                "metrics": metrics
            }, f, indent=2, default=str)

        print(f"Summary saved to: {results_path}")

    finally:
        processor.close()


if __name__ == "__main__":
    main()
