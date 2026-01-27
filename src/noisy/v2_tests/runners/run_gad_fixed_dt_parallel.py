#!/usr/bin/env python
"""Parallel SCINE runner for fixed-dt GAD baseline.

This tests the hypothesis that adaptive dt control is responsible for stalls.
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
from src.noisy.v2_tests.baselines.gad_fixed_dt import run_gad_fixed_dt
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
    t0 = time.time()
    try:
        result, _trajectory, _logger = run_gad_fixed_dt(
            predict_fn,
            coords,
            atomic_nums,
            n_steps=n_steps,
            dt=params["dt"],
            max_atom_disp=params["max_atom_disp"],
            ts_eps=params["ts_eps"],
            stop_at_ts=params["stop_at_ts"],
            min_interatomic_dist=params["min_interatomic_dist"],
            tr_threshold=params["tr_threshold"],
            track_mode=params["track_mode"],
            scine_elements=params.get("scine_elements"),
            log_dir=params.get("log_dir"),
            sample_id=sample_id,
            formula=formula,
        )
        wall_time = time.time() - t0

        final_neg_vib = result.get("final_morse_index", -1)
        steps_taken = result.get("total_steps", n_steps)
        steps_to_ts = result.get("converged_step")

        return {
            "final_neg_vib": final_neg_vib,
            "steps_taken": steps_taken,
            "steps_to_ts": steps_to_ts,
            "success": bool(result.get("success")),
            "wall_time": wall_time,
            "error": None,
        }
    except Exception as e:
        wall_time = time.time() - t0
        return {
            "final_neg_vib": -1,
            "steps_taken": 0,
            "steps_to_ts": None,
            "success": False,
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
    n_success = sum(1 for r in results if r.get("success"))
    n_errors = sum(1 for r in results if r.get("error") is not None)

    steps_when_success = [r["steps_to_ts"] for r in results if r.get("steps_to_ts") is not None]
    wall_times = [r["wall_time"] for r in results]

    final_neg_vibs = [r["final_neg_vib"] for r in results if r.get("error") is None]
    neg_vib_counts: Dict[int, int] = {}
    for v in final_neg_vibs:
        neg_vib_counts[v] = neg_vib_counts.get(v, 0) + 1

    return {
        "n_samples": n_samples,
        "n_success": n_success,
        "n_errors": n_errors,
        "success_rate": n_success / max(n_samples, 1),
        "mean_steps_when_success": float(np.mean(steps_when_success)) if steps_when_success else float("nan"),
        "mean_wall_time": float(np.mean(wall_times)) if wall_times else float("nan"),
        "total_wall_time": float(sum(wall_times)),
        "neg_vib_counts": neg_vib_counts,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-dt GAD baseline (parallel, SCINE)")
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

    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--max-atom-disp", type=float, default=0.35)
    parser.add_argument("--min-interatomic-dist", type=float, default=0.5)
    parser.add_argument("--ts-eps", type=float, default=1e-5)
    parser.add_argument("--tr-threshold", type=float, default=1e-6)
    parser.add_argument("--track-mode", action="store_true")
    parser.add_argument("--no-track-mode", dest="track_mode", action="store_false")
    parser.set_defaults(track_mode=True)
    parser.add_argument("--stop-at-ts", action="store_true")
    parser.add_argument("--no-stop-at-ts", dest="stop_at_ts", action="store_false")
    parser.set_defaults(stop_at_ts=True)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    diag_dir = Path(args.out_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "dt": args.dt,
        "max_atom_disp": args.max_atom_disp,
        "min_interatomic_dist": args.min_interatomic_dist,
        "ts_eps": args.ts_eps,
        "tr_threshold": args.tr_threshold,
        "track_mode": args.track_mode,
        "stop_at_ts": args.stop_at_ts,
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
        results_path = Path(args.out_dir) / f"gad_fixed_dt_parallel_{job_id}_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "job_id": job_id,
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
