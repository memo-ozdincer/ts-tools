#!/usr/bin/env python
"""Parallel SCINE Multi-Mode Eckart-MW with Diagnostic Logging."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

warnings.filterwarnings("ignore")

# Import diagnostic runner
from src.noisy.v2_tests.runners.run_with_diagnostics import run_multi_mode_with_diagnostics

from src.dependencies.common_utils import Transition1xDataset, UsePos, parse_starting_geometry
from src.noisy.parallel.scine_parallel import ParallelSCINEProcessor
from src.noisy.parallel.utils import run_batch_parallel

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SuppressStdout:
    """Context manager to suppress stdout (for noisy library outputs)."""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self
    def __exit__(self, *args):
        sys.stdout = self._stdout


BASELINE_PARAMS = {
    # Default parameters from standard parallel runner
    "method": "euler",
    "dt": 0.004998616,
    "dt_control": "neg_eig_plateau",
    "dt_min": 1e-6,
    "dt_max": 0.0787679,
    "max_atom_disp": 0.3508912104455338,
    "plateau_patience": 20,
    "plateau_boost": 1.8482911248689458,
    "plateau_shrink": 0.6341627930334096,
    "escape_disp_threshold": 0.000724457,
    "escape_window": 10,
    "escape_neg_vib_std": 0.884664427,
    "escape_delta": 0.2990654358143095,
    "adaptive_delta": False,
    "min_interatomic_dist": 0.5247819497457936,
    "max_escape_cycles": 1000,
    "stop_at_ts": True,
    "ts_eps": 1e-5,
    "hip_vib_mode": "projected",
    "hip_rigid_tol": 1e-6,
    "hip_eigh_device": "cpu",
    "early_stop_patience": 500,
    "early_stop_min_steps": 200,
    # Diagnostic specific
    "tr_threshold": 1e-6,
}


def run_single_sample_diagnostics(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    params: Dict[str, Any],
    n_steps: int,
    sample_id: str,
    formula: str,
    out_dir: str,
) -> Dict[str, Any]:
    t0 = time.time()

    # Determine diagnostic directory for this worker
    # We save directly inside the worker to avoid passing large logs back
    diag_dir = Path(out_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    try:
        # We suppress stdout to keep the parallel progress bar clean, 
        # but the TrajectoryLogger will capture the important data
        with SuppressStdout():
            out_dict, traj_logger = run_multi_mode_with_diagnostics(
                predict_fn,
                coords,
                atomic_nums,
                n_steps=n_steps,
                dt=params["dt"],
                stop_at_ts=params["stop_at_ts"],
                ts_eps=params["ts_eps"],
                dt_control=params["dt_control"],
                dt_min=params["dt_min"],
                dt_max=params["dt_max"],
                max_atom_disp=params["max_atom_disp"],
                plateau_patience=params["plateau_patience"],
                plateau_boost=params["plateau_boost"],
                plateau_shrink=params["plateau_shrink"],
                escape_disp_threshold=params["escape_disp_threshold"],
                escape_window=params["escape_window"],
                tr_threshold=params.get("tr_threshold", 1e-6),
                escape_neg_vib_std=params["escape_neg_vib_std"],
                escape_delta=params["escape_delta"],
                adaptive_delta=params["adaptive_delta"],
                min_interatomic_dist=params["min_interatomic_dist"],
                max_escape_cycles=params["max_escape_cycles"],
                sample_id=sample_id,
                formula=formula,
            )
            
            # Save diagnostic files explicitly here in the worker
            traj_logger.save(diag_dir)
            
        wall_time = time.time() - t0

        final_neg_vib = out_dict.get("final_neg_vibrational", -1)
        steps_taken = out_dict.get("steps_taken", n_steps)
        steps_to_ts = out_dict.get("steps_to_ts")
        escape_cycles = out_dict.get("escape_cycles_used", 0)
        early_stopped = out_dict.get("early_stopped", False)

        return {
            "final_neg_vib": final_neg_vib,
            "steps_taken": steps_taken,
            "steps_to_ts": steps_to_ts,
            "escape_cycles": escape_cycles,
            "success": final_neg_vib == 1,
            "wall_time": wall_time,
            "early_stopped": early_stopped,
            "error": None,
        }

    except Exception as e:
        wall_time = time.time() - t0
        return {
            "final_neg_vib": -1,
            "steps_taken": 0,
            "steps_to_ts": None,
            "escape_cycles": 0,
            "success": False,
            "wall_time": wall_time,
            "early_stopped": False,
            "error": str(e),
        }


def scine_worker_sample_diagnostics(predict_fn, payload) -> Dict[str, Any]:
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
    
    result = run_single_sample_diagnostics(
        predict_fn, 
        start_coords, 
        atomic_nums, 
        params, 
        n_steps,
        sample_id=sample_id,
        formula=str(formula),
        out_dir=out_dir
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
    samples = []

    for i, batch in enumerate(dataloader):
        if i >= max_samples:
            break
        if i % 10 == 0:
            print(f"    [Sample {i}/{max_samples}]", file=sys.stderr, end="\r")
        
        # Pass out_dir to payload so worker knows where to save diagnostics
        payload = (i, batch, params, n_steps, start_from, noise_seed, out_dir)
        samples.append((i, payload))

    # Note: run_batch_parallel handles the distribution to workers
    results = run_batch_parallel(samples, processor)

    n_samples = len(results)
    n_success = sum(1 for r in results if r["success"])
    n_errors = sum(1 for r in results if r["error"] is not None)
    n_early_stopped = sum(1 for r in results if r.get("early_stopped", False))

    steps_when_success = [r["steps_to_ts"] for r in results if r["steps_to_ts"] is not None]
    escape_cycles_list = [r["escape_cycles"] for r in results if r["error"] is None]
    wall_times = [r["wall_time"] for r in results]

    final_neg_vibs = [r["final_neg_vib"] for r in results if r["error"] is None]
    neg_vib_counts = {}
    for v in final_neg_vibs:
        neg_vib_counts[v] = neg_vib_counts.get(v, 0) + 1

    return {
        "n_samples": n_samples,
        "n_success": n_success,
        "n_errors": n_errors,
        "n_early_stopped": n_early_stopped,
        "success_rate": n_success / max(n_samples, 1),
        "mean_steps_when_success": np.mean(steps_when_success) if steps_when_success else float("nan"),
        "mean_escape_cycles": np.mean(escape_cycles_list) if escape_cycles_list else float("nan"),
        "mean_wall_time": np.mean(wall_times) if wall_times else float("nan"),
        "total_wall_time": sum(wall_times),
        "neg_vib_counts": neg_vib_counts,
        "results": results,
    }


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
    
    
# ... (Util logger code could be reused but for brevity we'll skip or simple copy if essential) ...
# For now, minimal utilization logging to focus on the task.

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel SCINE Multi-Mode Eckart-MW with Diagnostics"
    )

    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--scine-functional", type=str, default="DFTB0")
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--start-from", type=str, default="midpoint_rt")
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--threads-per-worker", type=int, default=None)

    # GAD params
    parser.add_argument("--dt", type=float, default=BASELINE_PARAMS["dt"])
    parser.add_argument("--dt-control", type=str, default=BASELINE_PARAMS["dt_control"])
    parser.add_argument("--dt-min", type=float, default=BASELINE_PARAMS["dt_min"])
    parser.add_argument("--dt-max", type=float, default=BASELINE_PARAMS["dt_max"])
    parser.add_argument("--max-atom-disp", type=float, default=BASELINE_PARAMS["max_atom_disp"])
    parser.add_argument("--plateau-patience", type=int, default=BASELINE_PARAMS["plateau_patience"])
    parser.add_argument("--plateau-boost", type=float, default=BASELINE_PARAMS["plateau_boost"])
    parser.add_argument("--plateau-shrink", type=float, default=BASELINE_PARAMS["plateau_shrink"])
    parser.add_argument("--escape-disp-threshold", type=float, default=BASELINE_PARAMS["escape_disp_threshold"])
    parser.add_argument("--escape-window", type=int, default=BASELINE_PARAMS["escape_window"])
    parser.add_argument("--escape-neg-vib-std", type=float, default=BASELINE_PARAMS["escape_neg_vib_std"])
    parser.add_argument("--escape-delta", type=float, default=BASELINE_PARAMS["escape_delta"])
    parser.add_argument("--min-interatomic-dist", type=float, default=BASELINE_PARAMS["min_interatomic_dist"])
    parser.add_argument("--max-escape-cycles", type=int, default=BASELINE_PARAMS["max_escape_cycles"])
    parser.add_argument("--tr-threshold", type=float, default=BASELINE_PARAMS["tr_threshold"])
    
    parser.add_argument("--stop-at-ts", dest="stop_at_ts", action="store_true")
    parser.add_argument("--no-stop-at-ts", dest="stop_at_ts", action="store_false")
    parser.set_defaults(stop_at_ts=BASELINE_PARAMS["stop_at_ts"])

    adaptive_group = parser.add_mutually_exclusive_group()
    adaptive_group.add_argument("--adaptive-delta", dest="adaptive_delta", action="store_true")
    adaptive_group.add_argument("--no-adaptive-delta", dest="adaptive_delta", action="store_false")
    parser.set_defaults(adaptive_delta=BASELINE_PARAMS["adaptive_delta"])

    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Ensure diagnostics dir exists
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
        "plateau_patience": args.plateau_patience,
        "plateau_boost": args.plateau_boost,
        "plateau_shrink": args.plateau_shrink,
        "escape_disp_threshold": args.escape_disp_threshold,
        "escape_window": args.escape_window,
        "escape_neg_vib_std": args.escape_neg_vib_std,
        "escape_delta": args.escape_delta,
        "adaptive_delta": args.adaptive_delta,
        "min_interatomic_dist": args.min_interatomic_dist,
        "max_escape_cycles": args.max_escape_cycles,
        "stop_at_ts": args.stop_at_ts,
        "tr_threshold": args.tr_threshold,
    })

    print(f"Starting Parallel GAD Diagnostics:")
    print(f"  Workers: {args.n_workers} (Threads/worker: {threads_per_worker})")
    print(f"  Output: {args.out_dir}")

    # Initialize Processor with Diagnostic Worker
    processor = ParallelSCINEProcessor(
        functional=args.scine_functional,
        threads_per_worker=threads_per_worker,
        n_workers=args.n_workers,
        worker_fn=scine_worker_sample_diagnostics,
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
        print("SCINE Parallel Diagnostics Completed")
        print("=" * 70)
        print(f"  Samples: {metrics['n_samples']}")
        print(f"  Success: {metrics['n_success']} ({metrics['success_rate']*100:.1f}%)")
        
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        results_path = Path(args.out_dir) / f"parallel_diagnostics_{job_id}_summary.json"
        
        # Save summary
        with open(results_path, "w") as f:
            json.dump({
                "job_id": job_id,
                "params": params,
                "metrics": metrics
            }, f, indent=2, default=str)
            
        print(f"Summary saved to: {results_path}")

    finally:
        processor.close()


if __name__ == "__main__":
    main()
