#!/usr/bin/env python
"""Fixed-parameter SCINE Multi-Mode Eckart-MW (parallel)."""

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

from .multi_mode_eckartmw import run_multi_mode_escape
from ..dependencies.common_utils import Transition1xDataset, UsePos, parse_starting_geometry
from ..parallel.scine_parallel import ParallelSCINEProcessor
from ..parallel.utils import run_batch_parallel

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
    "project_gradient_and_v": False,
}


def run_single_sample(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    params: Dict[str, Any],
    n_steps: int,
) -> Dict[str, Any]:
    t0 = time.time()

    try:
        with SuppressStdout():
            out_dict, aux = run_multi_mode_escape(
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
                hip_vib_mode=params["hip_vib_mode"],
                hip_rigid_tol=params["hip_rigid_tol"],
                hip_eigh_device=params["hip_eigh_device"],
                escape_neg_vib_std=params["escape_neg_vib_std"],
                escape_delta=params["escape_delta"],
                adaptive_delta=params["adaptive_delta"],
                min_interatomic_dist=params["min_interatomic_dist"],
                max_escape_cycles=params["max_escape_cycles"],
                profile_every=0,
                early_stop_patience=params.get("early_stop_patience", 0),
                early_stop_min_steps=params.get("early_stop_min_steps", 100),
                project_gradient_and_v=params.get("project_gradient_and_v", False),
            )
        wall_time = time.time() - t0

        final_neg_vib = out_dict.get("final_neg_vibrational", -1)
        steps_taken = out_dict.get("steps_taken", n_steps)
        steps_to_ts = aux.get("steps_to_ts")
        escape_cycles = aux.get("escape_cycles_used", 0)
        early_stopped = aux.get("early_stopped", False)

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
    result = run_single_sample(predict_fn, start_coords, atomic_nums, params, n_steps)
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
) -> Dict[str, Any]:
    samples = []

    for i, batch in enumerate(dataloader):
        if i >= max_samples:
            break
        if i % 10 == 0:
            print(f"    [Sample {i}/{max_samples}]", file=sys.stderr, end="\r")
        payload = (i, batch, params, n_steps, start_from, noise_seed)
        samples.append((i, payload))

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


def _get_cpu_stats() -> str:
    try:
        import psutil

        cpu_pct = psutil.cpu_percent(interval=None)
        mem_pct = psutil.virtual_memory().percent
        return f"cpu={cpu_pct:.1f}% mem={mem_pct:.1f}%"
    except Exception:
        load1, load5, load15 = os.getloadavg()
        cpu_count = os.cpu_count() or 1
        return f"load1={load1:.2f} load5={load5:.2f} load15={load15:.2f} cores={cpu_count}"


def _get_gpu_stats() -> str:
    try:
        import subprocess

        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not output:
            return "gpu=none"
        lines = []
        for line in output.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            idx, gpu_util, mem_util, mem_used, mem_total = parts[:5]
            lines.append(
                f"gpu{idx}={gpu_util}% mem={mem_util}% vram={mem_used}/{mem_total}MiB"
            )
        return " | ".join(lines) if lines else "gpu=none"
    except Exception:
        return "gpu=unavailable"


def start_util_logger(interval_s: int, stop_event: Event) -> Optional[Thread]:
    if interval_s <= 0:
        return None

    def _loop() -> None:
        while not stop_event.is_set():
            cpu_stats = _get_cpu_stats()
            gpu_stats = _get_gpu_stats()
            print(f"[UTIL] {cpu_stats} | {gpu_stats}", file=sys.stderr)
            stop_event.wait(interval_s)

    thread = Thread(target=_loop, daemon=True)
    thread.start()
    return thread


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-parameter SCINE Multi-Mode Eckart-MW (parallel)"
    )

    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--scine-functional", type=str, default="DFTB0")

    parser.add_argument("--n-steps", type=int, default=4000)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--start-from", type=str, default="midpoint_rt_noise2.0A")
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--threads-per-worker", type=int, default=None)
    parser.add_argument(
        "--util-log-every",
        type=int,
        default=0,
        help="Log CPU/GPU utilization every N seconds (0 disables).",
    )

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
    parser.add_argument("--stop-at-ts", dest="stop_at_ts", action="store_true")
    parser.add_argument("--no-stop-at-ts", dest="stop_at_ts", action="store_false")
    parser.set_defaults(stop_at_ts=BASELINE_PARAMS["stop_at_ts"])

    adaptive_group = parser.add_mutually_exclusive_group()
    adaptive_group.add_argument("--adaptive-delta", dest="adaptive_delta", action="store_true")
    adaptive_group.add_argument("--no-adaptive-delta", dest="adaptive_delta", action="store_false")
    parser.set_defaults(adaptive_delta=BASELINE_PARAMS["adaptive_delta"])

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="noisy-multi-mode-eckartmw")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument(
        "--project-gradient-and-v",
        action="store_true",
        help="Project gradient and guide vector v (Eckart-MW full projection).",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    total_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1))
    if args.threads_per_worker is None:
        threads_per_worker = max(1, total_cpus // max(args.n_workers, 1))
    else:
        threads_per_worker = args.threads_per_worker

    params = BASELINE_PARAMS.copy()
    params.update(
        {
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
            "project_gradient_and_v": bool(args.project_gradient_and_v),
        }
    )

    run_name = args.wandb_name or f"scine-multi-mode-parallel-{os.environ.get('SLURM_JOB_ID', 'local')}"

    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "n_steps": args.n_steps,
                "max_samples": args.max_samples,
                "start_from": args.start_from,
                "noise_seed": args.noise_seed,
                "scine_functional": args.scine_functional,
                "n_workers": args.n_workers,
                "threads_per_worker": threads_per_worker,
                "project_gradient_and_v": args.project_gradient_and_v,
                "params": params,
            },
            tags=["scine", "multi-mode-eckartmw", "parallel"],
        )

    processor = ParallelSCINEProcessor(
        functional=args.scine_functional,
        threads_per_worker=threads_per_worker,
        n_workers=args.n_workers,
        worker_fn=scine_worker_sample,
    )
    processor.start()
    util_stop_event = Event()
    util_thread = start_util_logger(args.util_log_every, util_stop_event)

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
        )

        print("\n" + "=" * 70)
        print("SCINE Multi-Mode Eckart-MW (Parallel) Results")
        print("=" * 70)
        print(f"  Samples: {metrics['n_samples']}")
        print(f"  Success (index-1): {metrics['n_success']} ({metrics['success_rate']*100:.1f}%)")
        print(f"  Errors: {metrics['n_errors']}")
        print(f"  Mean steps when success: {metrics['mean_steps_when_success']:.1f}")
        print(f"  Mean escape cycles: {metrics['mean_escape_cycles']:.2f}")
        print(f"  Total wall time: {metrics['total_wall_time']:.1f}s")
        print(f"  Neg vib distribution: {metrics['neg_vib_counts']}")
        print("=" * 70 + "\n")

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        results_path = Path(args.out_dir) / f"scine_multi_mode_parallel_{job_id}_results.json"
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
                        "threads_per_worker": threads_per_worker,
                        "split": args.split,
                    },
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        print(f"Results saved to: {results_path}")

        if args.wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "success_rate": metrics["success_rate"],
                    "n_success": metrics["n_success"],
                    "n_samples": metrics["n_samples"],
                    "n_errors": metrics["n_errors"],
                    "mean_steps_when_success": metrics["mean_steps_when_success"],
                    "mean_escape_cycles": metrics["mean_escape_cycles"],
                    "total_wall_time": metrics["total_wall_time"],
                }
            )
            wandb.finish()
    finally:
        util_stop_event.set()
        if util_thread is not None:
            util_thread.join(timeout=1.0)
        processor.close()


if __name__ == "__main__":
    main()
