#!/usr/bin/env python
"""Parallel SCINE runner for kick strategy comparison."""

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
from src.parallel.scine_parallel import ParallelSCINEProcessor
from src.parallel.utils import run_batch_parallel
from src.noisy.v2_tests.kick_experiments.kick_strategies import KICK_STRATEGIES
from src.noisy.v2_tests.runners.run_kick_comparison import run_gad_with_kick_strategy


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


def _parse_strategies(raw: Optional[str]) -> List[str]:
    if raw is None or raw.strip() == "" or raw.strip().lower() == "all":
        return list(KICK_STRATEGIES.keys())
    return [s.strip() for s in raw.split(",") if s.strip()]


def run_single_sample(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    strategies: List[str],
    params: Dict[str, Any],
    *,
    sample_id: str,
    formula: str,
) -> Dict[str, Any]:
    t0 = time.time()
    results = {}
    for strategy in strategies:
        out = run_gad_with_kick_strategy(
            predict_fn,
            coords.clone(),
            atomic_nums,
            strategy,
            n_steps=params["n_steps"],
            dt=params["dt"],
            dt_min=params["dt_min"],
            dt_max=params["dt_max"],
            max_atom_disp=params["max_atom_disp"],
            escape_window=params["escape_window"],
            escape_disp_threshold=params["escape_disp_threshold"],
            escape_neg_vib_std=params["escape_neg_vib_std"],
            escape_delta=params["escape_delta"],
            max_escape_cycles=params["max_escape_cycles"],
            min_interatomic_dist=params["min_interatomic_dist"],
            force_escape_after=params.get("force_escape_after"),
            ts_eps=params["ts_eps"],
            stop_at_ts=params["stop_at_ts"],
            tr_threshold=params["tr_threshold"],
            project_gradient_and_v=params["project_gradient_and_v"],
            sample_id=sample_id,
            formula=formula,
            log_dir=params.get("log_dir"),
        )
        results[strategy] = {
            "success": out.success,
            "final_morse_index": out.final_morse_index,
            "total_steps": out.total_steps,
            "escape_cycles": out.escape_cycles,
            "wall_time": out.wall_time,
            "error": out.error,
        }

    return {
        "results": results,
        "wall_time": time.time() - t0,
    }


def scine_worker_sample(predict_fn, payload) -> Dict[str, Any]:
    sample_idx, batch, strategies, params, start_from, noise_seed = payload
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
        strategies,
        params,
        sample_id=f"sample_{sample_idx:03d}",
        formula=str(formula),
    )
    result["sample_idx"] = sample_idx
    result["formula"] = getattr(batch, "formula", "")
    return result


def aggregate_results(all_results: List[Dict[str, Any]], strategies: List[str]) -> Dict[str, Any]:
    aggregated: Dict[str, Any] = {}
    for strategy in strategies:
        strat_results = [r["results"][strategy] for r in all_results if strategy in r["results"]]
        successes = [r for r in strat_results if r.get("success")]
        aggregated[strategy] = {
            "n_samples": len(strat_results),
            "n_success": len(successes),
            "success_rate": len(successes) / len(strat_results) if strat_results else 0.0,
            "mean_steps_success": float(np.mean([r["total_steps"] for r in successes])) if successes else float("nan"),
            "mean_escapes_success": float(np.mean([r["escape_cycles"] for r in successes])) if successes else float("nan"),
            "mean_wall_time": float(np.mean([r["wall_time"] for r in strat_results])) if strat_results else float("nan"),
            "final_index_distribution": {},
        }

        for r in strat_results:
            idx = r.get("final_morse_index", -1)
            aggregated[strategy]["final_index_distribution"][idx] = aggregated[strategy]["final_index_distribution"].get(idx, 0) + 1

    return aggregated


def run_batch(
    processor: ParallelSCINEProcessor,
    dataloader,
    strategies: List[str],
    params: Dict[str, Any],
    max_samples: int,
    start_from: str,
    noise_seed: Optional[int],
) -> Dict[str, Any]:
    samples = []
    for i, batch in enumerate(dataloader):
        if i >= max_samples:
            break
        payload = (i, batch, strategies, params, start_from, noise_seed)
        samples.append((i, payload))

    results = run_batch_parallel(samples, processor)
    return {
        "samples": results,
        "aggregated": aggregate_results(results, strategies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kick strategy comparison (parallel, SCINE)")
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

    parser.add_argument("--strategies", type=str, default="all", help="Comma-separated list or 'all'")

    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--dt-min", type=float, default=1e-6)
    parser.add_argument("--dt-max", type=float, default=0.08)
    parser.add_argument("--max-atom-disp", type=float, default=0.35)

    parser.add_argument("--escape-window", type=int, default=10)
    parser.add_argument("--escape-disp-threshold", type=float, default=1e-4)
    parser.add_argument("--escape-neg-vib-std", type=float, default=0.5)
    parser.add_argument("--escape-delta", type=float, default=0.3)
    parser.add_argument("--max-escape-cycles", type=int, default=500)
    parser.add_argument("--min-interatomic-dist", type=float, default=0.5)
    parser.add_argument("--force-escape-after", type=int, default=None)

    parser.add_argument("--ts-eps", type=float, default=1e-5)
    parser.add_argument("--stop-at-ts", action="store_true")
    parser.add_argument("--no-stop-at-ts", dest="stop_at_ts", action="store_false")
    parser.set_defaults(stop_at_ts=True)
    parser.add_argument("--tr-threshold", type=float, default=1e-6)
    parser.add_argument(
        "--project-gradient-and-v",
        action="store_true",
        default=False,
        help="Project gradient and guide vector into vibrational subspace (prevents TR leakage).",
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    diag_dir = Path(args.out_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    strategies = _parse_strategies(args.strategies)

    params = {
        "n_steps": args.n_steps,
        "dt": args.dt,
        "dt_min": args.dt_min,
        "dt_max": args.dt_max,
        "max_atom_disp": args.max_atom_disp,
        "escape_window": args.escape_window,
        "escape_disp_threshold": args.escape_disp_threshold,
        "escape_neg_vib_std": args.escape_neg_vib_std,
        "escape_delta": args.escape_delta,
        "max_escape_cycles": args.max_escape_cycles,
        "min_interatomic_dist": args.min_interatomic_dist,
        "force_escape_after": args.force_escape_after,
        "ts_eps": args.ts_eps,
        "stop_at_ts": args.stop_at_ts,
        "tr_threshold": args.tr_threshold,
        "project_gradient_and_v": bool(args.project_gradient_and_v),
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
        results = run_batch(
            processor,
            dataloader,
            strategies,
            params,
            args.max_samples,
            args.start_from,
            args.noise_seed,
        )

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        results_path = Path(args.out_dir) / f"kick_comparison_parallel_{job_id}_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "strategies": strategies,
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
                        "tr_threshold": args.tr_threshold,
                        "project_gradient_and_v": bool(args.project_gradient_and_v),
                    },
                    "aggregated": results["aggregated"],
                    "samples": results["samples"],
                },
                f,
                indent=2,
            )
        print(f"Results saved to: {results_path}")
    finally:
        processor.close()


if __name__ == "__main__":
    main()
