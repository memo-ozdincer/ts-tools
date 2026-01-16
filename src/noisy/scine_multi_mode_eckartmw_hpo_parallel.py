#!/usr/bin/env python
"""Bayesian Hyperparameter Optimization for SCINE Multi-Mode Eckart-MW (parallel)."""

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


class SuppressStdout:
    """Context manager to suppress stdout (for noisy library outputs)."""

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout


import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

from .multi_mode_eckartmw import run_multi_mode_escape
from ..dependencies.common_utils import Transition1xDataset, UsePos, parse_starting_geometry
from ..parallel.scine_parallel import ParallelSCINEProcessor
from ..parallel.utils import run_batch_parallel

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


BASELINE_PARAMS = {
    "method": "euler",
    "dt": 0.0021305320115760995,
    "dt_control": "neg_eig_plateau",
    "dt_min": 1e-6,
    "dt_max": 0.02292394234778395,
    "max_atom_disp": 0.43218830360353944,
    "plateau_patience": 9,
    "plateau_boost": 1.8080728236469248,
    "plateau_shrink": 0.4359629239850771,
    "escape_disp_threshold": 0.00019284383743673793,
    "escape_window": 27,
    "escape_neg_vib_std": 0.8596534261919356,
    "escape_delta": 0.2155875777745253,
    "adaptive_delta": False,
    "min_interatomic_dist": 0.4121103074412086,
    "max_escape_cycles": 1000,
    "stop_at_ts": True,
    "ts_eps": 1e-5,
    "hip_vib_mode": "projected",
    "hip_rigid_tol": 1e-6,
    "hip_eigh_device": "cpu",
    "early_stop_patience": 500,
    "early_stop_min_steps": 200,
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


def compute_objective(batch_metrics: Dict[str, Any]) -> float:
    success_rate = batch_metrics["success_rate"]
    mean_steps = batch_metrics["mean_steps_when_success"]

    if np.isfinite(mean_steps) and mean_steps > 0:
        speed_score = 1000.0 / mean_steps
    else:
        speed_score = 0.0

    score = success_rate * 1.0 + speed_score * 0.01
    return -score


def sample_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    params = BASELINE_PARAMS.copy()

    params["dt"] = trial.suggest_float("dt", 0.0005, 0.005, log=True)
    params["dt_max"] = trial.suggest_float("dt_max", 0.01, 0.1, log=True)
    params["max_atom_disp"] = trial.suggest_float("max_atom_disp", 0.1, 0.5)
    params["plateau_patience"] = trial.suggest_int("plateau_patience", 3, 20)
    params["plateau_boost"] = trial.suggest_float("plateau_boost", 1.2, 3.0)
    params["plateau_shrink"] = trial.suggest_float("plateau_shrink", 0.3, 0.7)
    params["escape_disp_threshold"] = trial.suggest_float("escape_disp_threshold", 1e-4, 1e-3, log=True)
    params["escape_window"] = trial.suggest_int("escape_window", 10, 50)
    params["escape_neg_vib_std"] = trial.suggest_float("escape_neg_vib_std", 0.2, 1.0)
    params["escape_delta"] = trial.suggest_float("escape_delta", 0.05, 0.3)
    params["adaptive_delta"] = trial.suggest_categorical("adaptive_delta", [True, False])
    params["min_interatomic_dist"] = trial.suggest_float("min_interatomic_dist", 0.3, 0.7)

    return params


def run_verification(
    processor: ParallelSCINEProcessor,
    dataloader,
    n_steps: int,
    max_samples: int,
    start_from: str,
    noise_seed: Optional[int],
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("VERIFICATION RUN: Using EXACT baseline parameters from SLURM script")
    print("=" * 70)
    print(f"Parameters: {json.dumps(BASELINE_PARAMS, indent=2)}")
    print("=" * 70 + "\n")

    metrics = run_batch(
        processor,
        dataloader,
        BASELINE_PARAMS,
        n_steps,
        max_samples,
        start_from,
        noise_seed,
    )

    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)
    print(f"  Samples: {metrics['n_samples']}")
    print(f"  Success (index-1): {metrics['n_success']} ({metrics['success_rate']*100:.1f}%)")
    print(f"  Errors: {metrics['n_errors']}")
    print(f"  Mean steps when success: {metrics['mean_steps_when_success']:.1f}")
    print(f"  Mean escape cycles: {metrics['mean_escape_cycles']:.2f}")
    print(f"  Total wall time: {metrics['total_wall_time']:.1f}s")
    print(f"  Neg vib distribution: {metrics['neg_vib_counts']}")
    print("=" * 70 + "\n")

    return metrics


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
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not output:
            return "gpu=none"
        parts = output.split(",")
        gpu_util = parts[0].strip()
        mem_util = parts[1].strip()
        mem_used = parts[2].strip()
        mem_total = parts[3].strip()
        return f"gpu={gpu_util}% mem={mem_util}% vram={mem_used}/{mem_total}MiB"
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


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian HPO for SCINE Multi-Mode Eckart-MW (parallel)"
    )

    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--scine-functional", type=str, default="DFTB0")

    parser.add_argument("--n-steps", type=int, default=4000)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--start-from", type=str, default="midpoint_rt_noise1.0A")
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

    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-startup-trials", type=int, default=5)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from existing study")
    parser.add_argument("--skip-verification", action="store_true")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="scine-multi-mode-hpo")
    parser.add_argument("--wandb-entity", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cpu"

    total_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1))
    if args.threads_per_worker is None:
        threads_per_worker = max(1, total_cpus // max(args.n_workers, 1))
    else:
        threads_per_worker = args.threads_per_worker

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    study_name = args.study_name or f"scine_multi_mode_hpo_{job_id}"

    db_path = Path(args.out_dir).resolve() / f"{study_name}.db"
    storage_url = f"sqlite:///{db_path}"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"HPO-{study_name}",
            config={
                "study_name": study_name,
                "n_trials": args.n_trials,
                "n_steps": args.n_steps,
                "max_samples": args.max_samples,
                "start_from": args.start_from,
                "noise_seed": args.noise_seed,
                "scine_functional": args.scine_functional,
                "baseline_params": BASELINE_PARAMS,
                "n_workers": args.n_workers,
                "threads_per_worker": threads_per_worker,
            },
            tags=["hpo", "scine", "multi-mode-eckartmw", "parallel"],
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
        VERIFICATION_SAMPLES = 5
        if not args.skip_verification:
            print(
                f"\n[Verification] Running on {VERIFICATION_SAMPLES} samples "
                f"(HPO will use {args.max_samples})",
                file=sys.stderr,
            )
            verification_dataloader = create_dataloader(
                args.h5_path,
                args.split,
                VERIFICATION_SAMPLES,
            )
            verification_metrics = run_verification(
                processor,
                verification_dataloader,
                args.n_steps,
                VERIFICATION_SAMPLES,
                args.start_from,
                args.noise_seed,
            )
            if args.wandb and WANDB_AVAILABLE:
                wandb.log(
                    {
                        "verification/success_rate": verification_metrics["success_rate"],
                        "verification/n_success": verification_metrics["n_success"],
                        "verification/mean_steps": verification_metrics["mean_steps_when_success"],
                        "verification/mean_escape_cycles": verification_metrics["mean_escape_cycles"],
                        "verification/total_time": verification_metrics["total_wall_time"],
                    }
                )

        sampler = TPESampler(
            n_startup_trials=args.n_startup_trials,
            seed=42,
        )

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=args.resume,
            direction="minimize",
            sampler=sampler,
        )

        if not args.resume and len(study.trials) == 0:
            baseline_trial_params = {
                "dt": BASELINE_PARAMS["dt"],
                "dt_max": BASELINE_PARAMS["dt_max"],
                "max_atom_disp": BASELINE_PARAMS["max_atom_disp"],
                "plateau_patience": BASELINE_PARAMS["plateau_patience"],
                "plateau_boost": BASELINE_PARAMS["plateau_boost"],
                "plateau_shrink": BASELINE_PARAMS["plateau_shrink"],
                "escape_disp_threshold": BASELINE_PARAMS["escape_disp_threshold"],
                "escape_window": BASELINE_PARAMS["escape_window"],
                "escape_neg_vib_std": BASELINE_PARAMS["escape_neg_vib_std"],
                "escape_delta": BASELINE_PARAMS["escape_delta"],
                "adaptive_delta": BASELINE_PARAMS["adaptive_delta"],
                "min_interatomic_dist": BASELINE_PARAMS["min_interatomic_dist"],
            }
            try:
                study.enqueue_trial(baseline_trial_params)
                print(">>> Enqueued baseline trial")
            except TypeError as e:
                print(f"WARNING: Could not enqueue baseline trial: {e}")

        def objective(trial: optuna.Trial) -> float:
            params = sample_hyperparameters(trial)

            print(f"\n>>> Trial {trial.number}: Running with params...")
            for k, v in params.items():
                if k not in BASELINE_PARAMS or params[k] != BASELINE_PARAMS[k]:
                    print(f"    {k}: {v}")

            dataloader_fresh = create_dataloader(
                args.h5_path,
                args.split,
                args.max_samples,
            )

            metrics = run_batch(
                processor,
                dataloader_fresh,
                params,
                args.n_steps,
                args.max_samples,
                args.start_from,
                args.noise_seed,
            )

            score = compute_objective(metrics)

            print(f"    Success rate: {metrics['success_rate']*100:.1f}%")
            print(f"    Mean steps: {metrics['mean_steps_when_success']:.1f}")
            print(f"    Score: {-score:.4f}")

            print(
                f"[DB] Trial {trial.number} completed: score={-score:.4f}, "
                f"success_rate={metrics['success_rate']*100:.1f}%",
                file=sys.stderr,
            )

            trial.set_user_attr("success_rate", metrics["success_rate"])
            trial.set_user_attr("n_success", metrics["n_success"])
            trial.set_user_attr("n_samples", metrics["n_samples"])
            trial.set_user_attr("n_errors", metrics["n_errors"])
            trial.set_user_attr("n_early_stopped", metrics["n_early_stopped"])
            trial.set_user_attr(
                "mean_steps_when_success",
                float(metrics["mean_steps_when_success"])
                if np.isfinite(metrics["mean_steps_when_success"])
                else None,
            )
            trial.set_user_attr(
                "mean_escape_cycles",
                float(metrics["mean_escape_cycles"])
                if np.isfinite(metrics["mean_escape_cycles"])
                else None,
            )
            trial.set_user_attr("mean_wall_time", metrics["mean_wall_time"])
            trial.set_user_attr("total_wall_time", metrics["total_wall_time"])
            trial.set_user_attr("neg_vib_distribution", metrics["neg_vib_counts"])

            if args.wandb and WANDB_AVAILABLE:
                log_dict = {
                    "trial/success_rate": metrics["success_rate"],
                    "trial/n_success": metrics["n_success"],
                    "trial/mean_steps": metrics["mean_steps_when_success"],
                    "trial/mean_escape_cycles": metrics["mean_escape_cycles"],
                    "trial/total_time": metrics["total_wall_time"],
                    "trial/score": -score,
                    "trial/number": trial.number,
                }
                for k, v in params.items():
                    if isinstance(v, (int, float, bool)):
                        log_dict[f"hparams/{k}"] = v
                wandb.log(log_dict)

            return score

        print(f"\n>>> Starting HPO with {args.n_trials} trials...", file=sys.stderr)
        print(f"    Study: {study_name}", file=sys.stderr)
        print(f"    Database: {db_path}", file=sys.stderr)

        def exception_callback(study, frozen_trial):
            if frozen_trial.state != optuna.trial.TrialState.FAIL:
                return
            print(
                f"[ERROR] Trial {frozen_trial.number} failed with exception:",
                file=sys.stderr,
            )
            print(
                f"        {frozen_trial.user_attrs.get('exception', 'Unknown error')}",
                file=sys.stderr,
            )
            exc_info = frozen_trial.system_attrs.get("fail_reason", "No traceback available")
            print(f"        {exc_info}", file=sys.stderr)

        try:
            study.optimize(
                objective,
                n_trials=args.n_trials,
                show_progress_bar=True,
                callbacks=[exception_callback],
            )
        except KeyboardInterrupt:
            print("\n>>> HPO interrupted by user. Saving results...", file=sys.stderr)
        except Exception as e:
            print(f"\n[ERROR] HPO failed with exception: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
            print("Attempting to save partial results...", file=sys.stderr)

        print(f"\n[DB] After optimization:", file=sys.stderr)
        print(f"[DB]   Database exists: {db_path.exists()}", file=sys.stderr)
        if db_path.exists():
            print(f"[DB]   Database size: {db_path.stat().st_size} bytes", file=sys.stderr)
        print(f"[DB]   Number of trials in study: {len(study.trials)}", file=sys.stderr)

        print("\n" + "=" * 70)
        print("BEST TRIAL")
        print("=" * 70)
        best_trial = study.best_trial
        print(f"  Trial number: {best_trial.number}")
        print(f"  Score: {-best_trial.value:.4f}")
        print(f"  Parameters:")
        for k, v in best_trial.params.items():
            print(f"    {k}: {v}")
        print("=" * 70)

        results_path = Path(args.out_dir) / f"{study_name}_results.json"
        results = {
            "study_name": study_name,
            "best_trial": {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
            },
            "n_trials": len(study.trials),
            "baseline_params": BASELINE_PARAMS,
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        if args.wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "best/trial_number": best_trial.number,
                    "best/score": -best_trial.value,
                    **{f"best/{k}": v for k, v in best_trial.params.items()},
                }
            )

            try:
                artifact = wandb.Artifact(
                    name=study_name,
                    type="optuna-study",
                    description=f"SCINE Multi-Mode HPO Optuna study: {study_name}",
                    metadata={
                        "n_trials": len(study.trials),
                        "best_score": -best_trial.value,
                    },
                )
                artifact.add_file(str(db_path))
                wandb.log_artifact(artifact)
                print(f"[W&B] Logged artifact: {study_name}")
            except Exception as e:
                print(f"[W&B] Failed to log artifact: {e}")

            wandb.finish()

        print("\nHPO complete!")
    finally:
        util_stop_event.set()
        if util_thread is not None:
            util_thread.join(timeout=1.0)
        processor.close()


if __name__ == "__main__":
    main()
