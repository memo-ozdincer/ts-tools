#!/usr/bin/env python
"""Bayesian Hyperparameter Optimization for SCINE GAD (plain, parallel)."""

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

from .run_gad_baselines_hpo_parallel import create_dataloader, run_batch, scine_worker_sample
from ..parallel.scine_parallel import ParallelSCINEProcessor

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


BASELINE_PARAMS: Dict[str, Any] = {
    "dt": 1.0e-5,
    "dt_control": "adaptive",
    "dt_min": 1.0e-6,
    "dt_max": 8.0e-2,
    "max_atom_disp": 0.35,
    "min_interatomic_dist": 0.5,
    "ts_eps": 1.0e-5,
    "tr_threshold": 1.0e-6,
    "stop_at_ts": True,
    "track_mode": False,
    "log_dir": None,
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

    params["dt_control"] = trial.suggest_categorical("dt_control", ["adaptive", "fixed"])
    params["dt_min"] = trial.suggest_float("dt_min", 1e-7, 5e-5, log=True)
    params["dt_max"] = trial.suggest_float("dt_max", 5e-3, 2e-1, log=True)
    if params["dt_max"] <= params["dt_min"]:
        params["dt_max"] = params["dt_min"] * 10.0

    params["dt"] = trial.suggest_float("dt", params["dt_min"], params["dt_max"], log=True)
    params["max_atom_disp"] = trial.suggest_float("max_atom_disp", 0.1, 0.7)
    params["min_interatomic_dist"] = trial.suggest_float("min_interatomic_dist", 0.3, 0.8)
    params["ts_eps"] = trial.suggest_float("ts_eps", 1e-6, 1e-3, log=True)
    params["tr_threshold"] = trial.suggest_float("tr_threshold", 1e-7, 1e-4, log=True)
    params["stop_at_ts"] = trial.suggest_categorical("stop_at_ts", [True, False])

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
    print(f"  Total wall time: {metrics['total_wall_time']:.1f}s")
    print(f"  Neg vib distribution: {metrics['neg_vib_counts']}")
    print("=" * 70 + "\n")

    return metrics


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


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian HPO for SCINE GAD (plain, parallel)"
    )

    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--scine-functional", type=str, default="DFTB0")

    parser.add_argument("--n-steps", type=int, default=15000)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--start-from", type=str, default="midpoint_rt_noise2.0A")
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--n-workers", type=int, default=8)
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
    parser.add_argument("--wandb-project", type=str, default="scine-gad-plain-hpo")
    parser.add_argument("--wandb-entity", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    total_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1))
    if args.threads_per_worker is None:
        threads_per_worker = max(1, total_cpus // max(args.n_workers, 1))
    else:
        threads_per_worker = args.threads_per_worker

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    study_name = args.study_name or f"scine_gad_plain_hpo_{job_id}"

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
            tags=["hpo", "scine", "gad", "plain", "parallel"],
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
                "dt_control": BASELINE_PARAMS["dt_control"],
                "dt_min": BASELINE_PARAMS["dt_min"],
                "dt_max": BASELINE_PARAMS["dt_max"],
                "max_atom_disp": BASELINE_PARAMS["max_atom_disp"],
                "min_interatomic_dist": BASELINE_PARAMS["min_interatomic_dist"],
                "ts_eps": BASELINE_PARAMS["ts_eps"],
                "tr_threshold": BASELINE_PARAMS["tr_threshold"],
                "stop_at_ts": BASELINE_PARAMS["stop_at_ts"],
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
            trial.set_user_attr(
                "mean_steps_when_success",
                float(metrics["mean_steps_when_success"])
                if np.isfinite(metrics["mean_steps_when_success"])
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
                    description=f"SCINE GAD HPO Optuna study: {study_name}",
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
