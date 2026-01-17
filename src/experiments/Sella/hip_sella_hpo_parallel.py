"""HIP Sella Hyperparameter Optimization using Optuna (parallel)."""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler
import torch

warnings.filterwarnings("ignore")


class SuppressStdout:
    """Context manager to suppress stdout (including HIP's edge_vec_0_distance errors)."""

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout


from ...dependencies.common_utils import (
    add_common_args,
    parse_starting_geometry,
    setup_experiment,
)
from ...dependencies.hessian import (
    vibrational_eigvals,
    get_scine_elements_from_predict_output,
)
from ...logging import finish_wandb, init_wandb_run, log_artifact, log_sample, log_summary
from ...parallel.hip_parallel import ParallelHIPProcessor
from ...parallel.utils import run_batch_parallel
from .sella_ts import run_sella_ts

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class HPOConfig:
    """Configuration for a single hyperparameter combination."""

    delta0: float = 0.2
    rho_inc: float = 1.035
    rho_dec: float = 20.0
    sigma_inc: float = 1.3
    sigma_dec: float = 0.85
    fmax: float = 1e-3
    apply_eckart: bool = False

    gamma: float = 0.0
    internal: bool = True
    order: int = 1
    use_exact_hessian: bool = True
    diag_every_n: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_str(self) -> str:
        eckart_str = "eckart" if self.apply_eckart else "noeckart"
        return (
            f"d0{self.delta0:.3f}_rdec{self.rho_dec:.1f}_"
            f"sdec{self.sigma_dec:.2f}_fmax{self.fmax:.0e}_{eckart_str}"
        )


@dataclass
class HPOResult:
    """Results for a single HPO trial."""

    config: HPOConfig
    n_samples: int = 0
    n_sella_converged: int = 0
    n_eigenvalue_ts: int = 0
    n_both: int = 0
    n_sella_only: int = 0

    final_fmax_list: List[float] = field(default_factory=list)
    steps_list: List[int] = field(default_factory=list)
    wall_time_list: List[float] = field(default_factory=list)
    neg_eigval_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def sella_convergence_rate(self) -> float:
        return self.n_sella_converged / self.n_samples if self.n_samples > 0 else 0.0

    @property
    def eigenvalue_ts_rate(self) -> float:
        return self.n_eigenvalue_ts / self.n_samples if self.n_samples > 0 else 0.0

    @property
    def both_rate(self) -> float:
        return self.n_both / self.n_samples if self.n_samples > 0 else 0.0

    @property
    def avg_steps(self) -> float:
        return float(np.mean(self.steps_list)) if self.steps_list else 0.0

    @property
    def avg_wall_time(self) -> float:
        return float(np.mean(self.wall_time_list)) if self.wall_time_list else 0.0

    @property
    def avg_final_fmax(self) -> float:
        return float(np.mean(self.final_fmax_list)) if self.final_fmax_list else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "n_samples": self.n_samples,
            "n_sella_converged": self.n_sella_converged,
            "n_eigenvalue_ts": self.n_eigenvalue_ts,
            "n_both": self.n_both,
            "n_sella_only": self.n_sella_only,
            "sella_convergence_rate": self.sella_convergence_rate,
            "eigenvalue_ts_rate": self.eigenvalue_ts_rate,
            "both_rate": self.both_rate,
            "avg_steps": self.avg_steps,
            "avg_wall_time": self.avg_wall_time,
            "avg_final_fmax": self.avg_final_fmax,
            "hard_sample_indices": list(getattr(self, "hard_sample_indices", [])),
            "neg_eigval_distribution": dict(self.neg_eigval_counts),
        }


HIP_GOOD_STARTING_CONFIG = HPOConfig(
    delta0=0.5,
    rho_dec=50.0,
    sigma_dec=0.9,
    rho_inc=1.035,
    sigma_inc=1.15,
    fmax=1e-3,
    apply_eckart=False,
)


def hip_sella_worker_sample(predict_fn, calculator, payload, device: str) -> Dict[str, Any]:
    sample_idx, batch, config, max_steps, start_from, noise_seed = payload
    batch = batch.to(device)
    atomic_nums = batch.z.detach().cpu().to(device)

    start_coords = parse_starting_geometry(
        start_from,
        batch,
        noise_seed=noise_seed,
        sample_index=sample_idx,
    ).detach().to(device)

    t0 = time.time()
    try:
        with SuppressStdout():
            out_dict, _ = run_sella_ts(
                calculator,
                "hip",
                start_coords,
                atomic_nums,
                fmax=config.fmax,
                max_steps=max_steps,
                internal=config.internal,
                delta0=config.delta0,
                order=config.order,
                device=device,
                save_trajectory=False,
                trajectory_dir=None,
                sample_index=sample_idx,
                logfile=None,
                verbose=False,
                use_exact_hessian=config.use_exact_hessian,
                diag_every_n=config.diag_every_n,
                gamma=config.gamma,
                rho_inc=config.rho_inc,
                rho_dec=config.rho_dec,
                sigma_inc=config.sigma_inc,
                sigma_dec=config.sigma_dec,
                apply_eckart=config.apply_eckart,
            )
        wall_time = time.time() - t0
    except Exception as exc:
        return {
            "sample_idx": sample_idx,
            "error": str(exc),
            "steps_taken": 0,
            "final_fmax": None,
            "sella_converged": False,
            "final_neg": -1,
            "is_ts": False,
            "wall_time": time.time() - t0,
        }

    final_neg = -1
    try:
        final_coords = out_dict["final_coords"].to(device)
        final_out = predict_fn(final_coords, atomic_nums, do_hessian=True, require_grad=False)
        final_scine_elements = get_scine_elements_from_predict_output(final_out)
        final_vib = vibrational_eigvals(
            final_out["hessian"],
            final_coords,
            atomic_nums,
            scine_elements=final_scine_elements,
        )
        final_neg = int((final_vib < 0).sum().item())
    except Exception:
        final_neg = -1

    sella_converged = bool(out_dict.get("converged", False))
    return {
        "sample_idx": sample_idx,
        "error": None,
        "steps_taken": out_dict.get("steps_taken", 0),
        "final_fmax": out_dict.get("final_fmax"),
        "sella_converged": sella_converged,
        "final_neg": final_neg,
        "is_ts": final_neg == 1,
        "wall_time": wall_time,
    }


def _compute_intermediate_score(results: List[Any], max_steps: int) -> float:
    if not results:
        return 0.0
    partial = [r[1] for r in results]
    n_samples = len(partial)
    ts_rate = sum(1 for r in partial if r.get("is_ts")) / n_samples
    sella_rate = sum(1 for r in partial if r.get("sella_converged")) / n_samples
    steps = [r.get("steps_taken", 0) for r in partial if r.get("steps_taken")]
    avg_steps = float(np.mean(steps)) if steps else 0.0
    step_bonus = 0.01 * (1 - avg_steps / max_steps) if avg_steps > 0 else 0.0
    sella_bonus = 0.001 * sella_rate
    return ts_rate + step_bonus + sella_bonus


def run_trial_evaluation_parallel(
    processor: ParallelHIPProcessor,
    dataloader,
    device: str,
    config: HPOConfig,
    max_steps: int,
    max_samples: int,
    start_from: str,
    noise_seed: Optional[int],
    verbose: bool = False,
    trial: Optional[optuna.Trial] = None,
    prune_after_n: int = 10,
    sample_indices: Optional[List[int]] = None,
) -> HPOResult:
    result = HPOResult(config=config)

    all_batches = list(dataloader)
    if sample_indices is not None:
        indices_to_run = [idx for idx in sample_indices if idx < len(all_batches)][
            :max_samples
        ]
    else:
        indices_to_run = list(range(min(max_samples, len(all_batches))))

    samples = []
    for sample_idx in indices_to_run:
        batch = all_batches[sample_idx]
        payload = (sample_idx, batch, config, max_steps, start_from, noise_seed)
        samples.append((sample_idx, payload))

    def intermediate_score_fn(partial_results):
        return _compute_intermediate_score(partial_results, max_steps)

    sample_results = run_batch_parallel(
        samples,
        processor,
        trial=trial,
        prune_after_n=prune_after_n,
        intermediate_score_fn=intermediate_score_fn,
    )

    for sample_res in sample_results:
        if sample_res.get("error") and verbose:
            print(f"[WARN] Sample {sample_res.get('sample_idx')} failed: {sample_res['error']}")

        result.n_samples += 1
        result.steps_list.append(sample_res.get("steps_taken", 0))
        result.wall_time_list.append(sample_res.get("wall_time", 0.0))

        if sample_res.get("final_fmax") is not None:
            result.final_fmax_list.append(sample_res["final_fmax"])

        sella_converged = bool(sample_res.get("sella_converged", False))
        if sella_converged:
            result.n_sella_converged += 1

        final_neg = int(sample_res.get("final_neg", -1))
        result.neg_eigval_counts[final_neg] += 1

        is_ts = bool(sample_res.get("is_ts", False))
        if is_ts:
            result.n_eigenvalue_ts += 1

        if sella_converged and is_ts:
            result.n_both += 1
        elif sella_converged and not is_ts:
            result.n_sella_only += 1

    return result


def compute_score(result: HPOResult, max_steps: int) -> float:
    ts_rate = result.eigenvalue_ts_rate
    step_bonus = 0.01 * (1 - result.avg_steps / max_steps) if result.avg_steps > 0 else 0
    sella_bonus = 0.001 * result.sella_convergence_rate
    return ts_rate + step_bonus + sella_bonus


def run_verification(
    processor: ParallelHIPProcessor,
    dataloader,
    device: str,
    config: HPOConfig,
    max_steps: int,
    max_samples: int,
    start_from: str,
    noise_seed: Optional[int],
) -> HPOResult:
    print("\n" + "=" * 70)
    print("VERIFICATION RUN: Using baseline parameters")
    print("=" * 70)
    print(f"Config: {config.to_str()}")
    print("=" * 70 + "\n")

    result = run_trial_evaluation_parallel(
        processor=processor,
        dataloader=dataloader,
        device=device,
        config=config,
        max_steps=max_steps,
        max_samples=max_samples,
        start_from=start_from,
        noise_seed=noise_seed,
        verbose=True,
        trial=None,
        prune_after_n=max_samples + 1,
        sample_indices=None,
    )

    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)
    print(f"  Samples: {result.n_samples}")
    print(f"  Eigenvalue TS (1 neg): {result.n_eigenvalue_ts} ({result.eigenvalue_ts_rate*100:.1f}%)")
    print(f"  Sella converged: {result.n_sella_converged} ({result.sella_convergence_rate*100:.1f}%)")
    print(f"  Both: {result.n_both} ({result.both_rate*100:.1f}%)")
    print(f"  Avg steps: {result.avg_steps:.1f}")
    print(f"  Avg wall time: {result.avg_wall_time:.2f}s")
    print(f"  Neg eigenvalue distribution: {dict(result.neg_eigval_counts)}")
    print("=" * 70 + "\n")

    return result


def prescreen_samples(
    calculator,
    dataloader,
    device: str,
    config: HPOConfig,
    max_steps: int,
    max_samples: int,
    start_from: str,
    noise_seed: Optional[int],
    verbose: bool = True,
) -> tuple[List[int], List[int]]:
    from ...runners._predict import make_predict_fn_from_calculator

    predict_fn = make_predict_fn_from_calculator(calculator, "hip")
    hard_indices = []
    easy_indices = []

    if verbose:
        print(f"\n{'='*60}")
        print("PRE-SCREENING: Identifying hard samples")
        print(f"Config: {config.to_str()}")
        print(f"{'='*60}")

    for i, batch in enumerate(dataloader):
        if i >= max_samples:
            break

        batch = batch.to(device)
        atomic_nums = batch.z.detach().cpu().to(device)
        start_coords = parse_starting_geometry(
            start_from,
            batch,
            noise_seed=noise_seed,
            sample_index=i,
        ).detach().to(device)

        try:
            with SuppressStdout():
                out_dict, _ = run_sella_ts(
                    calculator,
                    "hip",
                    start_coords,
                    atomic_nums,
                    fmax=config.fmax,
                    max_steps=max_steps,
                    internal=config.internal,
                    delta0=config.delta0,
                    order=config.order,
                    device=device,
                    save_trajectory=False,
                    trajectory_dir=None,
                    sample_index=i,
                    logfile=None,
                    verbose=False,
                    use_exact_hessian=config.use_exact_hessian,
                    diag_every_n=config.diag_every_n,
                    gamma=config.gamma,
                    rho_inc=config.rho_inc,
                    rho_dec=config.rho_dec,
                    sigma_inc=config.sigma_inc,
                    sigma_dec=config.sigma_dec,
                    apply_eckart=config.apply_eckart,
                )

            final_coords = out_dict["final_coords"].to(device)
            final_out = predict_fn(final_coords, atomic_nums, do_hessian=True, require_grad=False)
            final_scine_elements = get_scine_elements_from_predict_output(final_out)
            final_vib = vibrational_eigvals(
                final_out["hessian"],
                final_coords,
                atomic_nums,
                scine_elements=final_scine_elements,
            )
            final_neg = int((final_vib < 0).sum().item())

            is_ts = final_neg == 1
            if is_ts:
                easy_indices.append(i)
                if verbose:
                    print(f"  Sample {i}: EASY (TS found, 1 neg eigenvalue)")
            else:
                hard_indices.append(i)
                if verbose:
                    print(f"  Sample {i}: HARD ({final_neg} neg eigenvalues)")
        except Exception as e:
            hard_indices.append(i)
            if verbose:
                print(f"  Sample {i}: HARD (error: {e})")

    if verbose:
        print(f"\n{'='*60}")
        print("Pre-screening complete:")
        print(
            f"  Easy samples: {len(easy_indices)} "
            f"({100*len(easy_indices)/max(1, len(easy_indices)+len(hard_indices)):.1f}%)"
        )
        print(
            f"  Hard samples: {len(hard_indices)} "
            f"({100*len(hard_indices)/max(1, len(easy_indices)+len(hard_indices)):.1f}%)"
        )
        print(f"{'='*60}\n")

    return hard_indices, easy_indices


def create_objective(
    processor: ParallelHIPProcessor,
    dataloader,
    device: str,
    max_steps: int,
    max_samples: int,
    start_from: str,
    noise_seed: Optional[int],
    verbose: bool,
    use_wandb: bool,
    prune_after_n: int = 10,
    sample_indices: Optional[List[int]] = None,
):
    trial_count = [0]

    def objective(trial: optuna.Trial) -> float:
        trial_count[0] += 1

        delta0 = trial.suggest_float("delta0", 0.15, 0.8, log=True)
        rho_dec = trial.suggest_float("rho_dec", 15.0, 80.0)
        rho_inc = trial.suggest_float("rho_inc", 1.01, 1.1)
        sigma_dec = trial.suggest_float("sigma_dec", 0.75, 0.95)
        sigma_inc = trial.suggest_float("sigma_inc", 1.1, 1.8)
        fmax = trial.suggest_float("fmax", 1e-4, 1e-2, log=True)
        apply_eckart = trial.suggest_categorical("apply_eckart", [True, False])

        config = HPOConfig(
            delta0=delta0,
            rho_inc=rho_inc,
            rho_dec=rho_dec,
            sigma_inc=sigma_inc,
            sigma_dec=sigma_dec,
            fmax=fmax,
            apply_eckart=apply_eckart,
        )

        if verbose:
            print(f"\n[Trial {trial_count[0]}] Testing: {config.to_str()}")

        try:
            result = run_trial_evaluation_parallel(
                processor=processor,
                dataloader=dataloader,
                device=device,
                config=config,
                max_steps=max_steps,
                max_samples=max_samples,
                start_from=start_from,
                noise_seed=noise_seed,
                verbose=verbose,
                trial=trial,
                prune_after_n=prune_after_n,
                sample_indices=sample_indices,
            )
        except optuna.TrialPruned:
            if verbose:
                print(f"[Trial {trial_count[0]}] PRUNED after {prune_after_n} samples")
            raise

        score = compute_score(result, max_steps)

        if verbose:
            print(
                f"[Trial {trial_count[0]}] TS rate: {result.eigenvalue_ts_rate:.1%}, "
                f"Sella conv: {result.sella_convergence_rate:.1%}, "
                f"Both: {result.both_rate:.1%}, "
                f"Avg steps: {result.avg_steps:.1f}, "
                f"Score: {score:.4f}"
            )

        print(
            f"[DB] Trial {trial_count[0]} completed: score={score:.4f}, "
            f"ts_rate={result.eigenvalue_ts_rate:.1%}",
            file=sys.stderr,
        )

        trial.set_user_attr("eigenvalue_ts_rate", result.eigenvalue_ts_rate)
        trial.set_user_attr("sella_convergence_rate", result.sella_convergence_rate)
        trial.set_user_attr("both_rate", result.both_rate)
        trial.set_user_attr("avg_steps", result.avg_steps)
        trial.set_user_attr("avg_wall_time", result.avg_wall_time)
        trial.set_user_attr("n_samples", result.n_samples)
        trial.set_user_attr("neg_eigval_distribution", dict(result.neg_eigval_counts))

        if use_wandb:
            log_sample(
                trial_count[0] - 1,
                {
                    "trial/number": trial_count[0],
                    "trial/config_str": config.to_str(),
                    "hparams/delta0": delta0,
                    "hparams/rho_inc": rho_inc,
                    "hparams/rho_dec": rho_dec,
                    "hparams/sigma_inc": sigma_inc,
                    "hparams/sigma_dec": sigma_dec,
                    "hparams/fmax": fmax,
                    "hparams/apply_eckart": int(apply_eckart),
                    "metrics/eigenvalue_ts_rate": result.eigenvalue_ts_rate,
                    "metrics/sella_convergence_rate": result.sella_convergence_rate,
                    "metrics/both_rate": result.both_rate,
                    "metrics/score": score,
                    "counts/n_samples": result.n_samples,
                    "counts/n_eigenvalue_ts": result.n_eigenvalue_ts,
                    "counts/n_sella_converged": result.n_sella_converged,
                    "counts/n_both": result.n_both,
                    "counts/n_sella_only": result.n_sella_only,
                    "perf/avg_steps": result.avg_steps,
                    "perf/avg_wall_time": result.avg_wall_time,
                    "perf/avg_final_fmax": result.avg_final_fmax,
                    **{f"neg_eigval_dist/{k}": v for k, v in result.neg_eigval_counts.items()},
                },
            )

        return score

    return objective


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
            return "gpu=unknown"
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


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian HPO for Sella TS search with HIP calculator (parallel)."
    )
    parser = add_common_args(parser)
    parser.set_defaults(calculator="hip")

    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--start-from", type=str, default="midpoint_rt_noise1.0A")
    parser.add_argument("--optuna-seed", type=int, default=42)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prescreen", action="store_true")
    parser.add_argument("--prescreen-samples", type=int, default=100)
    parser.add_argument("--hard-samples-file", type=str, default=None)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument(
        "--device-ids",
        type=str,
        default=None,
        help="Comma-separated CUDA device IDs to use (e.g. 0,1,2,3).",
    )
    parser.add_argument(
        "--util-log-every",
        type=int,
        default=0,
        help="Log CPU/GPU utilization every N seconds (0 disables).",
    )

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="sella-hpo")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)

    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args(argv)
    args.calculator = "hip"
    device_ids = None
    if args.device_ids:
        device_ids = [int(item) for item in args.device_ids.split(",") if item.strip()]

    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)

    print(f"\n{'='*80}")
    print("HIP SELLA HYPERPARAMETER OPTIMIZATION (Optuna, parallel)")
    print(f"{'='*80}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Samples per trial: {args.max_samples}")
    print(f"Max steps per sample: {args.max_steps}")
    print(f"Starting geometry: {args.start_from}")
    print(f"Device: {device}")
    print(f"Workers: {args.n_workers}")
    print(f"{'='*80}")

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    study_name = args.study_name or f"hip_sella_hpo_{int(time.time())}"
    db_path = Path(out_dir) / f"{study_name}.db"
    storage_url = f"sqlite:///{db_path}"

    if args.wandb:
        wandb_name = args.wandb_name or f"hip-sella-hpo-{args.n_trials}trials-job{job_id}"
        init_wandb_run(
            project=args.wandb_project,
            name=wandb_name,
            config={
                "calculator": "hip",
                "n_trials": args.n_trials,
                "max_samples": args.max_samples,
                "max_steps": args.max_steps,
                "start_from": args.start_from,
                "noise_seed": getattr(args, "noise_seed", None),
                "optuna_seed": args.optuna_seed,
                "prune_after_n": 10,
                "study_name": study_name,
                "resume_mode": args.resume,
                "device": device,
                "slurm_job_id": job_id,
                "n_workers": args.n_workers,
                "gamma": 0.0,
                "internal": True,
                "use_exact_hessian": True,
                "diag_every_n": 1,
                "order": 1,
                "hpo/delta0_range": [0.15, 0.8],
                "hpo/delta0_log_scale": True,
                "hpo/rho_dec_range": [15.0, 80.0],
                "hpo/rho_inc_range": [1.01, 1.1],
                "hpo/sigma_dec_range": [0.75, 0.95],
                "hpo/sigma_inc_range": [1.1, 1.8],
                "hpo/fmax_range": [1e-4, 1e-2],
                "hpo/fmax_log_scale": True,
                "hpo/apply_eckart_options": [True, False],
                "objective/ts_rate_weight": 1.0,
                "objective/speed_weight": 0.01,
                "objective/sella_conv_weight": 0.001,
            },
            entity=args.wandb_entity,
            tags=["hpo", "sella", "hip", "optuna", "bayesian", "parallel", f"job-{job_id}"],
            run_dir=out_dir,
        )

    sampler = TPESampler(seed=args.optuna_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=0,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=args.resume,
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"Loaded {n_existing} existing trials from {db_path}")
        n_remaining = max(0, args.n_trials - n_existing)
        print(f"Will run {n_remaining} more trials")

    hard_sample_indices = None
    if args.hard_samples_file:
        with open(args.hard_samples_file, "r") as f:
            hard_sample_indices = json.load(f)
        print(f"\nLoaded {len(hard_sample_indices)} hard sample indices from {args.hard_samples_file}")
    elif args.prescreen:
        print(f"\nPre-screening {args.prescreen_samples} samples with good defaults...")
        hard_sample_indices, easy_sample_indices = prescreen_samples(
            calculator=calculator,
            dataloader=dataloader,
            device=device,
            config=HIP_GOOD_STARTING_CONFIG,
            max_steps=args.max_steps,
            max_samples=args.prescreen_samples,
            start_from=args.start_from,
            noise_seed=getattr(args, "noise_seed", None),
            verbose=args.verbose,
        )

        hard_samples_path = Path(out_dir) / "hard_sample_indices.json"
        with open(hard_samples_path, "w") as f:
            json.dump(hard_sample_indices, f)
        print(f"Saved hard sample indices to: {hard_samples_path}")

        if not hard_sample_indices:
            print("WARNING: No hard samples found. Proceeding with all samples...")
            hard_sample_indices = None

    if hard_sample_indices is not None:
        print(f"\nHPO will focus on {len(hard_sample_indices)} hard samples")

    if n_existing == 0:
        print("\nSeeding TPE with known-good starting configs...")
        good_configs = [
            {"delta0": 0.5, "rho_dec": 50.0, "rho_inc": 1.035, "sigma_dec": 0.9,
             "sigma_inc": 1.15, "fmax": 1e-3, "apply_eckart": False},
            {"delta0": 0.5, "rho_dec": 50.0, "rho_inc": 1.035, "sigma_dec": 0.9,
             "sigma_inc": 1.15, "fmax": 1e-3, "apply_eckart": True},
            {"delta0": 0.4, "rho_dec": 40.0, "rho_inc": 1.05, "sigma_dec": 0.85,
             "sigma_inc": 1.2, "fmax": 1e-3, "apply_eckart": False},
        ]
        for cfg in good_configs:
            try:
                study.enqueue_trial(cfg)
            except TypeError as e:
                print(f"WARNING: Could not enqueue seed trial: {e}")
                break

    processor = ParallelHIPProcessor(
        checkpoint_path=args.checkpoint_path,
        device=device,
        n_workers=args.n_workers,
        worker_fn=hip_sella_worker_sample,
        worker_kwargs={"device": device},
        device_ids=device_ids,
    )
    processor.start()
    util_stop_event = Event()
    util_thread = start_util_logger(args.util_log_every, util_stop_event)

    try:
        objective = create_objective(
            processor=processor,
            dataloader=dataloader,
            device=device,
            max_steps=args.max_steps,
            max_samples=args.max_samples,
            start_from=args.start_from,
            noise_seed=getattr(args, "noise_seed", None),
            verbose=args.verbose,
            use_wandb=args.wandb,
            sample_indices=hard_sample_indices,
        )

        try:
            n_existing = len(study.trials)
            n_to_run = max(0, args.n_trials - n_existing) if args.resume else args.n_trials
            if n_to_run > 0:
                study.optimize(objective, n_trials=n_to_run)
            else:
                print(f"Already have {n_existing} trials, skipping optimization")
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Optimization stopped by user. Saving results...")
        except Exception as e:
            print(f"\n[ERROR] Optimization failed: {e}")
            print(traceback.format_exc())
            print("Saving partial results...")

        print(f"\n{'='*80}")
        print("HPO RESULTS SUMMARY")
        print(f"{'='*80}")

        completed_trials = [t for t in study.trials if t.value is not None]
        if not completed_trials:
            print("\nNo completed trials. Check logs for errors.")
            print(f"{'='*80}")
            if args.wandb:
                finish_wandb()
            return

        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best score: {study.best_trial.value:.4f}")
        print(f"\nBest hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")

        print(f"\nTop 5 trials:")
        trials_sorted = sorted(
            study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True
        )
        print(f"{'Rank':<6} {'Score':<10} {'delta0':<10} {'rho_dec':<10} {'sigma_dec':<10} {'fmax':<10} {'eckart':<8}")
        print("-" * 74)
        for rank, trial in enumerate(trials_sorted[:5], 1):
            if trial.value is not None:
                print(
                    f"{rank:<6} {trial.value:<10.4f} "
                    f"{trial.params['delta0']:<10.4f} {trial.params['rho_dec']:<10.1f} "
                    f"{trial.params['sigma_dec']:<10.3f} {trial.params['fmax']:<10.2e} "
                    f"{trial.params['apply_eckart']!s:<8}"
                )

        print(f"\n{'='*80}")

        results_path = Path(out_dir) / "hip_hpo_results.json"
        results_data = {
            "study_name": study_name,
            "best_trial": study.best_trial.number,
            "best_score": study.best_trial.value,
            "best_params": study.best_trial.params,
            "best_user_attrs": dict(study.best_trial.user_attrs),
            "n_trials_total": len(study.trials),
            "n_trials_completed": len([t for t in study.trials if t.value is not None]),
            "n_trials_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "all_trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "state": str(t.state),
                    "params": t.params,
                    "user_attrs": dict(t.user_attrs),
                }
                for t in study.trials
            ],
        }
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"\nSaved detailed results to: {results_path}")

        best_config_path = Path(out_dir) / "hip_best_config.json"
        best_config = HPOConfig(
            delta0=study.best_trial.params["delta0"],
            rho_inc=study.best_trial.params["rho_inc"],
            rho_dec=study.best_trial.params["rho_dec"],
            sigma_inc=study.best_trial.params["sigma_inc"],
            sigma_dec=study.best_trial.params["sigma_dec"],
            fmax=study.best_trial.params["fmax"],
            apply_eckart=study.best_trial.params["apply_eckart"],
        )
        with open(best_config_path, "w") as f:
            json.dump(best_config.to_dict(), f, indent=2)
        print(f"Saved best config to: {best_config_path}")

        if args.wandb:
            n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            n_completed = len([t for t in study.trials if t.value is not None])
            n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
            best_trial = study.best_trial
            best_ts_rate = best_trial.user_attrs.get("eigenvalue_ts_rate", 0)
            best_sella_rate = best_trial.user_attrs.get("sella_convergence_rate", 0)
            best_both_rate = best_trial.user_attrs.get("both_rate", 0)
            best_avg_steps = best_trial.user_attrs.get("avg_steps", 0)

            all_scores = [t.value for t in study.trials if t.value is not None]
            all_ts_rates = [
                t.user_attrs.get("eigenvalue_ts_rate", 0)
                for t in study.trials
                if t.value is not None
            ]

            summary = {
                "best/trial_number": best_trial.number,
                "best/score": best_trial.value,
                "best/eigenvalue_ts_rate": best_ts_rate,
                "best/sella_convergence_rate": best_sella_rate,
                "best/both_rate": best_both_rate,
                "best/avg_steps": best_avg_steps,
                "best/delta0": best_trial.params["delta0"],
                "best/rho_inc": best_trial.params["rho_inc"],
                "best/rho_dec": best_trial.params["rho_dec"],
                "best/sigma_inc": best_trial.params["sigma_inc"],
                "best/sigma_dec": best_trial.params["sigma_dec"],
                "best/fmax": best_trial.params["fmax"],
                "best/apply_eckart": best_trial.params["apply_eckart"],
                "trials/n_completed": n_completed,
                "trials/n_pruned": n_pruned,
                "trials/n_failed": n_failed,
                "trials/n_total": len(study.trials),
                "aggregate/mean_score": float(np.mean(all_scores)) if all_scores else 0,
                "aggregate/std_score": float(np.std(all_scores)) if all_scores else 0,
                "aggregate/max_score": float(max(all_scores)) if all_scores else 0,
                "aggregate/mean_ts_rate": float(np.mean(all_ts_rates)) if all_ts_rates else 0,
                "aggregate/max_ts_rate": float(max(all_ts_rates)) if all_ts_rates else 0,
            }
            log_summary(summary)

            log_artifact(
                file_path=str(db_path),
                artifact_name=study_name,
                artifact_type="optuna-study",
                description=f"HIP Sella HPO Optuna study: {study_name}",
                metadata={
                    "n_trials": len(study.trials),
                    "n_completed": n_completed,
                    "n_pruned": n_pruned,
                    "best_score": study.best_trial.value,
                    "best_ts_rate": best_ts_rate,
                },
            )
            finish_wandb()

        print(f"{'='*80}")
        print(f"Optuna study saved to: {db_path}")
        print(f"To resume: --resume --study-name {study_name}")
    finally:
        util_stop_event.set()
        if util_thread is not None:
            util_thread.join(timeout=1.0)
        processor.close()


if __name__ == "__main__":
    main()
