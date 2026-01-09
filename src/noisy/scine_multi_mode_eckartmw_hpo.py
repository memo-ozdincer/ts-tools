#!/usr/bin/env python
"""Bayesian Hyperparameter Optimization for SCINE Multi-Mode Eckart-MW.

This script performs HPO directly on top of the working multi_mode_eckartmw.py
implementation, using the exact same run_multi_mode_escape function.

CRITICAL: The first trial uses EXACTLY the same parameters as the working
scine_multi_mode_eckartmw.slurm script to verify the setup before HPO begins.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import io
import numpy as np
import torch

# Suppress warnings (including edge_vec_0_distance errors)
warnings.filterwarnings('ignore')


class SuppressStdout:
    """Context manager to suppress stdout (including HIP's edge_vec_0_distance errors)."""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self
    
    def __exit__(self, *args):
        sys.stdout = self._stdout

# Import Optuna
import optuna
from optuna.samplers import TPESampler

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import from the working implementation
from .multi_mode_eckartmw import (
    run_multi_mode_escape,
    vibrational_eigvals,
    get_scine_elements_from_predict_output,
)
from ..dependencies.common_utils import setup_experiment
from ..dependencies.hessian import vibrational_eigvals
from ..runners._predict import make_predict_fn_from_calculator

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# EXACT BASELINE PARAMETERS FROM scine_multi_mode_eckartmw.slurm
# Updated with HPO results (trial 14)
# =============================================================================
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
    # SCINE doesn't use HIP-specific params, but we need placeholders
    "hip_vib_mode": "projected",
    "hip_rigid_tol": 1e-6,
    "hip_eigh_device": "cpu",
    # Early stopping: stop if avg neg_vib unchanged for 500 steps
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
    """Run a single sample with given parameters.
    
    Returns dict with:
        - final_neg_vib: Final number of negative vibrational eigenvalues
        - steps_taken: Number of steps taken
        - steps_to_ts: Steps to reach TS (if found)
        - escape_cycles: Number of escape cycles used
        - success: Whether we reached index-1 saddle
        - wall_time: Wall clock time
    """
    t0 = time.time()
    
    try:
        # Suppress stdout to hide any noisy error messages
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


def run_batch(
    predict_fn,
    dataloader,
    device,
    params: Dict[str, Any],
    n_steps: int,
    max_samples: int,
    start_from: str,
    noise_seed: Optional[int],
) -> Dict[str, Any]:
    """Run a batch of samples with given parameters.
    
    Returns aggregated metrics.
    """
    from ..dependencies.common_utils import parse_starting_geometry
    
    results = []
    
    for i, batch in enumerate(dataloader):
        if i >= max_samples:
            break
            
        batch = batch.to(device)
        atomic_nums = batch.z.detach().to(device)
        
        start_coords = parse_starting_geometry(
            start_from,
            batch,
            noise_seed=noise_seed,
            sample_index=i,
        ).detach().to(device)
        
        result = run_single_sample(
            predict_fn, start_coords, atomic_nums, params, n_steps
        )
        result["sample_idx"] = i
        results.append(result)
    
    # Aggregate metrics
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
        "mean_wall_time": np.mean(wall_times),
        "total_wall_time": sum(wall_times),
        "neg_vib_counts": neg_vib_counts,
        "results": results,
    }


def compute_objective(batch_metrics: Dict[str, Any]) -> float:
    """Compute objective value (higher is better).
    
    Priority:
    1. Success rate (index-1 saddle) - weight 1.0
    2. Speed (fewer steps when successful) - weight 0.01
    
    Returns negative value for minimization.
    """
    success_rate = batch_metrics["success_rate"]
    mean_steps = batch_metrics["mean_steps_when_success"]
    
    # Speed score: inversely proportional to steps (normalized)
    if np.isfinite(mean_steps) and mean_steps > 0:
        speed_score = 1000.0 / mean_steps  # Normalize to roughly 0-1 range
    else:
        speed_score = 0.0
    
    # Combined score (higher is better)
    score = success_rate * 1.0 + speed_score * 0.01
    
    # Return negative for minimization
    return -score


def sample_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters for a trial."""
    params = BASELINE_PARAMS.copy()
    
    # dt: base time step
    params["dt"] = trial.suggest_float("dt", 0.0005, 0.005, log=True)
    
    # dt_max: maximum allowed time step
    params["dt_max"] = trial.suggest_float("dt_max", 0.01, 0.1, log=True)
    
    # max_atom_disp: safety cap on per-atom displacement
    params["max_atom_disp"] = trial.suggest_float("max_atom_disp", 0.1, 0.5)
    
    # Plateau controller parameters
    params["plateau_patience"] = trial.suggest_int("plateau_patience", 3, 20)
    params["plateau_boost"] = trial.suggest_float("plateau_boost", 1.2, 3.0)
    params["plateau_shrink"] = trial.suggest_float("plateau_shrink", 0.3, 0.7)
    
    # Escape detection parameters
    params["escape_disp_threshold"] = trial.suggest_float("escape_disp_threshold", 1e-4, 1e-3, log=True)
    params["escape_window"] = trial.suggest_int("escape_window", 10, 50)
    params["escape_neg_vib_std"] = trial.suggest_float("escape_neg_vib_std", 0.2, 1.0)
    
    # Escape perturbation parameters
    params["escape_delta"] = trial.suggest_float("escape_delta", 0.05, 0.3)
    params["adaptive_delta"] = trial.suggest_categorical("adaptive_delta", [True, False])
    params["min_interatomic_dist"] = trial.suggest_float("min_interatomic_dist", 0.3, 0.7)
    
    return params


def run_verification(
    predict_fn,
    dataloader,
    device,
    n_steps: int,
    max_samples: int,
    start_from: str,
    noise_seed: Optional[int],
) -> Dict[str, Any]:
    """Run verification with EXACT baseline parameters.
    
    This confirms the setup matches the working SLURM script behavior.
    """
    print("\n" + "=" * 70)
    print("VERIFICATION RUN: Using EXACT baseline parameters from SLURM script")
    print("=" * 70)
    print(f"Parameters: {json.dumps(BASELINE_PARAMS, indent=2)}")
    print("=" * 70 + "\n")
    
    metrics = run_batch(
        predict_fn, dataloader, device, BASELINE_PARAMS,
        n_steps, max_samples, start_from, noise_seed
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


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian HPO for SCINE Multi-Mode Eckart-MW"
    )
    
    # Data paths
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    
    # SCINE configuration
    parser.add_argument("--scine-functional", type=str, default="DFTB0")
    
    # Run configuration
    parser.add_argument("--n-steps", type=int, default=4000)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--start-from", type=str, default="midpoint_rt_noise1.0A")
    parser.add_argument("--noise-seed", type=int, default=42)
    
    # HPO configuration
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-startup-trials", type=int, default=5)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from existing study")
    parser.add_argument("--skip-verification", action="store_true")
    
    # W&B configuration
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="scine-multi-mode-hpo")
    parser.add_argument("--wandb-entity", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate study name
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    study_name = args.study_name or f"scine_multi_mode_hpo_{job_id}"
    db_path = out_dir / f"{study_name}.db"
    storage_url = f"sqlite:///{db_path}"
    
    # Setup calculator and dataloader
    # Create a minimal args namespace for setup_experiment
    class SetupArgs:
        def __init__(self, args):
            self.h5_path = args.h5_path
            self.checkpoint_path = None  # SCINE doesn't need checkpoint
            self.out_dir = args.out_dir
            self.calculator = "scine"
            self.max_samples = args.max_samples
            self.start_from = args.start_from
            self.noise_seed = args.noise_seed
            self.scine_functional = args.scine_functional
            # These are needed by setup_experiment
            self.sample_index_file = None
            self.device = "cpu"  # SCINE always runs on CPU
            self.split = "test"
    
    setup_args = SetupArgs(args)
    calculator, dataloader, device, _ = setup_experiment(setup_args, shuffle=False)
    predict_fn = make_predict_fn_from_calculator(calculator, "scine")
    
    # Force CPU for SCINE
    device = "cpu"
    
    # Initialize W&B
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
            },
            tags=["hpo", "scine", "multi-mode-eckartmw"],
        )
    
    # Run verification (unless skipped)
    if not args.skip_verification:
        verification_metrics = run_verification(
            predict_fn, dataloader, device,
            args.n_steps, args.max_samples,
            args.start_from, args.noise_seed
        )
        
        if args.wandb and WANDB_AVAILABLE:
            wandb.log({
                "verification/success_rate": verification_metrics["success_rate"],
                "verification/n_success": verification_metrics["n_success"],
                "verification/mean_steps": verification_metrics["mean_steps_when_success"],
                "verification/mean_escape_cycles": verification_metrics["mean_escape_cycles"],
                "verification/total_time": verification_metrics["total_wall_time"],
            })
    
    # Create Optuna study
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
    
    # Enqueue baseline as first trial if starting fresh
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
        study.enqueue_trial(baseline_trial_params)
    
    # Define objective function
    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial)
        
        print(f"\n>>> Trial {trial.number}: Running with params...")
        for k, v in params.items():
            if k not in BASELINE_PARAMS or params[k] != BASELINE_PARAMS[k]:
                print(f"    {k}: {v}")
        
        # Reload dataloader (it may be exhausted)
        _, dataloader_fresh, _, _ = setup_experiment(setup_args, shuffle=False)
        
        metrics = run_batch(
            predict_fn, dataloader_fresh, device, params,
            args.n_steps, args.max_samples,
            args.start_from, args.noise_seed
        )
        
        score = compute_objective(metrics)
        
        print(f"    Success rate: {metrics['success_rate']*100:.1f}%")
        print(f"    Mean steps: {metrics['mean_steps_when_success']:.1f}")
        print(f"    Score: {-score:.4f}")
        
        # Log to W&B
        if args.wandb and WANDB_AVAILABLE:
            log_dict = {
                f"trial/success_rate": metrics["success_rate"],
                f"trial/n_success": metrics["n_success"],
                f"trial/mean_steps": metrics["mean_steps_when_success"],
                f"trial/mean_escape_cycles": metrics["mean_escape_cycles"],
                f"trial/total_time": metrics["total_wall_time"],
                f"trial/score": -score,
                f"trial/number": trial.number,
            }
            # Log hyperparameters
            for k, v in params.items():
                if isinstance(v, (int, float, bool)):
                    log_dict[f"hparams/{k}"] = v
            wandb.log(log_dict)
        
        return score
    
    # Run optimization
    print(f"\n>>> Starting HPO with {args.n_trials} trials...")
    print(f"    Study: {study_name}")
    print(f"    Database: {db_path}")
    
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            show_progress_bar=True,
            catch=(Exception,),
        )
    except KeyboardInterrupt:
        print("\n>>> HPO interrupted by user. Saving results...")
    
    # Print best results
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
    
    # Save results
    results_path = out_dir / f"{study_name}_results.json"
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
    
    # Log final results to W&B
    if args.wandb and WANDB_AVAILABLE:
        wandb.log({
            "best/trial_number": best_trial.number,
            "best/score": -best_trial.value,
            **{f"best/{k}": v for k, v in best_trial.params.items()},
        })
        wandb.finish()
    
    print("\nHPO complete!")


if __name__ == "__main__":
    main()
