"""
HIP Multi-Mode Eckart-MW HPO - INFORMED SEARCH
==============================================
Narrowed search ranges based on 800-step HPO results (job 1820826).

Best trial from 800-step run:
  - Score: -0.487 (~51% success rate at 800 steps)
  - Key findings:
    * adaptive_delta=True performed best
    * plateau_patience ~8 optimal
    * dt ~0.001, dt_max ~0.05
    * escape_window ~35

This script uses TIGHTER ranges centered on those optima.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler
import torch

from src.noisy.multi_mode_eckartmw_gad import multi_mode_eckartmw_gad
from src.experiments.common import setup_experiment
from src.core_algos.eckart_projection import eckart_mw_project_hip as eckart_project
from src.core_algos.geometry import parse_starting_geometry
from src.core_algos.predict_fn import make_predict_fn_from_calculator

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# INFORMED BASELINE - from best 800-step trial
# =============================================================================
BASELINE_PARAMS = {
    # Core dynamics (from best trial)
    "dt": 0.001,
    "dt_max": 0.05,
    "max_atom_disp": 0.173,  # from best trial

    # Plateau controller (from best trial)
    "plateau_patience": 8,
    "plateau_boost": 2.0,
    "plateau_shrink": 0.5,

    # Escape detection (from best trial)
    "escape_disp_threshold": 3e-4,
    "escape_window": 35,
    "escape_neg_vib_std": 0.5,

    # Escape perturbation (key finding: adaptive_delta=True)
    "escape_delta": 0.15,
    "adaptive_delta": True,  # CRITICAL: best trials used True
    "min_interatomic_dist": 0.5,

    # Convergence
    "fmax": 1e-3,
    "early_stop_neg_vibs": 3,
}

# =============================================================================
# PREVIOUS BEST TRIALS TO ENQUEUE
# (Extracted from 800-step HPO database)
# =============================================================================
KNOWN_GOOD_CONFIGS = [
    {
        # Best trial from 800-step run
        "dt": 0.001,
        "dt_max": 0.05,
        "max_atom_disp": 0.173,
        "plateau_patience": 8,
        "plateau_boost": 2.0,
        "plateau_shrink": 0.5,
        "escape_disp_threshold": 3e-4,
        "escape_window": 35,
        "escape_neg_vib_std": 0.5,
        "escape_delta": 0.15,
        "adaptive_delta": True,
        "min_interatomic_dist": 0.5,
    },
    {
        # Second-best configuration variant
        "dt": 0.0015,
        "dt_max": 0.04,
        "max_atom_disp": 0.20,
        "plateau_patience": 10,
        "plateau_boost": 1.8,
        "plateau_shrink": 0.55,
        "escape_disp_threshold": 4e-4,
        "escape_window": 30,
        "escape_neg_vib_std": 0.4,
        "escape_delta": 0.12,
        "adaptive_delta": True,
        "min_interatomic_dist": 0.5,
    },
]


def run_single_sample(predict_fn, start_coords, atomic_nums, params, n_steps):
    """Run multi-mode on a single sample."""
    try:
        start_time = time.perf_counter()
        
        trajectory, info = multi_mode_eckartmw_gad(
            predict_fn=predict_fn,
            start_coords=start_coords,
            atomic_nums=atomic_nums,
            eckart_fn=eckart_project,
            n_steps=n_steps,
            **params
        )
        
        wall_time = time.perf_counter() - start_time
        
        n_imag = info.get("final_neg_vib", 0)
        success = (n_imag == 1)
        
        return {
            "success": success,
            "final_neg_vib": n_imag,
            "steps_to_ts": info.get("steps") if success else None,
            "escape_cycles": info.get("escape_cycles", 0),
            "wall_time": wall_time,
            "early_stopped": info.get("early_stopped", False),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "final_neg_vib": None,
            "steps_to_ts": None,
            "escape_cycles": 0,
            "wall_time": 0,
            "early_stopped": False,
            "error": str(e),
        }


def run_batch(predict_fn, dataloader, device, params, n_steps, max_samples, start_from, noise_seed):
    """Run multi-mode on a batch of samples."""
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
    
    if np.isfinite(mean_steps) and mean_steps > 0:
        speed_score = 1000.0 / mean_steps
    else:
        speed_score = 0.0
    
    score = success_rate * 1.0 + speed_score * 0.01
    return -score


def sample_hyperparameters_informed(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters with NARROWED ranges based on 800-step HPO.
    
    Ranges are centered around best-performing values with ±30-50% margins.
    """
    params = BASELINE_PARAMS.copy()
    
    # dt: narrowed from [0.0005, 0.005] to [0.0007, 0.002]
    params["dt"] = trial.suggest_float("dt", 0.0007, 0.002, log=True)
    
    # dt_max: narrowed from [0.01, 0.1] to [0.03, 0.07]
    params["dt_max"] = trial.suggest_float("dt_max", 0.03, 0.07, log=True)
    
    # max_atom_disp: narrowed from [0.1, 0.5] to [0.12, 0.25]
    params["max_atom_disp"] = trial.suggest_float("max_atom_disp", 0.12, 0.25)
    
    # plateau_patience: narrowed from [3, 20] to [5, 12]
    params["plateau_patience"] = trial.suggest_int("plateau_patience", 5, 12)
    
    # plateau_boost: narrowed from [1.2, 3.0] to [1.5, 2.5]
    params["plateau_boost"] = trial.suggest_float("plateau_boost", 1.5, 2.5)
    
    # plateau_shrink: narrowed from [0.3, 0.7] to [0.4, 0.6]
    params["plateau_shrink"] = trial.suggest_float("plateau_shrink", 0.4, 0.6)
    
    # escape_disp_threshold: narrowed from [1e-4, 1e-3] to [2e-4, 5e-4]
    params["escape_disp_threshold"] = trial.suggest_float("escape_disp_threshold", 2e-4, 5e-4, log=True)
    
    # escape_window: narrowed from [10, 50] to [25, 45]
    params["escape_window"] = trial.suggest_int("escape_window", 25, 45)
    
    # escape_neg_vib_std: narrowed from [0.2, 1.0] to [0.3, 0.7]
    params["escape_neg_vib_std"] = trial.suggest_float("escape_neg_vib_std", 0.3, 0.7)
    
    # escape_delta: narrowed from [0.05, 0.3] to [0.08, 0.2]
    params["escape_delta"] = trial.suggest_float("escape_delta", 0.08, 0.2)
    
    # FIXED: adaptive_delta=True (best from 800-step run)
    # If you want to still explore, uncomment below:
    # params["adaptive_delta"] = trial.suggest_categorical("adaptive_delta", [True, False])
    params["adaptive_delta"] = True
    
    # min_interatomic_dist: narrowed from [0.3, 0.7] to [0.4, 0.6]
    params["min_interatomic_dist"] = trial.suggest_float("min_interatomic_dist", 0.4, 0.6)
    
    return params


def run_verification(predict_fn, dataloader, device, n_steps, max_samples, start_from, noise_seed):
    """Run verification with baseline parameters."""
    print("\n" + "=" * 70)
    print("VERIFICATION RUN: Using INFORMED baseline parameters")
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
    print("=" * 70 + "\n")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="INFORMED Bayesian HPO for HIP Multi-Mode Eckart-MW (4000 steps)"
    )
    
    # Data paths
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    
    # Run configuration (defaults for 4000-step regime)
    parser.add_argument("--n-steps", type=int, default=4000)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--start-from", type=str, default="midpoint_rt_noise1.0A")
    parser.add_argument("--noise-seed", type=int, default=42)
    
    # HPO configuration
    parser.add_argument("--n-trials", type=int, default=30)  # Fewer trials needed with informed ranges
    parser.add_argument("--n-startup-trials", type=int, default=3)  # Fewer random trials needed
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-verification", action="store_true")
    
    # W&B configuration
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="hip-multi-mode-hpo-informed")
    parser.add_argument("--wandb-entity", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate study name
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    study_name = args.study_name or f"hip_multi_mode_hpo_informed_{job_id}"
    db_path = out_dir / f"{study_name}.db"
    storage_url = f"sqlite:///{db_path}"
    
    # Setup calculator and dataloader
    class SetupArgs:
        def __init__(self, args):
            self.h5_path = args.h5_path
            self.checkpoint_path = args.checkpoint_path
            self.out_dir = args.out_dir
            self.calculator = "hip"
            self.max_samples = args.max_samples
            self.start_from = args.start_from
            self.noise_seed = args.noise_seed
            self.sample_index_file = None
            self.scine_functional = None
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.split = "test"
    
    setup_args = SetupArgs(args)
    calculator, dataloader, device, _ = setup_experiment(setup_args, shuffle=False)
    predict_fn = make_predict_fn_from_calculator(calculator, "hip")
    
    # Initialize W&B
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"HPO-Informed-{study_name}",
            config={
                "study_name": study_name,
                "n_trials": args.n_trials,
                "n_steps": args.n_steps,
                "max_samples": args.max_samples,
                "informed_search": True,
                "baseline_params": BASELINE_PARAMS,
            },
            tags=["hpo", "hip", "multi-mode-eckartmw", "informed"],
        )
    
    # Run verification
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
    
    # ENQUEUE KNOWN-GOOD CONFIGURATIONS
    if not args.resume and len(study.trials) == 0:
        print("\n>>> Enqueuing known-good configurations from 800-step HPO...")
        for i, config in enumerate(KNOWN_GOOD_CONFIGS):
            # Map to trial parameter names
            trial_params = {
                "dt": config["dt"],
                "dt_max": config["dt_max"],
                "max_atom_disp": config["max_atom_disp"],
                "plateau_patience": config["plateau_patience"],
                "plateau_boost": config["plateau_boost"],
                "plateau_shrink": config["plateau_shrink"],
                "escape_disp_threshold": config["escape_disp_threshold"],
                "escape_window": config["escape_window"],
                "escape_neg_vib_std": config["escape_neg_vib_std"],
                "escape_delta": config["escape_delta"],
                "min_interatomic_dist": config["min_interatomic_dist"],
            }
            study.enqueue_trial(trial_params)
            print(f"    Enqueued config {i+1}: dt={config['dt']}, patience={config['plateau_patience']}")
    
    # Define objective function
    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters_informed(trial)
        
        print(f"\n>>> Trial {trial.number}: Running with INFORMED params...")
        for k, v in params.items():
            if k not in BASELINE_PARAMS or params[k] != BASELINE_PARAMS[k]:
                print(f"    {k}: {v}")
        
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
        
        if args.wandb and WANDB_AVAILABLE:
            log_dict = {
                "trial/success_rate": metrics["success_rate"],
                "trial/n_success": metrics["n_success"],
                "trial/mean_steps": metrics["mean_steps_when_success"],
                "trial/score": -score,
                "trial/number": trial.number,
            }
            for k, v in params.items():
                if isinstance(v, (int, float, bool)):
                    log_dict[f"hparams/{k}"] = v
            wandb.log(log_dict)
        
        return score
    
    # Run optimization
    print(f"\n>>> Starting INFORMED HPO with {args.n_trials} trials...")
    print(f"    Study: {study_name}")
    print(f"    Database: {db_path}")
    print(f"    Narrowed search ranges based on 800-step HPO results")
    
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            show_progress_bar=True,
            catch=(Exception,),
        )
    except KeyboardInterrupt:
        print("\n>>> HPO interrupted. Saving results...")
    
    # Print best results
    print("\n" + "=" * 70)
    print("BEST TRIAL (4000-step INFORMED search)")
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
        "informed_search": True,
        "based_on_800_step_hpo": True,
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
        wandb.log({
            "best/trial_number": best_trial.number,
            "best/score": -best_trial.value,
            **{f"best/{k}": v for k, v in best_trial.params.items()},
        })
        wandb.finish()
    
    print("\nINFORMED HPO complete!")


if __name__ == "__main__":
    main()
