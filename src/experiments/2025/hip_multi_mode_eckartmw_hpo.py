from __future__ import annotations

"""Optuna Bayesian Hyperparameter Optimization for HIP multi-mode Eckart-MW GAD.

This script uses Optuna to find optimal hyperparameters that maximize convergence rate,
with special focus on samples that currently fail to converge.

Strategy:
1. Use Optuna's TPE (Tree-structured Parzen Estimator) sampler for Bayesian optimization
2. Focus on difficult samples (those that don't converge with default params)
3. Optimize: dt control, plateau detection, escape parameters, trust radius, adaptive deltas
4. Objective: Maximize convergence rate + minimize steps to converge for successful runs
5. Run ~800 steps per sample to keep trials fast (~30-60 samples per trial)

Hyperparameters optimized:
- dt_min, dt_max: Time step bounds
- plateau_patience: Steps before dt adjustment
- plateau_boost, plateau_shrink: dt adjustment factors
- escape_disp_threshold: Displacement threshold for plateau detection
- escape_window: Window size for plateau detection
- escape_neg_vib_std: Stability threshold for saddle index
- escape_delta: Base perturbation magnitude
- adaptive_delta_scale: Scale factor for adaptive perturbations (replaces binary adaptive_delta)
- trust_radius_max: Maximum allowed displacement per step
- gad_beta: Mixing parameter for mode tracking (0=no mix, 1=full update)

Features:
- W&B logging for detailed tracking (no plots, just stats)
- SQLite storage for crash recovery
- Graceful error handling (saves partial results on interrupt)
- Resume support for continuing from previous runs
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
import torch
from optuna.samplers import TPESampler

# Suppress Optuna's verbose logging (we log to W&B instead)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Relative imports: ... goes up from 2025 -> experiments -> src
# IMPORTANT: Import run_multi_mode_escape from NOISY module (the working version)
# NOT from experiments, to ensure identical algorithm behavior
from ...noisy.multi_mode_eckartmw import run_multi_mode_escape
from .multi_mode_eckartmw import compute_convergence_diagnostics
from ...dependencies.common_utils import add_common_args, setup_experiment, parse_starting_geometry
from ...dependencies.hessian import vibrational_eigvals, get_scine_elements_from_predict_output
from ...runners._predict import make_predict_fn_from_calculator
from ...logging import finish_wandb, init_wandb_run, log_sample, log_summary


def load_difficult_samples(
    dataloader,
    device: str,
    predict_fn,
    start_from: str,
    noise_seed: Optional[int],
    target_count: int = 30,
    difficulty_threshold: float = 0.5,
) -> list[Dict[str, Any]]:
    """Identify difficult samples that fail to converge with baseline parameters.
    
    This runs a quick baseline test on all samples (100 steps each) to identify
    which ones struggle, then returns those sample indices for focused HPO.
    
    Args:
        dataloader: Dataset loader
        device: Device to run on
        predict_fn: Prediction function
        start_from: Starting geometry type
        noise_seed: Random seed for noise
        target_count: Target number of difficult samples to return
        difficulty_threshold: Fraction of samples to consider "difficult" (0.5 = top 50% hardest)
        
    Returns:
        List of dicts with sample info (index, batch, start_coords, initial_neg)
    """
    print("\n" + "=" * 80)
    print("IDENTIFYING DIFFICULT SAMPLES")
    print("=" * 80)
    
    # Quick baseline: 200 steps with default parameters
    baseline_results = []
    
    for i, batch in enumerate(dataloader):
        if i >= 100:  # Test on first 100 samples max
            break
            
        batch = batch.to(device)
        atomic_nums = batch.z.detach().cpu().to(device)
        
        start_coords = parse_starting_geometry(
            start_from,
            batch,
            noise_seed=noise_seed,
            sample_index=i,
        ).detach().to(device)
        
        # Get initial neg vib
        try:
            init_out = predict_fn(start_coords, atomic_nums, do_hessian=True, require_grad=False)
            init_scine_elements = get_scine_elements_from_predict_output(init_out)
            init_vib = vibrational_eigvals(init_out["hessian"], start_coords, atomic_nums, scine_elements=init_scine_elements)
            initial_neg = int((init_vib < 0).sum().item())
        except Exception:
            initial_neg = -1
        
        # Run quick baseline test
        try:
            out_dict, aux = run_multi_mode_escape(
                predict_fn,
                start_coords,
                atomic_nums,
                n_steps=200,  # Quick test
                dt=0.001,
                stop_at_ts=True,
                ts_eps=1e-5,
                dt_control="neg_eig_plateau",
                dt_min=1e-6,
                dt_max=0.05,
                max_atom_disp=0.25,
                plateau_patience=5,   # Matches noisy SLURM script
                plateau_boost=2.0,    # Matches noisy SLURM script
                plateau_shrink=0.5,
                escape_disp_threshold=5e-4,
                escape_window=20,
                hip_vib_mode="proj_tol",
                hip_rigid_tol=1e-6,
                hip_eigh_device="cpu",
                escape_neg_vib_std=0.5,
                escape_delta=0.1,
                adaptive_delta=True,
                min_interatomic_dist=0.5,
                max_escape_cycles=1000,
                profile_every=0,
            )
            
            converged = aux.get("steps_to_ts") is not None
            steps_to_converge = aux.get("steps_to_ts", 200)
            final_neg = out_dict.get("final_neg_vibrational", -1)
            
        except Exception as e:
            print(f"[WARN] Sample {i} failed baseline: {e}")
            converged = False
            steps_to_converge = 200
            final_neg = -1
        
        # Difficulty score: higher = more difficult
        # Non-convergence is most difficult, then slow convergence
        if not converged:
            difficulty = 1000.0  # Very difficult
        else:
            difficulty = steps_to_converge
            
        baseline_results.append({
            "index": i,
            "batch": batch,
            "start_coords": start_coords,
            "initial_neg": initial_neg,
            "converged": converged,
            "steps_to_converge": steps_to_converge,
            "final_neg": final_neg,
            "difficulty": difficulty,
        })
        
        status = "✓" if converged else "✗"
        print(f"  Sample {i:3d}: {status} converged={converged} steps={steps_to_converge} difficulty={difficulty:.1f}")
    
    # Sort by difficulty (hardest first)
    baseline_results.sort(key=lambda x: x["difficulty"], reverse=True)
    
    # Take top difficult samples
    n_difficult = int(len(baseline_results) * difficulty_threshold)
    n_difficult = max(n_difficult, target_count)
    n_difficult = min(n_difficult, len(baseline_results))
    
    difficult_samples = baseline_results[:n_difficult]
    
    n_converged = sum(1 for r in baseline_results if r["converged"])
    n_total = len(baseline_results)
    baseline_rate = n_converged / n_total if n_total > 0 else 0.0
    
    n_difficult_converged = sum(1 for r in difficult_samples if r["converged"])
    difficult_rate = n_difficult_converged / len(difficult_samples) if difficult_samples else 0.0
    
    print(f"\nBaseline convergence rate: {n_converged}/{n_total} = {baseline_rate * 100:.1f}%")
    print(f"Selected {len(difficult_samples)} difficult samples")
    print(f"Difficult sample baseline convergence: {n_difficult_converged}/{len(difficult_samples)} = {difficult_rate * 100:.1f}%")
    print("=" * 80 + "\n")
    
    return difficult_samples


def objective(
    trial: optuna.Trial,
    difficult_samples: list[Dict[str, Any]],
    predict_fn,
    device: str,
    n_steps_per_sample: int,
    use_wandb: bool = False,
) -> float:
    """Optuna objective function.
    
    Returns a score where higher is better:
    - Primary: Convergence rate (% of samples that converge)
    - Secondary: Mean steps to converge for successful samples (lower is better)
    
    Combined score = convergence_rate - 0.0001 * mean_steps_to_converge
    This heavily prioritizes convergence rate while using steps as a tiebreaker.
    """
    
    # Sample hyperparameters
    dt_min = trial.suggest_float("dt_min", 1e-7, 1e-5, log=True)
    dt_max = trial.suggest_float("dt_max", 0.01, 0.1, log=True)
    
    plateau_patience = trial.suggest_int("plateau_patience", 3, 20)
    plateau_boost = trial.suggest_float("plateau_boost", 1.2, 3.0)
    plateau_shrink = trial.suggest_float("plateau_shrink", 0.3, 0.7)
    
    escape_disp_threshold = trial.suggest_float("escape_disp_threshold", 1e-5, 1e-3, log=True)
    escape_window = trial.suggest_int("escape_window", 10, 50)
    escape_neg_vib_std = trial.suggest_float("escape_neg_vib_std", 0.1, 1.0)
    
    escape_delta = trial.suggest_float("escape_delta", 0.05, 0.5)
    adaptive_delta_scale = trial.suggest_float("adaptive_delta_scale", 0.0, 2.0)
    
    # Trust radius (max displacement per atom per step)
    trust_radius_max = trial.suggest_float("trust_radius_max", 0.1, 0.5)
    
    # GAD mode tracking beta (0=sticky, 1=always update to best match)
    # This is handled internally by passing it to gad_euler_step_projected
    # For now, we'll keep it at 1.0 and not optimize (adds complexity to multi_mode_eckartmw.py)
    # gad_beta = trial.suggest_float("gad_beta", 0.7, 1.0)
    gad_beta = 1.0  # Fixed for now
    
    # Run on all difficult samples
    results = []
    
    for sample_info in difficult_samples:
        try:
            # Convert adaptive_delta_scale to bool + scale
            # If scale > 0, use adaptive with that scale
            # If scale == 0, don't use adaptive
            adaptive_delta = adaptive_delta_scale > 0
            
            out_dict, aux = run_multi_mode_escape(
                predict_fn,
                sample_info["start_coords"],
                sample_info["batch"].z.detach().cpu().to(device),
                n_steps=n_steps_per_sample,
                dt=0.001,  # Initial dt
                stop_at_ts=True,
                ts_eps=1e-5,
                dt_control="neg_eig_plateau",
                dt_min=dt_min,
                dt_max=dt_max,
                max_atom_disp=trust_radius_max,  # Use trust_radius_max as max_atom_disp
                plateau_patience=plateau_patience,
                plateau_boost=plateau_boost,
                plateau_shrink=plateau_shrink,
                escape_disp_threshold=escape_disp_threshold,
                escape_window=escape_window,
                hip_vib_mode="proj_tol",  # Fast mode for HPO
                hip_rigid_tol=1e-6,
                hip_eigh_device="cpu",
                escape_neg_vib_std=escape_neg_vib_std,
                escape_delta=escape_delta * (adaptive_delta_scale if adaptive_delta else 1.0),
                adaptive_delta=adaptive_delta,
                min_interatomic_dist=0.5,
                max_escape_cycles=1000,
                profile_every=0,
            )
            
            converged = aux.get("steps_to_ts") is not None
            steps_to_converge = aux.get("steps_to_ts", n_steps_per_sample)
            
            results.append({
                "converged": converged,
                "steps_to_converge": steps_to_converge,
            })
            
        except Exception as e:
            # Failed runs count as non-converged
            results.append({
                "converged": False,
                "steps_to_converge": n_steps_per_sample,
            })
    
    # Compute metrics
    n_converged = sum(1 for r in results if r["converged"])
    n_total = len(results)
    convergence_rate = n_converged / n_total if n_total > 0 else 0.0
    
    converged_steps = [r["steps_to_converge"] for r in results if r["converged"]]
    mean_steps = np.mean(converged_steps) if converged_steps else n_steps_per_sample
    
    # Combined score: prioritize convergence rate, use steps as tiebreaker
    # Normalize steps to [0, 1] range for fair comparison
    normalized_steps = mean_steps / n_steps_per_sample
    score = convergence_rate - 0.01 * normalized_steps  # Small penalty for slow convergence
    
    # Log intermediate results to Optuna
    trial.set_user_attr("convergence_rate", convergence_rate)
    trial.set_user_attr("mean_steps_to_converge", mean_steps)
    trial.set_user_attr("n_converged", n_converged)
    trial.set_user_attr("n_total", n_total)
    
    print(f"Trial {trial.number}: convergence_rate={convergence_rate:.3f} ({n_converged}/{n_total}) "
          f"mean_steps={mean_steps:.1f} score={score:.4f}")
    
    # Log to W&B with detailed metrics (no plots)
    if use_wandb:
        log_sample(
            trial.number,
            {
                # Trial info
                "trial/number": trial.number,
                # Hyperparameters
                "hparams/dt_min": dt_min,
                "hparams/dt_max": dt_max,
                "hparams/plateau_patience": plateau_patience,
                "hparams/plateau_boost": plateau_boost,
                "hparams/plateau_shrink": plateau_shrink,
                "hparams/escape_disp_threshold": escape_disp_threshold,
                "hparams/escape_window": escape_window,
                "hparams/escape_neg_vib_std": escape_neg_vib_std,
                "hparams/escape_delta": escape_delta,
                "hparams/adaptive_delta_scale": adaptive_delta_scale,
                "hparams/trust_radius_max": trust_radius_max,
                # Metrics
                "metrics/convergence_rate": convergence_rate,
                "metrics/mean_steps": mean_steps,
                "metrics/score": score,
                # Counts
                "counts/n_converged": n_converged,
                "counts/n_total": n_total,
            },
        )
    
    return score


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian HPO for HIP multi-mode Eckart-MW GAD convergence optimization."
    )
    parser = add_common_args(parser)
    parser.set_defaults(calculator="hip", noise_seed=42)  # Override defaults
    
    # HPO parameters
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--n-steps-per-sample", type=int, default=800, help="Steps per sample per trial")
    parser.add_argument("--n-samples", type=int, default=15, help="Number of samples per trial (from difficult set)")
    parser.add_argument("--difficulty-threshold", type=float, default=0.5, help="Fraction of samples to consider difficult")
    parser.add_argument("--start-from", type=str, default="midpoint_rt_noise1.0A")
    parser.add_argument("--study-name", type=str, default=None, help="Optuna study name. Default: auto-generated")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing study")
    
    # W&B arguments
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="gad-hpo", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    
    args = parser.parse_args(argv)
    
    # Setup
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    device = str(device)  # Ensure string
    predict_fn = make_predict_fn_from_calculator(calculator, "hip")
    
    # Generate study name if not provided
    job_id = os.environ.get("SLURM_JOB_ID", str(int(time.time())))
    study_name = args.study_name or f"hip-gad-hpo-{job_id}"
    
    # Setup storage (default to SQLite for crash recovery)
    storage_url = args.storage
    if storage_url is None:
        db_path = Path(out_dir) / f"{study_name}.db"
        storage_url = f"sqlite:///{db_path}"
    
    print(f"\n{'='*80}")
    print("HIP MULTI-MODE ECKART-MW GAD HPO")
    print(f"{'='*80}")
    print(f"Study: {study_name}")
    print(f"Storage: {storage_url}")
    print(f"Trials: {args.n_trials}")
    print(f"Steps per sample: {args.n_steps_per_sample}")
    print(f"Samples per trial: {args.n_samples}")
    print(f"Difficulty threshold: {args.difficulty_threshold}")
    if args.resume:
        print("Mode: RESUME (continuing from existing study)")
    print(f"{'='*80}")
    
    # Initialize W&B if requested
    if args.wandb:
        wandb_name = args.wandb_name or f"hip-gad-hpo-{job_id}"
        init_wandb_run(
            project=args.wandb_project,
            name=wandb_name,
            config={
                "calculator": "hip",
                "study_name": study_name,
                "n_trials": args.n_trials,
                "n_steps_per_sample": args.n_steps_per_sample,
                "n_samples": args.n_samples,
                "difficulty_threshold": args.difficulty_threshold,
                "start_from": args.start_from,
                "noise_seed": args.noise_seed,
                "device": device,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
                # HPO search ranges
                "hpo/dt_min_range": [1e-7, 1e-5],
                "hpo/dt_max_range": [0.01, 0.1],
                "hpo/plateau_patience_range": [3, 20],
                "hpo/plateau_boost_range": [1.2, 3.0],
                "hpo/plateau_shrink_range": [0.3, 0.7],
                "hpo/escape_disp_threshold_range": [1e-5, 1e-3],
                "hpo/escape_window_range": [10, 50],
                "hpo/escape_neg_vib_std_range": [0.1, 1.0],
                "hpo/escape_delta_range": [0.05, 0.5],
                "hpo/adaptive_delta_scale_range": [0.0, 2.0],
                "hpo/trust_radius_max_range": [0.1, 0.5],
            },
            entity=args.wandb_entity,
            tags=["hpo", "gad", "hip", "optuna", "multi-mode-eckartmw"],
            run_dir=out_dir,
        )
    
    # Identify difficult samples
    difficult_samples_full = load_difficult_samples(
        dataloader,
        device,
        predict_fn,
        args.start_from,
        args.noise_seed,
        target_count=args.n_samples * 2,  # Get more than needed
        difficulty_threshold=args.difficulty_threshold,
    )
    
    # Save difficult sample indices for reproducibility
    difficult_indices = [s["index"] for s in difficult_samples_full]
    indices_path = Path(out_dir) / "difficult_sample_indices.json"
    with open(indices_path, "w") as f:
        json.dump(difficult_indices, f)
    print(f"Saved difficult sample indices to: {indices_path}")
    
    # Use subset for each trial
    difficult_samples = difficult_samples_full[:args.n_samples]
    
    print(f"\nUsing {len(difficult_samples)} difficult samples for HPO")
    print(f"Running {args.n_trials} trials with {args.n_steps_per_sample} steps per sample")
    
    # Create Optuna study with SQLite storage for crash recovery
    sampler = TPESampler(seed=42, n_startup_trials=10)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=args.resume or True,  # Always allow loading for crash recovery
        direction="maximize",
        sampler=sampler,
    )
    
    # Report existing progress if resuming
    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"Loaded {n_existing} existing trials from storage")
        n_remaining = max(0, args.n_trials - n_existing)
        print(f"Will run {n_remaining} more trials")
    
    # Run optimization with graceful error handling
    try:
        # Calculate how many trials to run
        n_to_run = max(0, args.n_trials - n_existing) if args.resume else args.n_trials
        
        if n_to_run > 0:
            study.optimize(
                lambda trial: objective(
                    trial,
                    difficult_samples,
                    predict_fn,
                    device,
                    args.n_steps_per_sample,
                    use_wandb=args.wandb,
                ),
                n_trials=n_to_run,
                show_progress_bar=True,
            )
        else:
            print(f"Already have {n_existing} trials, skipping optimization")
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Optimization stopped by user. Saving results...")
    except Exception as e:
        print(f"\n[ERROR] Optimization failed: {e}")
        print(traceback.format_exc())
        print("Saving partial results...")
    
    # Print results (even if interrupted)
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    # Check if we have any completed trials
    completed_trials = [t for t in study.trials if t.value is not None]
    if not completed_trials:
        print("\nNo completed trials. Check logs for errors.")
        print(f"{'='*80}")
        if args.wandb:
            finish_wandb()
        return
    
    print(f"Total trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best convergence rate: {study.best_trial.user_attrs['convergence_rate']:.3f}")
    print(f"Best mean steps: {study.best_trial.user_attrs['mean_steps_to_converge']:.1f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save detailed results
    results_path = Path(out_dir) / "hpo_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "study_name": study_name,
            "best_trial": study.best_trial.number,
            "best_score": study.best_value,
            "best_params": study.best_params,
            "best_convergence_rate": study.best_trial.user_attrs["convergence_rate"],
            "best_mean_steps": study.best_trial.user_attrs["mean_steps_to_converge"],
            "n_trials_total": len(study.trials),
            "n_trials_completed": len(completed_trials),
            "difficult_sample_indices": difficult_indices,
            "all_trials": [
                {
                    "number": t.number,
                    "score": t.value,
                    "state": str(t.state),
                    "params": t.params,
                    "convergence_rate": t.user_attrs.get("convergence_rate"),
                    "mean_steps": t.user_attrs.get("mean_steps_to_converge"),
                    "n_converged": t.user_attrs.get("n_converged"),
                    "n_total": t.user_attrs.get("n_total"),
                }
                for t in study.trials
            ],
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    # Log final summary to W&B
    if args.wandb:
        best_trial = study.best_trial
        log_summary({
            # Best trial info
            "best/trial_number": best_trial.number,
            "best/score": best_trial.value,
            "best/convergence_rate": best_trial.user_attrs.get("convergence_rate", 0),
            "best/mean_steps": best_trial.user_attrs.get("mean_steps_to_converge", 0),
            # Best hyperparameters
            **{f"best/{k}": v for k, v in best_trial.params.items()},
            # Trial statistics
            "trials/n_completed": len(completed_trials),
            "trials/n_total": len(study.trials),
        })
        finish_wandb()
    
    print(f"{'='*80}")
    print(f"Optuna study saved to: {storage_url}")
    print(f"To resume: --resume --study-name {study_name}")


if __name__ == "__main__":
    main()
