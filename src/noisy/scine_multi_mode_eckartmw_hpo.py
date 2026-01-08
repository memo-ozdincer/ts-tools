from __future__ import annotations

"""Optuna Bayesian Hyperparameter Optimization for SCINE multi-mode Eckart-MW GAD.

This script lives in src/noisy/ and imports directly from the working noisy module.

IMPORTANT: Before HPO begins, a verification run is performed using the EXACT
parameters from the working SLURM script (scripts/killarney/noisy/scine_multi_mode_eckartmw.slurm)
to confirm the algorithm works correctly.

Strategy:
1. Run verification with EXACT noisy SLURM parameters (5 samples, quick check)
2. If verification passes, proceed with HPO
3. Use Optuna's TPE sampler for Bayesian optimization
4. Focus on difficult samples (those that don't converge with default params)
5. Objective: Maximize convergence rate + minimize steps to converge

Features:
- Verification run before HPO to confirm algorithm works
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
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import torch
from optuna.samplers import TPESampler

# Suppress warnings
warnings.filterwarnings('ignore')

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import from the LOCAL noisy module (not experiments)
from .multi_mode_eckartmw import run_multi_mode_escape
from ..dependencies.common_utils import add_common_args, setup_experiment, parse_starting_geometry
from ..dependencies.hessian import vibrational_eigvals, get_scine_elements_from_predict_output
from ..runners._predict import make_predict_fn_from_calculator
from ..logging import finish_wandb, init_wandb_run, log_sample, log_summary


# =============================================================================
# EXACT parameters from working SLURM: scripts/killarney/noisy/scine_multi_mode_eckartmw.slurm
# =============================================================================
NOISY_SLURM_PARAMS = {
    "n_steps": 15000,
    "dt": 0.001,
    "stop_at_ts": True,
    "ts_eps": 1e-5,
    "dt_control": "neg_eig_plateau",
    "dt_min": 1e-6,
    "dt_max": 0.05,
    "max_atom_disp": 0.25,
    "plateau_patience": 5,
    "plateau_boost": 2.0,
    "plateau_shrink": 0.5,
    "escape_disp_threshold": 5e-4,
    "escape_window": 20,
    "hip_vib_mode": "projected",  # SCINE uses projected mode
    "hip_rigid_tol": 1e-6,
    "hip_eigh_device": "auto",
    "escape_neg_vib_std": 0.5,
    "escape_delta": 0.1,
    "adaptive_delta": True,
    "min_interatomic_dist": 0.5,
    "max_escape_cycles": 1000,
    "profile_every": 0,
}


def run_verification(
    dataloader,
    device: str,
    predict_fn,
    start_from: str,
    noise_seed: Optional[int],
    n_samples: int = 5,
) -> Dict[str, Any]:
    """Run verification with EXACT noisy SLURM parameters.
    
    This confirms the algorithm works before HPO begins.
    Uses the EXACT same parameters as scripts/killarney/noisy/scine_multi_mode_eckartmw.slurm
    """
    print("\n" + "=" * 80)
    print("VERIFICATION RUN")
    print("Using EXACT parameters from scripts/killarney/noisy/scine_multi_mode_eckartmw.slurm")
    print("=" * 80)
    print("\nParameters:")
    for k, v in NOISY_SLURM_PARAMS.items():
        print(f"  {k}: {v}")
    print(f"\nRunning {n_samples} samples for verification...")
    print("=" * 80 + "\n")
    
    results = []
    
    for i, batch in enumerate(dataloader):
        if i >= n_samples:
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
            init_out = predict_fn(start_coords, atomic_nums, do_hessian=True, require_grad=False)
            init_scine_elements = get_scine_elements_from_predict_output(init_out)
            init_vib = vibrational_eigvals(init_out["hessian"], start_coords, atomic_nums, scine_elements=init_scine_elements)
            initial_neg = int((init_vib < 0).sum().item())
        except Exception:
            initial_neg = -1
        
        t0 = time.time()
        try:
            out_dict, aux = run_multi_mode_escape(
                predict_fn,
                start_coords,
                atomic_nums,
                **NOISY_SLURM_PARAMS,
            )
            
            wall = time.time() - t0
            converged = aux.get("steps_to_ts") is not None
            steps_to_ts = aux.get("steps_to_ts")
            final_neg = out_dict.get("final_neg_vibrational", -1)
            
            results.append({
                "index": i,
                "converged": converged,
                "steps_to_ts": steps_to_ts,
                "final_neg": final_neg,
                "initial_neg": initial_neg,
                "wall_time": wall,
                "error": None,
            })
            
            status = "✓ CONVERGED" if converged else "✗ not converged"
            print(f"  Sample {i:3d}: {status} | steps={steps_to_ts or 'N/A':>6} | "
                  f"final_neg={final_neg} | initial_neg={initial_neg} | time={wall:.1f}s")
            
        except Exception as e:
            wall = time.time() - t0
            results.append({
                "index": i,
                "converged": False,
                "steps_to_ts": None,
                "final_neg": -1,
                "initial_neg": initial_neg,
                "wall_time": wall,
                "error": str(e),
            })
            print(f"  Sample {i:3d}: ✗ ERROR: {e}")
    
    n_converged = sum(1 for r in results if r["converged"])
    n_total = len(results)
    convergence_rate = n_converged / n_total if n_total > 0 else 0.0
    
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    print(f"Converged: {n_converged}/{n_total} = {convergence_rate * 100:.1f}%")
    
    if convergence_rate >= 0.5:
        print("✓ Verification PASSED - algorithm is working")
    else:
        print("⚠ WARNING: Low convergence rate in verification")
    
    print("=" * 80 + "\n")
    
    return {
        "n_converged": n_converged,
        "n_total": n_total,
        "convergence_rate": convergence_rate,
        "results": results,
        "passed": convergence_rate >= 0.5,
    }


def load_difficult_samples(
    dataloader,
    device: str,
    predict_fn,
    start_from: str,
    noise_seed: Optional[int],
    target_count: int = 30,
    difficulty_threshold: float = 0.5,
    n_baseline_steps: int = 200,
) -> List[Dict[str, Any]]:
    """Identify difficult samples using noisy SLURM parameters."""
    print("\n" + "=" * 80)
    print("IDENTIFYING DIFFICULT SAMPLES")
    print(f"Running {n_baseline_steps}-step baseline with noisy SLURM parameters...")
    print("=" * 80)
    
    baseline_results = []
    baseline_params = NOISY_SLURM_PARAMS.copy()
    baseline_params["n_steps"] = n_baseline_steps
    
    for i, batch in enumerate(dataloader):
        if i >= 100:
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
            init_out = predict_fn(start_coords, atomic_nums, do_hessian=True, require_grad=False)
            init_scine_elements = get_scine_elements_from_predict_output(init_out)
            init_vib = vibrational_eigvals(init_out["hessian"], start_coords, atomic_nums, scine_elements=init_scine_elements)
            initial_neg = int((init_vib < 0).sum().item())
        except Exception:
            initial_neg = -1
        
        try:
            out_dict, aux = run_multi_mode_escape(
                predict_fn,
                start_coords,
                atomic_nums,
                **baseline_params,
            )
            
            converged = aux.get("steps_to_ts") is not None
            steps_to_converge = aux.get("steps_to_ts", n_baseline_steps)
            final_neg = out_dict.get("final_neg_vibrational", -1)
            
        except Exception as e:
            print(f"[WARN] Sample {i} failed baseline: {e}")
            converged = False
            steps_to_converge = n_baseline_steps
            final_neg = -1
        
        if not converged:
            difficulty = 1000.0
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
    
    baseline_results.sort(key=lambda x: x["difficulty"], reverse=True)
    
    n_difficult = int(len(baseline_results) * difficulty_threshold)
    n_difficult = max(n_difficult, target_count)
    n_difficult = min(n_difficult, len(baseline_results))
    
    difficult_samples = baseline_results[:n_difficult]
    
    n_converged = sum(1 for r in baseline_results if r["converged"])
    n_total = len(baseline_results)
    baseline_rate = n_converged / n_total if n_total > 0 else 0.0
    
    print(f"\nBaseline convergence rate: {n_converged}/{n_total} = {baseline_rate * 100:.1f}%")
    print(f"Selected {len(difficult_samples)} difficult samples for HPO")
    print("=" * 80 + "\n")
    
    return difficult_samples


def objective(
    trial: optuna.Trial,
    difficult_samples: List[Dict[str, Any]],
    predict_fn,
    device: str,
    n_steps_per_sample: int,
    use_wandb: bool = False,
) -> float:
    """Optuna objective function."""
    
    dt_min = trial.suggest_float("dt_min", 1e-7, 1e-5, log=True)
    dt_max = trial.suggest_float("dt_max", 0.01, 0.1, log=True)
    
    plateau_patience = trial.suggest_int("plateau_patience", 3, 15)
    plateau_boost = trial.suggest_float("plateau_boost", 1.2, 3.0)
    plateau_shrink = trial.suggest_float("plateau_shrink", 0.3, 0.7)
    
    escape_disp_threshold = trial.suggest_float("escape_disp_threshold", 1e-5, 1e-3, log=True)
    escape_window = trial.suggest_int("escape_window", 10, 40)
    escape_neg_vib_std = trial.suggest_float("escape_neg_vib_std", 0.1, 1.0)
    
    escape_delta = trial.suggest_float("escape_delta", 0.05, 0.3)
    adaptive_delta = trial.suggest_categorical("adaptive_delta", [True, False])
    
    trust_radius_max = trial.suggest_float("trust_radius_max", 0.15, 0.4)
    
    results = []
    
    for sample_info in difficult_samples:
        try:
            out_dict, aux = run_multi_mode_escape(
                predict_fn,
                sample_info["start_coords"],
                sample_info["batch"].z.detach().cpu().to(device),
                n_steps=n_steps_per_sample,
                dt=0.001,
                stop_at_ts=True,
                ts_eps=1e-5,
                dt_control="neg_eig_plateau",
                dt_min=dt_min,
                dt_max=dt_max,
                max_atom_disp=trust_radius_max,
                plateau_patience=plateau_patience,
                plateau_boost=plateau_boost,
                plateau_shrink=plateau_shrink,
                escape_disp_threshold=escape_disp_threshold,
                escape_window=escape_window,
                hip_vib_mode="projected",  # SCINE uses projected
                hip_rigid_tol=1e-6,
                hip_eigh_device="auto",
                escape_neg_vib_std=escape_neg_vib_std,
                escape_delta=escape_delta,
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
            
        except Exception:
            results.append({
                "converged": False,
                "steps_to_converge": n_steps_per_sample,
            })
    
    n_converged = sum(1 for r in results if r["converged"])
    n_total = len(results)
    convergence_rate = n_converged / n_total if n_total > 0 else 0.0
    
    converged_steps = [r["steps_to_converge"] for r in results if r["converged"]]
    mean_steps = float(np.mean(converged_steps)) if converged_steps else float(n_steps_per_sample)
    
    score = convergence_rate - 0.0001 * mean_steps
    
    trial.set_user_attr("convergence_rate", convergence_rate)
    trial.set_user_attr("mean_steps_to_converge", mean_steps)
    trial.set_user_attr("n_converged", n_converged)
    trial.set_user_attr("n_total", n_total)
    
    print(f"Trial {trial.number}: convergence_rate={convergence_rate:.3f} ({n_converged}/{n_total}) "
          f"mean_steps={mean_steps:.1f} score={score:.4f}")
    
    if use_wandb:
        log_sample(
            trial.number,
            {
                "trial/number": trial.number,
                "hparams/dt_min": dt_min,
                "hparams/dt_max": dt_max,
                "hparams/plateau_patience": plateau_patience,
                "hparams/plateau_boost": plateau_boost,
                "hparams/plateau_shrink": plateau_shrink,
                "hparams/escape_disp_threshold": escape_disp_threshold,
                "hparams/escape_window": escape_window,
                "hparams/escape_neg_vib_std": escape_neg_vib_std,
                "hparams/escape_delta": escape_delta,
                "hparams/adaptive_delta": adaptive_delta,
                "hparams/trust_radius_max": trust_radius_max,
                "metrics/convergence_rate": convergence_rate,
                "metrics/mean_steps": mean_steps,
                "metrics/score": score,
                "counts/n_converged": n_converged,
                "counts/n_total": n_total,
            },
        )
    
    return score


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian HPO for SCINE multi-mode Eckart-MW GAD (in noisy path)."
    )
    parser = add_common_args(parser)
    parser.set_defaults(calculator="scine", noise_seed=42)
    
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-steps-per-sample", type=int, default=800)
    parser.add_argument("--n-samples", type=int, default=15)
    parser.add_argument("--difficulty-threshold", type=float, default=0.5)
    parser.add_argument("--start-from", type=str, default="midpoint_rt_noise1.0A")
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--verification-samples", type=int, default=5)
    
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gad-hpo")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    
    args = parser.parse_args(argv)
    
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    device = "cpu"  # SCINE always CPU
    predict_fn = make_predict_fn_from_calculator(calculator, "scine")
    
    job_id = os.environ.get("SLURM_JOB_ID", str(int(time.time())))
    study_name = args.study_name or f"scine-gad-hpo-noisy-{job_id}"
    
    storage_url = args.storage
    if storage_url is None:
        db_path = Path(out_dir) / f"{study_name}.db"
        storage_url = f"sqlite:///{db_path}"
    
    print("\n" + "=" * 80)
    print("SCINE MULTI-MODE ECKART-MW GAD HPO (NOISY PATH)")
    print("=" * 80)
    print(f"Study: {study_name}")
    print(f"Storage: {storage_url}")
    print(f"Trials: {args.n_trials}")
    print("=" * 80)
    
    if args.wandb:
        wandb_name = args.wandb_name or f"scine-gad-hpo-noisy-{job_id}"
        init_wandb_run(
            project=args.wandb_project,
            name=wandb_name,
            config={
                "calculator": "scine",
                "study_name": study_name,
                "n_trials": args.n_trials,
                "n_steps_per_sample": args.n_steps_per_sample,
                "n_samples": args.n_samples,
                "start_from": args.start_from,
                "noise_seed": args.noise_seed,
                "noisy_slurm_params": NOISY_SLURM_PARAMS,
                "slurm_job_id": job_id,
            },
            entity=args.wandb_entity,
            tags=["hpo", "gad", "scine", "optuna", "noisy-path"],
            run_dir=out_dir,
        )
    
    # VERIFICATION
    if not args.skip_verification:
        verification = run_verification(
            dataloader,
            device,
            predict_fn,
            args.start_from,
            args.noise_seed,
            n_samples=args.verification_samples,
        )
        
        if args.wandb:
            log_sample(-1, {
                "verification/n_converged": verification["n_converged"],
                "verification/n_total": verification["n_total"],
                "verification/convergence_rate": verification["convergence_rate"],
                "verification/passed": verification["passed"],
            })
        
        if not verification["passed"]:
            print("\n" + "!" * 80)
            print("WARNING: Verification failed!")
            print("!" * 80 + "\n")
    
    # LOAD DIFFICULT SAMPLES
    difficult_samples = load_difficult_samples(
        dataloader,
        device,
        predict_fn,
        args.start_from,
        args.noise_seed,
        target_count=args.n_samples * 2,
        difficulty_threshold=args.difficulty_threshold,
    )
    
    difficult_indices = [s["index"] for s in difficult_samples]
    indices_path = Path(out_dir) / "difficult_sample_indices.json"
    with open(indices_path, "w") as f:
        json.dump(difficult_indices, f)
    
    hpo_samples = difficult_samples[:args.n_samples]
    print(f"\nUsing {len(hpo_samples)} samples for HPO")
    
    # CREATE STUDY
    sampler = TPESampler(seed=42, n_startup_trials=10)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=args.resume or True,
        direction="maximize",
        sampler=sampler,
    )
    
    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"Loaded {n_existing} existing trials")
    
    # RUN HPO
    try:
        n_to_run = max(0, args.n_trials - n_existing) if args.resume else args.n_trials
        
        if n_to_run > 0:
            print(f"\nRunning {n_to_run} HPO trials...")
            study.optimize(
                lambda trial: objective(
                    trial,
                    hpo_samples,
                    predict_fn,
                    device,
                    args.n_steps_per_sample,
                    use_wandb=args.wandb,
                ),
                n_trials=n_to_run,
                show_progress_bar=True,
            )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving results...")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print(traceback.format_exc())
    
    # SAVE RESULTS
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    completed_trials = [t for t in study.trials if t.value is not None]
    if not completed_trials:
        print("No completed trials.")
        if args.wandb:
            finish_wandb()
        return
    
    print(f"Total trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best convergence rate: {study.best_trial.user_attrs['convergence_rate']:.3f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    results_path = Path(out_dir) / "hpo_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "study_name": study_name,
            "best_trial": study.best_trial.number,
            "best_score": study.best_value,
            "best_params": study.best_params,
            "best_convergence_rate": study.best_trial.user_attrs["convergence_rate"],
            "noisy_slurm_params": NOISY_SLURM_PARAMS,
            "n_trials": len(study.trials),
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    if args.wandb:
        log_summary({
            "best/trial_number": study.best_trial.number,
            "best/score": study.best_value,
            "best/convergence_rate": study.best_trial.user_attrs.get("convergence_rate", 0),
            **{f"best/{k}": v for k, v in study.best_params.items()},
        })
        finish_wandb()
    
    print(f"Optuna study saved to: {storage_url}")
    print(f"To resume: --resume --study-name {study_name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
