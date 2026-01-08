"""SCINE Sella Hyperparameter Optimization using Optuna.

This module implements Bayesian hyperparameter optimization for Sella TS search
with the SCINE (semiempirical) calculator, optimizing for transition state convergence.

Key hyperparameters being optimized:
- delta0: Initial trust radius
- rho_inc, rho_dec: Trust radius adjustment thresholds
- sigma_inc, sigma_dec: Trust radius adjustment factors
- fmax: Force convergence threshold [1e-4, 1e-2]
- apply_eckart: Whether to apply Eckart projection before Sella internal conversion
  (Eckart removes trans/rot modes from Cartesian Hessian before Sella's internal
   coord conversion. NOT mass-weighted in final output.)

Fixed parameters:
- gamma = 0.0 (tightest Hessian convergence)
- max_steps = 100 per sample
- samples per trial = 30
- internal = True (ALWAYS use Sella's internal coordinates)
- use_exact_hessian = True (SCINE provides analytical Hessian)
- diag_every_n = 1 (fresh Hessian every step)

Objective function (in priority order):
1. PRIMARY: eigenvalue_ts_rate (fraction with exactly 1 negative eigenvalue)
2. SECONDARY: convergence speed (fewer steps is better)
3. TERTIARY: sella_convergence_rate (fmax reached)

Uses Optuna pruning to stop bad trials early after 10 samples.

Note: SCINE can use broader ranges than HIP since the default parameters are more
reasonable for SCINE's analytical Hessians.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler
import torch

from ...dependencies.common_utils import (
    add_common_args,
    parse_starting_geometry,
    setup_experiment,
)
from ...dependencies.hessian import (
    vibrational_eigvals,
    get_scine_elements_from_predict_output,
)
from ...logging import finish_wandb, init_wandb_run, log_sample, log_summary
from ...runners._predict import make_predict_fn_from_calculator
from .sella_ts import run_sella_ts

# Suppress Optuna's verbose logging (we log to W&B instead)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class HPOConfig:
    """Configuration for a single hyperparameter combination."""
    # Hyperparameters being optimized
    delta0: float = 0.1
    rho_inc: float = 1.035
    rho_dec: float = 10.0
    sigma_inc: float = 1.15
    sigma_dec: float = 0.75
    fmax: float = 1e-3  # Now a hyperparameter [1e-4, 1e-2]
    apply_eckart: bool = False  # Eckart project Hessian before Sella internal conversion
    
    # Fixed parameters
    gamma: float = 0.0
    internal: bool = True  # ALWAYS use internal coordinates
    order: int = 1
    use_exact_hessian: bool = True
    diag_every_n: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_str(self) -> str:
        """Short string representation for logging."""
        eckart_str = "eckart" if self.apply_eckart else "noeckart"
        return f"d0{self.delta0:.3f}_rdec{self.rho_dec:.1f}_sdec{self.sigma_dec:.2f}_fmax{self.fmax:.0e}_{eckart_str}"


@dataclass
class HPOResult:
    """Results for a single HPO trial."""
    config: HPOConfig
    n_samples: int = 0
    n_sella_converged: int = 0
    n_eigenvalue_ts: int = 0  # Exactly 1 negative eigenvalue
    n_both: int = 0  # Sella converged AND eigenvalue TS
    n_sella_only: int = 0  # Sella converged but NOT eigenvalue TS
    
    # Per-sample metrics
    final_fmax_list: List[float] = field(default_factory=list)
    steps_list: List[int] = field(default_factory=list)
    wall_time_list: List[float] = field(default_factory=list)
    neg_eigval_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def sella_convergence_rate(self) -> float:
        return self.n_sella_converged / self.n_samples if self.n_samples > 0 else 0.0
    
    @property
    def eigenvalue_ts_rate(self) -> float:
        """Primary metric: fraction with exactly 1 negative eigenvalue."""
        return self.n_eigenvalue_ts / self.n_samples if self.n_samples > 0 else 0.0
    
    @property
    def both_rate(self) -> float:
        """Both Sella converged AND eigenvalue TS."""
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
            "neg_eigval_distribution": dict(self.neg_eigval_counts),
        }


def run_trial_evaluation(
    calculator,
    dataloader,
    device: str,
    config: HPOConfig,
    max_steps: int,
    max_samples: int,
    start_from: str,
    noise_seed: Optional[int],
    verbose: bool = False,
    trial: Optional[optuna.Trial] = None,  # For pruning support
    prune_after_n: int = 10,  # Check for pruning after this many samples
) -> HPOResult:
    """Run Sella optimization for a batch of samples with given config.
    
    Supports Optuna pruning: if trial is provided, reports intermediate values
    and may raise TrialPruned if the trial looks unpromising.
    """
    
    # Create predict function for eigenvalue validation
    predict_fn = make_predict_fn_from_calculator(calculator, "scine")
    
    result = HPOResult(config=config)
    
    for i, batch in enumerate(dataloader):
        if i >= max_samples:
            break
        
        batch = batch.to(device)
        atomic_nums = batch.z.detach().cpu().to(device)
        
        # Get starting geometry
        start_coords = parse_starting_geometry(
            start_from,
            batch,
            noise_seed=noise_seed,
            sample_index=i,
        ).detach().to(device)
        
        t0 = time.time()
        
        try:
            out_dict, aux = run_sella_ts(
                calculator,
                "scine",
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
            wall_time = time.time() - t0
            
            # Track metrics
            result.n_samples += 1
            result.steps_list.append(out_dict["steps_taken"])
            result.wall_time_list.append(wall_time)
            
            if out_dict.get("final_fmax") is not None:
                result.final_fmax_list.append(out_dict["final_fmax"])
            
            sella_converged = bool(out_dict.get("converged", False))
            if sella_converged:
                result.n_sella_converged += 1
            
            # Eigenvalue validation
            final_coords = out_dict["final_coords"].to(device)
            try:
                final_out = predict_fn(final_coords, atomic_nums, do_hessian=True, require_grad=False)
                final_scine_elements = get_scine_elements_from_predict_output(final_out)
                final_vib = vibrational_eigvals(
                    final_out["hessian"], final_coords, atomic_nums, 
                    scine_elements=final_scine_elements
                )
                final_neg = int((final_vib < 0).sum().item())
            except Exception:
                final_neg = -1
            
            result.neg_eigval_counts[final_neg] += 1
            
            is_ts = final_neg == 1
            if is_ts:
                result.n_eigenvalue_ts += 1
            
            if sella_converged and is_ts:
                result.n_both += 1
            elif sella_converged and not is_ts:
                result.n_sella_only += 1
                
        except Exception as e:
            if verbose:
                print(f"[WARN] Sample {i} failed: {e}")
            result.n_samples += 1
            result.neg_eigval_counts[-1] += 1
        
        # Pruning check after prune_after_n samples
        if trial is not None and result.n_samples == prune_after_n:
            intermediate_score = compute_score(result, max_steps)
            trial.report(intermediate_score, step=prune_after_n)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    return result


def compute_score(result: HPOResult, max_steps: int) -> float:
    """Compute composite score for HPO objective.
    
    Priority order:
    1. PRIMARY: eigenvalue_ts_rate (fraction with exactly 1 neg eigenvalue) - weight 1.0
    2. SECONDARY: speed bonus (fewer steps) - weight 0.01
    3. TERTIARY: sella_convergence_rate - weight 0.001
    """
    ts_rate = result.eigenvalue_ts_rate
    step_bonus = 0.01 * (1 - result.avg_steps / max_steps) if result.avg_steps > 0 else 0
    sella_bonus = 0.001 * result.sella_convergence_rate
    return ts_rate + step_bonus + sella_bonus


def create_objective(
    calculator,
    dataloader,
    device: str,
    max_steps: int,
    max_samples: int,
    start_from: str,
    noise_seed: Optional[int],
    verbose: bool,
    use_wandb: bool,
    prune_after_n: int = 10,
):
    """Create Optuna objective function for SCINE HPO.
    
    Uses MedianPruner-compatible intermediate reporting after prune_after_n samples.
    """
    
    trial_count = [0]  # Mutable counter for trial tracking
    
    def objective(trial: optuna.Trial) -> float:
        """Single trial evaluation for Optuna."""
        trial_count[0] += 1
        
        # Sample hyperparameters - SCINE uses broader ranges since defaults are OK
        # but HIP-optimal values may be even better for SCINE
        # Range: [0.03, 0.8] for delta0 (includes default 0.048)
        
        delta0 = trial.suggest_float("delta0", 0.03, 0.8, log=True)
        rho_dec = trial.suggest_float("rho_dec", 3.0, 80.0)
        rho_inc = trial.suggest_float("rho_inc", 1.01, 1.1)
        sigma_dec = trial.suggest_float("sigma_dec", 0.5, 0.95)
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
        
        # Run evaluation with pruning support
        try:
            result = run_trial_evaluation(
                calculator=calculator,
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
            )
        except optuna.TrialPruned:
            if verbose:
                print(f"[Trial {trial_count[0]}] PRUNED after {prune_after_n} samples")
            raise
        
        # Compute composite score (priority: TS rate > speed > Sella convergence)
        score = compute_score(result, max_steps)
        
        if verbose:
            print(f"[Trial {trial_count[0]}] TS rate: {result.eigenvalue_ts_rate:.1%}, "
                  f"Sella conv: {result.sella_convergence_rate:.1%}, "
                  f"Both: {result.both_rate:.1%}, "
                  f"Avg steps: {result.avg_steps:.1f}, "
                  f"Score: {score:.4f}")
        
        # Store detailed user attrs on trial for later analysis
        trial.set_user_attr("eigenvalue_ts_rate", result.eigenvalue_ts_rate)
        trial.set_user_attr("sella_convergence_rate", result.sella_convergence_rate)
        trial.set_user_attr("both_rate", result.both_rate)
        trial.set_user_attr("avg_steps", result.avg_steps)
        trial.set_user_attr("avg_wall_time", result.avg_wall_time)
        trial.set_user_attr("n_samples", result.n_samples)
        trial.set_user_attr("neg_eigval_distribution", dict(result.neg_eigval_counts))
        
        # Log to W&B with detailed metrics (no plots)
        if use_wandb:
            log_sample(
                trial_count[0] - 1,  # 0-indexed
                {
                    # Trial identification
                    "trial/number": trial_count[0],
                    "trial/config_str": config.to_str(),
                    # Hyperparameters (prefixed for grouping)
                    "hparams/delta0": delta0,
                    "hparams/rho_inc": rho_inc,
                    "hparams/rho_dec": rho_dec,
                    "hparams/sigma_inc": sigma_inc,
                    "hparams/sigma_dec": sigma_dec,
                    "hparams/fmax": fmax,
                    "hparams/apply_eckart": int(apply_eckart),
                    # Primary metrics
                    "metrics/eigenvalue_ts_rate": result.eigenvalue_ts_rate,
                    "metrics/sella_convergence_rate": result.sella_convergence_rate,
                    "metrics/both_rate": result.both_rate,
                    "metrics/score": score,
                    # Counts
                    "counts/n_samples": result.n_samples,
                    "counts/n_eigenvalue_ts": result.n_eigenvalue_ts,
                    "counts/n_sella_converged": result.n_sella_converged,
                    "counts/n_both": result.n_both,
                    "counts/n_sella_only": result.n_sella_only,
                    # Performance
                    "perf/avg_steps": result.avg_steps,
                    "perf/avg_wall_time": result.avg_wall_time,
                    "perf/avg_final_fmax": result.avg_final_fmax,
                    # Negative eigenvalue distribution
                    **{f"neg_eigval_dist/{k}": v for k, v in result.neg_eigval_counts.items()},
                },
            )
        
        return score
    
    return objective


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for SCINE Sella HPO experiment."""
    parser = argparse.ArgumentParser(
        description="Bayesian HPO for Sella TS search with SCINE calculator."
    )
    parser = add_common_args(parser)
    parser.set_defaults(calculator="scine")
    
    # HPO parameters
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run. Default: 50",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum Sella optimization steps per sample. Default: 100",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name (for resuming). Default: auto-generated",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing study if it exists (requires --study-name)",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default="midpoint_rt_noise1.0A",
        help="Starting geometry. Default: midpoint_rt_noise1.0A",
    )
    parser.add_argument(
        "--optuna-seed",
        type=int,
        default=42,
        help="Random seed for Optuna TPE sampler. Default: 42",
    )
    
    # W&B arguments
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="sella-hpo")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    
    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
    )
    
    args = parser.parse_args(argv)
    
    # Force SCINE calculator
    args.calculator = "scine"
    
    # Set up experiment
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    device = "cpu"  # SCINE runs on CPU only
    
    print(f"\n{'='*80}")
    print("SCINE SELLA HYPERPARAMETER OPTIMIZATION (Optuna)")
    print(f"{'='*80}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Samples per trial: {args.max_samples}")
    print(f"Max steps per sample: {args.max_steps}")
    print(f"Starting geometry: {args.start_from}")
    print(f"Device: {device}")
    print(f"SCINE functional: {getattr(args, 'scine_functional', 'DFTB0')}")
    print(f"{'='*80}")
    print("\nFixed parameters:")
    print(f"  fmax: 1e-3")
    print(f"  gamma: 0.0")
    print(f"  internal: True")
    print(f"  use_exact_hessian: True")
    print(f"  diag_every_n: 1")
    print(f"{'='*80}\n")
    
    # Create SQLite storage for crash recovery
    study_name = args.study_name or f"scine_sella_hpo_{int(time.time())}"
    db_path = Path(out_dir) / f"{study_name}.db"
    storage_url = f"sqlite:///{db_path}"
    
    print(f"Optuna storage: {db_path}")
    print(f"Study name: {study_name}")
    if args.resume:
        print("Resume mode: will continue from existing study if present")
    
    # Initialize W&B if requested
    if args.wandb:
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        wandb_name = args.wandb_name or f"scine-sella-hpo-{args.n_trials}trials-job{job_id}"
        init_wandb_run(
            project=args.wandb_project,
            name=wandb_name,
            config={
                # Experiment setup
                "calculator": "scine",
                "scine_functional": getattr(args, "scine_functional", "DFTB0"),
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
                # Fixed Sella parameters
                "gamma": 0.0,
                "internal": True,  # ALWAYS use internal coords
                "use_exact_hessian": True,  # SCINE analytical Hessian
                "diag_every_n": 1,
                "order": 1,  # First-order saddle point (TS)
                # HPO search ranges (broader for SCINE)
                "hpo/delta0_range": [0.03, 0.8],
                "hpo/delta0_log_scale": True,
                "hpo/rho_dec_range": [3.0, 80.0],
                "hpo/rho_inc_range": [1.01, 1.1],
                "hpo/sigma_dec_range": [0.5, 0.95],
                "hpo/sigma_inc_range": [1.1, 1.8],
                "hpo/fmax_range": [1e-4, 1e-2],
                "hpo/fmax_log_scale": True,
                "hpo/apply_eckart_options": [True, False],
                # Objective function weights
                "objective/ts_rate_weight": 1.0,
                "objective/speed_weight": 0.01,
                "objective/sella_conv_weight": 0.001,
            },
            entity=args.wandb_entity,
            tags=["hpo", "sella", "scine", "optuna", "bayesian", f"job-{job_id}"],
            run_dir=out_dir,
        )
    
    # Create Optuna study with TPE sampler and MedianPruner
    sampler = TPESampler(seed=args.optuna_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Don't prune first 5 trials
        n_warmup_steps=0,  # Start pruning immediately after prune_after_n samples
    )
    
    # Load or create study
    load_if_exists = args.resume
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=load_if_exists,
    )
    
    # Report existing progress if resuming
    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"Loaded {n_existing} existing trials from {db_path}")
        n_remaining = max(0, args.n_trials - n_existing)
        print(f"Will run {n_remaining} more trials")
    
    # Create objective function
    objective = create_objective(
        calculator=calculator,
        dataloader=dataloader,
        device=device,
        max_steps=args.max_steps,
        max_samples=args.max_samples,
        start_from=args.start_from,
        noise_seed=getattr(args, "noise_seed", None),
        verbose=args.verbose,
        use_wandb=args.wandb,
    )
    
    # Run optimization with graceful error handling
    # Note: SQLite storage ensures trials are saved after each completion
    try:
        # Calculate how many trials to run (accounting for existing trials if resuming)
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
    
    # Print results summary (even if interrupted)
    print(f"\n{'='*80}")
    print("HPO RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Check if we have any completed trials
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
    
    # Top 5 trials
    print(f"\nTop 5 trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)
    print(f"{'Rank':<6} {'Score':<10} {'delta0':<10} {'rho_dec':<10} {'sigma_dec':<10} {'fmax':<10} {'eckart':<8}")
    print("-" * 74)
    for rank, trial in enumerate(trials_sorted[:5], 1):
        if trial.value is not None:
            print(f"{rank:<6} {trial.value:<10.4f} "
                  f"{trial.params['delta0']:<10.4f} {trial.params['rho_dec']:<10.1f} "
                  f"{trial.params['sigma_dec']:<10.3f} {trial.params['fmax']:<10.2e} "
                  f"{trial.params['apply_eckart']!s:<8}")
    
    print(f"\n{'='*80}")
    
    # Save detailed results to JSON
    results_path = Path(out_dir) / "scine_hpo_results.json"
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
    
    # Save best config
    best_config_path = Path(out_dir) / "scine_best_config.json"
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
    
    # Log final summary to W&B
    if args.wandb:
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_completed = len([t for t in study.trials if t.value is not None])
        n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        # Get best trial's detailed metrics from user attrs
        best_trial = study.best_trial
        best_ts_rate = best_trial.user_attrs.get("eigenvalue_ts_rate", 0)
        best_sella_rate = best_trial.user_attrs.get("sella_convergence_rate", 0)
        best_both_rate = best_trial.user_attrs.get("both_rate", 0)
        best_avg_steps = best_trial.user_attrs.get("avg_steps", 0)
        
        # Compute aggregate statistics across all completed trials
        all_scores = [t.value for t in study.trials if t.value is not None]
        all_ts_rates = [t.user_attrs.get("eigenvalue_ts_rate", 0) for t in study.trials if t.value is not None]
        
        summary = {
            # Best trial info
            "best/trial_number": best_trial.number,
            "best/score": best_trial.value,
            "best/eigenvalue_ts_rate": best_ts_rate,
            "best/sella_convergence_rate": best_sella_rate,
            "best/both_rate": best_both_rate,
            "best/avg_steps": best_avg_steps,
            # Best hyperparameters
            "best/delta0": best_trial.params["delta0"],
            "best/rho_inc": best_trial.params["rho_inc"],
            "best/rho_dec": best_trial.params["rho_dec"],
            "best/sigma_inc": best_trial.params["sigma_inc"],
            "best/sigma_dec": best_trial.params["sigma_dec"],
            "best/fmax": best_trial.params["fmax"],
            "best/apply_eckart": best_trial.params["apply_eckart"],
            # Trial statistics
            "trials/n_completed": n_completed,
            "trials/n_pruned": n_pruned,
            "trials/n_failed": n_failed,
            "trials/n_total": len(study.trials),
            # Aggregate metrics
            "aggregate/mean_score": float(np.mean(all_scores)) if all_scores else 0,
            "aggregate/std_score": float(np.std(all_scores)) if all_scores else 0,
            "aggregate/max_score": float(max(all_scores)) if all_scores else 0,
            "aggregate/mean_ts_rate": float(np.mean(all_ts_rates)) if all_ts_rates else 0,
            "aggregate/max_ts_rate": float(max(all_ts_rates)) if all_ts_rates else 0,
        }
        log_summary(summary)
        finish_wandb()
    
    print(f"{'='*80}")
    print(f"Optuna study saved to: {db_path}")
    print(f"To resume: --resume --study-name {study_name}")


if __name__ == "__main__":
    main()

