"""SCINE Sella Hyperparameter Optimization for TS Convergence.

This module implements a grid search over Sella hyperparameters to maximize
convergence to true first-order saddle points (exactly 1 negative eigenvalue).

Based on insights from:
- Wander et al. (2024) "Accessing Numerical Energy Hessians with Graph Neural
  Network Potentials and Their Application in Heterogeneous Catalysis"
  (arXiv:2410.01650v2)

Key findings from the paper informing this HPO:
1. Using exact Hessians improved TS convergence from ~65% to 93%
2. Trust radius management is critical for stability
3. Force convergence threshold affects saddle point quality
4. Hessian update frequency impacts both accuracy and cost

The primary metric optimized is the eigenvalue TS rate: the fraction of
converged structures with exactly 1 negative vibrational eigenvalue.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ...dependencies.common_utils import (
    add_common_args,
    parse_starting_geometry,
    setup_experiment,
)
from ...dependencies.experiment_logger import ExperimentLogger, RunResult
from ...dependencies.hessian import (
    vibrational_eigvals,
    get_scine_elements_from_predict_output,
)
from ...logging import finish_wandb, init_wandb_run, log_sample, log_summary
from ...runners._predict import make_predict_fn_from_calculator
from .sella_ts import run_sella_ts


@dataclass
class HPOConfig:
    """Configuration for a single hyperparameter combination.
    
    Trust radius parameters from Wander et al. (2024) arXiv:2410.01650v2.
    """
    fmax: float = 0.03
    delta0: float = 0.048  # Paper value: 4.8E-2
    use_exact_hessian: bool = True  # Always True (paper-recommended)
    diag_every_n: int = 1  # Always 1 (every step with exact Hessian)
    gamma: float = 0.0
    internal: bool = True  # Use internal coordinates
    order: int = 1  # Saddle order (1 for TS)
    # Trust radius parameters from paper (fixed, not grid-searched)
    rho_inc: float = 1.035
    rho_dec: float = 5.0
    sigma_inc: float = 1.15
    sigma_dec: float = 0.65
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_str(self) -> str:
        """Short string representation for logging."""
        coords = "int" if self.internal else "cart"
        return f"fmax{self.fmax}_d0{self.delta0}_{coords}"


@dataclass
class HPOResult:
    """Results for a single HPO configuration."""
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


def generate_hpo_grid(
    fmax_values: List[float],
    internal_values: List[bool] = [True],
) -> List[HPOConfig]:
    """Generate grid of HPO configurations.
    
    Based on paper insights (Wander et al. 2024, arXiv:2410.01650v2):
    - Tighter fmax helps saddle character verification
    - Exact Hessians crucial for ML/semiempirical potentials (65% -> 93% improvement)
    - diag_every_n=1 recommended with exact Hessians (always use every step)
    
    Grid search over: FMAX only
    
    Fixed parameters (from paper):
    - delta0: 0.048 (paper value: 4.8E-2)
    - use_exact_hessian: True (always)
    - diag_every_n: 1 (always, Hessian at every step)
    - rho_inc: 1.035, rho_dec: 5.0, sigma_inc: 1.15, sigma_dec: 0.65
    """
    configs = []
    
    for fmax, internal in itertools.product(
        fmax_values,
        internal_values,
    ):
        configs.append(HPOConfig(
            fmax=fmax,
            delta0=0.048,  # Fixed at paper value
            use_exact_hessian=True,  # Always True (paper)
            diag_every_n=1,  # Always 1 (paper)
            gamma=0.0,  # Not used with exact Hessians
            internal=internal,
            # Trust radius params are fixed at paper values in dataclass
        ))
    
    return configs


def run_hpo_experiment(
    calculator,
    dataloader,
    device: str,
    config: HPOConfig,
    max_steps: int,
    max_samples: int,
    start_from: str,
    noise_seed: Optional[int],
    traj_dir: str,
    verbose: bool = True,
) -> HPOResult:
    """Run experiment with a single HPO configuration."""
    
    # Create predict function for eigenvalue validation
    predict_fn = make_predict_fn_from_calculator(calculator, "scine")
    
    result = HPOResult(config=config)
    
    config_str = config.to_str()
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running HPO config: {config_str}")
        print(f"{'='*60}")
    
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
                save_trajectory=False,  # Skip trajectory saving for HPO
                trajectory_dir=None,
                sample_index=i,
                logfile=None,
                verbose=False,  # Quieter for HPO
                use_exact_hessian=config.use_exact_hessian,
                diag_every_n=config.diag_every_n,
                gamma=config.gamma,
                # Trust radius params from paper (Wander et al. 2024)
                rho_inc=config.rho_inc,
                rho_dec=config.rho_dec,
                sigma_inc=config.sigma_inc,
                sigma_dec=config.sigma_dec,
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
            result.neg_eigval_counts[-1] += 1  # -1 indicates error
    
    if verbose:
        print(f"Config {config_str}:")
        print(f"  Sella convergence: {result.sella_convergence_rate:.1%}")
        print(f"  Eigenvalue TS rate: {result.eigenvalue_ts_rate:.1%}")
        print(f"  Both criteria: {result.both_rate:.1%}")
        print(f"  Avg steps: {result.avg_steps:.1f}")
    
    return result


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for Sella HPO experiment."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for Sella TS search with SCINE."
    )
    parser = add_common_args(parser)
    parser.set_defaults(calculator="scine")
    
    # HPO grid parameters
    # FMAX: Force convergence threshold (eV/Å)
    # Very tight values for precise saddle point convergence
    parser.add_argument(
        "--fmax-values",
        type=float,
        nargs="+",
        default=[0.00005, 0.0001, 0.0003, 0.0005],
        help="Force convergence thresholds to test (eV/A). "
             "Default: [0.00005, 0.0001, 0.0003, 0.0005]",
    )
    parser.add_argument(
        "--test-cartesian",
        action="store_true",
        help="Also test Cartesian coordinates (in addition to internal).",
    )
    # NOTE: The following are FIXED based on paper (Wander et al. 2024):
    # - delta0: 0.048 (paper value: 4.8E-2)
    # - use_exact_hessian: True (always - 65% -> 93% improvement)
    # - diag_every_n: 1 (always - Hessian at every step)
    # - rho_inc=1.035, rho_dec=5.0, sigma_inc=1.15, sigma_dec=0.65
    
    # Experiment parameters
    parser.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="Maximum Sella optimization steps per sample. Default: 150",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default="midpoint_rt_noise1.0A",
        help="Starting geometry. Default: midpoint_rt_noise1.0A",
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
    
    # Parse coordinate modes
    internal_values = [True]
    if args.test_cartesian:
        internal_values.append(False)
    
    # Generate HPO grid (FMAX only)
    # All configs use: exact Hessian, diag_every_n=1, delta0=0.048, and paper's trust radius params
    configs = generate_hpo_grid(
        fmax_values=args.fmax_values,
        internal_values=internal_values,
    )
    
    print(f"\n{'='*80}")
    print("SELLA HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Total HPO configurations: {len(configs)}")
    print(f"Samples per config: {args.max_samples}")
    print(f"Max steps per sample: {args.max_steps}")
    print(f"Starting geometry: {args.start_from}")
    print(f"{'='*80}\n")
    
    # Initialize W&B if requested
    if args.wandb:
        wandb_name = args.wandb_name or f"sella-hpo-scine-{len(configs)}configs"
        init_wandb_run(
            project=args.wandb_project,
            name=wandb_name,
            config={
                "n_configs": len(configs),
                "max_samples": args.max_samples,
                "max_steps": args.max_steps,
                "start_from": args.start_from,
                "fmax_values": args.fmax_values,
                # Fixed parameters (from paper)
                "delta0": 0.048,
                "use_exact_hessian": True,
                "diag_every_n": 1,
                "rho_inc": 1.035,
                "rho_dec": 5.0,
                "sigma_inc": 1.15,
                "sigma_dec": 0.65,
            },
            entity=args.wandb_entity,
            tags=["hpo", "sella", "scine"],
            run_dir=out_dir,
        )
    
    # Trajectory directory (not used for HPO but create anyway)
    traj_dir = os.path.join(out_dir, "sella_hpo_trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    
    # Run all configurations
    all_results: List[HPOResult] = []
    
    for config_idx, config in enumerate(configs):
        print(f"\n[{config_idx + 1}/{len(configs)}] Testing config: {config.to_str()}")
        
        result = run_hpo_experiment(
            calculator=calculator,
            dataloader=dataloader,
            device=device,
            config=config,
            max_steps=args.max_steps,
            max_samples=args.max_samples,
            start_from=args.start_from,
            noise_seed=getattr(args, "noise_seed", None),
            traj_dir=traj_dir,
            verbose=args.verbose,
        )
        
        all_results.append(result)
        
        # Log to W&B
        if args.wandb:
            log_sample(
                config_idx,
                {
                    "config_str": config.to_str(),
                    "sella_convergence_rate": result.sella_convergence_rate,
                    "eigenvalue_ts_rate": result.eigenvalue_ts_rate,
                    "both_rate": result.both_rate,
                    "avg_steps": result.avg_steps,
                    "avg_wall_time": result.avg_wall_time,
                    "avg_final_fmax": result.avg_final_fmax,
                    "fmax": config.fmax,
                    "delta0": config.delta0,
                    "use_exact_hessian": int(config.use_exact_hessian),
                    "diag_every_n": config.diag_every_n,
                    "gamma": config.gamma,
                    "internal": int(config.internal),
                },
            )
    
    # Sort results by eigenvalue TS rate (primary metric)
    sorted_results = sorted(all_results, key=lambda r: r.eigenvalue_ts_rate, reverse=True)
    
    # Print summary
    print(f"\n{'='*80}")
    print("HPO RESULTS SUMMARY (sorted by eigenvalue TS rate)")
    print(f"{'='*80}")
    print(f"{'Config':<50} {'TS Rate':>10} {'Sella Conv':>12} {'Both':>10} {'Avg Steps':>12}")
    print("-" * 94)
    
    for result in sorted_results:
        config_str = result.config.to_str()[:48]
        print(
            f"{config_str:<50} "
            f"{result.eigenvalue_ts_rate:>9.1%} "
            f"{result.sella_convergence_rate:>11.1%} "
            f"{result.both_rate:>9.1%} "
            f"{result.avg_steps:>11.1f}"
        )
    
    print("-" * 94)
    
    # Best configuration
    best = sorted_results[0]
    print(f"\nBEST CONFIGURATION:")
    print(f"  Config: {best.config.to_str()}")
    print(f"  Eigenvalue TS rate: {best.eigenvalue_ts_rate:.1%}")
    print(f"  Sella convergence rate: {best.sella_convergence_rate:.1%}")
    print(f"  Both criteria satisfied: {best.both_rate:.1%}")
    print(f"  Average steps: {best.avg_steps:.1f}")
    print(f"  Average wall time: {best.avg_wall_time:.1f}s")
    print()
    print("Best config parameters:")
    for k, v in best.config.to_dict().items():
        print(f"  {k}: {v}")
    
    # Save results to JSON
    results_path = Path(out_dir) / "hpo_results.json"
    with open(results_path, "w") as f:
        json.dump([r.to_dict() for r in sorted_results], f, indent=2)
    print(f"\nSaved results to: {results_path}")
    
    # Save best config
    best_config_path = Path(out_dir) / "best_config.json"
    with open(best_config_path, "w") as f:
        json.dump(best.config.to_dict(), f, indent=2)
    print(f"Saved best config to: {best_config_path}")
    
    # Log final summary to W&B
    if args.wandb:
        summary = {
            "best_config": best.config.to_str(),
            "best_eigenvalue_ts_rate": best.eigenvalue_ts_rate,
            "best_sella_convergence_rate": best.sella_convergence_rate,
            "best_both_rate": best.both_rate,
            "best_avg_steps": best.avg_steps,
            "best_fmax": best.config.fmax,
            "best_delta0": best.config.delta0,
            "best_use_exact_hessian": best.config.use_exact_hessian,
            "best_diag_every_n": best.config.diag_every_n,
        }
        log_summary(summary)
        finish_wandb()
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

