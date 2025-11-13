# src/experiment_logger.py
"""
Unified logging infrastructure for GAD experiments.

Provides:
- Structured output directory management (scriptname/loss-type-flags/transition-types/)
- Transition-based graph sampling (max 10 random samples per transition)
- Aggregate statistics computation and logging
- Weights & Biases (W&B) integration for experiment tracking
"""
import os
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np

# W&B import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


@dataclass
class RunResult:
    """Single run result container."""
    sample_index: int
    formula: str
    initial_neg_eigvals: int
    final_neg_eigvals: int
    initial_neg_vibrational: Optional[int]
    final_neg_vibrational: Optional[int]
    steps_taken: int
    steps_to_ts: Optional[int]  # Steps to reach TS (1 neg eigenvalue), None if not reached
    final_time: Optional[float]
    final_eig0: Optional[float]
    final_eig1: Optional[float]
    final_eig_product: Optional[float]
    final_loss: Optional[float]
    rmsd_to_known_ts: Optional[float]
    stop_reason: Optional[str]
    plot_path: Optional[str]

    # Additional fields for specific experiments
    extra_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_data is None:
            self.extra_data = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, flattening extra_data."""
        base = asdict(self)
        extra = base.pop('extra_data', {})
        base.update(extra)
        return base

    @property
    def transition_key(self) -> str:
        """Return transition type key (e.g., '0neg-to-1neg')."""
        return f"{self.initial_neg_eigvals}neg-to-{self.final_neg_eigvals}neg"

    @property
    def reached_ts(self) -> bool:
        """Did this run reach a TS (1 negative eigenvalue)?"""
        return self.final_neg_eigvals == 1


class ExperimentLogger:
    """
    Manages experiment logging with structured output directories and sampling.

    Directory structure:
        base_dir/
            scriptname/                  # e.g., 'gad-rk45', 'gad-eigdescent'
                loss-type-flags/         # e.g., 'relu-loss', 'targeted-magnitude'
                    0neg-to-1neg/        # Transition type folders
                        sample_001.png
                        ...
                    aggregate_stats.json
                    all_runs.json
    """

    def __init__(
        self,
        base_dir: str,
        script_name: str,
        loss_type_flags: str,
        max_graphs_per_transition: int = 10,
        random_seed: Optional[int] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_tags: Optional[List[str]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            base_dir: Base output directory (e.g., 'results/')
            script_name: Name of script (e.g., 'gad-rk45', 'gad-eigdescent')
            loss_type_flags: Loss type and flags (e.g., 'relu-loss', 'targeted-magnitude-stopts')
            max_graphs_per_transition: Maximum number of graphs to save per transition type
            random_seed: Random seed for sampling (for reproducibility)
            use_wandb: Enable Weights & Biases logging
            wandb_project: W&B project name (required if use_wandb=True)
            wandb_entity: W&B entity/username (optional)
            wandb_tags: List of tags for this run
            wandb_config: Configuration dictionary to log to W&B
        """
        self.base_dir = Path(base_dir)
        self.script_name = self._sanitize_name(script_name)
        self.loss_type_flags = self._sanitize_name(loss_type_flags)
        self.max_graphs_per_transition = max_graphs_per_transition

        # Set up directory structure
        self.script_dir = self.base_dir / self.script_name
        self.run_dir = self.script_dir / self.loss_type_flags
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Track results and sampling
        self.results: List[RunResult] = []
        self.transition_samples: Dict[str, List[int]] = defaultdict(list)  # transition_key -> list of sample indices

        if random_seed is not None:
            random.seed(random_seed)

        # W&B setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_run = None

        if self.use_wandb:
            if not WANDB_AVAILABLE:
                print("[WARNING] W&B requested but not installed. Install with: pip install wandb")
                self.use_wandb = False
            elif wandb_project is None:
                print("[WARNING] W&B requested but no project name provided. Disabling W&B.")
                self.use_wandb = False
            else:
                # Initialize W&B run
                run_name = f"{script_name}_{loss_type_flags}"
                tags = wandb_tags or []
                tags.extend([script_name, loss_type_flags])

                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=run_name,
                    tags=tags,
                    config=wandb_config or {},
                    dir=str(self.run_dir),
                )
                print(f"[W&B] Initialized run: {self.wandb_run.name} ({self.wandb_run.url})")

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize directory/file names."""
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip("-") or "default"

    def add_result(self, result: RunResult) -> None:
        """Add a run result and track for sampling."""
        self.results.append(result)
        transition_key = result.transition_key
        self.transition_samples[transition_key].append(result.sample_index)

        # Log to W&B
        if self.use_wandb and self.wandb_run is not None:
            self._log_result_to_wandb(result)

    def should_save_graph(self, result: RunResult) -> bool:
        """
        Determine if we should save a graph for this result.

        Strategy: Reservoir sampling - maintain a random sample of up to max_graphs_per_transition.
        This ensures uniform random sampling even when we don't know total count ahead of time.
        """
        transition_key = result.transition_key
        samples = self.transition_samples[transition_key]
        n = len(samples)

        if n <= self.max_graphs_per_transition:
            # Haven't reached limit yet, always save
            return True
        else:
            # Reservoir sampling: with probability k/n, replace a random existing sample
            # For simplicity, we'll just track indices and decide at save time
            return False

    def get_graph_save_path(self, result: RunResult, filename: str) -> Optional[Path]:
        """
        Get the path where a graph should be saved, if at all.

        Returns None if this graph shouldn't be saved (over limit).
        """
        if not self.should_save_graph(result):
            return None

        transition_key = result.transition_key
        transition_dir = self.run_dir / transition_key
        transition_dir.mkdir(parents=True, exist_ok=True)

        return transition_dir / filename

    def save_graph(self, result: RunResult, fig, filename: str) -> Optional[str]:
        """
        Save a matplotlib figure if within sampling limit.

        Args:
            result: RunResult for this graph
            fig: matplotlib Figure object
            filename: Desired filename (will be sanitized)

        Returns:
            Path where graph was saved, or None if not saved
        """
        save_path = self.get_graph_save_path(result, filename)
        if save_path is None:
            return None

        fig.savefig(save_path, dpi=200)

        # Upload to W&B if enabled
        if self.use_wandb and self.wandb_run is not None:
            wandb.log({
                f"plots/{result.transition_key}/{filename}": wandb.Image(fig),
                "sample_index": result.sample_index,
            })

        return str(save_path)

    def _log_result_to_wandb(self, result: RunResult) -> None:
        """Log a single result to W&B."""
        if not self.use_wandb or self.wandb_run is None:
            return

        # Prepare metrics
        metrics = {
            "sample_index": result.sample_index,
            "transition_type": result.transition_key,
            "initial_neg_eigvals": result.initial_neg_eigvals,
            "final_neg_eigvals": result.final_neg_eigvals,
            "steps_taken": result.steps_taken,
            "reached_ts": int(result.reached_ts),
        }

        # Add optional metrics
        if result.steps_to_ts is not None:
            metrics["steps_to_ts"] = result.steps_to_ts
        if result.final_eig0 is not None:
            metrics["final_eig0"] = result.final_eig0
        if result.final_eig1 is not None:
            metrics["final_eig1"] = result.final_eig1
        if result.final_eig_product is not None:
            metrics["final_eig_product"] = result.final_eig_product
        if result.final_loss is not None:
            metrics["final_loss"] = result.final_loss
        if result.rmsd_to_known_ts is not None:
            metrics["rmsd_to_known_ts"] = result.rmsd_to_known_ts
        if result.final_time is not None:
            metrics["final_time"] = result.final_time

        # Log extra data
        if result.extra_data:
            for key, value in result.extra_data.items():
                if value is not None and not isinstance(value, (dict, list)):
                    metrics[f"extra/{key}"] = value

        wandb.log(metrics)

    def compute_aggregate_stats(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all runs."""
        if not self.results:
            return {}

        # Overall statistics
        all_eig0 = [r.final_eig0 for r in self.results if r.final_eig0 is not None]
        all_eig1 = [r.final_eig1 for r in self.results if r.final_eig1 is not None]
        all_eig_prod = [r.final_eig_product for r in self.results if r.final_eig_product is not None]
        all_rmsd = [r.rmsd_to_known_ts for r in self.results if r.rmsd_to_known_ts is not None]
        all_steps = [r.steps_taken for r in self.results]
        all_times = [r.final_time for r in self.results if r.final_time is not None]
        all_steps_to_ts = [r.steps_to_ts for r in self.results if r.steps_to_ts is not None]

        stats = {
            "total_runs": len(self.results),
            "avg_final_eig0": float(np.mean(all_eig0)) if all_eig0 else None,
            "std_final_eig0": float(np.std(all_eig0)) if all_eig0 else None,
            "avg_final_eig1": float(np.mean(all_eig1)) if all_eig1 else None,
            "std_final_eig1": float(np.std(all_eig1)) if all_eig1 else None,
            "avg_final_eig_product": float(np.mean(all_eig_prod)) if all_eig_prod else None,
            "std_final_eig_product": float(np.std(all_eig_prod)) if all_eig_prod else None,
            "avg_rmsd_to_ts": float(np.mean(all_rmsd)) if all_rmsd else None,
            "std_rmsd_to_ts": float(np.std(all_rmsd)) if all_rmsd else None,
            "avg_steps_taken": float(np.mean(all_steps)),
            "std_steps_taken": float(np.std(all_steps)),
            "avg_steps_to_ts": float(np.mean(all_steps_to_ts)) if all_steps_to_ts else None,
            "std_steps_to_ts": float(np.std(all_steps_to_ts)) if all_steps_to_ts else None,
            "avg_final_time": float(np.mean(all_times)) if all_times else None,
            "std_final_time": float(np.std(all_times)) if all_times else None,
        }

        # Per-transition statistics
        transition_stats = {}
        for transition_key in self.transition_samples.keys():
            matching_results = [r for r in self.results if r.transition_key == transition_key]
            if not matching_results:
                continue

            trans_eig0 = [r.final_eig0 for r in matching_results if r.final_eig0 is not None]
            trans_eig1 = [r.final_eig1 for r in matching_results if r.final_eig1 is not None]
            trans_eig_prod = [r.final_eig_product for r in matching_results if r.final_eig_product is not None]
            trans_rmsd = [r.rmsd_to_known_ts for r in matching_results if r.rmsd_to_known_ts is not None]
            trans_steps = [r.steps_taken for r in matching_results]
            trans_steps_to_ts = [r.steps_to_ts for r in matching_results if r.steps_to_ts is not None]

            transition_stats[transition_key] = {
                "count": len(matching_results),
                "avg_final_eig0": float(np.mean(trans_eig0)) if trans_eig0 else None,
                "avg_final_eig1": float(np.mean(trans_eig1)) if trans_eig1 else None,
                "avg_final_eig_product": float(np.mean(trans_eig_prod)) if trans_eig_prod else None,
                "avg_rmsd_to_ts": float(np.mean(trans_rmsd)) if trans_rmsd else None,
                "avg_steps_taken": float(np.mean(trans_steps)),
                "std_steps_taken": float(np.std(trans_steps)),
                "avg_steps_to_ts": float(np.mean(trans_steps_to_ts)) if trans_steps_to_ts else None,
                "std_steps_to_ts": float(np.std(trans_steps_to_ts)) if trans_steps_to_ts else None,
            }

        stats["per_transition"] = transition_stats

        # Success metrics
        ts_signature_count = sum(1 for r in self.results if r.final_neg_eigvals == 1)
        stats["ts_signature_count"] = ts_signature_count
        stats["ts_signature_rate"] = ts_signature_count / len(self.results)

        # Transition distribution
        transition_distribution = {}
        for transition_key, samples in self.transition_samples.items():
            transition_distribution[transition_key] = len(samples)
        stats["transition_distribution"] = transition_distribution

        return stats

    def save_all_results(self) -> Tuple[str, str]:
        """
        Save all results and aggregate statistics.

        Returns:
            Tuple of (all_runs_path, aggregate_stats_path)
        """
        # Save all individual run data
        all_runs_path = self.run_dir / "all_runs.json"
        all_runs_data = [r.to_dict() for r in self.results]
        with open(all_runs_path, "w") as f:
            json.dump(all_runs_data, f, indent=2)

        # Save aggregate statistics
        aggregate_stats_path = self.run_dir / "aggregate_stats.json"
        stats = self.compute_aggregate_stats()
        with open(aggregate_stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        # Log aggregate stats to W&B
        if self.use_wandb and self.wandb_run is not None:
            self._log_aggregate_stats_to_wandb(stats)

            # Upload JSON files as artifacts
            artifact = wandb.Artifact(
                name=f"{self.script_name}-{self.loss_type_flags}",
                type="results",
                description=f"Results for {self.script_name} with {self.loss_type_flags}"
            )
            artifact.add_file(str(all_runs_path), name="all_runs.json")
            artifact.add_file(str(aggregate_stats_path), name="aggregate_stats.json")
            self.wandb_run.log_artifact(artifact)

        return str(all_runs_path), str(aggregate_stats_path)

    def _log_aggregate_stats_to_wandb(self, stats: Dict[str, Any]) -> None:
        """Log aggregate statistics to W&B summary."""
        if not self.use_wandb or self.wandb_run is None:
            return

        # Log overall summary statistics
        summary_metrics = {
            "summary/total_runs": stats["total_runs"],
            "summary/ts_signature_count": stats["ts_signature_count"],
            "summary/ts_signature_rate": stats["ts_signature_rate"],
        }

        # Add overall averages
        for key in ["avg_final_eig0", "avg_final_eig1", "avg_final_eig_product",
                    "avg_rmsd_to_ts", "avg_steps_taken", "avg_steps_to_ts", "avg_final_time"]:
            if stats.get(key) is not None:
                summary_metrics[f"summary/{key}"] = stats[key]

        # Add per-transition stats
        if "per_transition" in stats:
            for transition_key, trans_stats in stats["per_transition"].items():
                prefix = f"transition/{transition_key}"
                for key, value in trans_stats.items():
                    if value is not None:
                        summary_metrics[f"{prefix}/{key}"] = value

        # Log to W&B summary (these persist after run completion)
        for key, value in summary_metrics.items():
            self.wandb_run.summary[key] = value

    def finish(self) -> None:
        """Finish W&B run if active."""
        if self.use_wandb and self.wandb_run is not None:
            wandb.finish()
            print("[W&B] Run finished")

    def print_summary(self) -> None:
        """Print a summary of the experiment results."""
        if not self.results:
            print("No results to summarize.")
            return

        stats = self.compute_aggregate_stats()

        print("\n" + "="*80)
        print(f"EXPERIMENT SUMMARY: {self.script_name} / {self.loss_type_flags}")
        print("="*80)

        print(f"\nTotal runs: {stats['total_runs']}")

        print("\n[Overall Statistics]")
        if stats['avg_final_eig0'] is not None:
            print(f"  λ₀: {stats['avg_final_eig0']:.6f} ± {stats['std_final_eig0']:.6f} eV/Å²")
        if stats['avg_final_eig1'] is not None:
            print(f"  λ₁: {stats['avg_final_eig1']:.6f} ± {stats['std_final_eig1']:.6f} eV/Å²")
        if stats['avg_final_eig_product'] is not None:
            print(f"  λ₀·λ₁: {stats['avg_final_eig_product']:.6e} ± {stats['std_final_eig_product']:.6e}")
        if stats['avg_rmsd_to_ts'] is not None:
            print(f"  RMSD to known TS: {stats['avg_rmsd_to_ts']:.4f} ± {stats['std_rmsd_to_ts']:.4f} Å")
        print(f"  Steps taken: {stats['avg_steps_taken']:.1f} ± {stats['std_steps_taken']:.1f}")
        if stats['avg_steps_to_ts'] is not None:
            print(f"  Steps to TS: {stats['avg_steps_to_ts']:.1f} ± {stats['std_steps_to_ts']:.1f}")
        if stats['avg_final_time'] is not None:
            print(f"  Final time: {stats['avg_final_time']:.3f} ± {stats['std_final_time']:.3f}")

        print(f"\n[Success Metrics]")
        print(f"  TS signature (1 neg eig): {stats['ts_signature_count']}/{stats['total_runs']} ({stats['ts_signature_rate']*100:.1f}%)")

        print(f"\n[Transition Distribution]")
        for transition_key, count in sorted(stats['transition_distribution'].items()):
            print(f"  {transition_key}: {count} samples")

        if stats.get('per_transition'):
            print(f"\n[Per-Transition Statistics]")
            for transition_key, trans_stats in sorted(stats['per_transition'].items()):
                print(f"\n  {transition_key} ({trans_stats['count']} samples):")
                if trans_stats['avg_final_eig0'] is not None:
                    print(f"    λ₀: {trans_stats['avg_final_eig0']:.6f} eV/Å²")
                if trans_stats['avg_final_eig1'] is not None:
                    print(f"    λ₁: {trans_stats['avg_final_eig1']:.6f} eV/Å²")
                if trans_stats['avg_rmsd_to_ts'] is not None:
                    print(f"    RMSD to TS: {trans_stats['avg_rmsd_to_ts']:.4f} Å")
                print(f"    Steps: {trans_stats['avg_steps_taken']:.1f} ± {trans_stats['std_steps_taken']:.1f}")
                if trans_stats['avg_steps_to_ts'] is not None:
                    print(f"    Steps to TS: {trans_stats['avg_steps_to_ts']:.1f} ± {trans_stats['std_steps_to_ts']:.1f}")

        print("="*80)


def build_loss_type_flags(args) -> str:
    """
    Build a descriptive string from arguments for the loss_type_flags directory.

    Args:
        args: argparse.Namespace with experiment configuration

    Returns:
        String like 'relu-loss' or 'targeted-magnitude-stopts-kick'
    """
    components = []

    # Add loss type if present
    if hasattr(args, 'loss_type'):
        components.append(args.loss_type)

    # Add relevant flags
    if hasattr(args, 'stop_at_ts') and args.stop_at_ts:
        components.append('stopts')

    if hasattr(args, 'enable_kick') and args.enable_kick:
        components.append('kick')

    if hasattr(args, 'eigenvector_following') and args.eigenvector_following:
        components.append('eigfollow')

    if hasattr(args, 'adaptive_targets') and args.adaptive_targets:
        components.append('adaptive')

    # Add start position
    if hasattr(args, 'start_from'):
        components.append(f"from-{args.start_from}")

    return '-'.join(components) if components else 'default'
