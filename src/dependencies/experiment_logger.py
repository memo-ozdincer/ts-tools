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

# ============================================================================
# Simple W&B Helper Functions (use these instead of class-based logging)
# ============================================================================

_wandb_run = None  # Module-level reference to active W&B run


def init_wandb_run(
    project: str,
    name: str,
    config: Dict[str, Any],
    entity: Optional[str] = None,
    tags: Optional[List[str]] = None,
    run_dir: Optional[str] = None,
    group: Optional[str] = None,
) -> bool:
    """
    Initialize a W&B run. Call once at the start of your experiment.
    
    Args:
        project: W&B project name
        name: Run name
        config: Configuration dictionary
        entity: W&B entity/username (optional)
        tags: List of tags
        run_dir: Directory for W&B files (optional)
    
    Returns:
        True if W&B was successfully initialized, False otherwise
    """
    global _wandb_run
    
    if not WANDB_AVAILABLE:
        print("[W&B] wandb not installed. Install with: pip install wandb")
        return False
    
    try:
        group_value = group or os.environ.get("WANDB_RUN_GROUP") or os.environ.get("WANDB_GROUP")
        _wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags or [],
            config=config,
            dir=run_dir,
            group=group_value,
        )
        print(f"[W&B] Initialized run: {_wandb_run.name} ({_wandb_run.url})")
        return True
    except Exception as e:
        print(f"[W&B] Failed to initialize: {e}")
        _wandb_run = None
        return False


def log_sample(
    sample_index: int,
    metrics: Dict[str, Any],
    fig=None,
    plot_name: Optional[str] = None,
) -> None:
    """
    Log metrics and optional plot for a single sample. Call once per sample.
    
    Args:
        sample_index: Index of the sample (used as step)
        metrics: Dictionary of metrics to log
        fig: Optional matplotlib figure to log as image
        plot_name: Name for the plot (default: "trajectory")
    """
    if _wandb_run is None:
        return
    
    # Prepare log dict
    log_dict = {"sample_index": sample_index}
    
    # Add all metrics
    for key, value in metrics.items():
        if value is not None and not isinstance(value, (dict, list)):
            log_dict[key] = value
    
    # Add plot if provided
    if fig is not None:
        plot_key = f"plots/{plot_name or 'trajectory'}"

        # If it looks like a Plotly figure, prefer logging it interactively.
        # In W&B, `wandb.Plotly(fig)` (when available) tends to behave better than
        # logging the raw figure object, and an explicit HTML embed enables scroll-zoom.
        is_plotly_like = hasattr(fig, "to_dict") or hasattr(fig, "to_plotly_json") or hasattr(fig, "write_html")
        if is_plotly_like:
            if hasattr(wandb, "Plotly"):
                try:
                    log_dict[plot_key] = wandb.Plotly(fig)
                except Exception:
                    # Fallback: raw figure (W&B may still render it)
                    log_dict[plot_key] = fig
            else:
                log_dict[plot_key] = fig

            # Extra: HTML embed with explicit Plotly config for better usability in W&B panels.
            # This is especially helpful for scroll-zoom + modebar controls.
            if hasattr(wandb, "Html") and hasattr(fig, "to_html"):
                try:
                    html = fig.to_html(
                        full_html=False,
                        include_plotlyjs="cdn",
                        config={
                            "scrollZoom": True,
                            "displayModeBar": True,
                            "displaylogo": False,
                            "responsive": True,
                            "doubleClick": "reset",
                        },
                    )
                    log_dict[f"{plot_key}_html"] = wandb.Html(html)
                except Exception:
                    pass
        else:
            log_dict[plot_key] = wandb.Image(fig)
    
    # Log everything together for this sample
    wandb.log(log_dict, step=sample_index)


def log_summary(summary_dict: Dict[str, Any]) -> None:
    """
    Log summary statistics at the end of the experiment.
    
    Args:
        summary_dict: Dictionary of summary metrics to log. All values are logged
                      directly to W&B summary (which persists after run completion).
                      
    Expected keys (all optional):
        - total_samples: Total number of samples processed
        - avg_steps: Average number of steps to convergence
        - avg_wallclock_time: Average wallclock time per sample (seconds)
        - ts_success_rate: Fraction that reached TS (order-1)
        - count_order_0, count_order_1, count_order_2, ...: Counts per final saddle order
        - avg_final_eig_product, avg_final_eig0, avg_final_eig1, etc.: Averages of metrics
    """
    if _wandb_run is None:
        return
    
    # Set all summary metrics
    for key, value in summary_dict.items():
        if value is not None:
            _wandb_run.summary[key] = value
    
    # Print summary with TS count
    total = summary_dict.get("total_samples", 0)
    avg_steps = summary_dict.get("avg_steps", 0)
    avg_time = summary_dict.get("avg_wallclock_time", 0)
    ts_count = summary_dict.get("ts_signature_count", 0)
    ts_rate = summary_dict.get("ts_success_rate", 0)
    print(f"[W&B] Logged summary: {total} samples, avg steps={avg_steps:.1f}, "
          f"avg time={avg_time:.2f}s, TS found={ts_count}/{total} ({ts_rate:.1%})")


def log_artifact(
    file_path: str,
    artifact_name: str,
    artifact_type: str = "optuna-study",
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Log a file as a W&B artifact (e.g., SQLite database).

    Args:
        file_path: Path to the file to upload
        artifact_name: Name for the artifact
        artifact_type: Type of artifact (default: "optuna-study")
        description: Optional description
        metadata: Optional metadata dict

    Returns:
        True if successful, False otherwise
    """
    if _wandb_run is None:
        return False

    try:
        path = Path(file_path)
        if not path.exists():
            print(f"[W&B] Artifact file not found: {file_path}")
            return False

        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=description or f"Artifact: {artifact_name}",
            metadata=metadata or {"file_size_mb": path.stat().st_size / (1024 * 1024)},
        )
        artifact.add_file(str(path))
        _wandb_run.log_artifact(artifact)
        print(f"[W&B] Logged artifact: {artifact_name} ({artifact_type})")
        return True
    except Exception as e:
        print(f"[W&B] Failed to log artifact: {e}")
        return False


def finish_wandb(
    artifact_path: Optional[str] = None,
    artifact_name: Optional[str] = None,
) -> None:
    """
    Finish the W&B run. Call at the end of your experiment.

    Args:
        artifact_path: Optional path to artifact (e.g., SQLite DB) to upload before finishing
        artifact_name: Name for the artifact (required if artifact_path provided)
    """
    global _wandb_run
    if _wandb_run is not None:
        # Upload artifact if provided
        if artifact_path and artifact_name:
            log_artifact(artifact_path, artifact_name)

        wandb.finish()
        print("[W&B] Run finished")
        _wandb_run = None


def is_wandb_active() -> bool:
    """Check if W&B is currently active."""
    return _wandb_run is not None


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
        """Did this run reach a TS (eigenvalue product < 0)?

        True TS signature: λ₀ < 0 and λ₁ > 0, meaning eig_product < 0.
        This is more reliable than counting negative eigenvalues.
        """
        if self.final_eig_product is None:
            return False
        return self.final_eig_product < 0


class ExperimentLogger:
    """
    Manages experiment logging with structured output directories and sampling.
    
    Note: For W&B logging, use the simple helper functions instead:
        init_wandb_run(), log_sample(), log_summary(), finish_wandb()

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
    ):
        """
        Args:
            base_dir: Base output directory (e.g., 'results/')
            script_name: Name of script (e.g., 'gad-rk45', 'gad-eigdescent')
            loss_type_flags: Loss type and flags (e.g., 'relu-loss', 'targeted-magnitude-stopts')
            max_graphs_per_transition: Maximum number of graphs to save per transition type
            random_seed: Random seed for sampling (for reproducibility)
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

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize directory/file names."""
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip("-") or "default"

    def add_result(self, result: RunResult) -> None:
        """Add a run result and track for sampling."""
        self.results.append(result)
        transition_key = result.transition_key
        self.transition_samples[transition_key].append(result.sample_index)

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
        return str(save_path)

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

        # Success metrics (TS signature: eigenvalue product < 0)
        # This means λ₀ < 0 and λ₁ > 0, which is the true TS signature
        ts_signature_count = sum(1 for r in self.results if r.reached_ts)
        stats["ts_signature_count"] = ts_signature_count
        stats["ts_signature_rate"] = ts_signature_count / len(self.results) if self.results else 0

        # Transition distribution
        transition_distribution = {}
        for transition_key, samples in self.transition_samples.items():
            transition_distribution[transition_key] = len(samples)
        stats["transition_distribution"] = transition_distribution

        # Add aliases for W&B log_summary() compatibility
        stats["total_samples"] = stats["total_runs"]
        stats["avg_steps"] = stats["avg_steps_taken"]
        stats["avg_wallclock_time"] = stats["avg_final_time"]
        stats["ts_success_rate"] = stats["ts_signature_rate"]

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

        return str(all_runs_path), str(aggregate_stats_path)

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
        print(f"  TS signature (λ₀·λ₁ < 0): {stats['ts_signature_count']}/{stats['total_runs']} ({stats['ts_signature_rate']*100:.1f}%)")

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

    if hasattr(args, 'adaptive_step_sizing') and args.adaptive_step_sizing:
        components.append('adaptive-steps')

    # Add start position
    if hasattr(args, 'start_from'):
        components.append(f"from-{args.start_from}")

    return '-'.join(components) if components else 'default'
