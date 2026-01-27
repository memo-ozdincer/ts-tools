"""Full trajectory logging for GAD failure diagnosis.

This module provides a TrajectoryLogger class that accumulates extended metrics
across all GAD steps, tracks escape events, and exports comprehensive logs for
post-hoc analysis.

The goal is to answer:
1. When does GAD fail? At what eigenvalue configurations?
2. Is failure correlated with λ₁ ≈ λ₂ (singularity proximity)?
3. Is failure correlated with Morse index > 1 (high-index saddles)?
4. When does v₂ kicking succeed vs fail?
5. What patterns predict successful TS discovery?
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .metrics import ExtendedMetrics, compute_extended_metrics
from .escape_logger import EscapeEvent, summarize_escape_events


@dataclass
class TrajectoryLogger:
    """Accumulates extended metrics and escape events for a full GAD trajectory.

    All metrics are state-based (computed from current geometry only).
    Path-dependent tracking has been removed.

    Usage:
        logger = TrajectoryLogger(sample_id="mol_001")

        for step in range(n_steps):
            # Compute GAD step...
            logger.log_step(step, coords, energy, forces, hessian_proj, gad_vec, dt_eff)

            if escape_triggered:
                logger.log_escape(escape_event)

        logger.finalize(final_coords, found_ts=True)
        logger.save("output_dir/")
    """

    sample_id: str
    formula: str = ""

    # Extended metrics for each step
    metrics: List[ExtendedMetrics] = field(default_factory=list)

    # Escape events
    escape_events: List[EscapeEvent] = field(default_factory=list)

    # Running statistics
    _step_count: int = 0
    _escape_count: int = 0

    # Final outcome
    final_coords: Optional[torch.Tensor] = None
    final_morse_index: Optional[int] = None
    converged_to_ts: bool = False
    total_steps: int = 0

    # Known TS for validation (optional)
    known_ts_coords: Optional[torch.Tensor] = None

    def log_step(
        self,
        step: int,
        coords: torch.Tensor,
        energy: float,
        forces: torch.Tensor,
        hessian_proj: torch.Tensor,
        gad_vec: torch.Tensor,
        dt_eff: float,
        *,
        mode_index: Optional[int] = None,
    ) -> ExtendedMetrics:
        """Log a single GAD step with extended metrics (state-based only).

        Args:
            step: Current step number
            coords: Current coordinates (N, 3)
            energy: Current energy
            forces: Current forces (N, 3)
            hessian_proj: Projected Hessian (3N, 3N)
            gad_vec: GAD direction (N, 3)
            dt_eff: Effective timestep
            mode_index: Which eigenvector is being tracked (optional)

        Returns:
            ExtendedMetrics for this step (state-based only)
        """
        metrics = compute_extended_metrics(
            step=step,
            coords=coords,
            energy=energy,
            forces=forces,
            hessian_proj=hessian_proj,
            gad_vec=gad_vec,
            dt_eff=dt_eff,
            mode_index=mode_index,
            known_ts_coords=self.known_ts_coords,
        )

        self.metrics.append(metrics)
        self._step_count = step + 1

        return metrics

    def log_escape(self, escape_event: EscapeEvent) -> None:
        """Log an escape event.

        Args:
            escape_event: Fully populated EscapeEvent
        """
        self.escape_events.append(escape_event)
        self._escape_count += 1

        # Reset mode tracking after escape (discontinuous jump)
        self.v1_prev = None
        self.v2_prev = None

    def finalize(
        self,
        final_coords: torch.Tensor,
        final_morse_index: int,
        converged_to_ts: bool,
    ) -> None:
        """Finalize the trajectory with outcome information.

        Args:
            final_coords: Final coordinates
            final_morse_index: Final Morse index
            converged_to_ts: Whether we converged to index-1 TS
        """
        self.final_coords = final_coords.detach().clone()
        self.final_morse_index = final_morse_index
        self.converged_to_ts = converged_to_ts
        self.total_steps = self._step_count

    def get_trajectory_dict(self) -> Dict[str, List[Any]]:
        """Export trajectory data as a dictionary of lists.

        Returns:
            Dictionary with metric name -> list of values
        """
        if not self.metrics:
            return {}

        result = {}
        for key in self.metrics[0].to_dict().keys():
            result[key] = [m.to_dict()[key] for m in self.metrics]

        return result

    def get_escape_summary(self) -> Dict[str, Any]:
        """Get summary statistics for escape events.

        Returns:
            Dictionary with escape summary statistics
        """
        return summarize_escape_events(self.escape_events)

    def get_failure_analysis(self) -> Dict[str, Any]:
        """Analyze failure patterns if trajectory didn't converge to TS.

        All analysis is based on state-based metrics (eigenvalues, gradient norm, etc.)
        with no path-dependent information.

        Returns:
            Dictionary with failure analysis metrics
        """
        if not self.metrics:
            return {"error": "No metrics logged"}

        # Analyze last N steps to understand failure mode
        n_analyze = min(50, len(self.metrics))
        recent = self.metrics[-n_analyze:]

        # Eigenvalue gap statistics (singularity proximity)
        gaps = [m.eig_gap_01 for m in recent if np.isfinite(m.eig_gap_01)]
        gap_stats = {
            "mean_eig_gap_01": float(np.mean(gaps)) if gaps else float("nan"),
            "min_eig_gap_01": float(np.min(gaps)) if gaps else float("nan"),
            "std_eig_gap_01": float(np.std(gaps)) if gaps else float("nan"),
        }

        # Morse index statistics
        indices = [m.morse_index for m in recent]
        index_stats = {
            "mean_morse_index": float(np.mean(indices)) if indices else float("nan"),
            "final_morse_index": indices[-1] if indices else -1,
            "morse_index_stable": float(np.std(indices)) < 0.5 if indices else False,
        }

        # Gradient statistics (state-based)
        grads = [m.grad_norm for m in recent if np.isfinite(m.grad_norm)]
        grad_stats = {
            "mean_grad_norm": float(np.mean(grads)) if grads else float("nan"),
            "final_grad_norm": grads[-1] if grads else float("nan"),
        }

        # Singularity metric
        sing_metrics = [m.singularity_metric for m in recent if np.isfinite(m.singularity_metric)]
        sing_stats = {
            "mean_singularity_metric": float(np.mean(sing_metrics)) if sing_metrics else float("nan"),
            "min_singularity_metric": float(np.min(sing_metrics)) if sing_metrics else float("nan"),
        }

        # Diagnose failure mode based on state-based metrics
        failure_mode = "unknown"
        if index_stats["final_morse_index"] > 1:
            if gap_stats["min_eig_gap_01"] < 0.01:
                failure_mode = "singularity_high_index"
            else:
                failure_mode = "high_index_saddle"
        elif index_stats["final_morse_index"] == 1:
            if grad_stats["final_grad_norm"] > 0.01:
                failure_mode = "grad_not_converged"
            else:
                failure_mode = "converged_ts"  # Actually success
        elif index_stats["final_morse_index"] == 0:
            failure_mode = "minimum"

        return {
            "failure_mode": failure_mode,
            **gap_stats,
            **index_stats,
            **grad_stats,
            **sing_stats,
            "n_analyzed_steps": n_analyze,
        }

    def get_full_summary(self) -> Dict[str, Any]:
        """Get complete summary of trajectory.

        Returns:
            Dictionary with all summary statistics
        """
        return {
            "sample_id": self.sample_id,
            "formula": self.formula,
            "total_steps": self.total_steps,
            "total_escapes": len(self.escape_events),
            "converged_to_ts": self.converged_to_ts,
            "final_morse_index": self.final_morse_index,
            "escape_summary": self.get_escape_summary(),
            "failure_analysis": self.get_failure_analysis() if not self.converged_to_ts else {},
        }

    def save(self, output_dir: str | Path, prefix: str = "") -> Dict[str, str]:
        """Save all trajectory data to files.

        Args:
            output_dir: Directory to save files
            prefix: Optional prefix for filenames

        Returns:
            Dictionary mapping data type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_prefix = f"{prefix}_" if prefix else ""
        file_prefix = f"{file_prefix}{self.sample_id}"

        paths = {}

        # Save trajectory metrics
        traj_path = output_dir / f"{file_prefix}_trajectory.json"
        with open(traj_path, "w") as f:
            json.dump(self.get_trajectory_dict(), f, indent=2, default=_json_serializer)
        paths["trajectory"] = str(traj_path)

        # Save escape events
        escape_path = output_dir / f"{file_prefix}_escapes.json"
        with open(escape_path, "w") as f:
            json.dump([e.to_dict() for e in self.escape_events], f, indent=2, default=_json_serializer)
        paths["escapes"] = str(escape_path)

        # Save summary
        summary_path = output_dir / f"{file_prefix}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.get_full_summary(), f, indent=2, default=_json_serializer)
        paths["summary"] = str(summary_path)

        return paths

    def print_summary(self) -> None:
        """Print a human-readable summary."""
        summary = self.get_full_summary()

        print(f"\n{'='*60}")
        print(f"Trajectory Summary: {self.sample_id}")
        print(f"{'='*60}")
        print(f"  Formula: {self.formula}")
        print(f"  Total steps: {summary['total_steps']}")
        print(f"  Total escapes: {summary['total_escapes']}")
        print(f"  Converged to TS: {summary['converged_to_ts']}")
        print(f"  Final Morse index: {summary['final_morse_index']}")

        if summary['escape_summary']['total_escapes'] > 0:
            esc = summary['escape_summary']
            print(f"\n  Escape Statistics:")
            print(f"    Success rate: {esc['success_rate']:.1%}")
            print(f"    Mean index improvement: {esc['mean_index_improvement']:.2f}")
            print(f"    Escapes near singularity: {esc['escapes_near_singularity']}")

        if not summary['converged_to_ts'] and summary.get('failure_analysis'):
            fa = summary['failure_analysis']
            print(f"\n  Failure Analysis:")
            print(f"    Failure mode: {fa['failure_mode']}")
            print(f"    Final Morse index: {fa['final_morse_index']}")
            print(f"    Min eigenvalue gap: {fa['min_eig_gap_01']:.2e}")
            print(f"    Final grad norm: {fa['final_grad_norm']:.2e}")

        print(f"{'='*60}\n")


def _json_serializer(obj):
    """JSON serializer for numpy/torch types."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_trajectory(path: str | Path) -> Dict[str, List[Any]]:
    """Load trajectory data from JSON file.

    Args:
        path: Path to trajectory JSON file

    Returns:
        Dictionary with metric name -> list of values
    """
    with open(path) as f:
        return json.load(f)


def load_escapes(path: str | Path) -> List[Dict[str, Any]]:
    """Load escape events from JSON file.

    Args:
        path: Path to escapes JSON file

    Returns:
        List of escape event dictionaries
    """
    with open(path) as f:
        return json.load(f)


def load_summary(path: str | Path) -> Dict[str, Any]:
    """Load summary from JSON file.

    Args:
        path: Path to summary JSON file

    Returns:
        Summary dictionary
    """
    with open(path) as f:
        return json.load(f)
