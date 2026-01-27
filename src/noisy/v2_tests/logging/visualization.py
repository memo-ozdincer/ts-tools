"""Visualization utilities for GAD trajectory analysis.

This module provides plotting functions for:
1. dt_eff trajectories overlaid by sample, grouped by chemical formula
2. Morse index coloring for trajectory analysis
3. TR mode eigenvalue verification plots
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def load_trajectory_data(trajectory_path: str | Path) -> Dict[str, List[Any]]:
    """Load trajectory JSON data from file.

    Args:
        trajectory_path: Path to trajectory JSON file

    Returns:
        Dictionary with metric name -> list of values
    """
    with open(trajectory_path) as f:
        return json.load(f)


def load_summary_data(summary_path: str | Path) -> Dict[str, Any]:
    """Load summary JSON data from file.

    Args:
        summary_path: Path to summary JSON file

    Returns:
        Summary dictionary
    """
    with open(summary_path) as f:
        return json.load(f)


def collect_trajectories_from_dir(
    diagnostics_dir: str | Path,
) -> List[Dict[str, Any]]:
    """Collect all trajectory data from a diagnostics directory.

    Args:
        diagnostics_dir: Directory containing trajectory JSON files

    Returns:
        List of dictionaries with trajectory data and metadata
    """
    diagnostics_dir = Path(diagnostics_dir)
    trajectories = []

    for traj_file in diagnostics_dir.glob("*_trajectory.json"):
        # Extract sample_id from filename (e.g., sample_001_trajectory.json)
        sample_id = traj_file.stem.replace("_trajectory", "")

        # Load trajectory data
        traj_data = load_trajectory_data(traj_file)

        # Try to load corresponding summary for formula
        summary_file = diagnostics_dir / f"{sample_id}_summary.json"
        formula = ""
        final_morse_index = -1
        converged = False

        if summary_file.exists():
            summary = load_summary_data(summary_file)
            formula = summary.get("formula", "")
            final_morse_index = summary.get("final_morse_index", -1)
            converged = summary.get("converged_to_ts", False)

        trajectories.append({
            "sample_id": sample_id,
            "formula": formula,
            "final_morse_index": final_morse_index,
            "converged": converged,
            "data": traj_data,
        })

    return trajectories


def plot_dt_eff_trajectories(
    trajectories: List[Dict[str, Any]],
    output_path: Optional[str | Path] = None,
    group_by_formula: bool = True,
    color_by_morse: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    log_scale: bool = True,
    max_steps: Optional[int] = None,
    alpha: float = 0.6,
) -> plt.Figure:
    """Plot dt_eff trajectories overlaid, optionally grouped by formula.

    Args:
        trajectories: List of trajectory dictionaries from collect_trajectories_from_dir
        output_path: Path to save figure (optional)
        group_by_formula: If True, create subplot per formula; otherwise single plot
        color_by_morse: If True, color lines by final Morse index
        figsize: Figure size
        log_scale: If True, use log scale for y-axis
        max_steps: Maximum steps to plot (None for all)
        alpha: Line transparency

    Returns:
        matplotlib Figure
    """
    if group_by_formula:
        # Group trajectories by formula
        by_formula: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for traj in trajectories:
            formula = traj.get("formula", "unknown") or "unknown"
            by_formula[formula].append(traj)

        n_formulas = len(by_formula)
        if n_formulas == 0:
            raise ValueError("No trajectories to plot")

        # Create subplots
        n_cols = min(3, n_formulas)
        n_rows = (n_formulas + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for idx, (formula, formula_trajs) in enumerate(sorted(by_formula.items())):
            ax = axes[idx]
            _plot_dt_eff_on_axis(
                ax, formula_trajs, formula, color_by_morse, log_scale, max_steps, alpha
            )

        # Hide unused subplots
        for idx in range(len(by_formula), len(axes)):
            axes[idx].set_visible(False)

        fig.tight_layout()

    else:
        # Single plot with all trajectories
        fig, ax = plt.subplots(figsize=figsize)
        _plot_dt_eff_on_axis(
            ax, trajectories, "All Samples", color_by_morse, log_scale, max_steps, alpha
        )
        fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def _plot_dt_eff_on_axis(
    ax: plt.Axes,
    trajectories: List[Dict[str, Any]],
    title: str,
    color_by_morse: bool,
    log_scale: bool,
    max_steps: Optional[int],
    alpha: float,
) -> None:
    """Helper to plot dt_eff trajectories on a single axis."""
    if not trajectories:
        return

    # Determine color mapping
    if color_by_morse:
        # Get unique Morse indices for colormap
        morse_indices = sorted(set(t.get("final_morse_index", -1) for t in trajectories))
        morse_to_idx = {m: i for i, m in enumerate(morse_indices)}
        n_colors = max(len(morse_indices), 1)
        cmap = cm.get_cmap("viridis", n_colors)

    for traj in trajectories:
        data = traj.get("data", {})
        dt_eff = data.get("step_size_eff", [])
        steps = data.get("step", list(range(len(dt_eff))))

        if not dt_eff:
            continue

        if max_steps is not None:
            dt_eff = dt_eff[:max_steps]
            steps = steps[:max_steps]

        # Determine color
        if color_by_morse:
            morse_idx = traj.get("final_morse_index", -1)
            color_idx = morse_to_idx.get(morse_idx, 0)
            color = cmap(color_idx / max(n_colors - 1, 1))
        else:
            color = None

        label = traj.get("sample_id", "")
        ax.plot(steps, dt_eff, alpha=alpha, color=color, linewidth=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("dt_eff")
    ax.set_title(title)

    if log_scale:
        ax.set_yscale("log")

    # Add colorbar legend for Morse index
    if color_by_morse and trajectories:
        morse_indices = sorted(set(t.get("final_morse_index", -1) for t in trajectories))
        if len(morse_indices) > 1:
            sm = cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=min(morse_indices), vmax=max(morse_indices))
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label="Final Morse Index")

    ax.grid(True, alpha=0.3)


def plot_tr_mode_diagnostics(
    trajectories: List[Dict[str, Any]],
    output_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (14, 8),
    max_steps: Optional[int] = None,
) -> plt.Figure:
    """Plot TR mode eigenvalue diagnostics to verify projection.

    Shows:
    - Number of TR modes over time (should be constant ~5-6)
    - Max TR eigenvalue magnitude (should stay near 0)
    - Mean TR eigenvalue magnitude

    Args:
        trajectories: List of trajectory dictionaries
        output_path: Path to save figure (optional)
        figsize: Figure size
        max_steps: Maximum steps to plot

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: n_tr_modes over time
    ax = axes[0, 0]
    for traj in trajectories:
        data = traj.get("data", {})
        n_tr = data.get("n_tr_modes", [])
        steps = data.get("step", list(range(len(n_tr))))

        if not n_tr:
            continue

        if max_steps is not None:
            n_tr = n_tr[:max_steps]
            steps = steps[:max_steps]

        ax.plot(steps, n_tr, alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("n_tr_modes")
    ax.set_title("Number of TR Modes (should be ~5-6)")
    ax.axhline(y=6, color="r", linestyle="--", alpha=0.5, label="Expected (linear)")
    ax.axhline(y=5, color="orange", linestyle="--", alpha=0.5, label="Expected (nonlinear)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: tr_eig_max over time
    ax = axes[0, 1]
    for traj in trajectories:
        data = traj.get("data", {})
        tr_max = data.get("tr_eig_max", [])
        steps = data.get("step", list(range(len(tr_max))))

        if not tr_max:
            continue

        if max_steps is not None:
            tr_max = tr_max[:max_steps]
            steps = steps[:max_steps]

        # Filter out nan values for plotting
        valid = [(s, v) for s, v in zip(steps, tr_max) if np.isfinite(v)]
        if valid:
            s, v = zip(*valid)
            ax.plot(s, v, alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("tr_eig_max")
    ax.set_title("Max TR Eigenvalue Magnitude (should be ~0)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Plot 3: tr_eig_mean over time
    ax = axes[1, 0]
    for traj in trajectories:
        data = traj.get("data", {})
        tr_mean = data.get("tr_eig_mean", [])
        steps = data.get("step", list(range(len(tr_mean))))

        if not tr_mean:
            continue

        if max_steps is not None:
            tr_mean = tr_mean[:max_steps]
            steps = steps[:max_steps]

        valid = [(s, v) for s, v in zip(steps, tr_mean) if np.isfinite(v)]
        if valid:
            s, v = zip(*valid)
            ax.plot(s, v, alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("tr_eig_mean")
    ax.set_title("Mean TR Eigenvalue Magnitude (should be ~0)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Plot 4: Histogram of tr_eig_max across all steps
    ax = axes[1, 1]
    all_tr_max = []
    for traj in trajectories:
        data = traj.get("data", {})
        tr_max = data.get("tr_eig_max", [])
        all_tr_max.extend([v for v in tr_max if np.isfinite(v)])

    if all_tr_max:
        # Use log bins
        log_vals = np.log10(np.array(all_tr_max) + 1e-20)
        ax.hist(log_vals, bins=50, alpha=0.7, edgecolor="black")
        ax.set_xlabel("log10(tr_eig_max)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Max TR Eigenvalue")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_morse_index_evolution(
    trajectories: List[Dict[str, Any]],
    output_path: Optional[str | Path] = None,
    group_by_formula: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    max_steps: Optional[int] = None,
    alpha: float = 0.6,
) -> plt.Figure:
    """Plot Morse index evolution over trajectory steps.

    Args:
        trajectories: List of trajectory dictionaries
        output_path: Path to save figure (optional)
        group_by_formula: If True, create subplot per formula
        figsize: Figure size
        max_steps: Maximum steps to plot
        alpha: Line transparency

    Returns:
        matplotlib Figure
    """
    if group_by_formula:
        by_formula: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for traj in trajectories:
            formula = traj.get("formula", "unknown") or "unknown"
            by_formula[formula].append(traj)

        n_formulas = len(by_formula)
        if n_formulas == 0:
            raise ValueError("No trajectories to plot")

        n_cols = min(3, n_formulas)
        n_rows = (n_formulas + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for idx, (formula, formula_trajs) in enumerate(sorted(by_formula.items())):
            ax = axes[idx]
            _plot_morse_on_axis(ax, formula_trajs, formula, max_steps, alpha)

        for idx in range(len(by_formula), len(axes)):
            axes[idx].set_visible(False)

        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        _plot_morse_on_axis(ax, trajectories, "All Samples", max_steps, alpha)
        fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def _plot_morse_on_axis(
    ax: plt.Axes,
    trajectories: List[Dict[str, Any]],
    title: str,
    max_steps: Optional[int],
    alpha: float,
) -> None:
    """Helper to plot Morse index on a single axis."""
    cmap = cm.get_cmap("tab10")

    for i, traj in enumerate(trajectories):
        data = traj.get("data", {})
        morse = data.get("morse_index", [])
        steps = data.get("step", list(range(len(morse))))

        if not morse:
            continue

        if max_steps is not None:
            morse = morse[:max_steps]
            steps = steps[:max_steps]

        color = cmap(i % 10)
        ax.plot(steps, morse, alpha=alpha, color=color, linewidth=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Morse Index")
    ax.set_title(title)
    ax.axhline(y=1, color="green", linestyle="--", alpha=0.5, label="Target (index-1 TS)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_summary_report(
    trajectories: List[Dict[str, Any]],
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Create a summary report of trajectory statistics.

    Args:
        trajectories: List of trajectory dictionaries
        output_path: Path to save JSON report (optional)

    Returns:
        Summary statistics dictionary
    """
    # Group by formula
    by_formula: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for traj in trajectories:
        formula = traj.get("formula", "unknown") or "unknown"
        by_formula[formula].append(traj)

    formula_stats = {}
    for formula, trajs in by_formula.items():
        n_samples = len(trajs)
        n_converged = sum(1 for t in trajs if t.get("converged", False))

        # Collect dt_eff stats
        all_dt_eff = []
        for traj in trajs:
            data = traj.get("data", {})
            all_dt_eff.extend(data.get("step_size_eff", []))

        dt_eff_stats = {}
        if all_dt_eff:
            dt_eff_arr = np.array([v for v in all_dt_eff if np.isfinite(v)])
            if len(dt_eff_arr) > 0:
                dt_eff_stats = {
                    "mean": float(np.mean(dt_eff_arr)),
                    "std": float(np.std(dt_eff_arr)),
                    "min": float(np.min(dt_eff_arr)),
                    "max": float(np.max(dt_eff_arr)),
                    "median": float(np.median(dt_eff_arr)),
                }

        # Collect TR mode stats
        all_tr_max = []
        all_n_tr = []
        for traj in trajs:
            data = traj.get("data", {})
            all_tr_max.extend(data.get("tr_eig_max", []))
            all_n_tr.extend(data.get("n_tr_modes", []))

        tr_stats = {}
        if all_tr_max:
            tr_arr = np.array([v for v in all_tr_max if np.isfinite(v)])
            if len(tr_arr) > 0:
                tr_stats["tr_eig_max_mean"] = float(np.mean(tr_arr))
                tr_stats["tr_eig_max_max"] = float(np.max(tr_arr))

        if all_n_tr:
            n_tr_arr = np.array([v for v in all_n_tr if np.isfinite(v)])
            if len(n_tr_arr) > 0:
                tr_stats["n_tr_modes_mean"] = float(np.mean(n_tr_arr))
                tr_stats["n_tr_modes_std"] = float(np.std(n_tr_arr))

        # Morse index distribution
        morse_dist = defaultdict(int)
        for traj in trajs:
            morse = traj.get("final_morse_index", -1)
            morse_dist[morse] += 1

        formula_stats[formula] = {
            "n_samples": n_samples,
            "n_converged": n_converged,
            "convergence_rate": n_converged / max(n_samples, 1),
            "dt_eff_stats": dt_eff_stats,
            "tr_mode_stats": tr_stats,
            "final_morse_distribution": dict(morse_dist),
        }

    report = {
        "total_samples": len(trajectories),
        "total_converged": sum(1 for t in trajectories if t.get("converged", False)),
        "n_formulas": len(by_formula),
        "by_formula": formula_stats,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    return report


def generate_all_plots(
    diagnostics_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Path]:
    """Generate all diagnostic plots from a diagnostics directory.

    Args:
        diagnostics_dir: Directory containing trajectory JSON files
        output_dir: Directory to save plots (defaults to diagnostics_dir/plots)
        max_steps: Maximum steps to plot

    Returns:
        Dictionary mapping plot name to file path
    """
    diagnostics_dir = Path(diagnostics_dir)
    if output_dir is None:
        output_dir = diagnostics_dir / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect trajectories
    trajectories = collect_trajectories_from_dir(diagnostics_dir)

    if not trajectories:
        print(f"No trajectories found in {diagnostics_dir}")
        return {}

    print(f"Found {len(trajectories)} trajectories")

    paths = {}

    # dt_eff trajectories grouped by formula
    dt_eff_path = output_dir / "dt_eff_by_formula.png"
    plot_dt_eff_trajectories(
        trajectories,
        output_path=dt_eff_path,
        group_by_formula=True,
        color_by_morse=True,
        max_steps=max_steps,
    )
    paths["dt_eff_by_formula"] = dt_eff_path
    print(f"Saved: {dt_eff_path}")

    # dt_eff all overlaid
    dt_eff_all_path = output_dir / "dt_eff_all.png"
    plot_dt_eff_trajectories(
        trajectories,
        output_path=dt_eff_all_path,
        group_by_formula=False,
        color_by_morse=True,
        max_steps=max_steps,
    )
    paths["dt_eff_all"] = dt_eff_all_path
    print(f"Saved: {dt_eff_all_path}")

    # TR mode diagnostics
    tr_path = output_dir / "tr_mode_diagnostics.png"
    plot_tr_mode_diagnostics(
        trajectories,
        output_path=tr_path,
        max_steps=max_steps,
    )
    paths["tr_diagnostics"] = tr_path
    print(f"Saved: {tr_path}")

    # Morse index evolution
    morse_path = output_dir / "morse_index_evolution.png"
    plot_morse_index_evolution(
        trajectories,
        output_path=morse_path,
        group_by_formula=True,
        max_steps=max_steps,
    )
    paths["morse_evolution"] = morse_path
    print(f"Saved: {morse_path}")

    # Summary report
    report_path = output_dir / "summary_report.json"
    create_summary_report(trajectories, output_path=report_path)
    paths["summary_report"] = report_path
    print(f"Saved: {report_path}")

    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate GAD trajectory diagnostic plots")
    parser.add_argument("diagnostics_dir", type=str, help="Directory with trajectory JSON files")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum steps to plot")

    args = parser.parse_args()
    generate_all_plots(args.diagnostics_dir, args.output_dir, args.max_steps)
