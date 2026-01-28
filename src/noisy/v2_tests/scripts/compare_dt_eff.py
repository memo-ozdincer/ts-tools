#!/usr/bin/env python
"""Compare dt_eff evolution between path-based and state-based adaptive timestep methods.

This script loads trajectory data from runs using:
1. Path-based adaptive control (gad_plain_run.slurm)
2. State-based adaptive strategies (gad_plain_run_nopath.slurm)

And compares how dt_eff changes over time for each trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np


def load_trajectory_data(diag_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """Load all trajectory JSON files from a diagnostics directory.

    Args:
        diag_dir: Path to diagnostics directory containing *_trajectory.json files

    Returns:
        Dictionary mapping sample_id -> trajectory data dict
    """
    trajectories = {}

    for traj_file in sorted(diag_dir.glob("*_trajectory.json")):
        # Extract sample_id from filename (e.g., "sample_000_trajectory.json" -> "sample_000")
        sample_id = traj_file.stem.replace("_trajectory", "")

        with open(traj_file) as f:
            traj_data = json.load(f)

        trajectories[sample_id] = traj_data

    return trajectories


def compare_dt_eff_single_trajectory(
    sample_id: str,
    dt_path: List[float],
    dt_nopath: List[float],
    output_path: Path,
) -> Dict[str, Any]:
    """Compare dt_eff evolution for a single trajectory.

    Args:
        sample_id: Sample identifier
        dt_path: dt_eff values from path-based method
        dt_nopath: dt_eff values from state-based method
        output_path: Where to save the comparison plot

    Returns:
        Dictionary with comparison statistics
    """
    n_steps_path = len(dt_path)
    n_steps_nopath = len(dt_nopath)
    n_steps = min(n_steps_path, n_steps_nopath)

    # Truncate to common length for comparison
    dt_path_trunc = dt_path[:n_steps]
    dt_nopath_trunc = dt_nopath[:n_steps]

    # Compute statistics
    stats = {
        "sample_id": sample_id,
        "n_steps_path": n_steps_path,
        "n_steps_nopath": n_steps_nopath,
        "n_steps_compared": n_steps,
        "path_method": {
            "mean_dt": float(np.mean(dt_path_trunc)),
            "std_dt": float(np.std(dt_path_trunc)),
            "min_dt": float(np.min(dt_path_trunc)),
            "max_dt": float(np.max(dt_path_trunc)),
            "final_dt": float(dt_path_trunc[-1]) if dt_path_trunc else None,
        },
        "state_method": {
            "mean_dt": float(np.mean(dt_nopath_trunc)),
            "std_dt": float(np.std(dt_nopath_trunc)),
            "min_dt": float(np.min(dt_nopath_trunc)),
            "max_dt": float(np.max(dt_nopath_trunc)),
            "final_dt": float(dt_nopath_trunc[-1]) if dt_nopath_trunc else None,
        },
    }

    # Create comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: dt_eff over time
    ax1 = axes[0]
    steps_path = np.arange(len(dt_path))
    steps_nopath = np.arange(len(dt_nopath))

    ax1.plot(steps_path, dt_path, label="Path-based (adaptive)", alpha=0.8, linewidth=1.5)
    ax1.plot(steps_nopath, dt_nopath, label="State-based (eigenvalue)", alpha=0.8, linewidth=1.5)
    ax1.set_ylabel("dt_eff", fontsize=12)
    ax1.set_title(f"Adaptive Timestep Evolution: {sample_id}", fontsize=14, fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot 2: Ratio of dt_eff (path / nopath)
    ax2 = axes[1]
    ratio = np.array(dt_path_trunc) / np.array(dt_nopath_trunc)
    ax2.plot(np.arange(n_steps), ratio, color="purple", alpha=0.7, linewidth=1.5)
    ax2.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Equal")
    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("dt_path / dt_nopath", fontsize=12)
    ax2.set_title("Ratio of Path-based to State-based dt_eff", fontsize=12)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return stats


def compare_all_trajectories(
    path_dir: Path,
    nopath_dir: Path,
    output_dir: Path,
) -> None:
    """Compare dt_eff evolution for all matching trajectories.

    Args:
        path_dir: Diagnostics directory from path-based runs
        nopath_dir: Diagnostics directory from state-based runs
        output_dir: Where to save comparison plots and statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trajectory data
    print(f"Loading path-based trajectories from: {path_dir}")
    traj_path = load_trajectory_data(path_dir)
    print(f"  Found {len(traj_path)} trajectories")

    print(f"Loading state-based trajectories from: {nopath_dir}")
    traj_nopath = load_trajectory_data(nopath_dir)
    print(f"  Found {len(traj_nopath)} trajectories")

    # Find common sample IDs
    common_samples = sorted(set(traj_path.keys()) & set(traj_nopath.keys()))
    print(f"\nComparing {len(common_samples)} common trajectories")

    if not common_samples:
        print("ERROR: No common sample IDs found between the two runs!")
        return

    # Compare each trajectory
    all_stats = []

    for sample_id in common_samples:
        print(f"  Processing {sample_id}...")

        dt_path = traj_path[sample_id].get("dt_eff") or traj_path[sample_id].get("step_size_eff", [])
        dt_nopath = traj_nopath[sample_id].get("dt_eff") or traj_nopath[sample_id].get("step_size_eff", [])

        if not dt_path or not dt_nopath:
            print(f"    WARNING: Missing dt_eff (or step_size_eff) data for {sample_id}")
            continue

        plot_path = output_dir / f"{sample_id}_dt_eff_comparison.png"
        stats = compare_dt_eff_single_trajectory(
            sample_id=sample_id,
            dt_path=dt_path,
            dt_nopath=dt_nopath,
            output_path=plot_path,
        )
        all_stats.append(stats)

        print(f"    Path method: mean={stats['path_method']['mean_dt']:.6f}, "
              f"std={stats['path_method']['std_dt']:.6f}")
        print(f"    State method: mean={stats['state_method']['mean_dt']:.6f}, "
              f"std={stats['state_method']['std_dt']:.6f}")

    # Save summary statistics
    summary_path = output_dir / "dt_eff_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nComparison complete!")
    print(f"  Individual plots saved to: {output_dir}")
    print(f"  Summary statistics saved to: {summary_path}")

    # Create aggregate comparison plot
    create_aggregate_plot(all_stats, output_dir)


def create_aggregate_plot(all_stats: List[Dict[str, Any]], output_dir: Path) -> None:
    """Create aggregate comparison plots across all trajectories.

    Args:
        all_stats: List of comparison statistics for each trajectory
        output_dir: Where to save the aggregate plot
    """
    if not all_stats:
        return

    # Extract statistics
    mean_dt_path = [s["path_method"]["mean_dt"] for s in all_stats]
    mean_dt_nopath = [s["state_method"]["mean_dt"] for s in all_stats]
    std_dt_path = [s["path_method"]["std_dt"] for s in all_stats]
    std_dt_nopath = [s["state_method"]["std_dt"] for s in all_stats]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean dt_eff comparison
    ax1 = axes[0]
    x = np.arange(len(all_stats))
    width = 0.35

    ax1.bar(x - width/2, mean_dt_path, width, label="Path-based", alpha=0.8)
    ax1.bar(x + width/2, mean_dt_nopath, width, label="State-based", alpha=0.8)
    ax1.set_xlabel("Trajectory", fontsize=12)
    ax1.set_ylabel("Mean dt_eff", fontsize=12)
    ax1.set_title("Mean Timestep Comparison Across Trajectories", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Scatter plot of mean dt values
    ax2 = axes[1]
    ax2.scatter(mean_dt_path, mean_dt_nopath, alpha=0.6, s=100)

    # Add diagonal line
    min_val = min(min(mean_dt_path), min(mean_dt_nopath))
    max_val = max(max(mean_dt_path), max(mean_dt_nopath))
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label="Equal")

    ax2.set_xlabel("Path-based mean dt_eff", fontsize=12)
    ax2.set_ylabel("State-based mean dt_eff", fontsize=12)
    ax2.set_title("Mean Timestep Correlation", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    plt.tight_layout()
    agg_plot_path = output_dir / "aggregate_dt_eff_comparison.png"
    plt.savefig(agg_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Aggregate plot saved to: {agg_plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare dt_eff evolution between path-based and state-based adaptive methods"
    )
    parser.add_argument(
        "--path-dir",
        type=str,
        required=True,
        help="Diagnostics directory from path-based run (gad_plain_run.slurm)",
    )
    parser.add_argument(
        "--nopath-dir",
        type=str,
        required=True,
        help="Diagnostics directory from state-based run (gad_plain_run_nopath.slurm)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for comparison plots and statistics",
    )

    args = parser.parse_args()

    path_dir = Path(args.path_dir)
    nopath_dir = Path(args.nopath_dir)
    output_dir = Path(args.output_dir)

    if not path_dir.exists():
        raise FileNotFoundError(f"Path-based diagnostics directory not found: {path_dir}")

    if not nopath_dir.exists():
        raise FileNotFoundError(f"State-based diagnostics directory not found: {nopath_dir}")

    compare_all_trajectories(path_dir, nopath_dir, output_dir)


if __name__ == "__main__":
    main()
