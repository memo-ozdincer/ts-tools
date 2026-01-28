#!/usr/bin/env python
"""Analyze TR mode diagnostics from trajectory logs.

This script inspects *_trajectory.json files and summarizes whether the
translation/rotation (TR) modes remain near-zero as expected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def load_trajectory_files(diag_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all trajectory JSON files from a diagnostics directory."""
    trajectories: Dict[str, Dict[str, Any]] = {}
    for traj_file in sorted(diag_dir.glob("*_trajectory.json")):
        sample_id = traj_file.stem.replace("_trajectory", "")
        with open(traj_file) as f:
            data = json.load(f)
        trajectories[sample_id] = data
    return trajectories


def _safe_array(values: Optional[List[float]]) -> np.ndarray:
    if not values:
        return np.array([], dtype=float)
    return np.array(values, dtype=float)


def summarize_tr_metrics(
    sample_id: str,
    traj: Dict[str, Any],
    *,
    expected_tr_modes: int,
    tr_threshold: float,
) -> Dict[str, Any]:
    """Compute summary statistics for TR diagnostics for one trajectory."""
    n_tr_modes = _safe_array(traj.get("n_tr_modes"))
    tr_eig_max = _safe_array(traj.get("tr_eig_max"))
    tr_eig_mean = _safe_array(traj.get("tr_eig_mean"))
    tr_eig_std = _safe_array(traj.get("tr_eig_std"))

    n_steps = int(max(len(n_tr_modes), len(tr_eig_max), len(tr_eig_mean), len(tr_eig_std)))
    if n_steps == 0:
        return {
            "sample_id": sample_id,
            "n_steps": 0,
            "error": "No TR diagnostics found in trajectory.",
        }

    if len(n_tr_modes) > 0:
        frac_expected = float(np.mean(n_tr_modes == expected_tr_modes))
        frac_off = float(np.mean(n_tr_modes != expected_tr_modes))
        n_tr_min = int(np.min(n_tr_modes))
        n_tr_max = int(np.max(n_tr_modes))
        n_tr_unique = sorted({int(x) for x in n_tr_modes.tolist()})
        n_tr_changes = int(np.sum(n_tr_modes[1:] != n_tr_modes[:-1])) if len(n_tr_modes) > 1 else 0
    else:
        frac_expected = float("nan")
        frac_off = float("nan")
        n_tr_min = -1
        n_tr_max = -1
        n_tr_unique = []
        n_tr_changes = 0

    if len(tr_eig_max) > 0:
        frac_over_threshold = float(np.mean(tr_eig_max > tr_threshold))
        max_tr_eig = float(np.max(tr_eig_max))
        mean_tr_eig = float(np.mean(tr_eig_max))
    else:
        frac_over_threshold = float("nan")
        max_tr_eig = float("nan")
        mean_tr_eig = float("nan")

    return {
        "sample_id": sample_id,
        "n_steps": n_steps,
        "expected_tr_modes": expected_tr_modes,
        "tr_threshold": tr_threshold,
        "n_tr_modes_min": n_tr_min,
        "n_tr_modes_max": n_tr_max,
        "n_tr_modes_unique": n_tr_unique,
        "n_tr_modes_changes": n_tr_changes,
        "frac_steps_expected_tr_modes": frac_expected,
        "frac_steps_off_tr_modes": frac_off,
        "frac_steps_tr_eig_max_over_threshold": frac_over_threshold,
        "tr_eig_max_max": max_tr_eig,
        "tr_eig_max_mean": mean_tr_eig,
        "tr_eig_mean_mean": float(np.mean(tr_eig_mean)) if len(tr_eig_mean) > 0 else float("nan"),
        "tr_eig_std_mean": float(np.mean(tr_eig_std)) if len(tr_eig_std) > 0 else float("nan"),
    }


def plot_tr_metrics(
    sample_id: str,
    traj: Dict[str, Any],
    output_dir: Path,
    *,
    tr_threshold: float,
    expected_tr_modes: int,
) -> None:
    """Create TR diagnostics plots for a single trajectory."""
    n_tr_modes = _safe_array(traj.get("n_tr_modes"))
    tr_eig_max = _safe_array(traj.get("tr_eig_max"))
    tr_eig_mean = _safe_array(traj.get("tr_eig_mean"))
    tr_eig_std = _safe_array(traj.get("tr_eig_std"))

    if len(n_tr_modes) == 0 and len(tr_eig_max) == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1 = axes[0]
    if len(tr_eig_max) > 0:
        ax1.plot(tr_eig_max, label="tr_eig_max", linewidth=1.5)
    if len(tr_eig_mean) > 0:
        ax1.plot(tr_eig_mean, label="tr_eig_mean", linewidth=1.0, alpha=0.7)
    if len(tr_eig_std) > 0:
        ax1.plot(tr_eig_std, label="tr_eig_std", linewidth=1.0, alpha=0.7)
    ax1.axhline(y=tr_threshold, color="red", linestyle="--", alpha=0.6, label="tr_threshold")
    ax1.set_ylabel("|TR eigenvalues|", fontsize=11)
    ax1.set_title(f"TR eigenvalue diagnostics: {sample_id}", fontsize=12, fontweight="bold")
    ax1.set_yscale("log")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    if len(n_tr_modes) > 0:
        ax2.plot(n_tr_modes, label="n_tr_modes", linewidth=1.5)
        ax2.axhline(y=expected_tr_modes, color="black", linestyle="--", alpha=0.6, label="expected")
    ax2.set_xlabel("Step", fontsize=11)
    ax2.set_ylabel("# TR modes", fontsize=11)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sample_id}_tr_diagnostics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze TR mode diagnostics from trajectory logs"
    )
    parser.add_argument(
        "--diag-dir",
        type=str,
        required=True,
        help="Diagnostics directory containing *_trajectory.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for summaries and plots",
    )
    parser.add_argument(
        "--tr-threshold",
        type=float,
        default=1e-6,
        help="Threshold for treating TR eigenvalues as near-zero",
    )
    parser.add_argument(
        "--expected-tr-modes",
        type=int,
        default=6,
        help="Expected number of TR modes (6 for nonlinear molecules)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate per-trajectory plots",
    )

    args = parser.parse_args()

    diag_dir = Path(args.diag_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectory_files(diag_dir)
    if not trajectories:
        raise FileNotFoundError(f"No *_trajectory.json files found in {diag_dir}")

    summaries: List[Dict[str, Any]] = []

    for sample_id, traj in trajectories.items():
        summary = summarize_tr_metrics(
            sample_id,
            traj,
            expected_tr_modes=args.expected_tr_modes,
            tr_threshold=args.tr_threshold,
        )
        summaries.append(summary)

        if args.plot:
            plot_tr_metrics(
                sample_id,
                traj,
                output_dir,
                tr_threshold=args.tr_threshold,
                expected_tr_modes=args.expected_tr_modes,
            )

    summary_path = output_dir / "tr_mode_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    # Aggregate report
    valid = [s for s in summaries if "error" not in s]
    aggregate = {
        "n_trajectories": len(valid),
        "tr_threshold": args.tr_threshold,
        "expected_tr_modes": args.expected_tr_modes,
        "mean_frac_expected_tr_modes": float(np.mean([s["frac_steps_expected_tr_modes"] for s in valid])) if valid else float("nan"),
        "mean_frac_tr_eig_over_threshold": float(np.mean([s["frac_steps_tr_eig_max_over_threshold"] for s in valid])) if valid else float("nan"),
        "max_tr_eig_overall": float(np.max([s["tr_eig_max_max"] for s in valid])) if valid else float("nan"),
        "mean_n_tr_modes_changes": float(np.mean([s["n_tr_modes_changes"] for s in valid])) if valid else float("nan"),
        "min_n_tr_modes_overall": int(np.min([s["n_tr_modes_min"] for s in valid])) if valid else -1,
        "max_n_tr_modes_overall": int(np.max([s["n_tr_modes_max"] for s in valid])) if valid else -1,
    }
    aggregate_path = output_dir / "tr_mode_aggregate.json"
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"Saved per-trajectory summary: {summary_path}")
    print(f"Saved aggregate summary: {aggregate_path}")
    if args.plot:
        print(f"Saved plots in: {output_dir}")


if __name__ == "__main__":
    main()
