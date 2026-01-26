"""Singularity analysis for GAD failure diagnosis.

This module analyzes whether GAD failures correlate with proximity to
singularity sets S = {x : λ₁(x) = λ₂(x)}, as predicted by the
Levitt-Ortner paper.

Key questions to answer:
1. When does GAD stall? At what eigenvalue gap values?
2. Is there a correlation between small |λ₂ - λ₁| and stalling?
3. Do escapes happen near singularities more often?
4. What is the typical eigenvalue spectrum when stuck?
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_batch_summary(path: str | Path) -> List[Dict[str, Any]]:
    """Load batch summary from diagnostic run."""
    with open(path) as f:
        return json.load(f)


def load_trajectory(path: str | Path) -> Dict[str, List[Any]]:
    """Load trajectory metrics from diagnostic run."""
    with open(path) as f:
        return json.load(f)


def analyze_stall_eigenvalues(
    trajectory: Dict[str, List[Any]],
    stall_threshold_disp: float = 1e-5,
    n_context_steps: int = 10,
) -> Dict[str, Any]:
    """Analyze eigenvalue characteristics when GAD stalls.

    A "stall" is defined as consecutive steps where displacement < threshold.

    Args:
        trajectory: Dictionary with metric lists
        stall_threshold_disp: Displacement threshold for stall detection
        n_context_steps: Number of steps around stall to analyze

    Returns:
        Dictionary with stall analysis results
    """
    if "x_disp_step" not in trajectory:
        # Try alternative names
        if "disp_from_last" in trajectory:
            disps = trajectory["disp_from_last"]
        else:
            return {"error": "No displacement data found"}
    else:
        disps = trajectory["x_disp_step"]

    # Find stall regions (consecutive low-displacement steps)
    stall_starts = []
    stall_ends = []
    in_stall = False
    stall_start = 0

    for i, d in enumerate(disps):
        if d < stall_threshold_disp:
            if not in_stall:
                in_stall = True
                stall_start = i
        else:
            if in_stall:
                in_stall = False
                if i - stall_start >= 5:  # At least 5 consecutive steps
                    stall_starts.append(stall_start)
                    stall_ends.append(i)

    if in_stall and len(disps) - stall_start >= 5:
        stall_starts.append(stall_start)
        stall_ends.append(len(disps))

    if not stall_starts:
        return {
            "n_stalls": 0,
            "stall_eigenvalue_gaps": [],
            "stall_morse_indices": [],
            "stall_singularity_metrics": [],
        }

    # Analyze eigenvalue characteristics during stalls
    stall_gaps = []
    stall_indices = []
    stall_singularities = []

    eig_gap_key = "eig_gap_01" if "eig_gap_01" in trajectory else None
    morse_key = "morse_index" if "morse_index" in trajectory else ("neg_vib" if "neg_vib" in trajectory else None)
    sing_key = "singularity_metric" if "singularity_metric" in trajectory else None

    for start, end in zip(stall_starts, stall_ends):
        if eig_gap_key:
            gaps = [trajectory[eig_gap_key][i] for i in range(start, end) if i < len(trajectory[eig_gap_key])]
            gaps = [g for g in gaps if np.isfinite(g)]
            if gaps:
                stall_gaps.append({
                    "mean": float(np.mean(gaps)),
                    "min": float(np.min(gaps)),
                    "max": float(np.max(gaps)),
                })

        if morse_key:
            indices = [trajectory[morse_key][i] for i in range(start, end) if i < len(trajectory[morse_key])]
            if indices:
                stall_indices.append({
                    "mean": float(np.mean(indices)),
                    "mode": int(np.bincount(indices).argmax()) if indices else -1,
                    "stable": float(np.std(indices)) < 0.5,
                })

        if sing_key:
            sings = [trajectory[sing_key][i] for i in range(start, end) if i < len(trajectory[sing_key])]
            sings = [s for s in sings if np.isfinite(s)]
            if sings:
                stall_singularities.append({
                    "mean": float(np.mean(sings)),
                    "min": float(np.min(sings)),
                })

    # Compute summary statistics
    all_gap_means = [g["mean"] for g in stall_gaps]
    all_gap_mins = [g["min"] for g in stall_gaps]
    all_indices = [i["mode"] for i in stall_indices]

    return {
        "n_stalls": len(stall_starts),
        "stall_durations": [e - s for s, e in zip(stall_starts, stall_ends)],
        "stall_eigenvalue_gaps": stall_gaps,
        "stall_morse_indices": stall_indices,
        "stall_singularity_metrics": stall_singularities,
        # Summary
        "mean_gap_at_stall": float(np.mean(all_gap_means)) if all_gap_means else float("nan"),
        "min_gap_at_stall": float(np.min(all_gap_mins)) if all_gap_mins else float("nan"),
        "dominant_morse_index_at_stall": int(np.bincount(all_indices).argmax()) if all_indices else -1,
        "fraction_near_singularity": sum(1 for g in all_gap_mins if g < 0.01) / len(all_gap_mins) if all_gap_mins else float("nan"),
    }


def analyze_escape_singularity_correlation(
    escapes: List[Dict[str, Any]],
    singularity_threshold: float = 0.01,
) -> Dict[str, Any]:
    """Analyze correlation between escape events and singularity proximity.

    Args:
        escapes: List of escape event dictionaries
        singularity_threshold: Gap threshold for "near singularity"

    Returns:
        Dictionary with correlation analysis
    """
    if not escapes:
        return {"n_escapes": 0}

    near_singularity = []
    successful_escapes = []
    gaps_before = []
    gaps_after = []
    indices_before = []
    indices_after = []

    for esc in escapes:
        gap = esc.get("pre_eig_gap_01", float("nan"))
        if np.isfinite(gap):
            gaps_before.append(gap)
            near_singularity.append(gap < singularity_threshold)

        gap_after = esc.get("post_eig_gap_01", float("nan"))
        if np.isfinite(gap_after):
            gaps_after.append(gap_after)

        indices_before.append(esc.get("pre_morse_index", -1))
        indices_after.append(esc.get("post_morse_index", -1))
        successful_escapes.append(esc.get("accepted", False))

    # Compute success rates
    n_near_sing = sum(near_singularity)
    n_far_sing = len(near_singularity) - n_near_sing

    success_near_sing = sum(s for s, n in zip(successful_escapes[:len(near_singularity)], near_singularity) if n)
    success_far_sing = sum(s for s, n in zip(successful_escapes[:len(near_singularity)], near_singularity) if not n)

    return {
        "n_escapes": len(escapes),
        "n_accepted": sum(successful_escapes),
        "acceptance_rate": sum(successful_escapes) / len(escapes),
        # Singularity correlation
        "n_near_singularity": n_near_sing,
        "n_far_singularity": n_far_sing,
        "success_rate_near_singularity": success_near_sing / n_near_sing if n_near_sing > 0 else float("nan"),
        "success_rate_far_singularity": success_far_sing / n_far_sing if n_far_sing > 0 else float("nan"),
        # Gap statistics
        "mean_gap_before": float(np.mean(gaps_before)) if gaps_before else float("nan"),
        "mean_gap_after": float(np.mean(gaps_after)) if gaps_after else float("nan"),
        "gap_improvement": float(np.mean(gaps_after)) - float(np.mean(gaps_before)) if gaps_before and gaps_after else float("nan"),
        # Index statistics
        "mean_index_before": float(np.mean(indices_before)) if indices_before else float("nan"),
        "mean_index_after": float(np.mean(indices_after)) if indices_after else float("nan"),
    }


def analyze_spectrum_evolution(
    trajectory: Dict[str, List[Any]],
    window_size: int = 50,
) -> Dict[str, Any]:
    """Analyze how eigenvalue spectrum evolves over trajectory.

    Args:
        trajectory: Dictionary with metric lists
        window_size: Size of sliding window for statistics

    Returns:
        Dictionary with spectrum evolution analysis
    """
    eig_keys = ["eig_0", "eig_1", "eig_2", "eig_3", "eig_4", "eig_5"]
    available_keys = [k for k in eig_keys if k in trajectory]

    if not available_keys:
        return {"error": "No eigenvalue data found"}

    n_steps = len(trajectory[available_keys[0]])

    # Compute statistics in windows
    windows = []
    for start in range(0, n_steps - window_size + 1, window_size // 2):
        end = start + window_size
        window_stats = {"start": start, "end": end}

        for key in available_keys:
            vals = trajectory[key][start:end]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                window_stats[f"{key}_mean"] = float(np.mean(vals))
                window_stats[f"{key}_std"] = float(np.std(vals))

        # Eigenvalue gap
        if "eig_gap_01" in trajectory:
            gaps = trajectory["eig_gap_01"][start:end]
            gaps = [g for g in gaps if np.isfinite(g)]
            if gaps:
                window_stats["gap_mean"] = float(np.mean(gaps))
                window_stats["gap_min"] = float(np.min(gaps))

        # Morse index
        if "morse_index" in trajectory:
            indices = trajectory["morse_index"][start:end]
            if indices:
                window_stats["index_mean"] = float(np.mean(indices))
                window_stats["index_std"] = float(np.std(indices))

        windows.append(window_stats)

    return {
        "n_windows": len(windows),
        "window_size": window_size,
        "windows": windows,
    }


def generate_singularity_report(
    diagnostics_dir: str | Path,
) -> str:
    """Generate a human-readable singularity analysis report.

    Args:
        diagnostics_dir: Path to diagnostics output directory

    Returns:
        Report string
    """
    diagnostics_dir = Path(diagnostics_dir)

    lines = [
        "=" * 70,
        "SINGULARITY ANALYSIS REPORT",
        "=" * 70,
        "",
    ]

    # Load batch summary
    batch_path = diagnostics_dir / "batch_summary.json"
    if batch_path.exists():
        summaries = load_batch_summary(batch_path)
        lines.append(f"Total samples analyzed: {len(summaries)}")
        lines.append(f"Converged to TS: {sum(1 for s in summaries if s.get('converged_to_ts', False))}")
        lines.append("")

    # Find all trajectory files
    traj_files = list(diagnostics_dir.glob("*_trajectory.json"))
    escape_files = list(diagnostics_dir.glob("*_escapes.json"))

    if not traj_files:
        lines.append("No trajectory files found.")
        return "\n".join(lines)

    # Aggregate stall analysis
    all_stall_gaps = []
    all_stall_indices = []
    total_stalls = 0

    for traj_file in traj_files:
        try:
            trajectory = load_trajectory(traj_file)
            analysis = analyze_stall_eigenvalues(trajectory)

            total_stalls += analysis.get("n_stalls", 0)
            for g in analysis.get("stall_eigenvalue_gaps", []):
                all_stall_gaps.append(g["min"])
            for i in analysis.get("stall_morse_indices", []):
                all_stall_indices.append(i["mode"])
        except Exception:
            continue

    lines.append("-" * 70)
    lines.append("STALL ANALYSIS")
    lines.append("-" * 70)
    lines.append(f"Total stall events: {total_stalls}")

    if all_stall_gaps:
        lines.append(f"Mean eigenvalue gap at stall: {np.mean(all_stall_gaps):.4e}")
        lines.append(f"Min eigenvalue gap at stall: {np.min(all_stall_gaps):.4e}")
        lines.append(f"Fraction of stalls near singularity (gap < 0.01): {sum(1 for g in all_stall_gaps if g < 0.01) / len(all_stall_gaps):.1%}")

    if all_stall_indices:
        lines.append(f"Dominant Morse index at stall: {int(np.bincount(all_stall_indices).argmax())}")

    # Aggregate escape analysis
    all_escapes = []
    for esc_file in escape_files:
        try:
            with open(esc_file) as f:
                escapes = json.load(f)
                all_escapes.extend(escapes)
        except Exception:
            continue

    if all_escapes:
        lines.append("")
        lines.append("-" * 70)
        lines.append("ESCAPE ANALYSIS")
        lines.append("-" * 70)

        esc_analysis = analyze_escape_singularity_correlation(all_escapes)
        lines.append(f"Total escape events: {esc_analysis['n_escapes']}")
        lines.append(f"Acceptance rate: {esc_analysis['acceptance_rate']:.1%}")
        lines.append(f"Near singularity: {esc_analysis['n_near_singularity']}")
        lines.append(f"Far from singularity: {esc_analysis['n_far_singularity']}")

        if np.isfinite(esc_analysis["success_rate_near_singularity"]):
            lines.append(f"Success rate near singularity: {esc_analysis['success_rate_near_singularity']:.1%}")
        if np.isfinite(esc_analysis["success_rate_far_singularity"]):
            lines.append(f"Success rate far from singularity: {esc_analysis['success_rate_far_singularity']:.1%}")

        lines.append(f"Mean gap before escape: {esc_analysis['mean_gap_before']:.4e}")
        lines.append(f"Mean gap after escape: {esc_analysis['mean_gap_after']:.4e}")
        lines.append(f"Mean index before: {esc_analysis['mean_index_before']:.2f}")
        lines.append(f"Mean index after: {esc_analysis['mean_index_after']:.2f}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("KEY FINDINGS")
    lines.append("=" * 70)

    # Determine key findings
    if all_stall_gaps:
        frac_near_sing = sum(1 for g in all_stall_gaps if g < 0.01) / len(all_stall_gaps)
        if frac_near_sing > 0.5:
            lines.append("- STALLS ARE CORRELATED WITH SINGULARITIES (>50% near λ₁≈λ₂)")
        else:
            lines.append("- Stalls are NOT strongly correlated with singularities")

    if all_stall_indices:
        dom_idx = int(np.bincount(all_stall_indices).argmax())
        if dom_idx > 1:
            lines.append(f"- STALLS OCCUR AT HIGH-INDEX SADDLES (dominant index = {dom_idx})")
        elif dom_idx == 1:
            lines.append("- Stalls occur at index-1 saddles (grad convergence issue?)")

    if all_escapes:
        near_sing_rate = esc_analysis.get("success_rate_near_singularity", float("nan"))
        far_sing_rate = esc_analysis.get("success_rate_far_singularity", float("nan"))
        if np.isfinite(near_sing_rate) and np.isfinite(far_sing_rate):
            if near_sing_rate < far_sing_rate - 0.1:
                lines.append("- v₂ KICKS ARE LESS EFFECTIVE NEAR SINGULARITIES")
            elif near_sing_rate > far_sing_rate + 0.1:
                lines.append("- v₂ kicks are MORE effective near singularities")
            else:
                lines.append("- v₂ kick effectiveness is similar near/far from singularities")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python singularity_analysis.py <diagnostics_dir>")
        sys.exit(1)

    report = generate_singularity_report(sys.argv[1])
    print(report)

    # Also save report
    report_path = Path(sys.argv[1]) / "singularity_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
