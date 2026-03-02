#!/usr/bin/env python3
"""Eigenvalue tolerance investigation for Newton-Raphson minimization trajectories.

Consumes trajectory JSON files from grid-search runs and produces evidence
FOR or AGAINST eigenvalue tolerance filtering.  Six analysis modules:

  A1. Band Evolution Statistics
  A2. Tolerance Sweep Experiment
  A3. Physical Significance Analysis
  A4. Stability / False Convergence Detection
  A5. Cascade Gap Analysis
  A6. Eigenvalue-Force-Energy Consistency Check

Usage:
  python analyze_nr_eigenvalue_justification.py \\
      --grid-dir <path> --output-dir <path> \\
      [--traj-glob "*/diagnostics/*_trajectory.json"] \\
      [--combo-tag <tag>] [--top-k-heatmaps 10]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOLERANCE_SWEEP = [0.0, 1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

MILESTONE_STEPS = [0, 10, 100, 500]  # "final" added dynamically

BAND_KEYS = [
    "n_eval_below_neg1e-1",
    "n_eval_neg1e-1_to_neg1e-2",
    "n_eval_neg1e-2_to_neg1e-3",
    "n_eval_neg1e-3_to_neg1e-4",
    "n_eval_neg1e-4_to_0",
    "n_eval_0_to_pos1e-4",
    "n_eval_pos1e-4_to_pos1e-3",
    "n_eval_above_pos1e-3",
]

BAND_LABELS = [
    "<-1e-1",
    "[-1e-1,-1e-2)",
    "[-1e-2,-1e-3)",
    "[-1e-3,-1e-4)",
    "[-1e-4,0)",
    "[0,+1e-4)",
    "[+1e-4,+1e-3)",
    ">+1e-3",
]

LOOKAHEAD_WINDOWS = [1, 5, 10, 50, 100, 500]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _get(step: Dict, key: str, default: Any = float("nan")) -> Any:
    """Safe field access from a trajectory step."""
    return step.get(key, default)


def _col(steps: List[Dict], key: str, default: float = float("nan")) -> np.ndarray:
    """Extract a column from trajectory steps as a numpy array."""
    return np.array([s.get(key, default) for s in steps], dtype=float)


def _finite(arr: np.ndarray) -> np.ndarray:
    """Return only finite elements."""
    return arr[np.isfinite(arr)]


def _is_converged(traj_data: Dict[str, Any]) -> bool:
    """Determine whether a trajectory converged (final_neg_vib == 0)."""
    fnv = traj_data.get("final_neg_vib")
    if fnv is not None:
        return fnv == 0
    # Fallback: check last step
    traj = traj_data.get("trajectory", [])
    if traj:
        return traj[-1].get("n_neg_evals", -1) == 0
    return False


def _first_zero_step(steps: List[Dict]) -> Optional[int]:
    """Return the first step index where n_neg_evals == 0, or None."""
    for i, s in enumerate(steps):
        if s.get("n_neg_evals", -1) == 0:
            return i
    return None


def _safe_log10(x: float) -> float:
    """Safe log10 for positive numbers."""
    if x <= 0:
        return float("nan")
    return math.log10(x)


def _band_populations(step: Dict) -> List[int]:
    """Extract the 8 band populations from a trajectory step."""
    return [step.get(k, 0) for k in BAND_KEYS]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_trajectories(
    grid_dir: Path, traj_glob: str, combo_tag: Optional[str] = None
) -> List[Tuple[str, Path]]:
    """Find trajectory JSON files, optionally filtered by combo tag.

    Returns list of (combo_tag_string, path) tuples.
    """
    results: List[Tuple[str, Path]] = []
    for p in sorted(grid_dir.glob(traj_glob)):
        # Derive combo tag from directory structure: first path component
        # relative to grid_dir is typically the combo tag folder.
        rel = p.relative_to(grid_dir)
        tag = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        if combo_tag is not None and tag != combo_tag:
            continue
        results.append((tag, p))
    return results


def load_trajectory(path: Path) -> Optional[Dict[str, Any]]:
    """Load a trajectory JSON, returning None on failure."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  WARNING: could not load {path}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# A1. Band Evolution Statistics
# ---------------------------------------------------------------------------

def analyze_band_evolution(
    all_trajs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """A1: Band evolution statistics.

    Returns a dict with:
      - milestone_stats: {milestone -> {"converged": {band: mean_pop}, "failed": ...}}
      - ghost_onset_converged: list of first-step indices where ghost-only negative evals appear
      - ghost_onset_failed: same for failed trajectories
    """
    milestone_stats: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    ghost_onset_converged: List[int] = []
    ghost_onset_failed: List[int] = []

    for traj_data in all_trajs:
        steps = traj_data.get("trajectory", [])
        if not steps:
            continue
        converged = _is_converged(traj_data)
        label = "converged" if converged else "failed"

        n_steps = len(steps)
        milestones = MILESTONE_STEPS + [n_steps - 1]

        for ms in milestones:
            idx = min(ms, n_steps - 1)
            if ms not in milestone_stats:
                milestone_stats[ms] = {"converged": defaultdict(list), "failed": defaultdict(list)}
            pops = _band_populations(steps[idx])
            for bk, pop in zip(BAND_KEYS, pops):
                milestone_stats[ms][label][bk].append(pop)

        # Ghost onset detection: first step where eigenvalues exist in [-1e-4, 0)
        # but NONE below -1e-4.
        ghost_onset = None
        for i, s in enumerate(steps):
            n_ghost_band = s.get("n_eval_neg1e-4_to_0", 0)
            n_below = (
                s.get("n_eval_below_neg1e-1", 0)
                + s.get("n_eval_neg1e-1_to_neg1e-2", 0)
                + s.get("n_eval_neg1e-2_to_neg1e-3", 0)
                + s.get("n_eval_neg1e-3_to_neg1e-4", 0)
            )
            if n_ghost_band > 0 and n_below == 0:
                ghost_onset = i
                break
        if ghost_onset is not None:
            if converged:
                ghost_onset_converged.append(ghost_onset)
            else:
                ghost_onset_failed.append(ghost_onset)

    # Aggregate milestone stats to means
    milestone_means: Dict[int, Dict[str, Dict[str, float]]] = {}
    for ms, groups in milestone_stats.items():
        milestone_means[ms] = {}
        for label, bands in groups.items():
            milestone_means[ms][label] = {}
            for bk, vals in bands.items():
                if vals:
                    milestone_means[ms][label][bk] = float(np.nanmean(vals))
                else:
                    milestone_means[ms][label][bk] = 0.0

    return {
        "milestone_means": milestone_means,
        "ghost_onset_converged": ghost_onset_converged,
        "ghost_onset_failed": ghost_onset_failed,
    }


# ---------------------------------------------------------------------------
# A2. Tolerance Sweep Experiment
# ---------------------------------------------------------------------------

def analyze_tolerance_sweep(
    all_trajs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """A2: Tolerance sweep experiment.

    For each tolerance T:
      - premature_stops: count of converged trajectories that would have stopped
        prematurely (all evals > -T but n_neg > 0, yet later achieved n_neg==0)
      - failed_recovery: count of failed trajectories whose final evals are all > -T
      - spectrum_before_convergence: eigenvalue stats N steps before n_neg==0
    """
    results_per_tol: Dict[float, Dict[str, Any]] = {}

    # Pre-compute per-trajectory info
    converged_trajs = [t for t in all_trajs if _is_converged(t)]
    failed_trajs = [t for t in all_trajs if not _is_converged(t)]

    spectrum_before: Dict[int, List[List[float]]] = {}  # offset -> list of bottom spectra
    for look_back in [1, 10, 100, 500]:
        spectrum_before[look_back] = []

    for traj_data in converged_trajs:
        steps = traj_data.get("trajectory", [])
        zero_idx = _first_zero_step(steps)
        if zero_idx is None:
            continue
        for look_back in [1, 10, 100, 500]:
            sb_idx = zero_idx - look_back
            if 0 <= sb_idx < len(steps):
                bs = steps[sb_idx].get("bottom_spectrum", [])
                if bs:
                    spectrum_before[look_back].append(bs)

    for T in TOLERANCE_SWEEP:
        premature_count = 0
        premature_details: List[Dict[str, Any]] = []

        for traj_data in converged_trajs:
            steps = traj_data.get("trajectory", [])
            zero_idx = _first_zero_step(steps)
            if zero_idx is None:
                continue

            # Check every step before convergence: all evals > -T but n_neg > 0?
            for i, s in enumerate(steps):
                if i >= zero_idx:
                    break
                n_neg = s.get("n_neg_evals", 0)
                if n_neg <= 0:
                    continue
                min_eval = s.get("min_vib_eval", float("-inf"))
                if math.isnan(min_eval):
                    continue
                # "all evals > -T" means min_eval > -T
                if min_eval > -T:
                    premature_count += 1
                    premature_details.append({
                        "sample_id": traj_data.get("sample_id", "?"),
                        "step": i,
                        "min_eval": min_eval,
                        "n_neg": n_neg,
                        "converged_at": zero_idx,
                    })
                    break  # one premature event per trajectory is enough

        # Failed recovery: final evals all > -T
        failed_recovery = 0
        for traj_data in failed_trajs:
            steps = traj_data.get("trajectory", [])
            if not steps:
                continue
            last = steps[-1]
            min_eval = last.get("min_vib_eval", float("-inf"))
            if math.isnan(min_eval):
                continue
            if min_eval > -T:
                failed_recovery += 1

        results_per_tol[T] = {
            "tolerance": T,
            "premature_stop_count": premature_count,
            "premature_stop_fraction": (
                premature_count / max(len(converged_trajs), 1)
            ),
            "failed_recovery_count": failed_recovery,
            "failed_recovery_fraction": (
                failed_recovery / max(len(failed_trajs), 1)
            ),
            "n_converged": len(converged_trajs),
            "n_failed": len(failed_trajs),
        }

    return {
        "sweep": results_per_tol,
        "spectrum_before_convergence": {
            str(k): {
                "n_samples": len(v),
                "all_evals": [ev for spec in v for ev in spec],
            }
            for k, v in spectrum_before.items()
        },
    }


# ---------------------------------------------------------------------------
# A3. Physical Significance Analysis
# ---------------------------------------------------------------------------

def analyze_physical_significance(
    all_trajs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """A3: Physical significance analysis.

    - Estimate Hessian numerical noise floor from stagnant tail regions.
    - For ghost eigenvalues: energy contribution vs gradient energy scale.
    - Ghost-magnitude / noise-floor ratio.
    """
    noise_floor_estimates: List[float] = []
    ghost_records: List[Dict[str, Any]] = []

    for traj_data in all_trajs:
        steps = traj_data.get("trajectory", [])
        if len(steps) < 10:
            continue

        # --- Noise floor estimation ---
        # Look at stagnant tail: steps where actual_step_disp < 0.001
        stagnant_evals: List[List[float]] = []
        for i in range(1, len(steps)):
            disp = steps[i].get("actual_step_disp", float("inf"))
            if disp < 0.001:
                prev_bs = steps[i - 1].get("bottom_spectrum", [])
                curr_bs = steps[i].get("bottom_spectrum", [])
                if prev_bs and curr_bs:
                    min_len = min(len(prev_bs), len(curr_bs))
                    diffs = [
                        abs(curr_bs[j] - prev_bs[j])
                        for j in range(min_len)
                    ]
                    if diffs:
                        noise_floor_estimates.append(float(np.std(diffs)))

        # --- Ghost eigenvalue analysis ---
        for i, s in enumerate(steps):
            bs = s.get("bottom_spectrum", [])
            force_norm = s.get("force_norm", float("nan"))
            trust_radius = s.get("trust_radius", float("nan"))
            energy = s.get("energy", float("nan"))

            for ev in bs:
                if -1e-4 < ev < 0:
                    # Ghost eigenvalue
                    d = trust_radius if math.isfinite(trust_radius) else 0.1
                    energy_contribution = 0.5 * abs(ev) * d * d
                    gradient_energy_scale = (
                        force_norm * d if math.isfinite(force_norm) else float("nan")
                    )
                    ghost_records.append({
                        "sample_id": traj_data.get("sample_id", "?"),
                        "step": i,
                        "eigenvalue": ev,
                        "abs_eigenvalue": abs(ev),
                        "energy_contribution": energy_contribution,
                        "gradient_energy_scale": gradient_energy_scale,
                        "force_norm": force_norm,
                        "trust_radius": d,
                        "energy": energy,
                    })

    # Aggregate noise floor
    if noise_floor_estimates:
        noise_floor = float(np.median(noise_floor_estimates))
    else:
        noise_floor = float("nan")

    # Compute ratios for ghost records
    for g in ghost_records:
        if math.isfinite(noise_floor) and noise_floor > 0:
            g["ratio_to_noise_floor"] = g["abs_eigenvalue"] / noise_floor
        else:
            g["ratio_to_noise_floor"] = float("nan")

    # Summary stats
    if ghost_records:
        ratios = [g["ratio_to_noise_floor"] for g in ghost_records
                  if math.isfinite(g["ratio_to_noise_floor"])]
        n_below_noise = sum(1 for r in ratios if r < 1.0)
        n_above_noise = sum(1 for r in ratios if r >= 1.0)
    else:
        ratios = []
        n_below_noise = 0
        n_above_noise = 0

    return {
        "noise_floor": noise_floor,
        "n_noise_floor_samples": len(noise_floor_estimates),
        "n_ghost_records": len(ghost_records),
        "ghost_records": ghost_records,
        "n_below_noise_floor": n_below_noise,
        "n_above_noise_floor": n_above_noise,
        "ratio_stats": {
            "mean": float(np.nanmean(ratios)) if ratios else float("nan"),
            "median": float(np.nanmedian(ratios)) if ratios else float("nan"),
            "min": float(np.nanmin(ratios)) if ratios else float("nan"),
            "max": float(np.nanmax(ratios)) if ratios else float("nan"),
        },
    }


# ---------------------------------------------------------------------------
# A4. Stability / False Convergence Detection
# ---------------------------------------------------------------------------

def analyze_false_convergence(
    all_trajs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """A4: Stability / false convergence detection.

    Simulate tolerance application:
      For each tolerance T at first qualifying step S, track steps S+1..S+N.
      Do eigenvalues stay within tolerance or breach it?
    Also check mode rotation at qualifying steps.
    """
    results_per_tol: Dict[float, Dict[int, Dict[str, Any]]] = {}

    for T in TOLERANCE_SWEEP:
        if T == 0:
            continue  # T=0 is exact, no false convergence possible
        results_per_tol[T] = {}

        for N in LOOKAHEAD_WINDOWS:
            n_qualifying = 0
            n_breaches = 0
            n_stable = 0
            mode_rotation_at_qualifying: List[int] = []
            continuity_at_qualifying: List[float] = []

            for traj_data in all_trajs:
                steps = traj_data.get("trajectory", [])
                if len(steps) < 2:
                    continue

                # Find first qualifying step: n_neg > 0 AND all evals > -T
                qualifying_idx = None
                for i, s in enumerate(steps):
                    n_neg = s.get("n_neg_evals", 0)
                    min_eval = s.get("min_vib_eval", float("-inf"))
                    if math.isnan(min_eval):
                        continue
                    if n_neg > 0 and min_eval > -T:
                        qualifying_idx = i
                        break

                if qualifying_idx is None:
                    continue

                n_qualifying += 1

                # Record mode rotation at qualifying step
                ec = steps[qualifying_idx].get("eigenvec_continuity", {})
                n_rot = ec.get("n_mode_rotation_events", 0)
                mode_rotation_at_qualifying.append(n_rot)
                cont_min = ec.get("mode_continuity_min", float("nan"))
                if math.isfinite(cont_min):
                    continuity_at_qualifying.append(cont_min)

                # Lookahead: check steps qualifying_idx+1 .. qualifying_idx+N
                breach_found = False
                for j in range(qualifying_idx + 1,
                               min(qualifying_idx + N + 1, len(steps))):
                    later_min_eval = steps[j].get("min_vib_eval", float("nan"))
                    if math.isnan(later_min_eval):
                        continue
                    if later_min_eval <= -T:
                        breach_found = True
                        break

                if breach_found:
                    n_breaches += 1
                else:
                    n_stable += 1

            false_conv_rate = n_breaches / max(n_qualifying, 1)
            results_per_tol[T][N] = {
                "tolerance": T,
                "lookahead": N,
                "n_qualifying": n_qualifying,
                "n_breaches": n_breaches,
                "n_stable": n_stable,
                "false_convergence_rate": false_conv_rate,
                "mean_mode_rotations": (
                    float(np.mean(mode_rotation_at_qualifying))
                    if mode_rotation_at_qualifying else float("nan")
                ),
                "mean_continuity_min": (
                    float(np.mean(continuity_at_qualifying))
                    if continuity_at_qualifying else float("nan")
                ),
            }

    return {"per_tolerance_lookahead": results_per_tol}


# ---------------------------------------------------------------------------
# A5. Cascade Gap Analysis
# ---------------------------------------------------------------------------

def analyze_cascade_gap(
    all_trajs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """A5: Cascade gap analysis.

    Per failed sample:
      - Required tolerance T = |min_final_eval|
      - CDF of required tolerances
      - Sample x tolerance heatmap data (binary pass/fail)
      - Identify natural clusters in the CDF
    """
    failed_trajs = [t for t in all_trajs if not _is_converged(t)]

    per_sample: List[Dict[str, Any]] = []

    for traj_data in failed_trajs:
        steps = traj_data.get("trajectory", [])
        if not steps:
            continue
        last = steps[-1]
        min_eval = last.get("min_vib_eval", float("nan"))
        sample_id = traj_data.get("sample_id", "?")
        formula = traj_data.get("formula", "?")

        required_tol = abs(min_eval) if math.isfinite(min_eval) else float("nan")

        # Determine pass/fail for each sweep tolerance
        tol_pass: Dict[float, bool] = {}
        for T in TOLERANCE_SWEEP:
            if math.isfinite(min_eval):
                tol_pass[T] = min_eval > -T
            else:
                tol_pass[T] = False

        per_sample.append({
            "sample_id": sample_id,
            "formula": formula,
            "min_final_eval": min_eval,
            "required_tolerance": required_tol,
            "n_final_neg": last.get("n_neg_evals", -1),
            "final_force_norm": last.get("force_norm", float("nan")),
            "final_energy": last.get("energy", float("nan")),
            "n_steps": len(steps),
            "tolerance_pass": tol_pass,
        })

    # Sort by required tolerance for CDF
    per_sample.sort(key=lambda x: x["required_tolerance"]
                    if math.isfinite(x["required_tolerance"]) else float("inf"))

    # CDF data
    valid_tols = [s["required_tolerance"] for s in per_sample
                  if math.isfinite(s["required_tolerance"])]

    # Gap detection: largest gap between consecutive sorted tolerances
    gaps: List[Dict[str, Any]] = []
    if len(valid_tols) >= 2:
        sorted_tols = sorted(valid_tols)
        for i in range(len(sorted_tols) - 1):
            gap_size = sorted_tols[i + 1] - sorted_tols[i]
            gaps.append({
                "lower": sorted_tols[i],
                "upper": sorted_tols[i + 1],
                "gap_size": gap_size,
                "gap_log_ratio": (
                    math.log10(sorted_tols[i + 1] / sorted_tols[i])
                    if sorted_tols[i] > 0 else float("nan")
                ),
            })
        gaps.sort(key=lambda x: x["gap_size"], reverse=True)

    # Per-tolerance summary: how many flip
    tol_summary: Dict[float, Dict[str, int]] = {}
    for T in TOLERANCE_SWEEP:
        n_pass = sum(1 for s in per_sample if s["tolerance_pass"].get(T, False))
        tol_summary[T] = {
            "n_pass": n_pass,
            "n_fail": len(per_sample) - n_pass,
            "n_total": len(per_sample),
        }

    return {
        "per_sample": per_sample,
        "valid_required_tolerances": valid_tols,
        "top_gaps": gaps[:10],
        "tolerance_summary": tol_summary,
    }


# ---------------------------------------------------------------------------
# A6. Eigenvalue-Force-Energy Consistency Check
# ---------------------------------------------------------------------------

def analyze_force_eigenvalue_consistency(
    all_trajs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """A6: Eigenvalue-force-energy consistency check.

    For all final states:
      - |min_eigenvalue| vs force_norm
      - For trajectories qualifying under tolerance T: force norm stats
      - Energy curvature check for ghost-mode samples
    """
    final_state_records: List[Dict[str, Any]] = []

    for traj_data in all_trajs:
        steps = traj_data.get("trajectory", [])
        if not steps:
            continue
        last = steps[-1]
        converged = _is_converged(traj_data)

        min_eval = last.get("min_vib_eval", float("nan"))
        force_norm = last.get("force_norm", float("nan"))
        energy = last.get("energy", float("nan"))
        n_neg = last.get("n_neg_evals", 0)

        # Energy curvature: check last few steps for flatness
        energy_flat = False
        if len(steps) >= 10:
            tail_energies = [s.get("energy", float("nan")) for s in steps[-10:]]
            valid_tail = [e for e in tail_energies if math.isfinite(e)]
            if len(valid_tail) >= 5:
                energy_range = max(valid_tail) - min(valid_tail)
                energy_flat = energy_range < 1e-8

        # Determine which tolerances this sample would pass under
        passes_tolerance: Dict[float, bool] = {}
        for T in TOLERANCE_SWEEP:
            if math.isfinite(min_eval) and n_neg > 0:
                passes_tolerance[T] = min_eval > -T
            elif n_neg == 0:
                passes_tolerance[T] = True
            else:
                passes_tolerance[T] = False

        final_state_records.append({
            "sample_id": traj_data.get("sample_id", "?"),
            "converged": converged,
            "min_eval": min_eval,
            "abs_min_eval": abs(min_eval) if math.isfinite(min_eval) else float("nan"),
            "force_norm": force_norm,
            "energy": energy,
            "n_neg": n_neg,
            "n_steps": len(steps),
            "energy_flat": energy_flat,
            "passes_tolerance": passes_tolerance,
        })

    # Per-tolerance force norm stats for qualifying samples
    tol_force_stats: Dict[float, Dict[str, float]] = {}
    for T in TOLERANCE_SWEEP:
        qualifying_forces = [
            r["force_norm"]
            for r in final_state_records
            if r["passes_tolerance"].get(T, False) and not r["converged"]
            and math.isfinite(r["force_norm"])
        ]
        if qualifying_forces:
            arr = np.array(qualifying_forces)
            tol_force_stats[T] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "max": float(np.max(arr)),
                "min": float(np.min(arr)),
                "n": len(qualifying_forces),
                "n_above_1e-4": int(np.sum(arr > 1e-4)),
                "n_above_1e-3": int(np.sum(arr > 1e-3)),
                "n_above_1e-2": int(np.sum(arr > 1e-2)),
            }
        else:
            tol_force_stats[T] = {
                "mean": float("nan"),
                "median": float("nan"),
                "max": float("nan"),
                "min": float("nan"),
                "n": 0,
                "n_above_1e-4": 0,
                "n_above_1e-3": 0,
                "n_above_1e-2": 0,
            }

    return {
        "final_state_records": final_state_records,
        "tolerance_force_stats": tol_force_stats,
    }


# ---------------------------------------------------------------------------
# CSV Writers
# ---------------------------------------------------------------------------

def write_eigenvalue_tolerance_sweep_csv(
    sweep_results: Dict[str, Any], output_dir: Path
) -> None:
    """Write eigenvalue_tolerance_sweep.csv."""
    path = output_dir / "eigenvalue_tolerance_sweep.csv"
    sweep = sweep_results["sweep"]
    rows = []
    for T in TOLERANCE_SWEEP:
        r = sweep[T]
        rows.append(r)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {path}")


def write_per_sample_required_tolerance_csv(
    cascade_results: Dict[str, Any], output_dir: Path
) -> None:
    """Write per_sample_required_tolerance.csv."""
    path = output_dir / "per_sample_required_tolerance.csv"
    samples = cascade_results["per_sample"]
    if not samples:
        return
    fieldnames = [
        "sample_id", "formula", "min_final_eval", "required_tolerance",
        "n_final_neg", "final_force_norm", "final_energy", "n_steps",
    ]
    # Add tolerance pass columns
    for T in TOLERANCE_SWEEP:
        fieldnames.append(f"pass_T={T:.0e}")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            row = {k: s[k] for k in fieldnames[:8]}
            for T in TOLERANCE_SWEEP:
                row[f"pass_T={T:.0e}"] = int(s["tolerance_pass"].get(T, False))
            writer.writerow(row)
    print(f"  Wrote {path}")


def write_band_evolution_summary_csv(
    band_results: Dict[str, Any], output_dir: Path
) -> None:
    """Write band_evolution_summary.csv."""
    path = output_dir / "band_evolution_summary.csv"
    milestone_means = band_results["milestone_means"]
    rows = []
    for ms in sorted(milestone_means.keys()):
        for label in ["converged", "failed"]:
            row = {"milestone_step": ms, "status": label}
            for bk in BAND_KEYS:
                row[bk] = milestone_means[ms].get(label, {}).get(bk, 0.0)
            rows.append(row)
    if not rows:
        return
    fieldnames = ["milestone_step", "status"] + BAND_KEYS
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {path}")


def write_ghost_mode_analysis_csv(
    phys_results: Dict[str, Any], output_dir: Path
) -> None:
    """Write ghost_mode_analysis.csv."""
    path = output_dir / "ghost_mode_analysis.csv"
    records = phys_results["ghost_records"]
    if not records:
        with open(path, "w") as f:
            f.write("# No ghost mode records found\n")
        print(f"  Wrote {path} (empty)")
        return
    fieldnames = [
        "sample_id", "step", "eigenvalue", "abs_eigenvalue",
        "energy_contribution", "gradient_energy_scale",
        "force_norm", "trust_radius", "ratio_to_noise_floor",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for g in records:
            row = {k: g.get(k, float("nan")) for k in fieldnames}
            writer.writerow(row)
    print(f"  Wrote {path}")


def write_false_convergence_csv(
    fc_results: Dict[str, Any], output_dir: Path
) -> None:
    """Write false_convergence_analysis.csv."""
    path = output_dir / "false_convergence_analysis.csv"
    per_tl = fc_results["per_tolerance_lookahead"]
    rows = []
    for T in TOLERANCE_SWEEP:
        if T == 0:
            continue
        for N in LOOKAHEAD_WINDOWS:
            r = per_tl.get(T, {}).get(N, {})
            if r:
                rows.append(r)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {path}")


def write_force_eigenvalue_consistency_csv(
    fe_results: Dict[str, Any], output_dir: Path
) -> None:
    """Write force_eigenvalue_consistency.csv."""
    path = output_dir / "force_eigenvalue_consistency.csv"
    records = fe_results["final_state_records"]
    if not records:
        return
    fieldnames = [
        "sample_id", "converged", "min_eval", "abs_min_eval",
        "force_norm", "energy", "n_neg", "n_steps", "energy_flat",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {k: r[k] for k in fieldnames}
            writer.writerow(row)
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# JSON summary writer
# ---------------------------------------------------------------------------

def write_justification_json(
    band_results: Dict[str, Any],
    sweep_results: Dict[str, Any],
    phys_results: Dict[str, Any],
    fc_results: Dict[str, Any],
    cascade_results: Dict[str, Any],
    fe_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Write eigenvalue_justification.json — a comprehensive summary."""

    def _sanitize(obj: Any) -> Any:
        """Make an object JSON-serializable (handle NaN, Inf, numpy types)."""
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if math.isnan(v):
                return None
            if math.isinf(v):
                return str(v)
            return v
        if isinstance(obj, float):
            if math.isnan(obj):
                return None
            if math.isinf(obj):
                return str(obj)
            return obj
        if isinstance(obj, np.ndarray):
            return _sanitize(obj.tolist())
        if isinstance(obj, bool):
            return obj
        return obj

    summary = {
        "A1_band_evolution": {
            "ghost_onset_converged_count": len(band_results["ghost_onset_converged"]),
            "ghost_onset_failed_count": len(band_results["ghost_onset_failed"]),
            "ghost_onset_converged_mean_step": (
                float(np.mean(band_results["ghost_onset_converged"]))
                if band_results["ghost_onset_converged"] else None
            ),
            "ghost_onset_failed_mean_step": (
                float(np.mean(band_results["ghost_onset_failed"]))
                if band_results["ghost_onset_failed"] else None
            ),
        },
        "A2_tolerance_sweep": {
            str(T): {
                "premature_stop_fraction": sweep_results["sweep"][T]["premature_stop_fraction"],
                "failed_recovery_fraction": sweep_results["sweep"][T]["failed_recovery_fraction"],
            }
            for T in TOLERANCE_SWEEP
        },
        "A3_physical_significance": {
            "noise_floor": phys_results["noise_floor"],
            "n_ghost_records": phys_results["n_ghost_records"],
            "n_below_noise_floor": phys_results["n_below_noise_floor"],
            "n_above_noise_floor": phys_results["n_above_noise_floor"],
            "ratio_stats": phys_results["ratio_stats"],
        },
        "A4_false_convergence": {
            str(T): {
                str(N): fc_results["per_tolerance_lookahead"]
                .get(T, {})
                .get(N, {})
                .get("false_convergence_rate", None)
                for N in LOOKAHEAD_WINDOWS
            }
            for T in TOLERANCE_SWEEP
            if T > 0
        },
        "A5_cascade_gap": {
            "n_failed_samples": len(cascade_results["per_sample"]),
            "top_gaps": cascade_results["top_gaps"][:5],
            "tolerance_summary": cascade_results["tolerance_summary"],
        },
        "A6_force_eigenvalue_consistency": {
            str(T): fe_results["tolerance_force_stats"].get(T, {})
            for T in TOLERANCE_SWEEP
        },
    }

    path = output_dir / "eigenvalue_justification.json"
    with open(path, "w") as f:
        json.dump(_sanitize(summary), f, indent=2)
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_eigenvalue_distribution_final(
    all_trajs: List[Dict[str, Any]], output_dir: Path
) -> None:
    """Plot 1: Histogram of final-step eigenvalues, converged vs failed."""
    converged_evals: List[float] = []
    failed_evals: List[float] = []

    for traj_data in all_trajs:
        steps = traj_data.get("trajectory", [])
        if not steps:
            continue
        last = steps[-1]
        bs = last.get("bottom_spectrum", [])
        converged = _is_converged(traj_data)
        for ev in bs:
            if math.isfinite(ev):
                if converged:
                    converged_evals.append(ev)
                else:
                    failed_evals.append(ev)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use abs for log-scale, distinguish sign by color/hatch
    all_evals = converged_evals + failed_evals
    if not all_evals:
        ax.text(0.5, 0.5, "No eigenvalue data", ha="center", va="center",
                transform=ax.transAxes)
    else:
        # Negative eigenvalues only for clarity
        neg_conv = [abs(e) for e in converged_evals if e < 0]
        neg_fail = [abs(e) for e in failed_evals if e < 0]

        bins = np.logspace(-8, 0, 60)
        if neg_conv:
            ax.hist(neg_conv, bins=bins, alpha=0.6, color="green",
                    label=f"Converged (n={len(neg_conv)})", edgecolor="darkgreen")
        if neg_fail:
            ax.hist(neg_fail, bins=bins, alpha=0.6, color="red",
                    label=f"Failed (n={len(neg_fail)})", edgecolor="darkred")

        ax.set_xscale("log")
        ax.set_xlabel("|Eigenvalue| (negative eigenvalues only)")
        ax.set_ylabel("Count")

    ax.set_title("Distribution of Final-Step Negative Eigenvalues")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "eigenvalue_distribution_final.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {output_dir / 'eigenvalue_distribution_final.png'}")


def plot_min_eval_cdf_failed(
    cascade_results: Dict[str, Any], output_dir: Path
) -> None:
    """Plot 2: CDF of |min eigenvalue| across failed trajectories."""
    valid_tols = sorted(cascade_results["valid_required_tolerances"])
    if not valid_tols:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No failed trajectories", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("CDF of |min eigenvalue| — Failed Trajectories")
        fig.tight_layout()
        fig.savefig(output_dir / "min_eval_cdf_failed.png", dpi=150)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(1, len(valid_tols) + 1) / len(valid_tols)
    ax.plot(valid_tols, y, "b-", linewidth=2)

    # Vertical lines at tolerance thresholds
    colors_cycle = plt.cm.tab10(np.linspace(0, 1, len(TOLERANCE_SWEEP)))
    for i, T in enumerate(TOLERANCE_SWEEP):
        if T == 0:
            continue
        ax.axvline(T, color=colors_cycle[i], linestyle="--", alpha=0.7,
                   label=f"T={T:.0e}")

    ax.set_xscale("log")
    ax.set_xlabel("|min eigenvalue| (required tolerance)")
    ax.set_ylabel("CDF (fraction of failed samples)")
    ax.set_title("CDF of Required Tolerance for Failed Trajectories")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "min_eval_cdf_failed.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {output_dir / 'min_eval_cdf_failed.png'}")


def plot_convergence_rate_vs_tolerance(
    sweep_results: Dict[str, Any],
    fc_results: Dict[str, Any],
    n_total: int,
    n_converged_exact: int,
    output_dir: Path,
) -> None:
    """Plot 3: Hypothetical convergence rate vs tolerance, with false-convergence overlay."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    tols = []
    hypo_conv_rates = []
    false_conv_rates_la100 = []

    for T in TOLERANCE_SWEEP:
        if T == 0:
            continue
        tols.append(T)
        sweep = sweep_results["sweep"][T]
        # Hypothetical convergence rate: exact converged + failed that would pass
        hypo_conv = n_converged_exact + sweep["failed_recovery_count"]
        hypo_conv_rates.append(hypo_conv / max(n_total, 1))

        # False convergence rate at lookahead 100
        fc_rate = (
            fc_results["per_tolerance_lookahead"]
            .get(T, {})
            .get(100, {})
            .get("false_convergence_rate", 0.0)
        )
        false_conv_rates_la100.append(fc_rate)

    if tols:
        ax1.plot(tols, hypo_conv_rates, "b-o", linewidth=2, label="Hypothetical convergence rate")
        ax1.axhline(n_converged_exact / max(n_total, 1), color="blue",
                    linestyle=":", alpha=0.5, label="Exact convergence rate")
        ax1.set_xlabel("Tolerance threshold")
        ax1.set_ylabel("Convergence rate", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.set_xscale("log")
        ax1.set_ylim(0, 1.05)

        ax2.plot(tols, false_conv_rates_la100, "r-s", linewidth=2,
                 label="False convergence rate (LA=100)")
        ax2.set_ylabel("False convergence rate", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.set_ylim(0, 1.05)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")

    ax1.set_title("Hypothetical Convergence Rate vs Tolerance Threshold")
    fig.tight_layout()
    fig.savefig(output_dir / "convergence_rate_vs_tolerance.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {output_dir / 'convergence_rate_vs_tolerance.png'}")


def plot_eigenvalue_timeline_per_sample(
    all_trajs: List[Dict[str, Any]], output_dir: Path, top_k: int = 10
) -> None:
    """Plot 4: Per-sample heatmap of eigenvalue evolution (top K hardest)."""
    # Select top-K failed trajectories by number of steps (longest / hardest)
    failed = [t for t in all_trajs if not _is_converged(t)]
    failed.sort(key=lambda t: len(t.get("trajectory", [])), reverse=True)
    selected = failed[:top_k]

    for traj_data in selected:
        steps = traj_data.get("trajectory", [])
        sample_id = traj_data.get("sample_id", "unknown")
        if not steps:
            continue

        # Build matrix: step x eigenvalue index from bottom_spectrum
        max_spec_len = max(len(s.get("bottom_spectrum", [])) for s in steps)
        if max_spec_len == 0:
            continue

        mat = np.full((len(steps), max_spec_len), float("nan"))
        for i, s in enumerate(steps):
            bs = s.get("bottom_spectrum", [])
            for j, val in enumerate(bs):
                mat[i, j] = val

        fig, ax = plt.subplots(figsize=(12, 6))
        # Signed magnitude color map: red for negative, blue for positive
        vmax = max(abs(np.nanmin(mat)) if np.any(np.isfinite(mat)) else 0.01,
                   abs(np.nanmax(mat)) if np.any(np.isfinite(mat)) else 0.01)
        vmax = max(vmax, 1e-6)
        im = ax.imshow(
            mat.T,
            aspect="auto",
            cmap="RdBu",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
            origin="lower",
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Eigenvalue Index (sorted)")
        ax.set_title(f"Eigenvalue Timeline — {sample_id}")
        plt.colorbar(im, ax=ax, label="Eigenvalue")
        fig.tight_layout()
        safe_id = sample_id.replace("/", "_").replace(" ", "_")
        fname = output_dir / f"eigenvalue_timeline_{safe_id}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Wrote {fname}")


def plot_band_evolution_converged_vs_failed(
    band_results: Dict[str, Any], output_dir: Path
) -> None:
    """Plot 5: Stacked area: band populations over time, two panels."""
    milestone_means = band_results["milestone_means"]
    milestones = sorted(milestone_means.keys())

    if not milestones:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_dir / "band_evolution_converged_vs_failed.png", dpi=150)
        plt.close(fig)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, label, title in [
        (ax1, "converged", "Converged Trajectories"),
        (ax2, "failed", "Failed Trajectories"),
    ]:
        band_data = {bk: [] for bk in BAND_KEYS}
        x_vals = []
        for ms in milestones:
            x_vals.append(ms)
            for bk in BAND_KEYS:
                band_data[bk].append(
                    milestone_means[ms].get(label, {}).get(bk, 0.0)
                )

        # Stacked area plot
        bottoms = np.zeros(len(x_vals))
        colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(BAND_KEYS)))
        for i, bk in enumerate(BAND_KEYS):
            vals = np.array(band_data[bk])
            ax.bar(range(len(x_vals)), vals, bottom=bottoms,
                   label=BAND_LABELS[i], color=colors[i], width=0.8)
            bottoms += vals

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([str(m) for m in x_vals], rotation=45)
        ax.set_xlabel("Milestone Step")
        ax.set_ylabel("Mean Band Population")
        ax.set_title(title)

    ax1.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 1))
    fig.suptitle("Eigenvalue Band Evolution: Converged vs Failed", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "band_evolution_converged_vs_failed.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {output_dir / 'band_evolution_converged_vs_failed.png'}")


def plot_sample_tolerance_heatmap(
    cascade_results: Dict[str, Any], output_dir: Path
) -> None:
    """Plot 6: Binary heatmap: sample x tolerance -> pass/fail."""
    per_sample = cascade_results["per_sample"]
    if not per_sample:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No failed samples", ha="center", va="center",
                transform=ax.transAxes)
        fig.savefig(output_dir / "sample_tolerance_heatmap.png", dpi=150)
        plt.close(fig)
        return

    # Sort by required tolerance
    sorted_samples = sorted(
        per_sample,
        key=lambda x: x["required_tolerance"]
        if math.isfinite(x["required_tolerance"]) else float("inf"),
    )

    tols_nonzero = [T for T in TOLERANCE_SWEEP if T > 0]
    mat = np.zeros((len(sorted_samples), len(tols_nonzero)))
    sample_labels = []
    for i, s in enumerate(sorted_samples):
        sample_labels.append(s["sample_id"])
        for j, T in enumerate(tols_nonzero):
            mat[i, j] = 1 if s["tolerance_pass"].get(T, False) else 0

    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_samples) * 0.3)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(range(len(tols_nonzero)))
    ax.set_xticklabels([f"{T:.0e}" for T in tols_nonzero], rotation=45)
    ax.set_xlabel("Tolerance Threshold")

    if len(sorted_samples) <= 50:
        ax.set_yticks(range(len(sorted_samples)))
        ax.set_yticklabels(sample_labels, fontsize=6)
    else:
        ax.set_ylabel(f"Samples (n={len(sorted_samples)}, sorted by required tol)")

    ax.set_title("Sample x Tolerance Heatmap (green=pass, red=fail)")
    plt.colorbar(im, ax=ax, label="Pass (1) / Fail (0)", ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(output_dir / "sample_tolerance_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {output_dir / 'sample_tolerance_heatmap.png'}")


def plot_ghost_onset_histogram(
    band_results: Dict[str, Any], output_dir: Path
) -> None:
    """Plot 7: When ghost eigenvalues first appear, converged vs failed."""
    conv = band_results["ghost_onset_converged"]
    fail = band_results["ghost_onset_failed"]

    fig, ax = plt.subplots(figsize=(10, 6))

    if not conv and not fail:
        ax.text(0.5, 0.5, "No ghost onset events detected",
                ha="center", va="center", transform=ax.transAxes)
    else:
        all_onsets = conv + fail
        if all_onsets:
            max_val = max(all_onsets)
            bins = np.linspace(0, max_val + 1, min(50, max_val + 2))
        else:
            bins = 20
        if conv:
            ax.hist(conv, bins=bins, alpha=0.6, color="green",
                    label=f"Converged (n={len(conv)})", edgecolor="darkgreen")
        if fail:
            ax.hist(fail, bins=bins, alpha=0.6, color="red",
                    label=f"Failed (n={len(fail)})", edgecolor="darkred")

    ax.set_xlabel("Step of Ghost Onset (first step with evals in [-1e-4,0) only)")
    ax.set_ylabel("Count")
    ax.set_title("Ghost Eigenvalue Onset: Converged vs Failed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ghost_onset_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {output_dir / 'ghost_onset_histogram.png'}")


def plot_false_convergence_vs_tolerance(
    fc_results: Dict[str, Any], output_dir: Path
) -> None:
    """Plot 8: False convergence rate vs tolerance, one line per lookahead window."""
    per_tl = fc_results["per_tolerance_lookahead"]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(LOOKAHEAD_WINDOWS)))

    for idx, N in enumerate(LOOKAHEAD_WINDOWS):
        tols_plot = []
        rates_plot = []
        for T in TOLERANCE_SWEEP:
            if T == 0:
                continue
            entry = per_tl.get(T, {}).get(N, {})
            if entry:
                tols_plot.append(T)
                rates_plot.append(entry.get("false_convergence_rate", 0.0))
        if tols_plot:
            ax.plot(tols_plot, rates_plot, "-o", color=colors[idx],
                    linewidth=2, label=f"LA={N}")

    ax.set_xscale("log")
    ax.set_xlabel("Tolerance Threshold")
    ax.set_ylabel("False Convergence Rate")
    ax.set_title("False Convergence Rate vs Tolerance (by Lookahead Window)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "false_convergence_vs_tolerance.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {output_dir / 'false_convergence_vs_tolerance.png'}")


def plot_force_vs_min_eigenvalue(
    fe_results: Dict[str, Any], output_dir: Path
) -> None:
    """Plot 9: Scatter: force_norm vs |min_eigenvalue|, colored by convergence."""
    records = fe_results["final_state_records"]

    fig, ax = plt.subplots(figsize=(10, 8))

    conv_x, conv_y = [], []
    fail_x, fail_y = [], []

    for r in records:
        abs_me = r["abs_min_eval"]
        fn = r["force_norm"]
        if not math.isfinite(abs_me) or not math.isfinite(fn):
            continue
        if abs_me <= 0:
            abs_me = 1e-12  # avoid log(0)
        if fn <= 0:
            fn = 1e-12
        if r["converged"]:
            conv_x.append(abs_me)
            conv_y.append(fn)
        else:
            fail_x.append(abs_me)
            fail_y.append(fn)

    if conv_x:
        ax.scatter(conv_x, conv_y, c="green", alpha=0.5, s=20,
                   label=f"Converged (n={len(conv_x)})", edgecolors="darkgreen",
                   linewidths=0.3)
    if fail_x:
        ax.scatter(fail_x, fail_y, c="red", alpha=0.5, s=20,
                   label=f"Failed (n={len(fail_x)})", edgecolors="darkred",
                   linewidths=0.3)

    # Reference lines
    for T in [1e-4, 1e-3, 1e-2]:
        ax.axvline(T, color="gray", linestyle="--", alpha=0.4,
                   label=f"|eval|={T:.0e}")
    ax.axhline(1e-4, color="blue", linestyle=":", alpha=0.4,
               label="force=1e-4 (convergence)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|min eigenvalue|")
    ax.set_ylabel("Force Norm")
    ax.set_title("Force Norm vs |Min Eigenvalue| — All Final States")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "force_vs_min_eigenvalue.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {output_dir / 'force_vs_min_eigenvalue.png'}")


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(
    band_results: Dict[str, Any],
    sweep_results: Dict[str, Any],
    phys_results: Dict[str, Any],
    fc_results: Dict[str, Any],
    cascade_results: Dict[str, Any],
    fe_results: Dict[str, Any],
    n_total: int,
    n_converged: int,
    n_failed: int,
) -> None:
    """Print a structured text report to stdout."""
    sep = "=" * 80
    sub = "-" * 60

    print()
    print(sep)
    print("  EIGENVALUE TOLERANCE JUSTIFICATION REPORT")
    print(sep)
    print(f"  Total trajectories:   {n_total}")
    print(f"  Converged (exact):    {n_converged}")
    print(f"  Failed:               {n_failed}")
    print(f"  Exact convergence rate: {n_converged / max(n_total, 1):.4f}")
    print()

    # --- A1 ---
    print(sep)
    print("  A1. BAND EVOLUTION STATISTICS")
    print(sep)
    print()
    print("  Ghost onset (first step with evals in [-1e-4,0) and NONE below -1e-4):")
    go_c = band_results["ghost_onset_converged"]
    go_f = band_results["ghost_onset_failed"]
    print(f"    Converged trajectories with ghost onset: {len(go_c)}")
    if go_c:
        print(f"      Mean onset step:   {np.mean(go_c):.1f}")
        print(f"      Median onset step: {np.median(go_c):.1f}")
        print(f"      Min / Max:         {min(go_c)} / {max(go_c)}")
    print(f"    Failed trajectories with ghost onset: {len(go_f)}")
    if go_f:
        print(f"      Mean onset step:   {np.mean(go_f):.1f}")
        print(f"      Median onset step: {np.median(go_f):.1f}")
        print(f"      Min / Max:         {min(go_f)} / {max(go_f)}")

    print()
    print("  Milestone band populations (mean):")
    mm = band_results["milestone_means"]
    for ms in sorted(mm.keys()):
        print(f"    Step {ms}:")
        for label in ["converged", "failed"]:
            vals = mm[ms].get(label, {})
            if not vals:
                continue
            neg_total = sum(vals.get(bk, 0) for bk in BAND_KEYS[:4])
            ghost_total = vals.get("n_eval_neg1e-4_to_0", 0)
            print(f"      {label:>10s}: neg_total={neg_total:.2f}, ghost_band={ghost_total:.2f}")
    print()

    # --- A2 ---
    print(sep)
    print("  A2. TOLERANCE SWEEP EXPERIMENT")
    print(sep)
    print()
    print("  Tolerance  |  Premature Stops  |  Frac  |  Failed Recovery  |  Frac")
    print(f"  {sub}")
    sweep = sweep_results["sweep"]
    for T in TOLERANCE_SWEEP:
        s = sweep[T]
        print(
            f"  {T:>9.0e}  |  {s['premature_stop_count']:>15d}  "
            f"|  {s['premature_stop_fraction']:.4f}  "
            f"|  {s['failed_recovery_count']:>15d}  "
            f"|  {s['failed_recovery_fraction']:.4f}"
        )

    print()
    print("  Spectrum before convergence (steps before n_neg==0):")
    spec_before = sweep_results["spectrum_before_convergence"]
    for offset_str in ["1", "10", "100", "500"]:
        info = spec_before.get(offset_str, {})
        n_samples = info.get("n_samples", 0)
        all_evals = info.get("all_evals", [])
        neg_evals = [e for e in all_evals if e < 0]
        if neg_evals:
            print(
                f"    {offset_str:>3s} steps before: {n_samples} samples, "
                f"{len(neg_evals)} neg evals, "
                f"min={min(neg_evals):.6e}, median={float(np.median(neg_evals)):.6e}"
            )
        else:
            print(f"    {offset_str:>3s} steps before: {n_samples} samples, 0 neg evals")
    print()

    # --- A3 ---
    print(sep)
    print("  A3. PHYSICAL SIGNIFICANCE ANALYSIS")
    print(sep)
    print()
    nf = phys_results["noise_floor"]
    print(f"  Estimated Hessian numerical noise floor: {nf:.6e}")
    print(f"  (from {phys_results['n_noise_floor_samples']} stagnant-tail samples)")
    print()
    print(f"  Ghost eigenvalue records: {phys_results['n_ghost_records']}")
    print(f"  Below noise floor (|eval| / noise < 1.0): {phys_results['n_below_noise_floor']}")
    print(f"  Above noise floor:                        {phys_results['n_above_noise_floor']}")
    rs = phys_results["ratio_stats"]
    print(f"  |eval|/noise ratio: mean={rs['mean']:.4f}, "
          f"median={rs['median']:.4f}, min={rs['min']:.6e}, max={rs['max']:.4f}")
    print()

    # --- A4 ---
    print(sep)
    print("  A4. STABILITY / FALSE CONVERGENCE DETECTION")
    print(sep)
    print()
    print("  False convergence rate by tolerance and lookahead window:")
    print()
    header = "  Tolerance  |  " + "  |  ".join(f"LA={N:>4d}" for N in LOOKAHEAD_WINDOWS)
    print(header)
    print(f"  {sub}")
    per_tl = fc_results["per_tolerance_lookahead"]
    for T in TOLERANCE_SWEEP:
        if T == 0:
            continue
        parts = [f"  {T:>9.0e}  "]
        for N in LOOKAHEAD_WINDOWS:
            entry = per_tl.get(T, {}).get(N, {})
            rate = entry.get("false_convergence_rate", float("nan"))
            n_q = entry.get("n_qualifying", 0)
            if math.isfinite(rate):
                parts.append(f"{rate:.3f}({n_q:>3d})")
            else:
                parts.append(f"  N/A      ")
        print("  |  ".join(parts))

    # Mode rotation summary
    print()
    print("  Mode rotation at qualifying steps (mean across qualifying events):")
    for T in [1e-4, 1e-3, 1e-2]:
        entry = per_tl.get(T, {}).get(100, {})
        rot = entry.get("mean_mode_rotations", float("nan"))
        cont = entry.get("mean_continuity_min", float("nan"))
        n_q = entry.get("n_qualifying", 0)
        if math.isfinite(rot):
            print(f"    T={T:.0e}: mean rotations={rot:.2f}, "
                  f"mean continuity_min={cont:.4f} (n={n_q})")
    print()

    # --- A5 ---
    print(sep)
    print("  A5. CASCADE GAP ANALYSIS")
    print(sep)
    print()
    per_sample = cascade_results["per_sample"]
    valid_tols = cascade_results["valid_required_tolerances"]
    print(f"  Total failed samples: {len(per_sample)}")
    if valid_tols:
        print(f"  Required tolerance range: [{min(valid_tols):.6e}, {max(valid_tols):.6e}]")
        print(f"  Median required tolerance: {float(np.median(valid_tols)):.6e}")

    print()
    print("  Top gaps in required-tolerance CDF:")
    for i, g in enumerate(cascade_results["top_gaps"][:5]):
        print(
            f"    Gap {i + 1}: [{g['lower']:.6e}, {g['upper']:.6e}] "
            f"size={g['gap_size']:.6e} log_ratio={g['gap_log_ratio']:.2f}"
        )

    print()
    print("  Per-tolerance pass/fail counts:")
    ts = cascade_results["tolerance_summary"]
    for T in TOLERANCE_SWEEP:
        if T == 0:
            continue
        s = ts.get(T, {})
        n_p = s.get("n_pass", 0)
        n_f = s.get("n_fail", 0)
        n_t = s.get("n_total", 0)
        print(f"    T={T:.0e}: pass={n_p}, fail={n_f}, total={n_t}")
    print()

    # --- A6 ---
    print(sep)
    print("  A6. EIGENVALUE-FORCE-ENERGY CONSISTENCY CHECK")
    print(sep)
    print()
    print("  Force norm stats for failed samples that would pass under tolerance T:")
    tfs = fe_results["tolerance_force_stats"]
    for T in TOLERANCE_SWEEP:
        if T == 0:
            continue
        s = tfs.get(T, {})
        n = s.get("n", 0)
        if n == 0:
            print(f"    T={T:.0e}: no qualifying samples")
        else:
            print(
                f"    T={T:.0e}: n={n}, mean_force={s['mean']:.6e}, "
                f"median_force={s['median']:.6e}, max_force={s['max']:.6e}, "
                f"n_force>1e-4={s['n_above_1e-4']}, n_force>1e-3={s['n_above_1e-3']}"
            )
    print()

    # --- FINDINGS ---
    print(sep)
    print("  FINDINGS")
    print(sep)
    print()

    # Evidence FOR tolerance
    print("  Evidence FOR eigenvalue tolerance filtering:")
    print()

    # 1. Ghost modes below noise floor
    nb = phys_results["n_below_noise_floor"]
    na = phys_results["n_above_noise_floor"]
    total_ghost = nb + na
    if total_ghost > 0:
        frac_below = nb / total_ghost
        print(f"    - {nb}/{total_ghost} ({frac_below:.1%}) ghost eigenvalue instances "
              f"are below the Hessian noise floor")
    else:
        print("    - No ghost eigenvalue instances found for noise floor comparison")

    # 2. Failed recovery potential
    for T in [1e-4, 5e-4, 1e-3]:
        s = sweep_results["sweep"][T]
        if s["failed_recovery_count"] > 0:
            print(f"    - At T={T:.0e}: {s['failed_recovery_count']} failed "
                  f"trajectories ({s['failed_recovery_fraction']:.1%}) "
                  f"would be reclassified as converged")

    # 3. Ghost onset in converged
    if go_c:
        print(f"    - {len(go_c)} converged trajectories passed through a "
              f"ghost-only phase (evals in [-1e-4,0) with none below)")
    print()

    # Evidence AGAINST tolerance
    print("  Evidence AGAINST eigenvalue tolerance filtering:")
    print()

    # 1. Premature stops
    for T in [1e-4, 5e-4, 1e-3]:
        s = sweep_results["sweep"][T]
        if s["premature_stop_count"] > 0:
            print(f"    - At T={T:.0e}: {s['premature_stop_count']} converged "
                  f"trajectories ({s['premature_stop_fraction']:.1%}) "
                  f"would have been stopped prematurely")

    # 2. False convergence
    for T in [1e-4, 5e-4, 1e-3]:
        entry = per_tl.get(T, {}).get(100, {})
        rate = entry.get("false_convergence_rate", 0)
        n_q = entry.get("n_qualifying", 0)
        if n_q > 0 and rate > 0:
            print(f"    - At T={T:.0e}: false convergence rate = {rate:.1%} "
                  f"(LA=100, n={n_q})")

    # 3. Large forces at qualifying geometries
    for T in [1e-4, 5e-4, 1e-3]:
        s = tfs.get(T, {})
        n_big_force = s.get("n_above_1e-3", 0)
        n = s.get("n", 0)
        if n > 0 and n_big_force > 0:
            print(f"    - At T={T:.0e}: {n_big_force}/{n} qualifying failed samples "
                  f"still have force_norm > 1e-3 (NOT at a minimum)")

    print()
    print(sep)
    print("  END OF REPORT")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eigenvalue tolerance investigation for NR minimization trajectories."
    )
    parser.add_argument(
        "--grid-dir",
        type=str,
        required=True,
        help="Root directory of the grid-search output.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output CSVs, JSONs, and PNGs.",
    )
    parser.add_argument(
        "--traj-glob",
        type=str,
        default="*/diagnostics/*_trajectory.json",
        help="Glob pattern for trajectory JSONs relative to grid-dir.",
    )
    parser.add_argument(
        "--combo-tag",
        type=str,
        default=None,
        help="If set, only process trajectories under this combo tag folder.",
    )
    parser.add_argument(
        "--top-k-heatmaps",
        type=int,
        default=10,
        help="Number of per-sample eigenvalue timeline heatmaps to generate.",
    )
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not grid_dir.is_dir():
        print(f"ERROR: --grid-dir {grid_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    # --- Discover and load trajectories ---
    print(f"Discovering trajectories in {grid_dir} with glob '{args.traj_glob}' ...")
    discovered = discover_trajectories(grid_dir, args.traj_glob, args.combo_tag)
    print(f"  Found {len(discovered)} trajectory files.")

    if not discovered:
        print("ERROR: No trajectory files found. Check --grid-dir and --traj-glob.",
              file=sys.stderr)
        sys.exit(1)

    all_trajs: List[Dict[str, Any]] = []
    for tag, path in discovered:
        traj_data = load_trajectory(path)
        if traj_data is not None:
            traj_data["_combo_tag"] = tag
            traj_data["_path"] = str(path)
            all_trajs.append(traj_data)

    print(f"  Successfully loaded {len(all_trajs)} trajectories.")
    n_total = len(all_trajs)
    n_converged = sum(1 for t in all_trajs if _is_converged(t))
    n_failed = n_total - n_converged
    print(f"  Converged: {n_converged}, Failed: {n_failed}")
    print()

    # --- Run analyses ---
    print("Running A1: Band Evolution Statistics ...")
    band_results = analyze_band_evolution(all_trajs)

    print("Running A2: Tolerance Sweep Experiment ...")
    sweep_results = analyze_tolerance_sweep(all_trajs)

    print("Running A3: Physical Significance Analysis ...")
    phys_results = analyze_physical_significance(all_trajs)

    print("Running A4: Stability / False Convergence Detection ...")
    fc_results = analyze_false_convergence(all_trajs)

    print("Running A5: Cascade Gap Analysis ...")
    cascade_results = analyze_cascade_gap(all_trajs)

    print("Running A6: Eigenvalue-Force-Energy Consistency Check ...")
    fe_results = analyze_force_eigenvalue_consistency(all_trajs)

    print()

    # --- Write CSVs ---
    print("Writing CSV outputs ...")
    write_eigenvalue_tolerance_sweep_csv(sweep_results, output_dir)
    write_per_sample_required_tolerance_csv(cascade_results, output_dir)
    write_band_evolution_summary_csv(band_results, output_dir)
    write_ghost_mode_analysis_csv(phys_results, output_dir)
    write_false_convergence_csv(fc_results, output_dir)
    write_force_eigenvalue_consistency_csv(fe_results, output_dir)
    print()

    # --- Write JSON ---
    print("Writing JSON summary ...")
    write_justification_json(
        band_results, sweep_results, phys_results,
        fc_results, cascade_results, fe_results,
        output_dir,
    )
    print()

    # --- Generate plots ---
    print("Generating plots ...")
    plot_eigenvalue_distribution_final(all_trajs, output_dir)
    plot_min_eval_cdf_failed(cascade_results, output_dir)
    plot_convergence_rate_vs_tolerance(
        sweep_results, fc_results, n_total, n_converged, output_dir
    )
    plot_eigenvalue_timeline_per_sample(all_trajs, output_dir, top_k=args.top_k_heatmaps)
    plot_band_evolution_converged_vs_failed(band_results, output_dir)
    plot_sample_tolerance_heatmap(cascade_results, output_dir)
    plot_ghost_onset_histogram(band_results, output_dir)
    plot_false_convergence_vs_tolerance(fc_results, output_dir)
    plot_force_vs_min_eigenvalue(fe_results, output_dir)
    print()

    # --- Print report ---
    print_report(
        band_results, sweep_results, phys_results,
        fc_results, cascade_results, fe_results,
        n_total, n_converged, n_failed,
    )


if __name__ == "__main__":
    main()
