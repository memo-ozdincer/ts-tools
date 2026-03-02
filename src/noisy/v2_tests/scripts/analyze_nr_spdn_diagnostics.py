#!/usr/bin/env python3
"""Comprehensive diagnostic analysis for Newton-Raphson minimization with
Spectrally-Partitioned DIIS-Newton (SPDN) mode.

Reads trajectory JSON files from grid-search output directories and produces
detailed diagnostic reports covering seven analysis modules:

  D1. SPDN vs Legacy Comparison
  D2. Spectral Conditioning Deep Dive
  D3. GDIIS Performance Analysis
  D4. Backtracking Line Search Analysis
  D5. Saddle Point Detection Diagnostics
  D6. Walking Out of Minima Analysis
  D7. Eigenvalue Stability & Change Rate Analysis

Usage:
  python analyze_nr_spdn_diagnostics.py \\
      --grid-dir <path> --output-dir <path> \\
      [--traj-glob "*/diagnostics/*_trajectory.json"] \\
      [--combo-tag <tag>] [--top-k 5]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("ggplot")
    except OSError:
        pass

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MILESTONE_STEPS = [0, 10, 50, 100, 500, 1000]

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

CASCADE_KEYS = [
    "n_neg_at_0.0",
    "n_neg_at_0.0001",
    "n_neg_at_0.0005",
    "n_neg_at_0.001",
    "n_neg_at_0.002",
    "n_neg_at_0.005",
    "n_neg_at_0.008",
    "n_neg_at_0.01",
]

N_INTERP_PTS = 100  # grid resolution for normalised-step interpolation


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _col(steps: List[Dict], key: str, default: float = float("nan")) -> np.ndarray:
    """Extract a column from trajectory steps as numpy array."""
    return np.array([s.get(key, default) for s in steps], dtype=float)


def _finite(arr: np.ndarray) -> np.ndarray:
    """Return only finite elements."""
    return arr[np.isfinite(arr)]


def _safe_mean(vals: Sequence[float]) -> float:
    f = [float(v) for v in vals if math.isfinite(float(v))]
    return float(np.mean(f)) if f else float("nan")


def _safe_median(vals: Sequence[float]) -> float:
    f = [float(v) for v in vals if math.isfinite(float(v))]
    return float(np.median(f)) if f else float("nan")


def _is_converged(traj_data: Dict[str, Any]) -> bool:
    """Determine convergence from trajectory data."""
    c = traj_data.get("converged")
    if c is not None:
        return bool(c)
    fnv = traj_data.get("final_neg_vib")
    if fnv is not None:
        return fnv == 0
    traj = traj_data.get("trajectory", [])
    if traj:
        return traj[-1].get("n_neg_evals", -1) == 0
    return False


def _is_spdn(traj_data: Dict[str, Any]) -> bool:
    """Check whether a trajectory used SPDN optimizer mode."""
    return traj_data.get("optimizer_mode", "") == "spdn"


def _band_populations(step: Dict) -> List[int]:
    """Extract the 8 band populations from a trajectory step."""
    return [step.get(k, 0) for k in BAND_KEYS]


def _interp_to_norm_grid(
    values: np.ndarray, n_pts: int = N_INTERP_PTS,
) -> np.ndarray:
    """Interpolate a per-step array to a normalised 0..1 grid."""
    n = len(values)
    if n == 0:
        return np.full(n_pts, float("nan"))
    if n == 1:
        return np.full(n_pts, values[0])
    x_orig = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, n_pts)
    return np.interp(x_new, x_orig, values)


def _median_iqr(
    matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise (axis=0) median and IQR from (n_trajectories, n_pts).

    Returns (median, q25, q75) each of shape (n_pts,).
    """
    if matrix.size == 0:
        n = matrix.shape[1] if matrix.ndim == 2 else N_INTERP_PTS
        nan_arr = np.full(n, float("nan"))
        return nan_arr.copy(), nan_arr.copy(), nan_arr.copy()
    med = np.nanmedian(matrix, axis=0)
    q25 = np.nanpercentile(matrix, 25, axis=0)
    q75 = np.nanpercentile(matrix, 75, axis=0)
    return med, q25, q75


def _write_csv(
    path: Path,
    rows: List[Dict],
    fieldnames: Optional[List[str]] = None,
) -> None:
    """Write a list of dicts to CSV."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(val: Any, fmt: str = ".4g") -> str:
    """Format a value for printing, handling NaN/None."""
    if val is None:
        return "N/A"
    try:
        if math.isnan(float(val)):
            return "N/A"
    except (TypeError, ValueError):
        return str(val)
    return format(float(val), fmt)


def _binned_mean(
    all_x: np.ndarray,
    all_y: np.ndarray,
    n_bins: int = 50,
) -> Tuple[List[float], List[float]]:
    """Bin *all_x* into *n_bins* and compute mean of *all_y* per bin.

    Returns (bin_centers, bin_means) with only non-empty bins.
    """
    x_fin = np.isfinite(all_x) & np.isfinite(all_y)
    ax, ay = all_x[x_fin], all_y[x_fin]
    if len(ax) == 0:
        return [], []
    xmin, xmax = float(np.min(ax)), float(np.max(ax))
    if xmax <= xmin:
        return [xmin], [float(np.mean(ay))]
    bins = np.linspace(xmin, xmax, n_bins + 1)
    idx = np.digitize(ax, bins)
    centers: List[float] = []
    means: List[float] = []
    for b in range(1, n_bins + 1):
        mask = idx == b
        v = ay[mask]
        vf = v[np.isfinite(v)]
        if len(vf) > 0:
            centers.append(0.5 * (bins[b - 1] + bins[b]))
            means.append(float(np.mean(vf)))
    return centers, means


# ---------------------------------------------------------------------------
# Trajectory discovery & loading
# ---------------------------------------------------------------------------

def discover_trajectories(
    grid_dir: Path,
    traj_glob: str,
    combo_tag: Optional[str] = None,
) -> List[Tuple[str, Path]]:
    """Find trajectory JSON files, optionally filtered to a single combo."""
    results: List[Tuple[str, Path]] = []
    for p in sorted(grid_dir.glob(traj_glob)):
        if not p.is_file():
            continue
        rel = p.relative_to(grid_dir)
        tag = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        if combo_tag is not None and tag != combo_tag:
            continue
        results.append((tag, p))
    return results


def load_all_trajectories(
    grid_dir: Path,
    traj_glob: str,
    combo_tag: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load all trajectory JSONs.

    Each element is a dict with keys: combo_tag, sample_id, traj_data.
    """
    entries = discover_trajectories(grid_dir, traj_glob, combo_tag)
    loaded: List[Dict[str, Any]] = []
    for tag, path in entries:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"  WARNING: could not load {path}: {exc}",
                file=sys.stderr,
            )
            continue
        sample_id = data.get("sample_id", path.stem)
        loaded.append({
            "combo_tag": tag,
            "sample_id": sample_id,
            "traj_data": data,
        })
    return loaded


# ===================================================================
# D1: SPDN vs Legacy Comparison
# ===================================================================

def analyze_d1_spdn_vs_legacy(
    all_entries: List[Dict[str, Any]],
) -> str:
    """Compare SPDN combos vs legacy (non-SPDN) combos."""
    lines: List[str] = []
    lines.append("Section D1: SPDN vs Legacy Comparison")
    lines.append("-" * 50)

    spdn_entries = [e for e in all_entries if _is_spdn(e["traj_data"])]
    legacy_entries = [e for e in all_entries if not _is_spdn(e["traj_data"])]

    # ------------------------------------------------------------------ #
    def _group_stats(entries: List[Dict[str, Any]], label: str) -> List[str]:
        out: List[str] = []
        if not entries:
            out.append(f"  No {label} trajectories found.")
            return out

        n_total = len(entries)
        n_conv_strict = sum(
            1 for e in entries if _is_converged(e["traj_data"])
        )

        # Relaxed convergence: n_neg_reliable == 0 at final step
        n_conv_relaxed = 0
        for e in entries:
            traj = e["traj_data"].get("trajectory", [])
            if traj:
                last = traj[-1]
                sd = last.get("saddle_diagnostic", {})
                n_neg_r = sd.get(
                    "n_neg_reliable", last.get("n_neg_evals", -1),
                )
                if n_neg_r == 0:
                    n_conv_relaxed += 1

        steps_conv: List[float] = []
        steps_fail: List[float] = []
        fnorm_conv: List[float] = []
        fnorm_fail: List[float] = []
        wall_times: List[float] = []
        trust_radii: List[float] = []
        linesearch_alphas: List[float] = []
        gdiis_attempts: List[int] = []
        gdiis_accepts: List[int] = []

        for e in entries:
            td = e["traj_data"]
            traj = td.get("trajectory", [])
            conv = _is_converged(td)
            n_steps = td.get("total_steps", len(traj))
            if conv:
                steps_conv.append(
                    td.get("converged_step", n_steps),
                )
                fn = td.get("final_force_norm")
                if fn is not None:
                    fnorm_conv.append(fn)
            else:
                steps_fail.append(n_steps)
                fn = td.get("final_force_norm")
                if fn is not None:
                    fnorm_fail.append(fn)
            wt = td.get("wall_time")
            if wt is not None:
                wall_times.append(wt)
            for s in traj:
                tr = s.get("trust_radius")
                if tr is not None:
                    trust_radii.append(tr)
                ls = s.get("spdn_linesearch", {})
                alpha = ls.get("alpha")
                if alpha is not None:
                    linesearch_alphas.append(alpha)
            ga = td.get("total_diis_attempts", 0)
            gac = td.get("total_diis_accepts", 0)
            if ga > 0:
                gdiis_attempts.append(ga)
                gdiis_accepts.append(gac)

        out.append(f"  {label} trajectories: {n_total}")
        out.append(
            f"    Convergence (strict, n_neg==0):    "
            f"{n_conv_strict}/{n_total} "
            f"({100 * n_conv_strict / max(n_total, 1):.1f}%)"
        )
        out.append(
            f"    Convergence (relaxed, n_neg_r==0): "
            f"{n_conv_relaxed}/{n_total} "
            f"({100 * n_conv_relaxed / max(n_total, 1):.1f}%)"
        )
        out.append(
            f"    Mean steps to converge: "
            f"{_fmt(_safe_mean(steps_conv))}"
        )
        out.append(
            f"    Mean steps (failed):    "
            f"{_fmt(_safe_mean(steps_fail))}"
        )
        if wall_times:
            out.append(
                f"    Mean wall time: {_fmt(_safe_mean(wall_times))} s"
            )
        out.append(
            f"    Final force norm (converged): "
            f"mean={_fmt(_safe_mean(fnorm_conv))}, "
            f"median={_fmt(_safe_median(fnorm_conv))}"
        )
        out.append(
            f"    Final force norm (failed):    "
            f"mean={_fmt(_safe_mean(fnorm_fail))}, "
            f"median={_fmt(_safe_median(fnorm_fail))}"
        )
        if trust_radii:
            out.append(
                f"    Trust radius: "
                f"mean={_fmt(_safe_mean(trust_radii))}, "
                f"median={_fmt(_safe_median(trust_radii))}"
            )
        if linesearch_alphas:
            out.append(
                f"    Line search alpha: "
                f"mean={_fmt(_safe_mean(linesearch_alphas))}, "
                f"median={_fmt(_safe_median(linesearch_alphas))}"
            )
        if gdiis_attempts:
            t_att = sum(gdiis_attempts)
            t_acc = sum(gdiis_accepts)
            rate = t_acc / max(t_att, 1)
            out.append(
                f"    GDIIS: {t_att} attempts, {t_acc} accepts "
                f"({100 * rate:.1f}% accept rate)"
            )
        return out
    # ------------------------------------------------------------------ #

    if not legacy_entries:
        lines.append(
            "  No legacy combos found for comparison, "
            "reporting SPDN stats only."
        )
        lines.append("")
    else:
        lines.extend(_group_stats(legacy_entries, "Legacy"))
        lines.append("")

    lines.extend(_group_stats(spdn_entries, "SPDN"))
    lines.append("")

    # Per-combo breakdown
    combos: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in all_entries:
        combos[e["combo_tag"]].append(e)

    if combos:
        lines.append("  Per-combo summary:")
        lines.append(
            f"  {'Combo':<50} {'Mode':<8} {'Conv':>6} "
            f"{'Total':>6} {'Rate':>8}"
        )
        for tag in sorted(combos.keys()):
            ents = combos[tag]
            n_t = len(ents)
            n_c = sum(1 for e in ents if _is_converged(e["traj_data"]))
            mode = "SPDN" if _is_spdn(ents[0]["traj_data"]) else "Legacy"
            rate_s = f"{100 * n_c / max(n_t, 1):.1f}%"
            lines.append(
                f"  {tag:<50} {mode:<8} {n_c:>6} {n_t:>6} {rate_s:>8}"
            )

    return "\n".join(lines)


# ===================================================================
# D2: Spectral Conditioning Deep Dive
# ===================================================================

def analyze_d2_spectral_conditioning(
    all_entries: List[Dict[str, Any]],
    output_dir: Path,
) -> str:
    """Spectral conditioning analysis for SPDN trajectories."""
    lines: List[str] = []
    lines.append("Section D2: Spectral Conditioning Deep Dive")
    lines.append("-" * 50)

    spdn_entries = [e for e in all_entries if _is_spdn(e["traj_data"])]
    if not spdn_entries:
        lines.append("  No SPDN trajectories found. Skipping D2.")
        return "\n".join(lines)

    csv_rows: List[Dict] = []

    # Normalised trajectory collectors for plotting
    conv_cond_full: List[np.ndarray] = []
    fail_cond_full: List[np.ndarray] = []
    conv_n_hard_neg: List[np.ndarray] = []
    fail_n_hard_neg: List[np.ndarray] = []
    conv_eval_change: List[np.ndarray] = []
    fail_eval_change: List[np.ndarray] = []
    conv_min_pos: List[np.ndarray] = []
    fail_min_pos: List[np.ndarray] = []
    conv_abs_max_neg: List[np.ndarray] = []
    fail_abs_max_neg: List[np.ndarray] = []

    milestones_for_report: Dict[int, Dict[str, List[float]]] = {}

    for entry in spdn_entries:
        td = entry["traj_data"]
        traj = td.get("trajectory", [])
        if not traj:
            continue
        converged = _is_converged(td)
        sample_id = entry["sample_id"]
        combo_tag = entry["combo_tag"]
        n_steps = len(traj)

        # Extract per-step spectral conditioning
        cf_arr = np.empty(n_steps, dtype=float)
        ch_arr = np.empty(n_steps, dtype=float)
        nhp_arr = np.empty(n_steps, dtype=float)
        nhn_arr = np.empty(n_steps, dtype=float)
        ns_arr = np.empty(n_steps, dtype=float)
        nn_arr = np.empty(n_steps, dtype=float)
        er_arr = np.empty(n_steps, dtype=float)
        ecr_arr = np.empty(n_steps, dtype=float)
        mp_arr = np.empty(n_steps, dtype=float)
        mn_arr = np.empty(n_steps, dtype=float)

        for idx, s in enumerate(traj):
            sc = s.get("spectral_conditioning", {})
            cf_arr[idx] = sc.get("cond_full", float("nan"))
            ch_arr[idx] = sc.get("cond_hard_only", float("nan"))
            nhp_arr[idx] = sc.get("n_hard_pos", float("nan"))
            nhn_arr[idx] = sc.get("n_hard_neg", float("nan"))
            ns_arr[idx] = sc.get("n_soft", float("nan"))
            nn_arr[idx] = sc.get("n_null", float("nan"))
            er_arr[idx] = sc.get("eval_range", float("nan"))
            ecr_arr[idx] = sc.get("eval_change_rate_mean", float("nan"))
            mp_arr[idx] = sc.get("min_pos_eval", float("nan"))
            mn_arr[idx] = sc.get("max_neg_eval", float("nan"))

        # Interpolate for normalised-step plots
        if converged:
            conv_cond_full.append(_interp_to_norm_grid(cf_arr))
            conv_n_hard_neg.append(_interp_to_norm_grid(nhn_arr))
            conv_eval_change.append(_interp_to_norm_grid(ecr_arr))
            conv_min_pos.append(_interp_to_norm_grid(mp_arr))
            conv_abs_max_neg.append(
                _interp_to_norm_grid(np.abs(mn_arr))
            )
        else:
            fail_cond_full.append(_interp_to_norm_grid(cf_arr))
            fail_n_hard_neg.append(_interp_to_norm_grid(nhn_arr))
            fail_eval_change.append(_interp_to_norm_grid(ecr_arr))
            fail_min_pos.append(_interp_to_norm_grid(mp_arr))
            fail_abs_max_neg.append(
                _interp_to_norm_grid(np.abs(mn_arr))
            )

        # Milestone CSV rows
        ms_indices = [ms for ms in MILESTONE_STEPS if ms < n_steps]
        ms_indices.append(n_steps - 1)
        for ms_step in ms_indices:
            si = min(ms_step, n_steps - 1)
            sc = traj[si].get("spectral_conditioning", {})
            row = {
                "sample_id": sample_id,
                "combo_tag": combo_tag,
                "converged": converged,
                "milestone_step": ms_step,
                "cond_full": sc.get("cond_full", float("nan")),
                "cond_hard_only": sc.get("cond_hard_only", float("nan")),
                "n_hard_pos": sc.get("n_hard_pos", float("nan")),
                "n_hard_neg": sc.get("n_hard_neg", float("nan")),
                "n_soft": sc.get("n_soft", float("nan")),
                "n_null": sc.get("n_null", float("nan")),
                "eval_range": sc.get("eval_range", float("nan")),
                "eval_change_rate_mean": sc.get(
                    "eval_change_rate_mean", float("nan"),
                ),
                "min_pos_eval": sc.get("min_pos_eval", float("nan")),
                "max_neg_eval": sc.get("max_neg_eval", float("nan")),
            }
            csv_rows.append(row)

            # Collect for milestone text
            key = ms_step
            if key not in milestones_for_report:
                milestones_for_report[key] = defaultdict(list)
            for fld in ("cond_full", "cond_hard_only", "n_hard_neg"):
                v = row[fld]
                try:
                    vf = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(vf):
                    lbl = "conv" if converged else "fail"
                    milestones_for_report[key][f"{fld}_{lbl}"].append(vf)

    _write_csv(output_dir / "spectral_conditioning_milestones.csv", csv_rows)

    # Text report
    n_conv = sum(1 for e in spdn_entries if _is_converged(e["traj_data"]))
    n_fail = len(spdn_entries) - n_conv
    lines.append(f"  SPDN trajectories: {len(spdn_entries)}")
    lines.append(f"  Converged: {n_conv}")
    lines.append(f"  Failed:    {n_fail}")
    lines.append("")

    for ms_step in sorted(milestones_for_report.keys()):
        data = milestones_for_report[ms_step]
        cf_c = data.get("cond_full_conv", [])
        cf_f = data.get("cond_full_fail", [])
        nh_c = data.get("n_hard_neg_conv", [])
        nh_f = data.get("n_hard_neg_fail", [])
        lines.append(f"  Milestone step {ms_step}:")
        lines.append(
            f"    cond_full  conv: mean={_fmt(_safe_mean(cf_c))}, "
            f"median={_fmt(_safe_median(cf_c))}  |  "
            f"fail: mean={_fmt(_safe_mean(cf_f))}, "
            f"median={_fmt(_safe_median(cf_f))}"
        )
        lines.append(
            f"    n_hard_neg conv: mean={_fmt(_safe_mean(nh_c))}  |  "
            f"fail: mean={_fmt(_safe_mean(nh_f))}"
        )

    # --- Eigenvalue spread at final step ---
    final_er_conv: List[float] = []
    final_er_fail: List[float] = []
    final_esp_conv: List[float] = []
    final_esp_fail: List[float] = []
    final_esn_conv: List[float] = []
    final_esn_fail: List[float] = []
    for entry in spdn_entries:
        traj = entry["traj_data"].get("trajectory", [])
        if not traj:
            continue
        sc = traj[-1].get("spectral_conditioning", {})
        conv = _is_converged(entry["traj_data"])
        for key, c_list, f_list in [
            ("eval_range", final_er_conv, final_er_fail),
            ("eval_spread_positive", final_esp_conv, final_esp_fail),
            ("eval_spread_negative", final_esn_conv, final_esn_fail),
        ]:
            v = sc.get(key, float("nan"))
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(vf):
                (c_list if conv else f_list).append(vf)

    lines.append("")
    lines.append("  Eigenvalue spread at final step:")
    lines.append(
        f"    eval_range          conv: {_fmt(_safe_mean(final_er_conv))}  |  "
        f"fail: {_fmt(_safe_mean(final_er_fail))}"
    )
    lines.append(
        f"    eval_spread_pos     conv: {_fmt(_safe_mean(final_esp_conv))}  |  "
        f"fail: {_fmt(_safe_mean(final_esp_fail))}"
    )
    lines.append(
        f"    eval_spread_neg     conv: {_fmt(_safe_mean(final_esn_conv))}  |  "
        f"fail: {_fmt(_safe_mean(final_esn_fail))}"
    )

    # Plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        x_norm = np.linspace(0.0, 1.0, N_INTERP_PTS)

        # (0,0) cond_full
        ax = axes[0, 0]
        if conv_cond_full:
            med, q25, q75 = _median_iqr(np.array(conv_cond_full))
            ax.plot(x_norm, med, color="green", label="Converged (median)")
            ax.fill_between(x_norm, q25, q75, color="green", alpha=0.2)
        if fail_cond_full:
            med, q25, q75 = _median_iqr(np.array(fail_cond_full))
            ax.plot(x_norm, med, color="red", label="Failed (median)")
            ax.fill_between(x_norm, q25, q75, color="red", alpha=0.2)
        ax.set_xlabel("Normalized step")
        ax.set_ylabel("cond_full")
        ax.set_title("Full Condition Number Evolution")
        ax.set_yscale("log")
        ax.legend(fontsize=8)

        # (0,1) n_hard_neg
        ax = axes[0, 1]
        if conv_n_hard_neg:
            med, q25, q75 = _median_iqr(np.array(conv_n_hard_neg))
            ax.plot(x_norm, med, color="green", label="Converged")
            ax.fill_between(x_norm, q25, q75, color="green", alpha=0.2)
        if fail_n_hard_neg:
            med, q25, q75 = _median_iqr(np.array(fail_n_hard_neg))
            ax.plot(x_norm, med, color="red", label="Failed")
            ax.fill_between(x_norm, q25, q75, color="red", alpha=0.2)
        ax.set_xlabel("Normalized step")
        ax.set_ylabel("n_hard_neg")
        ax.set_title("Hard Negative Eigenvalue Count")
        ax.legend(fontsize=8)

        # (1,0) eval_change_rate_mean
        ax = axes[1, 0]
        if conv_eval_change:
            med, q25, q75 = _median_iqr(np.array(conv_eval_change))
            ax.plot(x_norm, med, color="green", label="Converged")
            ax.fill_between(x_norm, q25, q75, color="green", alpha=0.2)
        if fail_eval_change:
            med, q25, q75 = _median_iqr(np.array(fail_eval_change))
            ax.plot(x_norm, med, color="red", label="Failed")
            ax.fill_between(x_norm, q25, q75, color="red", alpha=0.2)
        ax.set_xlabel("Normalized step")
        ax.set_ylabel("eval_change_rate_mean")
        ax.set_title("Eigenvalue Change Rate (Mean)")
        ax.legend(fontsize=8)

        # (1,1) min_pos_eval and |max_neg_eval|
        ax = axes[1, 1]
        if conv_min_pos:
            med_p, _, _ = _median_iqr(np.array(conv_min_pos))
            ax.plot(
                x_norm, med_p, color="green", linestyle="-",
                label="min_pos (conv)",
            )
        if fail_min_pos:
            med_p, _, _ = _median_iqr(np.array(fail_min_pos))
            ax.plot(
                x_norm, med_p, color="red", linestyle="-",
                label="min_pos (fail)",
            )
        if conv_abs_max_neg:
            med_n, _, _ = _median_iqr(np.array(conv_abs_max_neg))
            ax.plot(
                x_norm, med_n, color="green", linestyle="--",
                label="|max_neg| (conv)",
            )
        if fail_abs_max_neg:
            med_n, _, _ = _median_iqr(np.array(fail_abs_max_neg))
            ax.plot(
                x_norm, med_n, color="red", linestyle="--",
                label="|max_neg| (fail)",
            )
        ax.set_xlabel("Normalized step")
        ax.set_ylabel("Eigenvalue magnitude")
        ax.set_title("Smallest Pos / Largest |Neg| Eigenvalue")
        ax.set_yscale("log")
        ax.legend(fontsize=7)

        plt.tight_layout()
        fig.savefig(
            output_dir / "spectral_conditioning_evolution.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        lines.append("  Plot saved: spectral_conditioning_evolution.png")
    except Exception as exc:
        lines.append(f"  WARNING: plotting failed for D2: {exc}")

    lines.append("  CSV saved:  spectral_conditioning_milestones.csv")
    return "\n".join(lines)


# ===================================================================
# D3: GDIIS Performance Analysis
# ===================================================================

def analyze_d3_gdiis_performance(
    all_entries: List[Dict[str, Any]],
    output_dir: Path,
) -> str:
    """GDIIS performance analysis for SPDN trajectories."""
    lines: List[str] = []
    lines.append("Section D3: GDIIS Performance")
    lines.append("-" * 50)

    spdn_entries = [e for e in all_entries if _is_spdn(e["traj_data"])]
    if not spdn_entries:
        lines.append("  No SPDN trajectories found. Skipping D3.")
        return "\n".join(lines)

    combo_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "n_trajs": 0,
            "total_attempts": 0,
            "total_accepts": 0,
            "total_energy_accepts": 0,
        }
    )

    all_bcond_accepted: List[float] = []
    all_bcond_rejected: List[float] = []
    all_resid_accepted: List[float] = []
    all_resid_rejected: List[float] = []
    all_energy_changes: List[float] = []
    all_diis_attempt_steps: List[float] = []
    all_diis_accept_steps: List[float] = []
    steps_before_conv_last_diis: List[int] = []

    for entry in spdn_entries:
        td = entry["traj_data"]
        traj = td.get("trajectory", [])
        combo = entry["combo_tag"]
        converged = _is_converged(td)
        cs = combo_stats[combo]
        cs["n_trajs"] += 1
        cs["total_attempts"] += td.get("total_diis_attempts", 0)
        cs["total_accepts"] += td.get("total_diis_accepts", 0)
        cs["total_energy_accepts"] += td.get(
            "total_diis_energy_accepts", 0,
        )

        n_steps = len(traj)
        last_accept_step: Optional[int] = None

        for i, s in enumerate(traj):
            gi = s.get("gdiis_info", {})
            status = gi.get("status", "")
            if status in ("success", "failed"):
                norm_step = i / max(n_steps - 1, 1)
                all_diis_attempt_steps.append(norm_step)
                bcond = gi.get("B_cond", float("nan"))
                resid = gi.get("residual_ratio", float("nan"))

                accepted = s.get("gdiis_accepted", False)
                if accepted:
                    all_diis_accept_steps.append(norm_step)
                    last_accept_step = i
                    if math.isfinite(bcond):
                        all_bcond_accepted.append(bcond)
                    if math.isfinite(resid):
                        all_resid_accepted.append(resid)
                    ec = s.get("gdiis_energy_change", float("nan"))
                    if math.isfinite(ec):
                        all_energy_changes.append(ec)
                else:
                    if math.isfinite(bcond):
                        all_bcond_rejected.append(bcond)
                    if math.isfinite(resid):
                        all_resid_rejected.append(resid)

        if converged and last_accept_step is not None:
            conv_step = td.get("converged_step", n_steps)
            steps_before_conv_last_diis.append(
                conv_step - last_accept_step,
            )

    # CSV per combo
    csv_rows: List[Dict] = []
    for combo in sorted(combo_stats.keys()):
        cs = combo_stats[combo]
        rate = cs["total_accepts"] / max(cs["total_attempts"], 1)
        csv_rows.append({
            "combo_tag": combo,
            "n_trajectories": cs["n_trajs"],
            "total_diis_attempts": cs["total_attempts"],
            "total_diis_accepts": cs["total_accepts"],
            "total_diis_energy_accepts": cs["total_energy_accepts"],
            "accept_rate": rate,
        })
    _write_csv(output_dir / "gdiis_performance.csv", csv_rows)

    # Text report
    total_att = sum(c["total_attempts"] for c in combo_stats.values())
    total_acc = sum(c["total_accepts"] for c in combo_stats.values())
    total_eacc = sum(
        c["total_energy_accepts"] for c in combo_stats.values()
    )
    lines.append(f"  Total DIIS attempts:       {total_att}")
    lines.append(
        f"  Total DIIS accepts:        {total_acc} "
        f"({100 * total_acc / max(total_att, 1):.1f}%)"
    )
    lines.append(f"  Total energy accepts:      {total_eacc}")
    lines.append(
        f"  B_cond (accepted):  "
        f"mean={_fmt(_safe_mean(all_bcond_accepted))}, "
        f"median={_fmt(_safe_median(all_bcond_accepted))}"
    )
    lines.append(
        f"  B_cond (rejected):  "
        f"mean={_fmt(_safe_mean(all_bcond_rejected))}, "
        f"median={_fmt(_safe_median(all_bcond_rejected))}"
    )
    lines.append(
        f"  Residual ratio (accepted): "
        f"mean={_fmt(_safe_mean(all_resid_accepted))}"
    )
    lines.append(
        f"  Residual ratio (rejected): "
        f"mean={_fmt(_safe_mean(all_resid_rejected))}"
    )
    lines.append(
        f"  Energy change from DIIS:   "
        f"mean={_fmt(_safe_mean(all_energy_changes))}"
    )
    if steps_before_conv_last_diis:
        lines.append(
            f"  Steps from last DIIS accept to convergence: "
            f"mean={_fmt(_safe_mean(steps_before_conv_last_diis))}, "
            f"median={_fmt(_safe_median(steps_before_conv_last_diis))}"
        )

    # Plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (0,0) accept rate per combo
        ax = axes[0, 0]
        if csv_rows:
            combos_sorted = sorted(
                csv_rows, key=lambda r: r["accept_rate"], reverse=True,
            )
            tags = [r["combo_tag"][:30] for r in combos_sorted]
            rates = [r["accept_rate"] for r in combos_sorted]
            y_pos = np.arange(len(tags))
            ax.barh(y_pos, rates, color="steelblue")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(tags, fontsize=6)
            ax.set_xlabel("DIIS Accept Rate")
            ax.set_title("DIIS Accept Rate per Combo")
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes,
            )

        # (0,1) B_cond histogram
        ax = axes[0, 1]
        both_bc = all_bcond_accepted + all_bcond_rejected
        if both_bc:
            lo = max(1e-10, min(both_bc))
            hi = max(both_bc)
            bins = np.logspace(np.log10(lo), np.log10(max(hi, lo * 2)), 30)
            if all_bcond_accepted:
                ax.hist(
                    all_bcond_accepted, bins=bins, alpha=0.6,
                    color="green", label="Accepted",
                )
            if all_bcond_rejected:
                ax.hist(
                    all_bcond_rejected, bins=bins, alpha=0.6,
                    color="red", label="Rejected",
                )
            ax.set_xscale("log")
            ax.set_xlabel("B matrix condition number")
            ax.set_title("DIIS B-matrix Condition Number")
            ax.legend(fontsize=8)
        else:
            ax.text(
                0.5, 0.5, "No GDIIS data", ha="center", va="center",
                transform=ax.transAxes,
            )

        # (1,0) energy change histogram
        ax = axes[1, 0]
        if all_energy_changes:
            ax.hist(all_energy_changes, bins=50, color="steelblue", alpha=0.7)
            ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
            ax.set_xlabel("Energy change from DIIS")
            ax.set_ylabel("Count")
            ax.set_title("Energy Change from DIIS Extrapolation")
        else:
            ax.text(
                0.5, 0.5, "No GDIIS energy data", ha="center",
                va="center", transform=ax.transAxes,
            )

        # (1,1) step of attempts vs accepts
        ax = axes[1, 1]
        if all_diis_attempt_steps:
            ax.hist(
                all_diis_attempt_steps, bins=50, alpha=0.5,
                color="gray", label="Attempts",
            )
        if all_diis_accept_steps:
            ax.hist(
                all_diis_accept_steps, bins=50, alpha=0.6,
                color="green", label="Accepts",
            )
        ax.set_xlabel("Normalized step")
        ax.set_ylabel("Count")
        ax.set_title("DIIS Attempts vs Accepts over Optimization")
        ax.legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(
            output_dir / "gdiis_analysis.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        lines.append("  Plot saved: gdiis_analysis.png")
    except Exception as exc:
        lines.append(f"  WARNING: plotting failed for D3: {exc}")

    lines.append("  CSV saved:  gdiis_performance.csv")
    return "\n".join(lines)


# ===================================================================
# D4: Backtracking Line Search Analysis
# ===================================================================

def analyze_d4_linesearch(
    all_entries: List[Dict[str, Any]],
    output_dir: Path,
) -> str:
    """Backtracking line search analysis for SPDN trajectories."""
    lines: List[str] = []
    lines.append("Section D4: Backtracking Line Search")
    lines.append("-" * 50)

    spdn_entries = [e for e in all_entries if _is_spdn(e["traj_data"])]
    if not spdn_entries:
        lines.append("  No SPDN trajectories found. Skipping D4.")
        return "\n".join(lines)

    combo_ls: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "n_steps": 0,
            "total_backtracks": 0,
            "n_perfect": 0,
            "n_mild": 0,
            "n_aggressive": 0,
            "n_rejected": 0,
            "alphas": [],
            "energy_decreases": [],
        }
    )

    all_n_backtracks: List[int] = []
    all_alphas: List[float] = []
    all_energy_dec: List[float] = []
    scatter_bt_hn: List[Tuple[float, float]] = []

    for entry in spdn_entries:
        td = entry["traj_data"]
        traj = td.get("trajectory", [])
        combo = entry["combo_tag"]

        for s in traj:
            ls = s.get("spdn_linesearch", {})
            if not ls:
                continue
            nb = ls.get("n_backtracks", 0)
            alpha = ls.get("alpha", float("nan"))
            accepted = ls.get("accepted", True)
            e_before = ls.get("energy_before", float("nan"))
            e_after = ls.get("energy_after", float("nan"))

            cls = combo_ls[combo]
            cls["n_steps"] += 1
            cls["total_backtracks"] += nb

            if nb == 0:
                cls["n_perfect"] += 1
            elif 1 <= nb <= 3:
                cls["n_mild"] += 1
            else:
                cls["n_aggressive"] += 1

            if not accepted:
                cls["n_rejected"] += 1

            all_n_backtracks.append(nb)
            if math.isfinite(alpha):
                all_alphas.append(alpha)
                cls["alphas"].append(alpha)

            if math.isfinite(e_before) and math.isfinite(e_after):
                dec = e_after - e_before
                all_energy_dec.append(dec)
                cls["energy_decreases"].append(dec)

            sc = s.get("spectral_conditioning", {})
            n_hn = sc.get("n_hard_neg", float("nan"))
            if math.isfinite(n_hn):
                scatter_bt_hn.append((nb, n_hn))

    # CSV per combo
    csv_rows: List[Dict] = []
    for combo in sorted(combo_ls.keys()):
        cls = combo_ls[combo]
        n = cls["n_steps"]
        csv_rows.append({
            "combo_tag": combo,
            "n_steps_with_linesearch": n,
            "mean_backtracks": cls["total_backtracks"] / max(n, 1),
            "frac_perfect_0bt": cls["n_perfect"] / max(n, 1),
            "frac_mild_1to3bt": cls["n_mild"] / max(n, 1),
            "frac_aggressive_4plus": cls["n_aggressive"] / max(n, 1),
            "frac_rejected": cls["n_rejected"] / max(n, 1),
            "mean_alpha": _safe_mean(cls["alphas"]),
            "mean_energy_decrease": _safe_mean(cls["energy_decreases"]),
        })
    _write_csv(output_dir / "linesearch_stats.csv", csv_rows)

    # Text report
    total_steps = sum(c["n_steps"] for c in combo_ls.values())
    total_perfect = sum(c["n_perfect"] for c in combo_ls.values())
    total_mild = sum(c["n_mild"] for c in combo_ls.values())
    total_agg = sum(c["n_aggressive"] for c in combo_ls.values())
    total_rej = sum(c["n_rejected"] for c in combo_ls.values())

    lines.append(f"  Total steps with linesearch: {total_steps}")
    lines.append(
        f"    Perfect (0 backtracks):   {total_perfect} "
        f"({100 * total_perfect / max(total_steps, 1):.1f}%)"
    )
    lines.append(
        f"    Mild (1-3 backtracks):    {total_mild} "
        f"({100 * total_mild / max(total_steps, 1):.1f}%)"
    )
    lines.append(
        f"    Aggressive (4+):          {total_agg} "
        f"({100 * total_agg / max(total_steps, 1):.1f}%)"
    )
    lines.append(
        f"    Rejected (accepted=False): {total_rej} "
        f"({100 * total_rej / max(total_steps, 1):.1f}%)"
    )
    lines.append(
        f"  Alpha distribution: "
        f"mean={_fmt(_safe_mean(all_alphas))}, "
        f"median={_fmt(_safe_median(all_alphas))}"
    )
    lines.append(
        f"  Energy decrease:    "
        f"mean={_fmt(_safe_mean(all_energy_dec))}, "
        f"median={_fmt(_safe_median(all_energy_dec))}"
    )

    # Correlation n_backtracks vs cond_full
    all_bt_cf: List[Tuple[float, float]] = []
    for entry in spdn_entries:
        for s in entry["traj_data"].get("trajectory", []):
            ls = s.get("spdn_linesearch", {})
            sc = s.get("spectral_conditioning", {})
            nb = ls.get("n_backtracks")
            cf = sc.get("cond_full")
            if nb is not None and cf is not None:
                try:
                    if math.isfinite(float(nb)) and math.isfinite(float(cf)):
                        all_bt_cf.append((float(nb), float(cf)))
                except (TypeError, ValueError):
                    pass
    if len(all_bt_cf) > 5:
        bt_arr = np.array([x[0] for x in all_bt_cf])
        cf_arr = np.array([x[1] for x in all_bt_cf])
        corr = float(np.corrcoef(bt_arr, cf_arr)[0, 1])
        lines.append(
            f"  Correlation n_backtracks vs cond_full: {corr:.3f} "
            f"(n={len(all_bt_cf)})"
        )

    # Plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (0,0) n_backtracks histogram
        ax = axes[0, 0]
        if all_n_backtracks:
            max_bt = max(all_n_backtracks)
            bins = np.arange(-0.5, max_bt + 1.5, 1)
            ax.hist(
                all_n_backtracks, bins=bins, color="steelblue",
                alpha=0.7, edgecolor="black", linewidth=0.5,
            )
            ax.set_xlabel("Number of backtracks")
            ax.set_ylabel("Count")
            ax.set_title("Backtrack Count Distribution")
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes,
            )

        # (0,1) alpha distribution (log x)
        ax = axes[0, 1]
        if all_alphas:
            pos_alphas = [a for a in all_alphas if a > 0]
            if pos_alphas:
                bins = np.logspace(
                    np.log10(min(pos_alphas)),
                    np.log10(max(max(pos_alphas), min(pos_alphas) * 2)),
                    40,
                )
                ax.hist(pos_alphas, bins=bins, color="steelblue", alpha=0.7)
                ax.set_xscale("log")
            ax.set_xlabel("Final alpha (step size)")
            ax.set_ylabel("Count")
            ax.set_title("Step Size (alpha) Distribution")
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes,
            )

        # (1,0) n_backtracks vs n_hard_neg scatter
        ax = axes[1, 0]
        if scatter_bt_hn:
            bt_v = [x[0] for x in scatter_bt_hn]
            hn_v = [x[1] for x in scatter_bt_hn]
            ax.scatter(hn_v, bt_v, alpha=0.15, s=8, color="steelblue")
            ax.set_xlabel("n_hard_neg")
            ax.set_ylabel("n_backtracks")
            ax.set_title("Backtracks vs Hard Negative Count")
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes,
            )

        # (1,1) energy decrease histogram
        ax = axes[1, 1]
        if all_energy_dec:
            ax.hist(all_energy_dec, bins=50, color="steelblue", alpha=0.7)
            ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
            ax.set_xlabel("Energy change per step (E_after - E_before)")
            ax.set_ylabel("Count")
            ax.set_title("Energy Change per Line Search Step")
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes,
            )

        plt.tight_layout()
        fig.savefig(
            output_dir / "linesearch_analysis.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        lines.append("  Plot saved: linesearch_analysis.png")
    except Exception as exc:
        lines.append(f"  WARNING: plotting failed for D4: {exc}")

    lines.append("  CSV saved:  linesearch_stats.csv")
    return "\n".join(lines)


# ===================================================================
# D5: Saddle Point Detection Diagnostics (DIAGNOSTIC ONLY)
# ===================================================================

def analyze_d5_saddle_diagnostics(
    all_entries: List[Dict[str, Any]],
    output_dir: Path,
) -> str:
    """Saddle point detection diagnostics for SPDN trajectories."""
    lines: List[str] = []
    lines.append("Section D5: Saddle Point Detection (Diagnostic)")
    lines.append("-" * 50)

    spdn_entries = [e for e in all_entries if _is_spdn(e["traj_data"])]
    if not spdn_entries:
        lines.append("  No SPDN trajectories found. Skipping D5.")
        return "\n".join(lines)

    total_steps_all = 0
    total_saddle_steps = 0
    csv_rows: List[Dict] = []

    saddle_force_norms: List[float] = []
    saddle_n_neg_reliable: List[float] = []
    saddle_bottom_spectrum_neg: List[float] = []
    saddle_dist_reactant: List[float] = []
    saddle_dist_product: List[float] = []

    persistence_lengths: List[int] = []
    conv_saddle_relative_steps: List[float] = []

    for entry in spdn_entries:
        td = entry["traj_data"]
        traj = td.get("trajectory", [])
        if not traj:
            continue
        converged = _is_converged(td)
        sample_id = entry["sample_id"]
        combo_tag = entry["combo_tag"]
        n_steps = len(traj)

        total_steps_all += n_steps
        saddle_steps_this = 0
        consecutive_run = 0
        saddle_step_indices: List[int] = []
        local_persistence: List[int] = []

        for i, s in enumerate(traj):
            sd = s.get("saddle_diagnostic", {})
            is_saddle = sd.get("is_saddle_like", False)
            if is_saddle:
                saddle_steps_this += 1
                consecutive_run += 1
                saddle_step_indices.append(i)

                fn = sd.get(
                    "force_norm", s.get("force_norm", float("nan")),
                )
                try:
                    if math.isfinite(float(fn)):
                        saddle_force_norms.append(float(fn))
                except (TypeError, ValueError):
                    pass
                nr = sd.get("n_neg_reliable", float("nan"))
                try:
                    if math.isfinite(float(nr)):
                        saddle_n_neg_reliable.append(float(nr))
                except (TypeError, ValueError):
                    pass
                bs = s.get("bottom_spectrum", [])
                for ev in bs:
                    try:
                        if float(ev) < 0:
                            saddle_bottom_spectrum_neg.append(float(ev))
                    except (TypeError, ValueError):
                        pass
                dr = s.get("dist_to_reactant_rmsd", float("nan"))
                try:
                    if math.isfinite(float(dr)):
                        saddle_dist_reactant.append(float(dr))
                except (TypeError, ValueError):
                    pass
                dp = s.get("dist_to_product_rmsd", float("nan"))
                try:
                    if math.isfinite(float(dp)):
                        saddle_dist_product.append(float(dp))
                except (TypeError, ValueError):
                    pass
            else:
                if consecutive_run > 0:
                    persistence_lengths.append(consecutive_run)
                    local_persistence.append(consecutive_run)
                consecutive_run = 0

        if consecutive_run > 0:
            persistence_lengths.append(consecutive_run)
            local_persistence.append(consecutive_run)

        total_saddle_steps += saddle_steps_this

        conv_step = td.get("converged_step", n_steps)
        if converged and saddle_step_indices:
            for si in saddle_step_indices:
                conv_saddle_relative_steps.append(
                    si / max(conv_step, 1),
                )

        csv_rows.append({
            "sample_id": sample_id,
            "combo_tag": combo_tag,
            "converged": converged,
            "n_steps": n_steps,
            "n_saddle_like_steps": saddle_steps_this,
            "frac_saddle_like": saddle_steps_this / max(n_steps, 1),
            "n_persistence_runs": len(local_persistence),
            "max_persistence": (
                max(local_persistence) if local_persistence else 0
            ),
            "first_saddle_step": (
                saddle_step_indices[0] if saddle_step_indices else -1
            ),
            "last_saddle_step": (
                saddle_step_indices[-1] if saddle_step_indices else -1
            ),
        })

    _write_csv(output_dir / "saddle_diagnostics.csv", csv_rows)

    # Text report
    frac = total_saddle_steps / max(total_steps_all, 1)
    lines.append(f"  Total steps analysed:     {total_steps_all}")
    lines.append(
        f"  Saddle-like steps:        {total_saddle_steps} "
        f"({100 * frac:.2f}%)"
    )
    lines.append("")

    if saddle_force_norms:
        lines.append(
            f"  Saddle-like force norm:   "
            f"mean={_fmt(_safe_mean(saddle_force_norms))}, "
            f"median={_fmt(_safe_median(saddle_force_norms))}"
        )
    if saddle_n_neg_reliable:
        lines.append(
            f"  Saddle-like n_neg_reliable: "
            f"mean={_fmt(_safe_mean(saddle_n_neg_reliable))}"
        )
    if saddle_bottom_spectrum_neg:
        lines.append(
            f"  Negative evals at saddle-like steps: "
            f"mean={_fmt(_safe_mean(saddle_bottom_spectrum_neg))}, "
            f"min={_fmt(min(saddle_bottom_spectrum_neg))}, "
            f"max={_fmt(max(saddle_bottom_spectrum_neg))}"
        )

    lines.append("")
    lines.append("  Persistence (consecutive saddle-like steps):")
    if persistence_lengths:
        lines.append(f"    Number of runs:  {len(persistence_lengths)}")
        lines.append(
            f"    Mean length:     "
            f"{_fmt(_safe_mean(persistence_lengths))}"
        )
        lines.append(f"    Max length:      {max(persistence_lengths)}")
        lines.append(
            f"    Median length:   "
            f"{_fmt(_safe_median(persistence_lengths))}"
        )
        short = sum(1 for p in persistence_lengths if p <= 3)
        long_ = sum(1 for p in persistence_lengths if p > 10)
        lines.append(
            f"    Short (<=3 steps): {short}  |  "
            f"Long (>10 steps): {long_}"
        )
    else:
        lines.append("    No saddle-like persistence runs detected.")

    lines.append("")
    n_conv_with_saddle = sum(
        1 for r in csv_rows
        if r["converged"] and r["n_saddle_like_steps"] > 0
    )
    n_conv_total = sum(1 for r in csv_rows if r["converged"])
    lines.append(
        f"  Converged trajectories through saddle-like regions: "
        f"{n_conv_with_saddle}/{n_conv_total}"
    )
    if conv_saddle_relative_steps:
        lines.append(
            f"    Relative step positions: "
            f"mean={_fmt(_safe_mean(conv_saddle_relative_steps))}, "
            f"median={_fmt(_safe_median(conv_saddle_relative_steps))}"
        )

    if saddle_dist_reactant or saddle_dist_product:
        lines.append("")
        lines.append("  Geometry context at saddle-like steps:")
        if saddle_dist_reactant:
            lines.append(
                f"    dist_to_reactant_rmsd: "
                f"mean={_fmt(_safe_mean(saddle_dist_reactant))}, "
                f"median={_fmt(_safe_median(saddle_dist_reactant))}"
            )
        if saddle_dist_product:
            lines.append(
                f"    dist_to_product_rmsd:  "
                f"mean={_fmt(_safe_mean(saddle_dist_product))}, "
                f"median={_fmt(_safe_median(saddle_dist_product))}"
            )

    lines.append("  CSV saved: saddle_diagnostics.csv")
    return "\n".join(lines)


# ===================================================================
# D6: Walking Out of Minima Analysis
# ===================================================================

def analyze_d6_walking_out_of_minima(
    all_entries: List[Dict[str, Any]],
    output_dir: Path,
) -> str:
    """For CONVERGED trajectories, trace the PES landscape from the minimum
    back toward the starting geometry.
    """
    lines: List[str] = []
    lines.append("Section D6: Walking Out of Minima")
    lines.append("-" * 50)

    conv_entries = [
        e for e in all_entries if _is_converged(e["traj_data"])
    ]
    if not conv_entries:
        lines.append("  No converged trajectories found. Skipping D6.")
        return "\n".join(lines)

    csv_rows: List[Dict] = []

    # For plotting
    all_min_eval_vs_dist: List[Tuple[np.ndarray, np.ndarray]] = []
    all_force_vs_dist: List[Tuple[np.ndarray, np.ndarray]] = []
    all_cond_vs_dist: List[Tuple[np.ndarray, np.ndarray]] = []
    all_bands_vs_dist: List[Tuple[np.ndarray, np.ndarray]] = []

    basin_radii: List[float] = []

    for entry in conv_entries:
        td = entry["traj_data"]
        traj = td.get("trajectory", [])
        if not traj:
            continue

        sample_id = entry["sample_id"]
        combo_tag = entry["combo_tag"]
        n_steps = len(traj)

        # Distance from minimum proxy: |disp_from_start[i] - disp_from_start[final]|
        disp_from_start = _col(traj, "disp_from_start_rmsd")
        final_disp = (
            disp_from_start[-1]
            if len(disp_from_start) > 0
            else float("nan")
        )
        dist_from_min = np.abs(disp_from_start - final_disp)

        min_vib_eval = _col(traj, "min_vib_eval")
        force_norm = _col(traj, "force_norm")
        energy = _col(traj, "energy")
        n_neg = _col(traj, "n_neg_evals")
        cond_num = _col(traj, "cond_num")

        bands = np.array(
            [_band_populations(s) for s in traj], dtype=float,
        )

        # Find the step closest to minimum where n_neg > 0
        # (search from end toward start)
        dist_at_first_neg = float("nan")
        force_at_first_neg = float("nan")
        eval_at_first_neg = float("nan")
        for i in range(n_steps - 1, -1, -1):
            nn_val = n_neg[i]
            if math.isfinite(nn_val) and nn_val > 0:
                dist_at_first_neg = dist_from_min[i]
                force_at_first_neg = force_norm[i]
                eval_at_first_neg = min_vib_eval[i]
                break

        if math.isfinite(dist_at_first_neg):
            basin_radii.append(dist_at_first_neg)

        energy_at_conv = (
            energy[-1] if len(energy) > 0 else float("nan")
        )

        csv_rows.append({
            "sample_id": sample_id,
            "combo_tag": combo_tag,
            "n_steps_to_converge": n_steps,
            "dist_at_first_neg_eval": dist_at_first_neg,
            "force_at_first_neg_eval": force_at_first_neg,
            "eval_at_first_neg_eval": eval_at_first_neg,
            "energy_at_convergence": energy_at_conv,
            "basin_radius_estimate": dist_at_first_neg,
        })

        # Collect for plotting
        mask = np.isfinite(dist_from_min)
        if mask.any():
            d = dist_from_min[mask]
            all_min_eval_vs_dist.append((d, min_vib_eval[mask]))
            all_force_vs_dist.append((d, force_norm[mask]))
            all_cond_vs_dist.append((d, cond_num[mask]))
            if bands.shape[0] == len(dist_from_min):
                all_bands_vs_dist.append((d, bands[mask]))

    _write_csv(output_dir / "walking_out_of_minima.csv", csv_rows)

    # Text report
    lines.append(
        f"  Converged trajectories analysed: {len(conv_entries)}"
    )
    valid_basins = [b for b in basin_radii if math.isfinite(b)]
    if valid_basins:
        lines.append(
            "  Basin radius estimate "
            "(dist where first neg eval appears):"
        )
        lines.append(
            f"    mean={_fmt(_safe_mean(valid_basins))} A, "
            f"median={_fmt(_safe_median(valid_basins))} A, "
            f"min={_fmt(min(valid_basins))}, "
            f"max={_fmt(max(valid_basins))}"
        )
    else:
        lines.append(
            "  No basin radius estimates "
            "(all converged without negative evals)."
        )

    lines.append("")
    lines.append("  Per-sample convergence stats:")
    for row in csv_rows[:10]:
        lines.append(
            f"    {row['sample_id']}: "
            f"basin_r={_fmt(row['basin_radius_estimate'])}, "
            f"E_conv={_fmt(row['energy_at_convergence'])}, "
            f"n_steps={row['n_steps_to_converge']}"
        )
    if len(csv_rows) > 10:
        lines.append(f"    ... ({len(csv_rows) - 10} more)")

    # Plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (0,0) Min eigenvalue vs distance from min
        ax = axes[0, 0]
        for dist, eval_arr in all_min_eval_vs_dist:
            sort_idx = np.argsort(dist)
            ax.plot(
                dist[sort_idx], eval_arr[sort_idx], alpha=0.15,
                linewidth=0.5, color="steelblue",
            )
        if all_min_eval_vs_dist:
            all_d = np.concatenate([d for d, _ in all_min_eval_vs_dist])
            all_e = np.concatenate([e for _, e in all_min_eval_vs_dist])
            centers, means = _binned_mean(all_d, all_e)
            if centers:
                ax.plot(
                    centers, means, color="black",
                    linewidth=2, label="Mean",
                )
                ax.legend(fontsize=8)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Distance from minimum (RMSD, A)")
        ax.set_ylabel("min_vib_eval")
        ax.set_title("Eigenvalue Profile from Minimum")

        # (0,1) Force norm vs distance from min
        ax = axes[0, 1]
        for dist, fn_arr in all_force_vs_dist:
            sort_idx = np.argsort(dist)
            ax.plot(
                dist[sort_idx], fn_arr[sort_idx], alpha=0.15,
                linewidth=0.5, color="steelblue",
            )
        if all_force_vs_dist:
            all_d = np.concatenate([d for d, _ in all_force_vs_dist])
            all_fn = np.concatenate([e for _, e in all_force_vs_dist])
            centers, means = _binned_mean(all_d, all_fn)
            if centers:
                ax.plot(
                    centers, means, color="black",
                    linewidth=2, label="Mean",
                )
                ax.legend(fontsize=8)
        ax.set_xlabel("Distance from minimum (RMSD, A)")
        ax.set_ylabel("force_norm")
        ax.set_title("Force Norm vs Distance from Minimum")

        # (1,0) Band population stacked area (averaged)
        ax = axes[1, 0]
        if all_bands_vs_dist:
            all_d = np.concatenate([d for d, _ in all_bands_vs_dist])
            all_b = np.vstack([b for _, b in all_bands_vs_dist])
            d_max = np.nanmax(all_d) if len(all_d) > 0 else 0.0
            if math.isfinite(d_max) and d_max > 0 and len(all_d) > 0:
                n_bins = 40
                bins = np.linspace(0, d_max, n_bins + 1)
                bin_idx = np.digitize(all_d, bins)
                band_means = np.zeros((n_bins, 8))
                for b in range(1, n_bins + 1):
                    mask_b = bin_idx == b
                    if mask_b.any():
                        band_means[b - 1] = np.nanmean(
                            all_b[mask_b], axis=0,
                        )
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                colors = [
                    "#d32f2f", "#e57373", "#ffb74d", "#fff176",
                    "#aed581", "#81d4fa", "#64b5f6", "#1565c0",
                ]
                ax.stackplot(
                    bin_centers, band_means.T,
                    labels=BAND_LABELS, colors=colors, alpha=0.8,
                )
                ax.set_xlabel("Distance from minimum (RMSD, A)")
                ax.set_ylabel("Mean eigenvalue count")
                ax.set_title(
                    "Band Population vs Distance from Minimum"
                )
                ax.legend(fontsize=5, loc="upper left", ncol=2)
            else:
                ax.text(
                    0.5, 0.5, "Insufficient data", ha="center",
                    va="center", transform=ax.transAxes,
                )
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes,
            )

        # (1,1) Condition number vs distance from min
        ax = axes[1, 1]
        for dist, cn_arr in all_cond_vs_dist:
            sort_idx = np.argsort(dist)
            ax.plot(
                dist[sort_idx], cn_arr[sort_idx], alpha=0.15,
                linewidth=0.5, color="steelblue",
            )
        if all_cond_vs_dist:
            all_d = np.concatenate([d for d, _ in all_cond_vs_dist])
            all_cn = np.concatenate([e for _, e in all_cond_vs_dist])
            centers, means = _binned_mean(all_d, all_cn)
            if centers:
                ax.plot(
                    centers, means, color="black",
                    linewidth=2, label="Mean",
                )
                ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("Distance from minimum (RMSD, A)")
        ax.set_ylabel("Condition number")
        ax.set_title("Condition Number vs Distance from Minimum")

        plt.tight_layout()
        fig.savefig(
            output_dir / "walking_out_of_minima.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        lines.append("  Plot saved: walking_out_of_minima.png")
    except Exception as exc:
        lines.append(f"  WARNING: plotting failed for D6: {exc}")

    lines.append("  CSV saved:  walking_out_of_minima.csv")
    return "\n".join(lines)


# ===================================================================
# D7: Eigenvalue Stability & Change Rate Analysis
# ===================================================================

def analyze_d7_eigenvalue_stability(
    all_entries: List[Dict[str, Any]],
    output_dir: Path,
) -> str:
    """Eigenvalue stability and change rate for ALL trajectories."""
    lines: List[str] = []
    lines.append("Section D7: Eigenvalue Stability & Change Rates")
    lines.append("-" * 50)

    if not all_entries:
        lines.append("  No trajectories found. Skipping D7.")
        return "\n".join(lines)

    csv_rows: List[Dict] = []

    conv_change_rate: List[np.ndarray] = []
    fail_change_rate: List[np.ndarray] = []
    all_scatter_eval_force: List[Tuple[float, float, bool]] = []
    conv_eval_ratios: List[float] = []
    fail_eval_ratios: List[float] = []
    conv_band_trans: List[np.ndarray] = []
    fail_band_trans: List[np.ndarray] = []

    for entry in all_entries:
        td = entry["traj_data"]
        traj = td.get("trajectory", [])
        if len(traj) < 2:
            continue

        converged = _is_converged(td)
        sample_id = entry["sample_id"]
        combo_tag = entry["combo_tag"]
        n_steps = len(traj)

        # --- Step-to-step eigenvalue change ---
        eval_changes: List[float] = []
        max_eval_change = 0.0

        for i in range(1, n_steps):
            bs_prev = traj[i - 1].get("bottom_spectrum", [])
            bs_curr = traj[i].get("bottom_spectrum", [])
            if bs_prev and bs_curr:
                n_common = min(len(bs_prev), len(bs_curr))
                for j in range(n_common):
                    try:
                        change = abs(float(bs_curr[j]) - float(bs_prev[j]))
                        eval_changes.append(change)
                        if change > max_eval_change:
                            max_eval_change = change
                    except (TypeError, ValueError):
                        pass

        mean_eval_change = (
            _safe_mean(eval_changes) if eval_changes else float("nan")
        )

        # --- Autocorrelation of min_vib_eval for oscillation detection ---
        min_eval_signal = _col(traj, "min_vib_eval")
        finite_signal = _finite(min_eval_signal)
        oscillation_period = float("nan")
        if len(finite_signal) > 20:
            try:
                centered = finite_signal - np.mean(finite_signal)
                norm = float(np.sum(centered ** 2))
                if norm > 1e-30:
                    acf = np.correlate(centered, centered, mode="full")
                    acf = acf[len(acf) // 2:]  # positive lags
                    acf = acf / (norm + 1e-30)
                    crossed_zero = False
                    for k in range(1, len(acf)):
                        if acf[k] <= 0:
                            crossed_zero = True
                        elif crossed_zero and acf[k] > acf[k - 1]:
                            oscillation_period = float(k)
                            break
            except Exception:
                pass

        # --- Band transition rates ---
        band_transitions_per_step: List[float] = []
        for i in range(1, n_steps):
            bp_prev = _band_populations(traj[i - 1])
            bp_curr = _band_populations(traj[i])
            trans = sum(abs(bp_curr[j] - bp_prev[j]) for j in range(8))
            band_transitions_per_step.append(float(trans))

        mean_band_trans = _safe_mean(band_transitions_per_step)
        bt_array = np.array(band_transitions_per_step, dtype=float)

        # --- Per-step change rate (mean across bottom spectrum) ---
        per_step_changes: List[float] = []
        for i in range(1, n_steps):
            bs_prev = traj[i - 1].get("bottom_spectrum", [])
            bs_curr = traj[i].get("bottom_spectrum", [])
            if bs_prev and bs_curr:
                n_common = min(len(bs_prev), len(bs_curr))
                step_ch = []
                for j in range(n_common):
                    try:
                        step_ch.append(
                            abs(float(bs_curr[j]) - float(bs_prev[j]))
                        )
                    except (TypeError, ValueError):
                        pass
                per_step_changes.append(
                    _safe_mean(step_ch) if step_ch else float("nan")
                )
            else:
                per_step_changes.append(float("nan"))

        change_rate_arr = np.array(per_step_changes, dtype=float)
        if len(change_rate_arr) > 0:
            interp_cr = _interp_to_norm_grid(change_rate_arr)
            (conv_change_rate if converged else fail_change_rate).append(
                interp_cr
            )

        if len(bt_array) > 0:
            interp_bt = _interp_to_norm_grid(bt_array)
            (conv_band_trans if converged else fail_band_trans).append(
                interp_bt
            )

        # --- |min_vib_eval| vs force_norm scatter ---
        for s in traj:
            mve = s.get("min_vib_eval", float("nan"))
            fn = s.get("force_norm", float("nan"))
            try:
                mve_f = float(mve)
                fn_f = float(fn)
                if math.isfinite(mve_f) and math.isfinite(fn_f):
                    all_scatter_eval_force.append(
                        (abs(mve_f), fn_f, converged)
                    )
            except (TypeError, ValueError):
                pass

        # --- Eigenvalue dominance ratio |l1|/|l2| ---
        for s in traj:
            bs = s.get("bottom_spectrum", [])
            if len(bs) >= 2:
                neg_evals = sorted(
                    [float(e) for e in bs
                     if isinstance(e, (int, float)) and float(e) < 0]
                )
                if len(neg_evals) >= 2:
                    l1 = abs(neg_evals[0])
                    l2 = abs(neg_evals[1])
                    if l2 > 1e-30:
                        ratio = l1 / l2
                        (
                            conv_eval_ratios if converged
                            else fail_eval_ratios
                        ).append(ratio)

        csv_rows.append({
            "sample_id": sample_id,
            "combo_tag": combo_tag,
            "converged": converged,
            "mean_eval_change": mean_eval_change,
            "max_eval_change": max_eval_change,
            "oscillation_period": oscillation_period,
            "mean_band_transitions_per_step": mean_band_trans,
        })

    _write_csv(output_dir / "eigenvalue_stability.csv", csv_rows)

    # Text report
    n_conv = sum(1 for r in csv_rows if r["converged"])
    n_fail = sum(1 for r in csv_rows if not r["converged"])
    lines.append(
        f"  Trajectories analysed: {len(csv_rows)} "
        f"(converged: {n_conv}, failed: {n_fail})"
    )

    conv_ch = [
        r["mean_eval_change"] for r in csv_rows if r["converged"]
    ]
    fail_ch = [
        r["mean_eval_change"] for r in csv_rows if not r["converged"]
    ]
    lines.append(
        f"  Mean eval change (converged): "
        f"{_fmt(_safe_mean(conv_ch))}"
    )
    lines.append(
        f"  Mean eval change (failed):    "
        f"{_fmt(_safe_mean(fail_ch))}"
    )

    conv_osc = [
        r["oscillation_period"] for r in csv_rows
        if r["converged"] and math.isfinite(r["oscillation_period"])
    ]
    fail_osc = [
        r["oscillation_period"] for r in csv_rows
        if not r["converged"] and math.isfinite(r["oscillation_period"])
    ]
    lines.append(
        f"  Oscillation period (converged): detected in "
        f"{len(conv_osc)} trajs, mean={_fmt(_safe_mean(conv_osc))}"
    )
    lines.append(
        f"  Oscillation period (failed):    detected in "
        f"{len(fail_osc)} trajs, mean={_fmt(_safe_mean(fail_osc))}"
    )

    conv_bt = [
        r["mean_band_transitions_per_step"]
        for r in csv_rows if r["converged"]
    ]
    fail_bt = [
        r["mean_band_transitions_per_step"]
        for r in csv_rows if not r["converged"]
    ]
    lines.append(
        f"  Mean band transitions/step (converged): "
        f"{_fmt(_safe_mean(conv_bt))}"
    )
    lines.append(
        f"  Mean band transitions/step (failed):    "
        f"{_fmt(_safe_mean(fail_bt))}"
    )

    if conv_eval_ratios or fail_eval_ratios:
        lines.append(
            f"  Eigenvalue ratio |l1|/|l2| (converged): "
            f"mean={_fmt(_safe_mean(conv_eval_ratios))}, "
            f"median={_fmt(_safe_median(conv_eval_ratios))}"
        )
        lines.append(
            f"  Eigenvalue ratio |l1|/|l2| (failed):    "
            f"mean={_fmt(_safe_mean(fail_eval_ratios))}, "
            f"median={_fmt(_safe_median(fail_eval_ratios))}"
        )

    # Plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        x_norm = np.linspace(0.0, 1.0, N_INTERP_PTS)

        # (0,0) Eigenvalue change rate
        ax = axes[0, 0]
        if conv_change_rate:
            med, q25, q75 = _median_iqr(np.array(conv_change_rate))
            ax.plot(x_norm, med, color="green", label="Converged")
            ax.fill_between(x_norm, q25, q75, color="green", alpha=0.2)
        if fail_change_rate:
            med, q25, q75 = _median_iqr(np.array(fail_change_rate))
            ax.plot(x_norm, med, color="red", label="Failed")
            ax.fill_between(x_norm, q25, q75, color="red", alpha=0.2)
        ax.set_xlabel("Normalized step")
        ax.set_ylabel("Mean eigenvalue change")
        ax.set_title("Eigenvalue Change Rate")
        ax.legend(fontsize=8)

        # (0,1) |min_vib_eval| vs force_norm scatter
        ax = axes[0, 1]
        if all_scatter_eval_force:
            data = all_scatter_eval_force
            if len(data) > 50000:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(data), 50000, replace=False)
                data = [data[int(i)] for i in idx]
            conv_pts = [(e, f) for e, f, c in data if c]
            fail_pts = [(e, f) for e, f, c in data if not c]
            if fail_pts:
                ax.scatter(
                    [p[0] for p in fail_pts],
                    [p[1] for p in fail_pts],
                    alpha=0.05, s=4, color="red", label="Failed",
                )
            if conv_pts:
                ax.scatter(
                    [p[0] for p in conv_pts],
                    [p[1] for p in conv_pts],
                    alpha=0.05, s=4, color="green", label="Converged",
                )
            ax.set_xlabel("|min_vib_eval|")
            ax.set_ylabel("force_norm")
            ax.set_title("|min eigenvalue| vs Force Norm")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize=8, markerscale=5)
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes,
            )

        # (1,0) Eigenvalue ratio distribution
        ax = axes[1, 0]
        combined_ratios = conv_eval_ratios + fail_eval_ratios
        upper = min(20.0, max(combined_ratios)) if combined_ratios else 20.0
        bins = np.linspace(0, upper, 50)
        if conv_eval_ratios:
            clipped = [min(r, upper) for r in conv_eval_ratios]
            ax.hist(
                clipped, bins=bins, alpha=0.6, color="green",
                label="Converged", density=True,
            )
        if fail_eval_ratios:
            clipped = [min(r, upper) for r in fail_eval_ratios]
            ax.hist(
                clipped, bins=bins, alpha=0.6, color="red",
                label="Failed", density=True,
            )
        ax.axvline(1.0, color="black", linestyle="--", linewidth=0.5)
        ax.set_xlabel("|lambda_1| / |lambda_2|")
        ax.set_ylabel("Density")
        ax.set_title("Eigenvalue Dominance Ratio")
        ax.legend(fontsize=8)

        # (1,1) Band transition rate
        ax = axes[1, 1]
        if conv_band_trans:
            med, q25, q75 = _median_iqr(np.array(conv_band_trans))
            ax.plot(x_norm, med, color="green", label="Converged")
            ax.fill_between(x_norm, q25, q75, color="green", alpha=0.2)
        if fail_band_trans:
            med, q25, q75 = _median_iqr(np.array(fail_band_trans))
            ax.plot(x_norm, med, color="red", label="Failed")
            ax.fill_between(x_norm, q25, q75, color="red", alpha=0.2)
        ax.set_xlabel("Normalized step")
        ax.set_ylabel("Band transitions per step")
        ax.set_title("Band Transition Rate")
        ax.legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(
            output_dir / "eigenvalue_stability.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        lines.append("  Plot saved: eigenvalue_stability.png")
    except Exception as exc:
        lines.append(f"  WARNING: plotting failed for D7: {exc}")

    lines.append("  CSV saved:  eigenvalue_stability.csv")
    return "\n".join(lines)


# ===================================================================
# Summary of Key Findings
# ===================================================================

def generate_summary(
    all_entries: List[Dict[str, Any]],
) -> str:
    """Produce a summary block of quantitative findings."""
    lines: List[str] = []
    lines.append("=" * 64)
    lines.append("SUMMARY OF KEY FINDINGS")
    lines.append("=" * 64)

    n_total = len(all_entries)
    if n_total == 0:
        lines.append("  No trajectories loaded.")
        return "\n".join(lines)

    n_spdn = sum(1 for e in all_entries if _is_spdn(e["traj_data"]))
    n_legacy = n_total - n_spdn
    n_conv = sum(
        1 for e in all_entries if _is_converged(e["traj_data"])
    )

    lines.append(
        f"  - Total trajectories: {n_total} "
        f"(SPDN: {n_spdn}, Legacy: {n_legacy})"
    )
    lines.append(
        f"  - Overall convergence: {n_conv}/{n_total} "
        f"({100 * n_conv / max(n_total, 1):.1f}%)"
    )

    # SPDN convergence
    spdn_entries = [
        e for e in all_entries if _is_spdn(e["traj_data"])
    ]
    if spdn_entries:
        sc = sum(1 for e in spdn_entries if _is_converged(e["traj_data"]))
        lines.append(
            f"  - SPDN convergence: {sc}/{len(spdn_entries)} "
            f"({100 * sc / max(len(spdn_entries), 1):.1f}%)"
        )

    # Legacy convergence
    legacy_entries = [
        e for e in all_entries if not _is_spdn(e["traj_data"])
    ]
    if legacy_entries:
        lc = sum(
            1 for e in legacy_entries if _is_converged(e["traj_data"])
        )
        lines.append(
            f"  - Legacy convergence: {lc}/{len(legacy_entries)} "
            f"({100 * lc / max(len(legacy_entries), 1):.1f}%)"
        )

    # Steps to converge
    conv_entries = [
        e for e in all_entries if _is_converged(e["traj_data"])
    ]
    if conv_entries:
        steps = [
            e["traj_data"].get(
                "converged_step",
                e["traj_data"].get(
                    "total_steps",
                    len(e["traj_data"].get("trajectory", [])),
                ),
            )
            for e in conv_entries
        ]
        lines.append(
            f"  - Mean steps to converge: {_fmt(_safe_mean(steps))}"
        )

    # GDIIS summary
    total_diis_att = sum(
        e["traj_data"].get("total_diis_attempts", 0)
        for e in spdn_entries
    )
    total_diis_acc = sum(
        e["traj_data"].get("total_diis_accepts", 0)
        for e in spdn_entries
    )
    if total_diis_att > 0:
        lines.append(
            f"  - GDIIS accept rate: {total_diis_acc}/{total_diis_att} "
            f"({100 * total_diis_acc / max(total_diis_att, 1):.1f}%)"
        )

    # Final force norms
    conv_fn = [
        e["traj_data"].get("final_force_norm", float("nan"))
        for e in conv_entries
    ]
    valid_fn = [
        v for v in conv_fn
        if v is not None and math.isfinite(float(v))
    ]
    if valid_fn:
        lines.append(
            f"  - Final force norm (converged): "
            f"mean={_fmt(_safe_mean(valid_fn))}, "
            f"median={_fmt(_safe_median(valid_fn))}"
        )

    fail_entries = [
        e for e in all_entries if not _is_converged(e["traj_data"])
    ]
    fail_fn = [
        e["traj_data"].get("final_force_norm", float("nan"))
        for e in fail_entries
    ]
    valid_fail_fn = [
        v for v in fail_fn
        if v is not None and math.isfinite(float(v))
    ]
    if valid_fail_fn:
        lines.append(
            f"  - Final force norm (failed): "
            f"mean={_fmt(_safe_mean(valid_fail_fn))}, "
            f"median={_fmt(_safe_median(valid_fail_fn))}"
        )

    combos = set(e["combo_tag"] for e in all_entries)
    lines.append(f"  - Number of parameter combos: {len(combos)}")

    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Comprehensive diagnostic analysis for NR minimization "
            "with SPDN mode."
        ),
    )
    parser.add_argument(
        "--grid-dir", required=True,
        help="Grid search output directory.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write analysis outputs.",
    )
    parser.add_argument(
        "--traj-glob",
        default="*/diagnostics/*_trajectory.json",
        help="Glob for trajectory files relative to --grid-dir.",
    )
    parser.add_argument(
        "--combo-tag", default=None,
        help="Filter to a single combo tag (optional).",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of per-sample detailed plots.",
    )
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Scanning {grid_dir} for trajectory files "
        f"(glob: {args.traj_glob})..."
    )
    all_entries = load_all_trajectories(
        grid_dir, args.traj_glob, args.combo_tag,
    )
    if not all_entries:
        print(
            "No trajectory files found. "
            "Check --grid-dir and --traj-glob."
        )
        return

    n_combos = len(set(e["combo_tag"] for e in all_entries))
    n_spdn = sum(1 for e in all_entries if _is_spdn(e["traj_data"]))
    n_legacy = len(all_entries) - n_spdn
    print(
        f"Loaded {len(all_entries)} trajectories across "
        f"{n_combos} combos "
        f"(SPDN: {n_spdn}, Legacy: {n_legacy}).\n"
    )

    # ==== Run all analysis modules ====
    print("=" * 64)
    print("SPDN DIAGNOSTIC ANALYSIS REPORT")
    print("=" * 64)
    print()

    # D1
    d1 = analyze_d1_spdn_vs_legacy(all_entries)
    print(d1)
    print()

    # D2
    d2 = analyze_d2_spectral_conditioning(all_entries, output_dir)
    print(d2)
    print()

    # D3
    d3 = analyze_d3_gdiis_performance(all_entries, output_dir)
    print(d3)
    print()

    # D4
    d4 = analyze_d4_linesearch(all_entries, output_dir)
    print(d4)
    print()

    # D5
    d5 = analyze_d5_saddle_diagnostics(all_entries, output_dir)
    print(d5)
    print()

    # D6
    d6 = analyze_d6_walking_out_of_minima(all_entries, output_dir)
    print(d6)
    print()

    # D7
    d7 = analyze_d7_eigenvalue_stability(all_entries, output_dir)
    print(d7)
    print()

    # Summary
    summary = generate_summary(all_entries)
    print(summary)
    print()

    print(f"All outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
