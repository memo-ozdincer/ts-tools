#!/usr/bin/env python3
"""Trajectory-level statistics for Newton-Raphson minimization grid runs.

Reads every *_trajectory.json under a grid directory and computes per-trajectory
and per-combo aggregate statistics on the quantities that are also plotted:

  1. Trust radius evolution       (trust_radius per step)
  2. Actual step displacement     (actual_step_disp per step)
  3. Hit-trust-radius rate        (hit_trust_radius flag)
  4. Step retries                 (retries per step)
  5. Condition number             (cond_num = |lambda_max| / |lambda_min|)
  6. Force norm decay             (force_norm per step)
  7. Negative vibrational modes   (n_neg_evals over time)

Per-trajectory stats are written to a CSV; per-combo aggregates are written to
JSON and printed as a report.

Expected trajectory JSON format (from run_minimization_parallel.py):
  {
    "sample_id": "sample_000",
    "method": "newton_raphson",
    "trajectory": [
      {"step": 0, "trust_radius": ..., "actual_step_disp": ...,
       "hit_trust_radius": ..., "retries": ..., "cond_num": ...,
       "force_norm": ..., "n_neg_evals": ..., ...},
      ...
    ]
  }
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


COMBO_RE = re.compile(
    r"mad(?P<mad>[^_]+)_tr(?P<tr>[^_]+)_pg(?P<pg>true|false)_ph(?P<ph>true|false)$"
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _col(steps: List[Dict], key: str) -> np.ndarray:
    return np.array([s.get(key, float("nan")) for s in steps], dtype=float)


def _finite(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


def _gmean(arr: np.ndarray) -> float:
    """Geometric mean (for log-normal quantities like condition number)."""
    pos = arr[arr > 0]
    if len(pos) == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(pos))))


def _stat(arr: np.ndarray) -> Dict[str, float]:
    f = _finite(arr)
    if len(f) == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean":   float(np.mean(f)),
        "median": float(np.median(f)),
        "std":    float(np.std(f)),
        "min":    float(np.min(f)),
        "max":    float(np.max(f)),
        "n":      int(len(f)),
    }


def _mean_safe(vals: List[float]) -> float:
    f = [v for v in vals if math.isfinite(v)]
    return float(np.mean(f)) if f else float("nan")


# ---------------------------------------------------------------------------
# Per-trajectory statistics
# ---------------------------------------------------------------------------

def compute_trajectory_stats(traj: List[Dict], converged: bool) -> Dict[str, Any]:
    n = len(traj)
    if n == 0:
        return {}

    trust   = _col(traj, "trust_radius")
    actual  = _col(traj, "actual_step_disp")
    hit     = np.array([bool(s.get("hit_trust_radius", False)) for s in traj])
    retries = _col(traj, "retries")
    cond    = _col(traj, "cond_num")
    fnorm   = _col(traj, "force_norm")
    n_neg   = _col(traj, "n_neg_evals")

    # --- Trust radius ---
    tr_s = _stat(trust)
    tr_growth  = int(np.sum(np.diff(_finite(trust)) > 0)) if len(_finite(trust)) > 1 else 0
    tr_shrink  = int(np.sum(np.diff(_finite(trust)) < 0)) if len(_finite(trust)) > 1 else 0
    tr_final   = float(trust[-1]) if np.isfinite(trust[-1]) else float("nan")
    hit_frac   = float(np.mean(hit)) if len(hit) > 0 else float("nan")

    # --- Displacement ---
    disp_s = _stat(actual)
    # Correlation between trust radius cap and actual step (when both finite)
    mask = np.isfinite(trust) & np.isfinite(actual) & (trust > 0)
    if mask.sum() > 2:
        ratio = actual[mask] / trust[mask]
        disp_trust_ratio = _stat(ratio)
    else:
        disp_trust_ratio = {"mean": float("nan")}

    # --- Retries ---
    ret_s = _stat(retries)
    ret_nonzero_frac = float(np.mean(retries[np.isfinite(retries)] > 0)) if np.any(np.isfinite(retries)) else float("nan")
    ret_dist = {
        "0": int(np.sum(retries == 0)),
        "1": int(np.sum(retries == 1)),
        "2": int(np.sum(retries == 2)),
        "3+": int(np.sum(retries >= 3)),
    }

    # --- Condition number ---
    cond_s    = _stat(cond)
    cond_gm   = _gmean(_finite(cond))
    # Correlation with retries
    mask_cr = np.isfinite(cond) & np.isfinite(retries)
    if mask_cr.sum() > 2:
        cond_retry_corr = float(np.corrcoef(cond[mask_cr], retries[mask_cr])[0, 1])
    else:
        cond_retry_corr = float("nan")

    # --- Force norm decay ---
    fn = _finite(fnorm)
    fn0 = float(fn[0]) if len(fn) > 0 else float("nan")
    fn_final = float(fn[-1]) if len(fn) > 0 else float("nan")
    fn_decay = fn_final / fn0 if fn0 > 0 else float("nan")
    fn_monotone_frac = float(np.mean(np.diff(fn) < 0)) if len(fn) > 1 else float("nan")
    # Half-life: first step where fnorm < 0.5 * fn0
    if math.isfinite(fn0) and len(fn) > 0:
        halflife_steps = next((i for i, v in enumerate(fn) if v < 0.5 * fn0), None)
    else:
        halflife_steps = None

    # --- Negative vibrational modes ---
    nn = _finite(n_neg)
    n_neg_initial = float(nn[0]) if len(nn) > 0 else float("nan")
    n_neg_final   = float(nn[-1]) if len(nn) > 0 else float("nan")
    # Steps at n_neg == 0 (already at minimum signature)
    frac_at_zero = float(np.mean(nn == 0)) if len(nn) > 0 else float("nan")
    # First step where n_neg == 0
    first_zero = next((i for i, v in enumerate(nn) if v == 0), None)

    return {
        "n_steps": n,
        "converged": converged,
        # Trust radius
        "tr_mean":       tr_s["mean"],
        "tr_median":     tr_s["median"],
        "tr_std":        tr_s["std"],
        "tr_min":        tr_s["min"],
        "tr_max":        tr_s["max"],
        "tr_final":      tr_final,
        "tr_growth_steps": tr_growth,
        "tr_shrink_steps": tr_shrink,
        "tr_hit_frac":   hit_frac,
        # Actual displacement
        "disp_mean":     disp_s["mean"],
        "disp_max":      disp_s["max"],
        "disp_trust_ratio_mean": disp_trust_ratio["mean"],
        # Retries
        "retry_mean":    ret_s["mean"],
        "retry_max":     ret_s["max"],
        "retry_nonzero_frac": ret_nonzero_frac,
        "retry_dist_0":  ret_dist["0"],
        "retry_dist_1":  ret_dist["1"],
        "retry_dist_2":  ret_dist["2"],
        "retry_dist_3+": ret_dist["3+"],
        # Condition number
        "cond_mean":     cond_s["mean"],
        "cond_gmean":    cond_gm,
        "cond_max":      cond_s["max"],
        "cond_std":      cond_s["std"],
        "cond_retry_corr": cond_retry_corr,
        # Force norm decay
        "fnorm_initial":       fn0,
        "fnorm_final":         fn_final,
        "fnorm_decay_ratio":   fn_decay,
        "fnorm_monotone_frac": fn_monotone_frac,
        "fnorm_halflife_step": halflife_steps,
        # Negative eigenvalue count
        "n_neg_initial": n_neg_initial,
        "n_neg_final":   n_neg_final,
        "n_neg_frac_zero": frac_at_zero,
        "n_neg_first_zero_step": first_zero,
    }


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------

def load_all_trajectories(grid_dir: Path, traj_glob: str) -> List[Dict[str, Any]]:
    """Load every trajectory file, attaching combo metadata."""
    rows = []
    for traj_path in sorted(grid_dir.rglob(traj_glob)):
        if not traj_path.is_file():
            continue

        # Combo tag is the direct parent of the diagnostics subfolder
        # Structure: <grid>/<combo_tag>/diagnostics/<sample>_trajectory.json
        parts = traj_path.relative_to(grid_dir).parts
        combo_tag = parts[0] if len(parts) >= 1 else "unknown"

        match = COMBO_RE.fullmatch(combo_tag)
        if not match:
            continue

        with open(traj_path) as f:
            data = json.load(f)

        traj = data.get("trajectory", [])
        sample_id = data.get("sample_id", traj_path.stem)
        converged = data.get("final_neg_vib") == 0

        stats = compute_trajectory_stats(traj, converged)
        stats.update({
            "combo_tag":            combo_tag,
            "sample_id":            sample_id,
            "max_atom_disp":        float(match.group("mad")),
            "tr_threshold":         float(match.group("tr")),
            "project_gradient_and_v": match.group("pg") == "true",
            "purify_hessian":       match.group("ph") == "true",
        })
        rows.append(stats)

    return rows


def aggregate_by_combo(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Mean (and std) of every numeric stat across samples for each combo."""
    numeric_keys = [k for k in rows[0].keys() if k not in
                    {"combo_tag", "sample_id", "max_atom_disp", "tr_threshold",
                     "project_gradient_and_v", "purify_hessian", "converged",
                     "retry_dist_0", "retry_dist_1", "retry_dist_2", "retry_dist_3+"}]

    by_combo: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_combo[row["combo_tag"]].append(row)

    result = {}
    for tag, samples in by_combo.items():
        agg: Dict[str, Any] = {
            "combo_tag": tag,
            "n_trajectories": len(samples),
            "n_converged": sum(1 for s in samples if s.get("converged")),
            "max_atom_disp": samples[0]["max_atom_disp"],
            "tr_threshold":  samples[0]["tr_threshold"],
        }
        for key in numeric_keys:
            vals = [s[key] for s in samples
                    if key in s and s[key] is not None and math.isfinite(float(s[key]))]
            agg[f"{key}__mean"] = float(np.mean(vals)) if vals else float("nan")
            agg[f"{key}__std"]  = float(np.std(vals))  if len(vals) > 1 else float("nan")

        # Aggregate retry distribution
        for rk in ["retry_dist_0", "retry_dist_1", "retry_dist_2", "retry_dist_3+"]:
            agg[f"{rk}__total"] = sum(s.get(rk, 0) for s in samples)

        result[tag] = agg
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

PRINT_KEYS = [
    ("Trust Radius",         ["tr_mean__mean", "tr_max__mean", "tr_final__mean",
                               "tr_hit_frac__mean", "tr_growth_steps__mean", "tr_shrink_steps__mean"]),
    ("Step Displacement",    ["disp_mean__mean", "disp_max__mean", "disp_trust_ratio_mean__mean"]),
    ("Retries",              ["retry_mean__mean", "retry_max__mean", "retry_nonzero_frac__mean"]),
    ("Condition Number",     ["cond_gmean__mean", "cond_mean__mean", "cond_max__mean",
                               "cond_retry_corr__mean"]),
    ("Force Norm Decay",     ["fnorm_initial__mean", "fnorm_final__mean",
                               "fnorm_decay_ratio__mean", "fnorm_monotone_frac__mean",
                               "fnorm_halflife_step__mean"]),
    ("Neg Eigenvalue Count", ["n_neg_initial__mean", "n_neg_final__mean",
                               "n_neg_frac_zero__mean", "n_neg_first_zero_step__mean"]),
]


def print_report(agg: Dict[str, Dict], top_k: int = 10) -> None:
    combos = sorted(agg.values(), key=lambda x: -x["n_converged"])

    print("=" * 70)
    print("Newton-Raphson Trajectory Statistics  (aggregated over samples)")
    print("=" * 70)

    for combo in combos[:top_k]:
        tag = combo["combo_tag"]
        nc  = combo["n_converged"]
        nt  = combo["n_trajectories"]
        print(f"\n--- {tag}  ({nc}/{nt} converged) ---")
        for section, keys in PRINT_KEYS:
            vals_str = []
            for k in keys:
                v = combo.get(k, float("nan"))
                if math.isfinite(v):
                    short = k.replace("__mean", "").replace("_", " ")
                    vals_str.append(f"{short}={v:.3g}")
            if vals_str:
                print(f"  [{section}]  " + ",  ".join(vals_str))

    print("\n" + "=" * 70)
    print("Retry distribution totals (all combos, all samples)")
    print("=" * 70)
    for combo in combos:
        r0  = combo.get("retry_dist_0__total", 0)
        r1  = combo.get("retry_dist_1__total", 0)
        r2  = combo.get("retry_dist_2__total", 0)
        r3p = combo.get("retry_dist_3+__total", 0)
        total = r0 + r1 + r2 + r3p
        if total == 0:
            continue
        print(f"  {combo['combo_tag']}: 0={r0}({r0/total:.0%})  "
              f"1={r1}({r1/total:.0%})  2={r2}({r2/total:.0%})  "
              f"3+={r3p}({r3p/total:.0%})")


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = [k for k in rows[0].keys() if k != "combo_tag"]
    keys = ["combo_tag"] + keys
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-trajectory statistics for NR minimization grid runs"
    )
    parser.add_argument("--grid-dir",   required=True,
                        help="Grid directory containing combo subdirectories")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for CSV/JSON output (default: <grid-dir>/traj_stats)")
    parser.add_argument("--traj-glob",  default="*/diagnostics/*_trajectory.json",
                        help="Glob for trajectory files relative to --grid-dir")
    parser.add_argument("--top-k", type=int, default=10,
                        help="How many combos to print in detail")
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    out_dir  = Path(args.output_dir) if args.output_dir else grid_dir / "traj_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {grid_dir} for trajectory files...")
    rows = load_all_trajectories(grid_dir, args.traj_glob)
    if not rows:
        print("No trajectory files found. Check --traj-glob or --grid-dir.")
        return
    print(f"Loaded {len(rows)} trajectories across "
          f"{len({r['combo_tag'] for r in rows})} combos.\n")

    agg = aggregate_by_combo(rows)
    print_report(agg, top_k=args.top_k)

    # Per-trajectory CSV
    write_csv(out_dir / "nr_traj_stats_per_sample.csv", rows)

    # Per-combo CSV
    combo_rows = sorted(agg.values(), key=lambda x: -x["n_converged"])
    write_csv(out_dir / "nr_traj_stats_per_combo.csv", combo_rows)

    # JSON summary
    with open(out_dir / "nr_traj_stats_summary.json", "w") as f:
        json.dump({"n_trajectories": len(rows), "combos": combo_rows}, f, indent=2)

    print(f"\nOutput written to: {out_dir}")


if __name__ == "__main__":
    main()
