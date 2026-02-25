#!/usr/bin/env python3
"""Trajectory-level statistics for GAD grid runs.

Reads every *_trajectory.json under a grid directory and computes per-trajectory
and per-combo aggregate statistics on the quantities that are also plotted:

  1.  Trust radius (step_size_eff)     — evolution, growth/shrink rate
  2.  Actual step displacement (x_disp_step)
  3.  Eigenvalues eig_0, eig_1         — lowest two vibrational modes
  4.  Eigenvalue gap (eig_gap_01)      — |λ₂ - λ₁|, singularity proximity
  5.  Eigenvalue ratio |eig_1|/|eig_0| — eccentricity of the saddle
  6.  Condition number                 — |eig_max_abs| / |eig_min_abs| (vib)
  7.  Morse index trajectory           — time at index 0, 1, >1
  8.  Mode tracking overlap            — ⟨v₁(t) | v₁(t-1)⟩, mode-flip events
  9.  Energy delta                     — sign, magnitude, uphill-step fraction
  10. TR modes filtered (n_tr_modes)   — how many modes are discarded per step
  11. Gradient norm (grad_norm)        — convergence diagnostic
  12. GAD direction quality            — grad_proj_v1, gad_grad_angle

Expected trajectory JSON format (TrajectoryLogger — column-oriented):
  {field_name: [value_at_step_0, value_at_step_1, ...], ...}

Fields used from ExtendedMetrics.to_dict():
  step, eig_0, eig_1, eig_2..eig_5, eig_gap_01, eig_gap_01_rel, eig_gap_12,
  morse_index, singularity_metric, step_size_eff, x_disp_step, x_disp_window,
  mode_overlap, v1_v2_overlap, grad_norm, energy_delta, n_tr_modes,
  tr_eig_max, tr_eig_mean, grad_proj_v1, gad_grad_angle
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


COMBO_RE = re.compile(
    r"mad(?P<mad>[^_]+)_tr(?P<tr>[^_]+)_bl(?P<bl>.+)_pg(?P<pg>true|false)$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(data: Dict, key: str) -> np.ndarray:
    """Pull a column from the column-oriented trajectory dict."""
    vals = data.get(key, [])
    return np.array(vals, dtype=float) if vals else np.array([], dtype=float)


def _finite(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


def _gmean(arr: np.ndarray) -> float:
    pos = arr[arr > 0]
    return float(np.exp(np.mean(np.log(pos)))) if len(pos) > 0 else float("nan")


def _stat(arr: np.ndarray) -> Dict[str, float]:
    f = _finite(arr)
    if len(f) == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean":   float(np.mean(f)),
        "median": float(np.median(f)),
        "std":    float(np.std(f)),
        "min":    float(np.min(f)),
        "max":    float(np.max(f)),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


# ---------------------------------------------------------------------------
# Per-trajectory statistics
# ---------------------------------------------------------------------------

def compute_trajectory_stats(data: Dict, converged: bool) -> Dict[str, Any]:
    n = len(data.get("step", []))
    if n == 0:
        return {}

    trust     = _col(data, "step_size_eff")
    disp      = _col(data, "x_disp_step")
    eig0      = _col(data, "eig_0")
    eig1      = _col(data, "eig_1")
    eig_gap   = _col(data, "eig_gap_01")
    eig_gap_r = _col(data, "eig_gap_01_rel")
    morse     = _col(data, "morse_index")
    overlap   = _col(data, "mode_overlap")
    v1v2_ov   = _col(data, "v1_v2_overlap")
    grad_norm = _col(data, "grad_norm")
    edelta    = _col(data, "energy_delta")
    n_tr      = _col(data, "n_tr_modes")
    gp_v1     = _col(data, "grad_proj_v1")
    gad_angle = _col(data, "gad_grad_angle")
    sing      = _col(data, "singularity_metric")

    # -- Trust radius --
    tr_s      = _stat(trust)
    tr_growth = int(np.sum(np.diff(_finite(trust)) > 0)) if len(_finite(trust)) > 1 else 0
    tr_shrink = int(np.sum(np.diff(_finite(trust)) < 0)) if len(_finite(trust)) > 1 else 0
    tr_final  = float(trust[-1]) if len(trust) > 0 and np.isfinite(trust[-1]) else float("nan")
    # Fraction of steps hitting the cap (disp ≥ 98% of trust radius)
    hit_mask  = (np.isfinite(trust) & np.isfinite(disp) & (trust > 0) & (disp >= 0.98 * trust))
    hit_frac  = float(np.mean(hit_mask)) if len(hit_mask) > 0 else float("nan")

    # -- Actual displacement --
    disp_s = _stat(disp)
    disp_trust_ratio = _stat(disp[np.isfinite(trust) & (trust > 0)] / trust[np.isfinite(trust) & (trust > 0)])

    # -- Eigenvalues --
    eig0_s = _stat(eig0)
    eig1_s = _stat(eig1)
    # Fraction of steps where eig0 < 0 (correct saddle signature for eig0)
    neg0_frac = float(np.mean(eig0 < 0)) if len(_finite(eig0)) > 0 else float("nan")
    # Eccentricity: |eig1| / |eig0| — how separated the saddle mode is from next mode
    with np.errstate(divide="ignore", invalid="ignore"):
        eccent = np.abs(eig1) / np.maximum(np.abs(eig0), 1e-12)
    eccent_s = _stat(eccent[np.isfinite(eig0) & (np.abs(eig0) > 1e-10)])

    # -- Eigenvalue gap (singularity proximity) --
    gap_s    = _stat(eig_gap)
    gap_r_s  = _stat(eig_gap_r)
    # Fraction of steps "near singularity" at various thresholds
    near_sing_01  = float(np.mean(_finite(eig_gap) < 0.01))  if len(_finite(eig_gap)) > 0 else float("nan")
    near_sing_10  = float(np.mean(_finite(eig_gap) < 0.10))  if len(_finite(eig_gap)) > 0 else float("nan")
    near_sing_50  = float(np.mean(_finite(eig_gap) < 0.50))  if len(_finite(eig_gap)) > 0 else float("nan")

    # -- Condition number of vibrational Hessian: singularity_metric is min adjacent gap;
    #    also compute as |eig_max_abs| / |eig_min_abs| from eig_0 and eig_5 if available --
    eig5 = _col(data, "eig_5")
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_num = np.abs(eig5) / np.maximum(np.abs(eig0), 1e-12)
    cond_s  = _stat(cond_num[np.isfinite(cond_num) & (cond_num < 1e8)])
    cond_gm = _gmean(_finite(cond_num[cond_num > 0]))

    # -- Morse index trajectory --
    mo = _finite(morse)
    frac_index0   = float(np.mean(mo == 0)) if len(mo) > 0 else float("nan")
    frac_index1   = float(np.mean(mo == 1)) if len(mo) > 0 else float("nan")
    frac_index_gt1= float(np.mean(mo > 1))  if len(mo) > 0 else float("nan")
    first_index1  = next((i for i, v in enumerate(mo) if v == 1), None)
    # Steps from first index-1 to convergence (if converged)
    if first_index1 is not None and converged:
        steps_in_index1 = n - first_index1
    else:
        steps_in_index1 = None

    # -- Mode tracking overlap --
    ov_s    = _stat(overlap)
    # Mode flip events: overlap drops below 0.5 (sudden direction change)
    ov_flip = float(np.mean(_finite(overlap) < 0.5)) if len(_finite(overlap)) > 0 else float("nan")
    # Auto-correlation of overlap at lag-1 (does it oscillate?)
    ov_f = _finite(overlap)
    if len(ov_f) > 3:
        ov_autocorr = float(np.corrcoef(ov_f[:-1], ov_f[1:])[0, 1])
    else:
        ov_autocorr = float("nan")
    # v1-v2 swap detection
    v1v2_s     = _stat(v1v2_ov)
    v1v2_high  = float(np.mean(_finite(v1v2_ov) > 0.5)) if len(_finite(v1v2_ov)) > 0 else float("nan")

    # -- Energy delta --
    ed = _finite(edelta)
    ed_abs = np.abs(ed)
    edelta_mean     = float(np.mean(ed))      if len(ed) > 0 else float("nan")
    edelta_abs_mean = float(np.mean(ed_abs))  if len(ed_abs) > 0 else float("nan")
    edelta_pos_frac = float(np.mean(ed > 0))  if len(ed) > 0 else float("nan")  # uphill steps
    edelta_std      = float(np.std(ed))       if len(ed) > 0 else float("nan")

    # -- TR modes filtered --
    tr_modes_s   = _stat(n_tr)
    tr_excess    = float(np.mean(_finite(n_tr) > 6)) if len(_finite(n_tr)) > 0 else float("nan")

    # -- Gradient norm --
    gn_s    = _stat(grad_norm)
    gn_gm   = _gmean(_finite(grad_norm))
    gn_final= float(_finite(grad_norm)[-1]) if len(_finite(grad_norm)) > 0 else float("nan")

    # -- GAD direction quality --
    gp1_s  = _stat(gp_v1)    # grad_proj_v1: frac of gradient along v1
    angle_s = _stat(gad_angle)

    # -- Correlations between plotted quantities --
    corr_trust_disp  = _corr(trust, disp)
    corr_gap_overlap = _corr(eig_gap, overlap)
    corr_gap_retry   = float("nan")  # retries not in GAD logger

    return {
        "n_steps":  n,
        "converged": converged,
        # Trust radius
        "tr_mean":            tr_s["mean"],
        "tr_median":          tr_s["median"],
        "tr_std":             tr_s["std"],
        "tr_max":             tr_s["max"],
        "tr_final":           tr_final,
        "tr_growth_steps":    tr_growth,
        "tr_shrink_steps":    tr_shrink,
        "tr_hit_frac":        hit_frac,
        # Displacement
        "disp_mean":          disp_s["mean"],
        "disp_max":           disp_s["max"],
        "disp_trust_ratio":   disp_trust_ratio["mean"],
        # Eigenvalues
        "eig0_mean":          eig0_s["mean"],
        "eig0_final":         float(_finite(eig0)[-1]) if len(_finite(eig0)) > 0 else float("nan"),
        "eig1_mean":          eig1_s["mean"],
        "eig0_neg_frac":      neg0_frac,
        "eccentricity_mean":  eccent_s["mean"],
        "eccentricity_gmean": _gmean(_finite(eccent)),
        # Eigenvalue gap
        "gap_mean":           gap_s["mean"],
        "gap_min":            gap_s["min"],
        "gap_std":            gap_s["std"],
        "gap_rel_mean":       gap_r_s["mean"],
        "near_sing_frac_001": near_sing_01,
        "near_sing_frac_010": near_sing_10,
        "near_sing_frac_050": near_sing_50,
        # Condition number
        "cond_gmean":         cond_gm,
        "cond_mean":          cond_s["mean"],
        "cond_max":           cond_s["max"],
        # Morse index
        "morse_frac_0":       frac_index0,
        "morse_frac_1":       frac_index1,
        "morse_frac_gt1":     frac_index_gt1,
        "morse_first_index1": first_index1,
        "morse_steps_in_index1": steps_in_index1,
        # Mode overlap
        "overlap_mean":       ov_s["mean"],
        "overlap_min":        ov_s["min"],
        "overlap_std":        ov_s["std"],
        "overlap_flip_frac":  ov_flip,
        "overlap_autocorr":   ov_autocorr,
        "v1v2_swap_frac":     v1v2_high,
        # Energy delta
        "edelta_mean":        edelta_mean,
        "edelta_abs_mean":    edelta_abs_mean,
        "edelta_pos_frac":    edelta_pos_frac,
        "edelta_std":         edelta_std,
        # TR modes filtered
        "n_tr_modes_mean":    tr_modes_s["mean"],
        "n_tr_modes_max":     tr_modes_s["max"],
        "n_tr_excess_frac":   tr_excess,
        # Gradient norm
        "grad_norm_gmean":    gn_gm,
        "grad_norm_final":    gn_final,
        # GAD direction quality
        "grad_proj_v1_mean":  gp1_s["mean"],
        "gad_angle_mean":     angle_s["mean"],
        # Correlations
        "corr_trust_disp":    corr_trust_disp,
        "corr_gap_overlap":   corr_gap_overlap,
    }


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------

def load_all_trajectories(grid_dir: Path, traj_glob: str) -> List[Dict[str, Any]]:
    rows = []
    for traj_path in sorted(grid_dir.rglob(traj_glob)):
        if not traj_path.is_file():
            continue

        parts = traj_path.relative_to(grid_dir).parts
        combo_tag = parts[0] if len(parts) >= 1 else "unknown"
        match = COMBO_RE.fullmatch(combo_tag)
        if not match:
            continue

        with open(traj_path) as f:
            data = json.load(f)

        # Column-oriented: data IS the trajectory dict
        if "step" not in data:
            continue

        sample_id = traj_path.stem.replace("_trajectory", "")
        # Infer convergence: any step where eig0 < 0 and eig1 > 0
        eig0 = np.array(data.get("eig_0", []), dtype=float)
        eig1 = np.array(data.get("eig_1", []), dtype=float)
        converged = bool(np.any((eig0 < 0) & (eig1 > 0)))

        stats = compute_trajectory_stats(data, converged)
        stats.update({
            "combo_tag":    combo_tag,
            "sample_id":    sample_id,
            "baseline":     match.group("bl"),
            "max_atom_disp": float(match.group("mad")),
            "tr_threshold":  float(match.group("tr")),
            "project_gradient_and_v": match.group("pg") == "true",
        })
        rows.append(stats)

    return rows


def aggregate_by_combo(rows: List[Dict]) -> Dict[str, Dict]:
    skip = {"combo_tag", "sample_id", "baseline", "max_atom_disp", "tr_threshold",
            "project_gradient_and_v", "converged"}
    numeric_keys = [k for k in rows[0] if k not in skip]

    by_combo: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_combo[row["combo_tag"]].append(row)

    result = {}
    for tag, samples in by_combo.items():
        agg: Dict[str, Any] = {
            "combo_tag":     tag,
            "baseline":      samples[0]["baseline"],
            "max_atom_disp": samples[0]["max_atom_disp"],
            "tr_threshold":  samples[0]["tr_threshold"],
            "n_trajectories": len(samples),
            "n_converged":   sum(1 for s in samples if s.get("converged")),
        }
        for key in numeric_keys:
            vals = [s[key] for s in samples
                    if key in s and s[key] is not None
                    and isinstance(s[key], (int, float)) and math.isfinite(float(s[key]))]
            agg[f"{key}__mean"] = float(np.mean(vals)) if vals else float("nan")
            agg[f"{key}__std"]  = float(np.std(vals))  if len(vals) > 1 else float("nan")
        result[tag] = agg
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

SECTIONS = [
    ("Trust Radius",           ["tr_mean__mean", "tr_max__mean", "tr_final__mean",
                                 "tr_hit_frac__mean", "tr_growth_steps__mean", "tr_shrink_steps__mean"]),
    ("Eigenvalue Gap (Sing.)", ["gap_mean__mean", "gap_min__mean", "gap_std__mean",
                                 "near_sing_frac_001__mean", "near_sing_frac_010__mean",
                                 "near_sing_frac_050__mean"]),
    ("Saddle Eccentricity",    ["eccentricity_gmean__mean", "eig0_mean__mean", "eig1_mean__mean",
                                 "eig0_neg_frac__mean"]),
    ("Condition Number",       ["cond_gmean__mean", "cond_mean__mean", "cond_max__mean"]),
    ("Morse Index",            ["morse_frac_0__mean", "morse_frac_1__mean", "morse_frac_gt1__mean",
                                 "morse_first_index1__mean", "morse_steps_in_index1__mean"]),
    ("Mode Tracking",          ["overlap_mean__mean", "overlap_min__mean",
                                 "overlap_flip_frac__mean", "overlap_autocorr__mean",
                                 "v1v2_swap_frac__mean"]),
    ("Energy Delta",           ["edelta_mean__mean", "edelta_abs_mean__mean",
                                 "edelta_pos_frac__mean", "edelta_std__mean"]),
    ("TR Modes Filtered",      ["n_tr_modes_mean__mean", "n_tr_modes_max__mean",
                                 "n_tr_excess_frac__mean"]),
    ("Gradient / GAD Quality", ["grad_norm_gmean__mean", "grad_norm_final__mean",
                                 "grad_proj_v1_mean__mean", "gad_angle_mean__mean"]),
    ("Correlations",           ["corr_trust_disp__mean", "corr_gap_overlap__mean"]),
]


def print_report(agg: Dict[str, Dict], top_k: int = 10) -> None:
    combos = sorted(agg.values(), key=lambda x: (-x["n_converged"], x["combo_tag"]))

    print("=" * 72)
    print("GAD Trajectory Statistics  (aggregated over samples per combo)")
    print("=" * 72)

    # ---- Ranked summary table ----
    print(f"\n{'Config':<48} {'Conv':>5} {'TR_mean':>8} {'Gap_min':>8} "
          f"{'Overlap':>8} {'Flips':>7} {'MorseT1':>8}")
    print("-" * 96)
    for c in combos[:top_k * 2]:
        tag  = c["combo_tag"][:47]
        nc   = c["n_converged"]
        nt   = c["n_trajectories"]
        trm  = c.get("tr_mean__mean", float("nan"))
        gm   = c.get("gap_min__mean", float("nan"))
        ovm  = c.get("overlap_mean__mean", float("nan"))
        flp  = c.get("overlap_flip_frac__mean", float("nan"))
        mi1  = c.get("morse_first_index1__mean", float("nan"))
        print(f"  {tag:<46} {nc}/{nt:>3}  "
              f"{trm:>8.3g}  {gm:>8.3g}  {ovm:>8.3f}  {flp:>7.2%}  {mi1:>8.1f}")

    print("\n" + "=" * 72)
    print("Detailed stats — top configurations")
    print("=" * 72)
    for c in combos[:top_k]:
        nc = c["n_converged"]
        nt = c["n_trajectories"]
        print(f"\n--- {c['combo_tag']}  ({nc}/{nt} converged, "
              f"baseline={c['baseline']}) ---")
        for section, keys in SECTIONS:
            vals_str = []
            for k in keys:
                v = c.get(k, float("nan"))
                if isinstance(v, float) and math.isfinite(v):
                    short = k.replace("__mean", "").replace("_", " ")
                    vals_str.append(f"{short}={v:.3g}")
            if vals_str:
                print(f"  [{section}]  " + ",  ".join(vals_str))

    # ---- Mode tracking: plain vs mode_tracked comparison ----
    print("\n" + "=" * 72)
    print("Mode tracking effect: plain vs mode_tracked")
    print(f"{'Baseline':<16} {'N':>5} {'Conv%':>7} {'Overlap':>9} {'Flips':>8} "
          f"{'Gap_min':>9} {'Steps@1':>9}")
    print("-" * 60)
    by_bl: Dict[str, List[Dict]] = defaultdict(list)
    for c in agg.values():
        by_bl[c["baseline"]].append(c)
    for bl, items in sorted(by_bl.items()):
        nc   = sum(i["n_converged"]   for i in items)
        nt   = sum(i["n_trajectories"] for i in items)
        ovm  = np.nanmean([i.get("overlap_mean__mean", float("nan")) for i in items])
        flp  = np.nanmean([i.get("overlap_flip_frac__mean", float("nan")) for i in items])
        gm   = np.nanmean([i.get("gap_min__mean", float("nan")) for i in items])
        mi1  = np.nanmean([i.get("morse_first_index1__mean", float("nan")) for i in items])
        print(f"  {bl:<14} {nt:>5} {nc/nt:>7.1%} {ovm:>9.3f} {flp:>8.2%} "
              f"{gm:>9.3g} {mi1:>9.1f}")


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    all_keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-trajectory statistics for GAD grid runs"
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

    print(f"Scanning {grid_dir} for GAD trajectory files...")
    rows = load_all_trajectories(grid_dir, args.traj_glob)
    if not rows:
        print("No trajectory files found. Check --traj-glob or --grid-dir.")
        return
    print(f"Loaded {len(rows)} trajectories across "
          f"{len({r['combo_tag'] for r in rows})} combos.\n")

    agg = aggregate_by_combo(rows)
    print_report(agg, top_k=args.top_k)

    write_csv(out_dir / "gad_traj_stats_per_sample.csv", rows)
    combo_rows = sorted(agg.values(), key=lambda x: (-x["n_converged"], x["combo_tag"]))
    write_csv(out_dir / "gad_traj_stats_per_combo.csv", combo_rows)

    with open(out_dir / "gad_traj_stats_summary.json", "w") as f:
        json.dump({"n_trajectories": len(rows), "combos": combo_rows}, f, indent=2)

    print(f"\nOutput written to: {out_dir}")


if __name__ == "__main__":
    main()
