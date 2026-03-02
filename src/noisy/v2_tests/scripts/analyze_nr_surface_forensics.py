#!/usr/bin/env python3
"""Surface / failure forensics for Newton-Raphson minimization trajectories.

Reads trajectory JSONs from grid-search output directories and performs deep
analysis of the potential energy surface behaviour and failure modes.

Seven analysis modules:
  F1  Trust-radius <-> eigenvalue correlation
  F2  Eigenvalue band transition matrices
  F3  Stagnation anatomy
  F4  Oscillation cycle detection (FFT-based)
  F5  Distance evolution comparison (converged vs failed)
  F6  Ghost-mode characterisation
  F7  Trust-crushed population analysis

Usage:
  python analyze_nr_surface_forensics.py \
      --grid-dir <path> --output-dir <path> \
      [--traj-glob "*/diagnostics/*_trajectory.json"] \
      [--combo-tag <tag>]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BAND_LABELS = [
    "below_neg1e-1",
    "neg1e-1_to_neg1e-2",
    "neg1e-2_to_neg1e-3",
    "neg1e-3_to_neg1e-4",
    "neg1e-4_to_0",
    "0_to_pos1e-4",
    "pos1e-4_to_pos1e-3",
    "above_pos1e-3",
]
N_BANDS = len(BAND_LABELS)

MILESTONES = [0, 100, 500, 1000, 3000]  # "final" handled separately

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(val: Any, default: float = float("nan")) -> float:
    """Return *val* as float, falling back to *default*."""
    if val is None:
        return default
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _eigenvalue_to_band(ev: float) -> int:
    """Map a single eigenvalue to one of the 8 band indices (0..7)."""
    if ev < -1e-1:
        return 0
    if ev < -1e-2:
        return 1
    if ev < -1e-3:
        return 2
    if ev < -1e-4:
        return 3
    if ev < 0:
        return 4
    if ev < 1e-4:
        return 5
    if ev < 1e-3:
        return 6
    return 7


def _atoms_from_formula(formula: str) -> int:
    """Estimate number of atoms from a molecular formula string like 'CH4O2'."""
    if not formula:
        return 0
    total = 0
    for _, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        total += int(count) if count else 1
    return total


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient; returns NaN if degenerate."""
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    sx, sy = np.std(x), np.std(y)
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_trajectories(
    grid_dir: Path,
    traj_glob: str,
    combo_tag: Optional[str],
) -> List[Dict[str, Any]]:
    """Load all trajectory JSONs that match *traj_glob* under *grid_dir*."""
    files = sorted(grid_dir.glob(traj_glob))
    if combo_tag:
        files = [f for f in files if combo_tag in str(f)]
    data: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with open(fp) as fh:
                d = json.load(fh)
            d["_source_file"] = str(fp)
            data.append(d)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] Could not load {fp}: {exc}", file=sys.stderr)
    return data


def _is_converged(traj_data: Dict[str, Any]) -> bool:
    fnv = traj_data.get("final_neg_vib")
    return fnv == 0 if fnv is not None else False


# ---------------------------------------------------------------------------
# F1  Trust Radius <-> Eigenvalue Correlation
# ---------------------------------------------------------------------------


def analyze_tr_eigenvalue_correlation(
    all_trajs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """F1: per-trajectory correlation + TR collapse events."""
    rows: List[Dict[str, Any]] = []
    collapse_events: List[Dict[str, Any]] = []

    for td in all_trajs:
        traj = td.get("trajectory", [])
        sid = td.get("sample_id", "unknown")
        if len(traj) < 3:
            continue

        tr_arr = np.array([_safe(s.get("trust_radius"), np.nan) for s in traj])
        nn_arr = np.array([_safe(s.get("n_neg_evals"), np.nan) for s in traj])
        me_arr = np.array([_safe(s.get("min_vib_eval"), np.nan) for s in traj])

        valid = np.isfinite(tr_arr) & np.isfinite(nn_arr) & np.isfinite(me_arr)
        if valid.sum() < 3:
            continue

        corr_nn = _pearson(tr_arr[valid], nn_arr[valid])
        corr_me = _pearson(tr_arr[valid], me_arr[valid])

        rows.append({
            "sample_id": sid,
            "converged": _is_converged(td),
            "n_steps": len(traj),
            "pearson_tr_vs_nneg": round(corr_nn, 5),
            "pearson_tr_vs_min_eval": round(corr_me, 5),
            "mean_tr": round(float(np.nanmean(tr_arr)), 6),
            "mean_nneg": round(float(np.nanmean(nn_arr)), 3),
        })

        # TR collapse: > 50 % drop in one step
        for i in range(1, len(tr_arr)):
            if not (np.isfinite(tr_arr[i]) and np.isfinite(tr_arr[i - 1])):
                continue
            if tr_arr[i - 1] > 0 and (tr_arr[i] / tr_arr[i - 1]) < 0.5:
                window_lo = max(0, i - 3)
                window_hi = min(len(traj), i + 4)
                nn_window = nn_arr[window_lo:window_hi].tolist()
                me_window = me_arr[window_lo:window_hi].tolist()
                collapse_events.append({
                    "sample_id": sid,
                    "step": i,
                    "tr_before": float(tr_arr[i - 1]),
                    "tr_after": float(tr_arr[i]),
                    "drop_frac": round(1.0 - tr_arr[i] / tr_arr[i - 1], 4),
                    "nneg_window": nn_window,
                    "min_eval_window": me_window,
                })
    return rows, collapse_events


# ---------------------------------------------------------------------------
# F2  Eigenvalue Band Transition Matrix
# ---------------------------------------------------------------------------


def analyze_band_transitions(
    all_trajs: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """F2: build 8x8 transition count matrix and compute mean residence times."""
    counts = np.zeros((N_BANDS, N_BANDS), dtype=np.float64)
    residence_steps: Dict[int, List[int]] = defaultdict(list)

    for td in all_trajs:
        traj = td.get("trajectory", [])
        if len(traj) < 2:
            continue

        for step_idx in range(len(traj) - 1):
            spec_cur = traj[step_idx].get("bottom_spectrum", [])
            spec_nxt = traj[step_idx + 1].get("bottom_spectrum", [])
            n_common = min(len(spec_cur), len(spec_nxt))
            for k in range(n_common):
                b_cur = _eigenvalue_to_band(spec_cur[k])
                b_nxt = _eigenvalue_to_band(spec_nxt[k])
                counts[b_cur, b_nxt] += 1

        # Residence time: consecutive steps in same band per eigenvalue index
        for td2 in [td]:  # iterate once, kept for clarity
            traj2 = td2.get("trajectory", [])
            if len(traj2) < 2:
                continue
            max_modes = max(len(s.get("bottom_spectrum", [])) for s in traj2)
            for k in range(max_modes):
                cur_band: Optional[int] = None
                run_len = 0
                for s in traj2:
                    spec = s.get("bottom_spectrum", [])
                    if k >= len(spec):
                        if cur_band is not None and run_len > 0:
                            residence_steps[cur_band].append(run_len)
                        cur_band = None
                        run_len = 0
                        continue
                    b = _eigenvalue_to_band(spec[k])
                    if b == cur_band:
                        run_len += 1
                    else:
                        if cur_band is not None and run_len > 0:
                            residence_steps[cur_band].append(run_len)
                        cur_band = b
                        run_len = 1
                if cur_band is not None and run_len > 0:
                    residence_steps[cur_band].append(run_len)

    mean_res = np.zeros(N_BANDS, dtype=np.float64)
    for b in range(N_BANDS):
        if residence_steps[b]:
            mean_res[b] = float(np.mean(residence_steps[b]))
    return counts, mean_res


# ---------------------------------------------------------------------------
# F3  Stagnation Anatomy
# ---------------------------------------------------------------------------


def analyze_stagnation(
    all_trajs: List[Dict[str, Any]],
    min_window: int = 50,
) -> List[Dict[str, Any]]:
    """F3: find stagnation windows (>= min_window steps with constant n_neg)."""
    windows: List[Dict[str, Any]] = []
    for td in all_trajs:
        traj = td.get("trajectory", [])
        sid = td.get("sample_id", "unknown")
        if len(traj) < min_window:
            continue
        nn = [s.get("n_neg_evals", -1) for s in traj]
        run_start = 0
        for i in range(1, len(nn) + 1):
            if i < len(nn) and nn[i] == nn[run_start]:
                continue
            run_len = i - run_start
            if run_len >= min_window:
                seg = traj[run_start:i]
                energies = [_safe(s.get("energy")) for s in seg]
                forces = [_safe(s.get("force_norm")) for s in seg]
                evals = [_safe(s.get("min_vib_eval")) for s in seg]
                trs = [_safe(s.get("trust_radius")) for s in seg]
                disps = [_safe(s.get("actual_step_disp")) for s in seg]
                e_valid = [v for v in energies if math.isfinite(v)]
                f_valid = [v for v in forces if math.isfinite(v)]
                ev_valid = [v for v in evals if math.isfinite(v)]
                tr_valid = [v for v in trs if math.isfinite(v)]
                d_valid = [v for v in disps if math.isfinite(v)]
                windows.append({
                    "sample_id": sid,
                    "converged": _is_converged(td),
                    "n_neg_value": nn[run_start],
                    "start_step": run_start,
                    "end_step": i - 1,
                    "length": run_len,
                    "energy_range": round(max(e_valid) - min(e_valid), 8) if e_valid else float("nan"),
                    "energy_mean": round(float(np.mean(e_valid)), 6) if e_valid else float("nan"),
                    "force_norm_range": round(max(f_valid) - min(f_valid), 8) if f_valid else float("nan"),
                    "force_norm_mean": round(float(np.mean(f_valid)), 6) if f_valid else float("nan"),
                    "min_eval_range": round(max(ev_valid) - min(ev_valid), 10) if ev_valid else float("nan"),
                    "min_eval_mean": round(float(np.mean(ev_valid)), 8) if ev_valid else float("nan"),
                    "tr_range": round(max(tr_valid) - min(tr_valid), 8) if tr_valid else float("nan"),
                    "tr_mean": round(float(np.mean(tr_valid)), 6) if tr_valid else float("nan"),
                    "mean_displacement": round(float(np.mean(d_valid)), 8) if d_valid else float("nan"),
                })
            run_start = i
    return windows


# ---------------------------------------------------------------------------
# F4  Oscillation Cycle Detection (FFT-based autocorrelation)
# ---------------------------------------------------------------------------


def _detect_oscillation(
    signal: np.ndarray,
    min_period: int = 4,
    max_period: int = 500,
) -> Dict[str, Any]:
    """Return dominant period, amplitude, confidence for a 1-D signal."""
    result: Dict[str, Any] = {
        "has_cycle": False,
        "period": float("nan"),
        "amplitude": float("nan"),
        "confidence": 0.0,
    }
    sig = signal.copy()
    valid = np.isfinite(sig)
    if valid.sum() < 2 * min_period:
        return result

    # Interpolate NaNs linearly for FFT
    xp = np.where(valid)[0]
    fp = sig[valid]
    sig = np.interp(np.arange(len(sig)), xp, fp)

    sig = sig - np.mean(sig)
    if np.std(sig) < 1e-15:
        return result

    # FFT-based autocorrelation
    n = len(sig)
    fft_sig = np.fft.rfft(sig, n=2 * n)
    power = np.abs(fft_sig) ** 2
    autocorr = np.fft.irfft(power)[:n]
    autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr

    # Find peaks in autocorrelation (local maxima)
    search_lo = min_period
    search_hi = min(max_period, n // 2)
    if search_lo >= search_hi:
        return result

    seg = autocorr[search_lo:search_hi]
    if len(seg) < 3:
        return result

    peaks: List[int] = []
    for i in range(1, len(seg) - 1):
        if seg[i] > seg[i - 1] and seg[i] > seg[i + 1]:
            peaks.append(i + search_lo)

    if not peaks:
        return result

    best_lag = peaks[int(np.argmax([autocorr[p] for p in peaks]))]
    confidence = float(autocorr[best_lag])

    if confidence < 0.15:
        return result

    # Amplitude: RMS of signal
    amplitude = float(np.std(signal[valid]))

    result.update({
        "has_cycle": True,
        "period": int(best_lag),
        "amplitude": round(amplitude, 8),
        "confidence": round(confidence, 4),
    })
    return result


def analyze_oscillations(
    all_trajs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """F4: detect oscillation cycles in n_neg, min_eval, TR, energy."""
    rows: List[Dict[str, Any]] = []
    for td in all_trajs:
        traj = td.get("trajectory", [])
        sid = td.get("sample_id", "unknown")
        if len(traj) < 20:
            continue

        signals = {
            "n_neg": np.array([_safe(s.get("n_neg_evals")) for s in traj]),
            "min_eval": np.array([_safe(s.get("min_vib_eval")) for s in traj]),
            "trust_radius": np.array([_safe(s.get("trust_radius")) for s in traj]),
            "energy": np.array([_safe(s.get("energy")) for s in traj]),
        }
        row: Dict[str, Any] = {
            "sample_id": sid,
            "converged": _is_converged(td),
            "n_steps": len(traj),
        }
        any_cycle = False
        for name, sig in signals.items():
            info = _detect_oscillation(sig)
            row[f"{name}_has_cycle"] = info["has_cycle"]
            row[f"{name}_period"] = info["period"]
            row[f"{name}_amplitude"] = info["amplitude"]
            row[f"{name}_confidence"] = info["confidence"]
            if info["has_cycle"]:
                any_cycle = True
        row["any_cycle"] = any_cycle
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# F5  Distance Evolution Comparison
# ---------------------------------------------------------------------------


def analyze_distance_evolution(
    all_trajs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """F5: distance-to-reactant at milestones for converged vs failed."""
    groups: Dict[str, Dict[int, List[float]]] = {
        "converged": defaultdict(list),
        "failed": defaultdict(list),
    }
    for td in all_trajs:
        traj = td.get("trajectory", [])
        if not traj:
            continue
        label = "converged" if _is_converged(td) else "failed"
        n = len(traj)
        milestones_full = MILESTONES + [n - 1]
        for m in milestones_full:
            if m >= n:
                continue
            d = _safe(traj[m].get("dist_to_reactant_rmsd"))
            if math.isfinite(d):
                groups[label][m].append(d)

    summary: Dict[str, Any] = {}
    for label in ("converged", "failed"):
        ms: Dict[str, Any] = {}
        for m, vals in sorted(groups[label].items()):
            arr = np.array(vals)
            ms[str(m)] = {
                "mean": round(float(np.mean(arr)), 6),
                "std": round(float(np.std(arr)), 6),
                "count": len(vals),
            }
        summary[label] = ms
    return summary


# ---------------------------------------------------------------------------
# F6  Ghost-Mode Characterisation
# ---------------------------------------------------------------------------


def analyze_ghost_modes(
    all_trajs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """F6: characterise ghost modes and correlate with molecule size."""
    rows: List[Dict[str, Any]] = []
    for td in all_trajs:
        traj = td.get("trajectory", [])
        sid = td.get("sample_id", "unknown")
        formula = td.get("formula", "")
        n_atoms = _atoms_from_formula(formula)
        if not traj:
            continue

        ghost_steps = 0
        rotation_events_total = 0
        mode_continuity_mins: List[float] = []
        ghost_eigenvalues: List[float] = []

        for s in traj:
            spec = s.get("bottom_spectrum", [])
            # Ghost eigenvalue: in [-1e-4, 0) and no eigenvalue below -1e-4
            has_deep_neg = any(ev < -1e-4 for ev in spec)
            ghost_evs = [ev for ev in spec if -1e-4 <= ev < 0]
            if ghost_evs and not has_deep_neg:
                ghost_steps += 1
                ghost_eigenvalues.extend(ghost_evs)

            evc = s.get("eigenvec_continuity", {})
            mc_min = _safe(evc.get("mode_continuity_min"))
            if math.isfinite(mc_min):
                mode_continuity_mins.append(mc_min)
            rotation_events_total += int(evc.get("n_mode_rotation_events", 0))

        # Persistence: are the same modes ghost, or do they rotate?
        persist_spans: List[int] = []
        cur_run = 0
        for s in traj:
            spec = s.get("bottom_spectrum", [])
            has_deep_neg = any(ev < -1e-4 for ev in spec)
            ghost_evs = [ev for ev in spec if -1e-4 <= ev < 0]
            if ghost_evs and not has_deep_neg:
                cur_run += 1
            else:
                if cur_run > 0:
                    persist_spans.append(cur_run)
                cur_run = 0
        if cur_run > 0:
            persist_spans.append(cur_run)

        rows.append({
            "sample_id": sid,
            "converged": _is_converged(td),
            "formula": formula,
            "n_atoms": n_atoms,
            "n_steps": len(traj),
            "ghost_steps": ghost_steps,
            "ghost_frac": round(ghost_steps / max(len(traj), 1), 4),
            "total_ghost_eigenvalues": len(ghost_eigenvalues),
            "mean_ghost_eigenvalue": round(float(np.mean(ghost_eigenvalues)), 10) if ghost_eigenvalues else float("nan"),
            "mode_continuity_min_mean": round(float(np.mean(mode_continuity_mins)), 5) if mode_continuity_mins else float("nan"),
            "total_rotation_events": rotation_events_total,
            "n_persist_spans": len(persist_spans),
            "max_persist_span": max(persist_spans) if persist_spans else 0,
            "mean_persist_span": round(float(np.mean(persist_spans)), 2) if persist_spans else 0.0,
        })
    return rows


# ---------------------------------------------------------------------------
# F7  Trust-Crushed Population Analysis
# ---------------------------------------------------------------------------

TR_FLOOR = 0.015


def analyze_trust_crushed(
    all_trajs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """F7: when TR is at floor, how much step is killed by TR capping?"""
    rows: List[Dict[str, Any]] = []
    for td in all_trajs:
        traj = td.get("trajectory", [])
        sid = td.get("sample_id", "unknown")
        if not traj:
            continue

        crushed_steps = 0
        neg_fracs: List[float] = []
        overlaps_min: List[float] = []
        n_neg_at_crush: List[int] = []
        evals_at_crush: List[float] = []

        for s in traj:
            tr = _safe(s.get("trust_radius"))
            if not math.isfinite(tr) or tr >= TR_FLOOR:
                continue
            crushed_steps += 1
            nmd = s.get("neg_mode_diag", {})
            frac = _safe(nmd.get("step_along_neg_frac"))
            if math.isfinite(frac):
                neg_fracs.append(frac)
            mno = _safe(nmd.get("min_neg_grad_overlap"))
            if math.isfinite(mno):
                overlaps_min.append(mno)
            nn = s.get("n_neg_evals", 0)
            n_neg_at_crush.append(int(nn))
            mev = _safe(s.get("min_vib_eval"))
            if math.isfinite(mev):
                evals_at_crush.append(mev)

        rows.append({
            "sample_id": sid,
            "converged": _is_converged(td),
            "n_steps": len(traj),
            "crushed_steps": crushed_steps,
            "crushed_frac": round(crushed_steps / max(len(traj), 1), 4),
            "mean_step_along_neg_frac": round(float(np.mean(neg_fracs)), 8) if neg_fracs else float("nan"),
            "std_step_along_neg_frac": round(float(np.std(neg_fracs)), 8) if neg_fracs else float("nan"),
            "mean_min_neg_grad_overlap": round(float(np.mean(overlaps_min)), 8) if overlaps_min else float("nan"),
            "mean_n_neg_at_crush": round(float(np.mean(n_neg_at_crush)), 3) if n_neg_at_crush else float("nan"),
            "mean_min_eval_at_crush": round(float(np.mean(evals_at_crush)), 10) if evals_at_crush else float("nan"),
        })
    return rows


# ---------------------------------------------------------------------------
# Population classification for Plot 6
# ---------------------------------------------------------------------------


def _classify_population(td: Dict[str, Any]) -> str:
    """Return 'A', 'B', 'C', or 'converged'."""
    if _is_converged(td):
        return "converged"
    traj = td.get("trajectory", [])
    if not traj:
        return "C"
    last = traj[-1]

    # Pop A: ghost modes -- only very shallow negatives at final step
    n_shallow = last.get("n_eval_neg1e-4_to_0", 0)
    n_deep = (
        last.get("n_eval_below_neg1e-1", 0)
        + last.get("n_eval_neg1e-1_to_neg1e-2", 0)
        + last.get("n_eval_neg1e-2_to_neg1e-3", 0)
        + last.get("n_eval_neg1e-3_to_neg1e-4", 0)
    )
    if n_shallow > 0 and n_deep == 0:
        return "A"

    # Pop B: trust-crushed -- TR at floor with eigenvalues below -1e-4
    tr = _safe(last.get("trust_radius"))
    has_deep = n_deep > 0
    if math.isfinite(tr) and tr < TR_FLOOR and has_deep:
        return "B"

    # Pop C: border cases
    return "C"


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# Plot 1 - TR vs n_neg scatter
def plot_tr_vs_nneg(all_trajs: List[Dict[str, Any]], outdir: Path) -> None:
    tr_vals: List[float] = []
    nn_vals: List[float] = []
    for td in all_trajs:
        if _is_converged(td):
            continue
        traj = td.get("trajectory", [])
        for i, s in enumerate(traj):
            if i % 10 != 0:
                continue
            tr = _safe(s.get("trust_radius"))
            nn = _safe(s.get("n_neg_evals"))
            if math.isfinite(tr) and math.isfinite(nn):
                tr_vals.append(tr)
                nn_vals.append(nn)
    if not tr_vals:
        print("  [WARN] No data for tr_vs_nneg scatter.")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(nn_vals, tr_vals, alpha=0.15, s=8, edgecolors="none")
    ax.set_xlabel("n_neg_evals")
    ax.set_ylabel("trust_radius")
    ax.set_title("Trust Radius vs n_neg (failed trajectories, every 10th step)")
    _save_fig(fig, outdir / "tr_vs_nneg_scatter.png")


# Plot 2 - Band transition matrix heatmap
def plot_band_transition_matrix(counts: np.ndarray, outdir: Path) -> None:
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normed = np.where(row_sums > 0, counts / row_sums, 0.0)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(normed, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(N_BANDS))
    ax.set_yticks(range(N_BANDS))
    short_labels = [
        "<-1e-1",
        "(-1e-1,-1e-2)",
        "(-1e-2,-1e-3)",
        "(-1e-3,-1e-4)",
        "(-1e-4,0)",
        "(0,1e-4)",
        "(1e-4,1e-3)",
        ">1e-3",
    ]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("To band")
    ax.set_ylabel("From band")
    ax.set_title("Eigenvalue Band Transition Probabilities (row-normalised)")
    # Annotate cells
    for i in range(N_BANDS):
        for j in range(N_BANDS):
            val = normed[i, j]
            if val > 0.005:
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)
    fig.colorbar(im, ax=ax, label="transition probability")
    _save_fig(fig, outdir / "band_transition_matrix.png")


# Plot 3 - Stagnation anatomy boxplots
def plot_stagnation_anatomy(windows: List[Dict[str, Any]], outdir: Path) -> None:
    if not windows:
        print("  [WARN] No stagnation windows for boxplots.")
        return
    energy_ranges = [w["energy_range"] for w in windows if math.isfinite(w.get("energy_range", float("nan")))]
    force_ranges = [w["force_norm_range"] for w in windows if math.isfinite(w.get("force_norm_range", float("nan")))]
    eval_ranges = [w["min_eval_range"] for w in windows if math.isfinite(w.get("min_eval_range", float("nan")))]

    data = []
    labels = []
    if energy_ranges:
        data.append(energy_ranges)
        labels.append("Energy range")
    if force_ranges:
        data.append(force_ranges)
        labels.append("Force-norm range")
    if eval_ranges:
        data.append(eval_ranges)
        labels.append("Min-eval range")
    if not data:
        print("  [WARN] Insufficient data for stagnation boxplots.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Range during stagnation window")
    ax.set_title("Stagnation Anatomy: Variation Within Constant-n_neg Windows")
    ax.set_yscale("symlog", linthresh=1e-6)
    _save_fig(fig, outdir / "stagnation_anatomy_boxplots.png")


# Plot 4 - Oscillation period histogram
def plot_oscillation_periods(osc_rows: List[Dict[str, Any]], outdir: Path) -> None:
    periods: List[float] = []
    for r in osc_rows:
        for name in ("n_neg", "min_eval", "trust_radius", "energy"):
            if r.get(f"{name}_has_cycle") and math.isfinite(r.get(f"{name}_period", float("nan"))):
                periods.append(r[f"{name}_period"])
    if not periods:
        print("  [WARN] No oscillation cycles detected for histogram.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(periods, bins=min(50, max(10, len(periods) // 5)), color="#4c72b0",
            edgecolor="white", alpha=0.8)
    ax.set_xlabel("Detected period (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Detected Oscillation Periods")
    _save_fig(fig, outdir / "oscillation_period_histogram.png")


# Plot 5 - Distance to reactant evolution
def plot_distance_evolution(
    all_trajs: List[Dict[str, Any]],
    outdir: Path,
    max_steps: int = 4000,
) -> None:
    """Shaded mean +/- std curves for converged vs failed."""
    groups: Dict[str, List[np.ndarray]] = {"converged": [], "failed": []}
    for td in all_trajs:
        traj = td.get("trajectory", [])
        if not traj:
            continue
        label = "converged" if _is_converged(td) else "failed"
        arr = np.full(max_steps, np.nan)
        for i, s in enumerate(traj):
            if i >= max_steps:
                break
            d = _safe(s.get("dist_to_reactant_rmsd"))
            arr[i] = d
        groups[label].append(arr)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"converged": "#55a868", "failed": "#c44e52"}
    for label in ("converged", "failed"):
        arrs = groups[label]
        if not arrs:
            continue
        mat = np.vstack(arrs)
        with np.errstate(all="ignore"):
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            count = np.sum(np.isfinite(mat), axis=0)
        valid = count >= 2
        steps = np.arange(max_steps)
        ax.plot(steps[valid], mean[valid], label=f"{label} (n={len(arrs)})",
                color=colors[label], linewidth=1.5)
        ax.fill_between(steps[valid],
                        (mean - std)[valid],
                        (mean + std)[valid],
                        alpha=0.2, color=colors[label])
    ax.set_xlabel("Step")
    ax.set_ylabel("dist_to_reactant_rmsd")
    ax.set_title("Distance to Reactant: Converged vs Failed")
    ax.legend()
    _save_fig(fig, outdir / "distance_to_reactant_evolution.png")


# Plot 6 - Population comparison (3-panel)
def plot_population_comparison(all_trajs: List[Dict[str, Any]], outdir: Path) -> None:
    pops: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": [], "C": []}
    for td in all_trajs:
        p = _classify_population(td)
        if p in pops:
            pops[p].append(td)

    def _final_values(pop: List[Dict[str, Any]], key: str) -> List[float]:
        vals: List[float] = []
        for td in pop:
            traj = td.get("trajectory", [])
            if not traj:
                continue
            v = _safe(traj[-1].get(key))
            if math.isfinite(v):
                vals.append(v)
        return vals

    def _final_neg_mode_values(pop: List[Dict[str, Any]], key: str) -> List[float]:
        vals: List[float] = []
        for td in pop:
            traj = td.get("trajectory", [])
            if not traj:
                continue
            nmd = traj[-1].get("neg_mode_diag", {})
            v = _safe(nmd.get(key))
            if math.isfinite(v):
                vals.append(v)
        return vals

    pop_labels = ["A: Ghost", "B: Trust-crushed", "C: Border"]
    pop_keys = ["A", "B", "C"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: min_vib_eval distributions
    ax = axes[0]
    data_eval = [_final_values(pops[k], "min_vib_eval") for k in pop_keys]
    non_empty_eval = [(d, l) for d, l in zip(data_eval, pop_labels) if d]
    if non_empty_eval:
        ax.boxplot([d for d, _ in non_empty_eval],
                   labels=[l for _, l in non_empty_eval],
                   patch_artist=True)
    ax.set_ylabel("min_vib_eval (final step)")
    ax.set_title("Eigenvalue Distribution")

    # Panel 2: trust_radius distributions
    ax = axes[1]
    data_tr = [_final_values(pops[k], "trust_radius") for k in pop_keys]
    non_empty_tr = [(d, l) for d, l in zip(data_tr, pop_labels) if d]
    if non_empty_tr:
        ax.boxplot([d for d, _ in non_empty_tr],
                   labels=[l for _, l in non_empty_tr],
                   patch_artist=True)
    ax.set_ylabel("trust_radius (final step)")
    ax.set_title("Trust Radius Distribution")

    # Panel 3: min_neg_grad_overlap distributions
    ax = axes[2]
    data_ov = [_final_neg_mode_values(pops[k], "min_neg_grad_overlap") for k in pop_keys]
    non_empty_ov = [(d, l) for d, l in zip(data_ov, pop_labels) if d]
    if non_empty_ov:
        ax.boxplot([d for d, _ in non_empty_ov],
                   labels=[l for _, l in non_empty_ov],
                   patch_artist=True)
    ax.set_ylabel("min_neg_grad_overlap (final step)")
    ax.set_title("Gradient-Mode Overlap")

    fig.suptitle("Population Comparison: Ghost / Trust-Crushed / Border", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_fig(fig, outdir / "population_comparison.png")


# ---------------------------------------------------------------------------
# CSV / JSON writers
# ---------------------------------------------------------------------------


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        print(f"  [WARN] No rows to write for {path.name}")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {path}  ({len(rows)} rows)")


def _json_safe(obj: Any) -> Any:
    """Make *obj* JSON-serialisable (handle numpy types and NaN)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------


def print_report(
    corr_rows: List[Dict[str, Any]],
    collapse_events: List[Dict[str, Any]],
    transition_counts: np.ndarray,
    mean_residence: np.ndarray,
    stag_windows: List[Dict[str, Any]],
    osc_rows: List[Dict[str, Any]],
    dist_summary: Dict[str, Any],
    ghost_rows: List[Dict[str, Any]],
    crush_rows: List[Dict[str, Any]],
    all_trajs: List[Dict[str, Any]],
) -> None:
    sep = "=" * 72

    n_total = len(all_trajs)
    n_conv = sum(1 for td in all_trajs if _is_converged(td))
    n_fail = n_total - n_conv

    print(f"\n{sep}")
    print("  SURFACE / FAILURE FORENSICS REPORT")
    print(f"{sep}")
    print(f"  Total trajectories loaded : {n_total}")
    print(f"  Converged                 : {n_conv}")
    print(f"  Failed                    : {n_fail}")

    # F1
    print(f"\n{sep}")
    print("  F1. Trust Radius <-> Eigenvalue Correlation")
    print(f"{sep}")
    if corr_rows:
        tr_nn = [r["pearson_tr_vs_nneg"] for r in corr_rows if math.isfinite(r.get("pearson_tr_vs_nneg", float("nan")))]
        tr_me = [r["pearson_tr_vs_min_eval"] for r in corr_rows if math.isfinite(r.get("pearson_tr_vs_min_eval", float("nan")))]
        if tr_nn:
            print(f"  Pearson(TR, n_neg)    : mean={np.mean(tr_nn):.4f}  std={np.std(tr_nn):.4f}  n={len(tr_nn)}")
        if tr_me:
            print(f"  Pearson(TR, min_eval) : mean={np.mean(tr_me):.4f}  std={np.std(tr_me):.4f}  n={len(tr_me)}")
        print(f"  TR collapse events (>50% drop in 1 step): {len(collapse_events)}")
        if collapse_events:
            drops = [e["drop_frac"] for e in collapse_events]
            print(f"    mean drop fraction: {np.mean(drops):.4f}")
    else:
        print("  No data.")

    # F2
    print(f"\n{sep}")
    print("  F2. Eigenvalue Band Transition Matrix")
    print(f"{sep}")
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normed = np.where(row_sums > 0, transition_counts / row_sums, 0.0)
    short_labels = [
        "<-1e-1", "(-1e-1,-1e-2)", "(-1e-2,-1e-3)", "(-1e-3,-1e-4)",
        "(-1e-4,0)", "(0,1e-4)", "(1e-4,1e-3)", ">1e-3",
    ]
    print("  Row-normalised transition probabilities (diagonal = stay):")
    for i in range(N_BANDS):
        diag = normed[i, i]
        print(f"    {short_labels[i]:>18s} : self-transition = {diag:.3f}")
    print("  Mean residence time per band (steps):")
    for i in range(N_BANDS):
        print(f"    {short_labels[i]:>18s} : {mean_residence[i]:.1f}")

    # F3
    print(f"\n{sep}")
    print("  F3. Stagnation Anatomy")
    print(f"{sep}")
    print(f"  Stagnation windows found (>= 50 steps constant n_neg): {len(stag_windows)}")
    if stag_windows:
        lengths = [w["length"] for w in stag_windows]
        print(f"    Length: mean={np.mean(lengths):.0f}  median={np.median(lengths):.0f}  max={max(lengths)}")
        e_ranges = [w["energy_range"] for w in stag_windows if math.isfinite(w.get("energy_range", float("nan")))]
        if e_ranges:
            print(f"    Energy range during stagnation: mean={np.mean(e_ranges):.6f}")
        f_ranges = [w["force_norm_range"] for w in stag_windows if math.isfinite(w.get("force_norm_range", float("nan")))]
        if f_ranges:
            print(f"    Force-norm range: mean={np.mean(f_ranges):.6f}")

    # F4
    print(f"\n{sep}")
    print("  F4. Oscillation Cycle Detection")
    print(f"{sep}")
    if osc_rows:
        n_with_cycle = sum(1 for r in osc_rows if r.get("any_cycle"))
        frac = n_with_cycle / max(len(osc_rows), 1)
        print(f"  Trajectories with detected cycles: {n_with_cycle}/{len(osc_rows)} ({frac:.1%})")
        for name in ("n_neg", "min_eval", "trust_radius", "energy"):
            periods = [r[f"{name}_period"] for r in osc_rows
                       if r.get(f"{name}_has_cycle") and math.isfinite(r.get(f"{name}_period", float("nan")))]
            if periods:
                print(f"    {name:>14s}: n_detected={len(periods)}  "
                      f"median_period={np.median(periods):.0f}  "
                      f"mean_period={np.mean(periods):.1f}")
    else:
        print("  No data.")

    # F5
    print(f"\n{sep}")
    print("  F5. Distance Evolution Comparison")
    print(f"{sep}")
    for label in ("converged", "failed"):
        ms = dist_summary.get(label, {})
        if not ms:
            continue
        print(f"  {label.upper()}:")
        for m, stats in sorted(ms.items(), key=lambda x: int(x[0])):
            print(f"    step {m:>5s}: mean={stats['mean']:.4f}  std={stats['std']:.4f}  n={stats['count']}")

    # F6
    print(f"\n{sep}")
    print("  F6. Ghost-Mode Characterisation")
    print(f"{sep}")
    if ghost_rows:
        has_ghost = [r for r in ghost_rows if r["ghost_steps"] > 0]
        print(f"  Trajectories with any ghost steps: {len(has_ghost)}/{len(ghost_rows)}")
        if has_ghost:
            ghost_fracs = [r["ghost_frac"] for r in has_ghost]
            print(f"    Ghost fraction: mean={np.mean(ghost_fracs):.3f}  max={max(ghost_fracs):.3f}")
            n_atoms_list = [r["n_atoms"] for r in has_ghost if r["n_atoms"] > 0]
            ghost_frac_for_corr = [r["ghost_frac"] for r in has_ghost if r["n_atoms"] > 0]
            if len(n_atoms_list) >= 3:
                corr = _pearson(np.array(n_atoms_list, dtype=float),
                                np.array(ghost_frac_for_corr, dtype=float))
                print(f"    Pearson(n_atoms, ghost_frac) = {corr:.4f}")
    else:
        print("  No data.")

    # F7
    print(f"\n{sep}")
    print("  F7. Trust-Crushed Population Analysis")
    print(f"{sep}")
    if crush_rows:
        has_crush = [r for r in crush_rows if r["crushed_steps"] > 0]
        print(f"  Trajectories with TR < {TR_FLOOR}: {len(has_crush)}/{len(crush_rows)}")
        if has_crush:
            fracs = [r["crushed_frac"] for r in has_crush]
            print(f"    Crushed fraction: mean={np.mean(fracs):.3f}  max={max(fracs):.3f}")
            neg_fracs = [r["mean_step_along_neg_frac"] for r in has_crush
                         if math.isfinite(r.get("mean_step_along_neg_frac", float("nan")))]
            if neg_fracs:
                print(f"    Mean step_along_neg_frac at crush: {np.mean(neg_fracs):.6f}")
    else:
        print("  No data.")

    # Population summary
    print(f"\n{sep}")
    print("  Population Summary (for Plot 6)")
    print(f"{sep}")
    pop_counts: Dict[str, int] = defaultdict(int)
    for td in all_trajs:
        pop_counts[_classify_population(td)] += 1
    for key, label in [("converged", "Converged"), ("A", "Pop A (Ghost modes)"),
                        ("B", "Pop B (Trust-crushed)"), ("C", "Pop C (Border)")]:
        print(f"  {label:>25s}: {pop_counts[key]}")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Surface / failure forensics for NR minimization trajectories.",
    )
    parser.add_argument(
        "--grid-dir",
        type=str,
        required=True,
        help="Root directory of grid-search outputs (contains combo subdirs).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where CSVs, JSONs, and PNGs are saved.",
    )
    parser.add_argument(
        "--traj-glob",
        type=str,
        default="*/diagnostics/*_trajectory.json",
        help="Glob pattern (relative to grid-dir) to find trajectory JSONs.",
    )
    parser.add_argument(
        "--combo-tag",
        type=str,
        default=None,
        help="Optional substring filter on file paths.",
    )
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Grid directory : {grid_dir}")
    print(f"Output directory: {outdir}")
    print(f"Trajectory glob : {args.traj_glob}")
    if args.combo_tag:
        print(f"Combo-tag filter: {args.combo_tag}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading trajectories ...")
    all_trajs = load_trajectories(grid_dir, args.traj_glob, args.combo_tag)
    print(f"  Loaded {len(all_trajs)} trajectories.")
    if not all_trajs:
        print("  No trajectories found -- exiting.")
        return

    # ------------------------------------------------------------------
    # F1: TR <-> eigenvalue correlation
    # ------------------------------------------------------------------
    print("\n[F1] Trust-radius <-> eigenvalue correlation ...")
    corr_rows, collapse_events = analyze_tr_eigenvalue_correlation(all_trajs)

    # ------------------------------------------------------------------
    # F2: Band transition matrix
    # ------------------------------------------------------------------
    print("[F2] Eigenvalue band transition matrix ...")
    transition_counts, mean_residence = analyze_band_transitions(all_trajs)

    # ------------------------------------------------------------------
    # F3: Stagnation anatomy
    # ------------------------------------------------------------------
    print("[F3] Stagnation anatomy ...")
    stag_windows = analyze_stagnation(all_trajs)

    # ------------------------------------------------------------------
    # F4: Oscillation detection
    # ------------------------------------------------------------------
    print("[F4] Oscillation cycle detection ...")
    osc_rows = analyze_oscillations(all_trajs)

    # ------------------------------------------------------------------
    # F5: Distance evolution
    # ------------------------------------------------------------------
    print("[F5] Distance evolution comparison ...")
    dist_summary = analyze_distance_evolution(all_trajs)

    # ------------------------------------------------------------------
    # F6: Ghost-mode characterisation
    # ------------------------------------------------------------------
    print("[F6] Ghost-mode characterisation ...")
    ghost_rows = analyze_ghost_modes(all_trajs)

    # ------------------------------------------------------------------
    # F7: Trust-crushed population
    # ------------------------------------------------------------------
    print("[F7] Trust-crushed population analysis ...")
    crush_rows = analyze_trust_crushed(all_trajs)

    # ------------------------------------------------------------------
    # Write CSVs
    # ------------------------------------------------------------------
    print("\nWriting CSVs ...")
    _write_csv(corr_rows, outdir / "tr_eigenvalue_correlation.csv")
    _write_csv(stag_windows, outdir / "stagnation_windows.csv")
    _write_csv(osc_rows, outdir / "oscillation_analysis.csv")
    _write_csv(ghost_rows, outdir / "ghost_mode_characterization.csv")
    _write_csv(crush_rows, outdir / "trust_crushed_analysis.csv")

    # ------------------------------------------------------------------
    # Write master JSON
    # ------------------------------------------------------------------
    master: Dict[str, Any] = {
        "n_trajectories": len(all_trajs),
        "n_converged": sum(1 for td in all_trajs if _is_converged(td)),
        "n_failed": sum(1 for td in all_trajs if not _is_converged(td)),
        "f1_tr_eigenvalue_correlation": {
            "n_rows": len(corr_rows),
            "n_collapse_events": len(collapse_events),
            "collapse_events_sample": collapse_events[:20],
        },
        "f2_band_transitions": {
            "transition_counts": transition_counts.tolist(),
            "mean_residence_time": mean_residence.tolist(),
            "band_labels": BAND_LABELS,
        },
        "f3_stagnation": {
            "n_windows": len(stag_windows),
            "window_lengths": [w["length"] for w in stag_windows],
        },
        "f4_oscillation": {
            "n_analysed": len(osc_rows),
            "n_with_any_cycle": sum(1 for r in osc_rows if r.get("any_cycle")),
        },
        "f5_distance_evolution": dist_summary,
        "f6_ghost_modes": {
            "n_with_ghost": sum(1 for r in ghost_rows if r["ghost_steps"] > 0),
            "n_total": len(ghost_rows),
        },
        "f7_trust_crushed": {
            "n_with_crush": sum(1 for r in crush_rows if r["crushed_steps"] > 0),
            "n_total": len(crush_rows),
        },
        "populations": {
            label: sum(1 for td in all_trajs if _classify_population(td) == label)
            for label in ("converged", "A", "B", "C")
        },
    }
    json_path = outdir / "surface_forensics.json"
    with open(json_path, "w") as f:
        json.dump(_json_safe(master), f, indent=2)
    print(f"  Saved {json_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nGenerating plots ...")
    plot_tr_vs_nneg(all_trajs, outdir)
    plot_band_transition_matrix(transition_counts, outdir)
    plot_stagnation_anatomy(stag_windows, outdir)
    plot_oscillation_periods(osc_rows, outdir)
    plot_distance_evolution(all_trajs, outdir)
    plot_population_comparison(all_trajs, outdir)

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------
    print_report(
        corr_rows,
        collapse_events,
        transition_counts,
        mean_residence,
        stag_windows,
        osc_rows,
        dist_summary,
        ghost_rows,
        crush_rows,
        all_trajs,
    )

    print("Done.")


if __name__ == "__main__":
    main()
