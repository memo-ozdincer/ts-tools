#!/usr/bin/env python3
"""Analyze GAD grid-search outputs.

Expected directory layout:
  <grid_dir>/mad*_tr*_bl*_pg*/gad_*_parallel_*_results.json

(Legacy layout mad*_tr*_bl*_pg* is still accepted.  New runs may also use
 ts_eps in the tag: mad*_tr*_bl*_pg*_tse*.)

New in v2:
  - Cascade evaluation table: rows = ts_eps (convergence strictness),
    columns = eval_threshold (how strictly we count n_neg at the final geometry).
    Two sub-tables:
      rate_eq1 — success rate counting n_neg_at_T == 1 (true TS)
      rate_le1 — success rate counting n_neg_at_T <= 1 (TS or minimum)
    Gap between strict ts_eps and loose eval_T reveals "false rejection" vs
    "genuine failure to reach Morse index 1".

  - Negative eigenvalue gap analysis (per combo and per sample):
      lambda_0        — most-negative eigenvalue at convergence (the climbing mode)
      lambda_1        — second eigenvalue (top of positive ladder)
      abs_lambda_0    — |lambda_0|
      lambda_gap_ratio — |lambda_0| / |lambda_1|
    If gap_ratio ≈ 1, the TS mode is buried in noise — the "convergence" is suspect.
    If gap_ratio >> 1, the TS mode is well-separated and reliable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Folder-name regex — accept both legacy and new (ts_eps) tags
# ---------------------------------------------------------------------------
COMBO_RE_NEW = re.compile(
    r"mad(?P<mad>[^_]+)_tr(?P<tr>[^_]+)_bl(?P<bl>plain|mode_tracked)_pg(?P<pg>true|false)"
    r"(?:_tse(?P<tse>[^_]+))?$"
)
COMBO_RE_LEGACY = re.compile(
    r"mad(?P<mad>[^_]+)_tr(?P<tr>[^_]+)_bl(?P<bl>.+)_pg(?P<pg>true|false)$"
)

# Cascade thresholds must match CASCADE_THRESHOLDS in run_gad_baselines_parallel.py
CASCADE_THRESHOLDS: List[float] = [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2]


def _parse_combo_tag(combo_tag: str) -> Optional[Dict[str, Any]]:
    m = COMBO_RE_NEW.fullmatch(combo_tag)
    if m:
        return {
            "mad": _safe_float(m.group("mad")),
            "tr": _safe_float(m.group("tr")),
            "bl": m.group("bl"),
            "pg": m.group("pg") == "true",
            "tse": _safe_float(m.group("tse") or "1e-5", 1e-5),
        }
    m = COMBO_RE_LEGACY.fullmatch(combo_tag)
    if m:
        return {
            "mad": _safe_float(m.group("mad")),
            "tr": _safe_float(m.group("tr")),
            "bl": m.group("bl"),
            "pg": m.group("pg") == "true",
            "tse": 1e-5,
        }
    return None


@dataclass
class ComboRecord:
    tag: str
    path: str
    max_atom_disp: float
    tr_threshold: float
    ts_eps: float
    baseline: str
    project_gradient_and_v: bool
    n_samples: int
    n_success: int
    n_errors: int
    success_rate: float
    mean_steps_when_success: float
    mean_wall_time: float
    total_wall_time: float
    neg_vib_counts: Dict[int, int]
    # cascade_table from results JSON (may be empty for old runs)
    cascade_table: Dict[str, Any] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    data = [v for v in values if math.isfinite(v)]
    if not data:
        return float("nan")
    return sum(data) / len(data)


def _sort_value(value: Any) -> Tuple[int, Any]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (1, value)
    return (2, str(value))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_records(grid_dir: Path, result_glob: str) -> List[ComboRecord]:
    records: List[ComboRecord] = []

    for result_path in sorted(grid_dir.glob(result_glob)):
        if not result_path.is_file():
            continue

        combo_tag = result_path.parent.name
        parsed = _parse_combo_tag(combo_tag)
        if parsed is None:
            print(f"  [warn] Cannot parse combo tag: {combo_tag} — skipping")
            continue

        with open(result_path) as f:
            payload = json.load(f)

        metrics = payload.get("metrics", {})
        neg_vib_raw = metrics.get("neg_vib_counts", {})
        neg_vib_counts = {int(k): int(v) for k, v in neg_vib_raw.items()}

        records.append(
            ComboRecord(
                tag=combo_tag,
                path=str(result_path),
                max_atom_disp=parsed["mad"],
                tr_threshold=parsed["tr"],
                ts_eps=parsed["tse"],
                baseline=parsed["bl"],
                project_gradient_and_v=parsed["pg"],
                n_samples=int(metrics.get("n_samples", 0)),
                n_success=int(metrics.get("n_success", 0)),
                n_errors=int(metrics.get("n_errors", 0)),
                success_rate=_safe_float(metrics.get("success_rate"), 0.0),
                mean_steps_when_success=_safe_float(metrics.get("mean_steps_when_success")),
                mean_wall_time=_safe_float(metrics.get("mean_wall_time")),
                total_wall_time=_safe_float(metrics.get("total_wall_time")),
                neg_vib_counts=neg_vib_counts,
                cascade_table=metrics.get("cascade_table", {}),
                results=list(metrics.get("results", [])),
            )
        )

    return records


# ---------------------------------------------------------------------------
# Ranking and main-effect summaries (unchanged from v1)
# ---------------------------------------------------------------------------

def rank_records(records: List[ComboRecord]) -> List[ComboRecord]:
    def _key(r: ComboRecord) -> Tuple[float, int, float, float]:
        steps = r.mean_steps_when_success
        steps_sort = steps if math.isfinite(steps) else float("inf")
        wall = r.mean_wall_time if math.isfinite(r.mean_wall_time) else float("inf")
        return (-r.success_rate, -r.n_success, steps_sort, wall)

    return sorted(records, key=_key)


def summarize_main_effect(records: List[ComboRecord], attr: str) -> List[Dict[str, Any]]:
    grouped: Dict[Any, List[ComboRecord]] = defaultdict(list)
    for record in records:
        grouped[getattr(record, attr)].append(record)

    rows: List[Dict[str, Any]] = []
    for value in sorted(grouped.keys(), key=_sort_value):
        bucket = grouped[value]
        rows.append(
            {
                "value": value,
                "n_configs": len(bucket),
                "mean_success_rate": _mean(r.success_rate for r in bucket),
                "mean_n_success": _mean(float(r.n_success) for r in bucket),
                "mean_steps_when_success": _mean(r.mean_steps_when_success for r in bucket),
                "mean_wall_time": _mean(r.mean_wall_time for r in bucket),
            }
        )
    return rows


def summarize_mad_tr_interaction(records: List[ComboRecord]) -> List[Dict[str, Any]]:
    mads = sorted({r.max_atom_disp for r in records})
    trs = sorted({r.tr_threshold for r in records})

    table: List[Dict[str, Any]] = []
    for mad in mads:
        row: Dict[str, Any] = {"max_atom_disp": mad}
        for tr in trs:
            bucket = [r for r in records if r.max_atom_disp == mad and r.tr_threshold == tr]
            row[f"tr_{tr:g}"] = _mean(r.success_rate for r in bucket)
        table.append(row)
    return table


def summarize_baseline_interaction(records: List[ComboRecord]) -> List[Dict[str, Any]]:
    """Compare plain vs mode_tracked at each (mad, tr) combo."""
    baselines = sorted({r.baseline for r in records})
    mads = sorted({r.max_atom_disp for r in records})
    trs = sorted({r.tr_threshold for r in records})

    rows: List[Dict[str, Any]] = []
    for mad in mads:
        for tr in trs:
            row: Dict[str, Any] = {"max_atom_disp": mad, "tr_threshold": tr}
            for bl in baselines:
                bucket = [
                    r for r in records
                    if r.max_atom_disp == mad and r.tr_threshold == tr and r.baseline == bl
                ]
                row[f"success_rate_{bl}"] = _mean(r.success_rate for r in bucket)
                row[f"mean_steps_{bl}"] = _mean(r.mean_steps_when_success for r in bucket)
            rows.append(row)
    return rows


def summarize_sample_hardness(records: List[ComboRecord]) -> List[Dict[str, Any]]:
    success_count: Dict[int, int] = defaultdict(int)
    total_count: Dict[int, int] = defaultdict(int)
    best_step: Dict[int, int] = {}
    best_tag: Dict[int, str] = {}

    for record in records:
        for row in record.results:
            idx = row.get("sample_idx")
            if idx is None:
                continue
            idx = int(idx)
            total_count[idx] += 1
            if bool(row.get("success")):
                success_count[idx] += 1
                step = row.get("steps_to_ts")
                if isinstance(step, int):
                    prev = best_step.get(idx)
                    if prev is None or step < prev:
                        best_step[idx] = step
                        best_tag[idx] = record.tag

    summary: List[Dict[str, Any]] = []
    for idx in sorted(total_count):
        successes = success_count[idx]
        total = total_count[idx]
        summary.append(
            {
                "sample_idx": idx,
                "n_success": successes,
                "n_total": total,
                "success_rate": successes / max(total, 1),
                "best_steps_to_ts": best_step.get(idx),
                "best_combo_tag": best_tag.get(idx),
            }
        )

    summary.sort(key=lambda x: (x["success_rate"], x["n_success"], x["sample_idx"]))
    return summary


def summarize_neg_vib_distribution(records: List[ComboRecord]) -> Dict[str, Any]:
    """Aggregate final Morse index (neg_vib) counts across the entire grid."""
    total: Dict[int, int] = defaultdict(int)
    for r in records:
        for k, v in r.neg_vib_counts.items():
            total[k] += v
    return dict(sorted(total.items()))


# ---------------------------------------------------------------------------
# NEW: cascade cross-table (optimizer ts_eps × eval_threshold → success_rate)
# ---------------------------------------------------------------------------

def build_cascade_cross_table(records: List[ComboRecord]) -> Dict[str, Any]:
    """Build a 2D table: rows = ts_eps (optimizer convergence gate),
    columns = eval_threshold (how strictly we count n_neg at final geometry).

    Two sub-tables:
      rate_eq1: success if n_neg_at_T == 1  (exactly one TS mode below -T)
      rate_le1: success if n_neg_at_T <= 1  (at most one mode below -T)

    Interpretation:
      rate_eq1 at T=0: strict baseline (same as optimizer success_rate).
      rate_eq1 at T=2e-3 >> rate_eq1 at T=0: false-rejection problem.
        The optimizer found Morse-1 geometry, but a tiny residual negative
        eigenvalue (|λ| < 2e-3) causes the product test to fail.
      rate_le1 >> rate_eq1 everywhere: optimizer overshoots into minimums.
        n_neg==0 at the final geometry; try a looser ts_eps or fewer steps.

    Also reports negative eigenvalue gap statistics per ts_eps value:
      mean_gap_ratio, mean_abs_lambda0 — from samples that actually converged.
      If gap_ratio ≈ 1 even when ts_eps=1e-5 succeeds, the TS eigenvalue is
      unresolved from noise and those "successes" may be spurious.
    """
    ts_eps_values = sorted({r.ts_eps for r in records})

    rows: List[Dict[str, Any]] = []
    for tse in ts_eps_values:
        bucket = [r for r in records if math.isclose(r.ts_eps, tse, rel_tol=1e-6)
                  and r.cascade_table]
        if not bucket:
            continue
        row: Dict[str, Any] = {"ts_eps": tse}
        for thr in CASCADE_THRESHOLDS:
            rates_eq1, rates_le1 = [], []
            for rec in bucket:
                ct = rec.cascade_table
                r_eq1 = ct.get("rate_eq1_at_thr", {}).get(str(thr))
                r_le1 = ct.get("rate_le1_at_thr", {}).get(str(thr))
                if r_eq1 is not None:
                    rates_eq1.append(float(r_eq1))
                if r_le1 is not None:
                    rates_le1.append(float(r_le1))
            row[f"eq1_eval_{thr}"] = _mean(rates_eq1)
            row[f"le1_eval_{thr}"] = _mean(rates_le1)

        # Strict success rate (from the optimizer's own ts_eps gate)
        row["optimizer_strict_rate"] = _mean(r.success_rate for r in bucket)

        # Negative eigenvalue gap stats — only from runs with cascade data
        gap_ratios = [
            float(ct.get("mean_gap_ratio_at_success", float("nan")))
            for r in bucket
            for ct in [r.cascade_table]
            if math.isfinite(ct.get("mean_gap_ratio_at_success", float("nan")))
        ]
        abs_lam0s = [
            float(ct.get("mean_abs_lambda0_at_success", float("nan")))
            for r in bucket
            for ct in [r.cascade_table]
            if math.isfinite(ct.get("mean_abs_lambda0_at_success", float("nan")))
        ]
        abs_lam1s = [
            float(ct.get("mean_abs_lambda1_at_success", float("nan")))
            for r in bucket
            for ct in [r.cascade_table]
            if math.isfinite(ct.get("mean_abs_lambda1_at_success", float("nan")))
        ]
        row["mean_gap_ratio_at_success"] = _mean(gap_ratios)
        row["mean_abs_lambda0_at_success"] = _mean(abs_lam0s)
        row["mean_abs_lambda1_at_success"] = _mean(abs_lam1s)

        rows.append(row)

    return {
        "eval_thresholds": CASCADE_THRESHOLDS,
        "ts_eps_values_tested": ts_eps_values,
        "table": rows,
    }


def print_cascade_table(cross: Dict[str, Any]) -> None:
    table = cross.get("table", [])
    thresholds = cross.get("eval_thresholds", CASCADE_THRESHOLDS)
    if not table:
        print("  (no cascade data — need results from run_gad_baselines_parallel.py v2)")
        return

    col_w = 8
    opt_w = 14
    # Print rate_eq1 sub-table (n_neg_at_T == 1)
    print("  [eq1] rate where n_neg_at_T == 1  (Morse index exactly 1):")
    header = f"  {'ts_eps':<{opt_w}}"
    for thr in thresholds:
        header += f"{'T='+str(thr):>{col_w}}"
    header += f"{'strict':>{col_w}}"
    print(header)
    print("  " + "-" * (opt_w + col_w * (len(thresholds) + 1)))

    for row in table:
        line = f"  {row['ts_eps']:<{opt_w}g}"
        for thr in thresholds:
            v = row.get(f"eq1_eval_{thr}", float("nan"))
            line += f"{v:>{col_w}.3f}" if math.isfinite(v) else f"{'nan':>{col_w}}"
        v_strict = row.get("optimizer_strict_rate", float("nan"))
        line += f"{v_strict:>{col_w}.3f}" if math.isfinite(v_strict) else f"{'nan':>{col_w}}"
        gap = row.get("mean_gap_ratio_at_success", float("nan"))
        lam0 = row.get("mean_abs_lambda0_at_success", float("nan"))
        lam1 = row.get("mean_abs_lambda1_at_success", float("nan"))
        extras = []
        if math.isfinite(gap):
            extras.append(f"gap={gap:.2f}")
        if math.isfinite(lam0):
            extras.append(f"|λ₀|={lam0:.4f}")
        if math.isfinite(lam1):
            extras.append(f"|λ₁|={lam1:.4f}")
        if extras:
            line += "  (" + ", ".join(extras) + ")"
        print(line)

    print("")
    print("  [le1] rate where n_neg_at_T <= 1  (TS or minimum — overshooting check):")
    print(header)
    print("  " + "-" * (opt_w + col_w * (len(thresholds) + 1)))
    for row in table:
        line = f"  {row['ts_eps']:<{opt_w}g}"
        for thr in thresholds:
            v = row.get(f"le1_eval_{thr}", float("nan"))
            line += f"{v:>{col_w}.3f}" if math.isfinite(v) else f"{'nan':>{col_w}}"
        v_strict = row.get("optimizer_strict_rate", float("nan"))
        line += f"{v_strict:>{col_w}.3f}" if math.isfinite(v_strict) else f"{'nan':>{col_w}}"
        print(line)

    print("")
    print("  Columns = eval threshold T (accept if n_neg_at_T == 1 or <= 1).")
    print("  'strict' = optimizer's own eig_product < -ts_eps gate.")
    print("  Large gap (eq1[T=2e-3] >> eq1[T=0]) → false-rejection problem.")
    print("  Large le1-eq1 difference → optimizer overshooting into minimums.")
    print("  gap ratio >> 1 at success → TS eigenvalue clearly separated from noise.")
    print("  gap ratio ≈ 1 at success → TS eigenvalue buried in noise (suspect).")


# ---------------------------------------------------------------------------
# Negative eigenvalue gap summary per combo
# ---------------------------------------------------------------------------

def summarize_neg_eval_gap(records: List[ComboRecord]) -> List[Dict[str, Any]]:
    """Per-combo summary of negative eigenvalue magnitude at convergence.

    Reads cascade_table.mean_gap_ratio_at_success and
    cascade_table.mean_abs_lambda0_at_success from each record's stored
    cascade_table (populated by run_gad_baselines_parallel.py v2).

    Returns rows sorted by mean_gap_ratio descending (best-separated first).
    """
    rows: List[Dict[str, Any]] = []
    for r in records:
        ct = r.cascade_table
        if not ct:
            continue
        gap = _safe_float(ct.get("mean_gap_ratio_at_success"), float("nan"))
        lam0 = _safe_float(ct.get("mean_abs_lambda0_at_success"), float("nan"))
        lam1 = _safe_float(ct.get("mean_abs_lambda1_at_success"), float("nan"))
        n_with_data = int(ct.get("n_successful_with_gap_data", 0))
        rows.append({
            "tag": r.tag,
            "ts_eps": r.ts_eps,
            "max_atom_disp": r.max_atom_disp,
            "baseline": r.baseline,
            "success_rate": r.success_rate,
            "n_success": r.n_success,
            "n_samples": r.n_samples,
            "mean_gap_ratio": gap,
            "mean_abs_lambda0": lam0,
            "mean_abs_lambda1": lam1,
            "n_with_gap_data": n_with_data,
        })
    rows.sort(key=lambda x: -x.get("mean_gap_ratio", float("-inf")) if math.isfinite(x.get("mean_gap_ratio", float("nan"))) else float("-inf"))
    return rows


def print_neg_eval_gap_report(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("  (no negative eigenvalue gap data — need v2 runner results)")
        return
    print(f"  {'tag':<45} {'success':>10} {'|λ₀|/|λ₁|':>12} {'|λ₀|':>10} {'|λ₁|':>10} {'n_data':>8}")
    print("  " + "-" * 100)
    for row in rows:
        gap = row.get("mean_gap_ratio", float("nan"))
        lam0 = row.get("mean_abs_lambda0", float("nan"))
        lam1 = row.get("mean_abs_lambda1", float("nan"))
        gap_str = f"{gap:.3f}" if math.isfinite(gap) else "nan"
        lam0_str = f"{lam0:.5f}" if math.isfinite(lam0) else "nan"
        lam1_str = f"{lam1:.5f}" if math.isfinite(lam1) else "nan"
        print(
            f"  {row['tag']:<45} "
            f"{row['n_success']}/{row['n_samples']:>{6}} "
            f"({row['success_rate']:.2f}) "
            f"{gap_str:>12} "
            f"{lam0_str:>10} "
            f"{lam1_str:>10} "
            f"{row['n_with_gap_data']:>8}"
        )


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_cascade_csv(path: Path, cross: Dict[str, Any]) -> None:
    table = cross.get("table", [])
    if not table:
        return
    thresholds = cross.get("eval_thresholds", CASCADE_THRESHOLDS)
    fieldnames = ["ts_eps"]
    fieldnames += [f"eq1_eval_{t}" for t in thresholds]
    fieldnames += [f"le1_eval_{t}" for t in thresholds]
    fieldnames += ["optimizer_strict_rate", "mean_gap_ratio_at_success", "mean_abs_lambda0_at_success", "mean_abs_lambda1_at_success"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in table:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_report(
    records: List[ComboRecord],
    ranked: List[ComboRecord],
    top_k: int,
    main_effects: Dict[str, List[Dict[str, Any]]],
    mad_tr_table: List[Dict[str, Any]],
    baseline_table: List[Dict[str, Any]],
    sample_hardness: List[Dict[str, Any]],
    neg_vib_dist: Dict[str, Any],
    cross_table: Dict[str, Any],
    neg_eval_gap: List[Dict[str, Any]],
) -> None:
    print(f"Loaded {len(records)} configurations.")
    print("Ranking: Success Rate (desc) → Total Success (desc) → Mean Steps (asc) → Wall Time (asc)")
    print("")

    if ranked:
        best = ranked[0]
        print(
            "Best overall configuration:\n"
            f"  {best.tag} | succeeded {best.n_success}/{best.n_samples} "
            f"({best.success_rate:.2f}), mean steps {best.mean_steps_when_success:.1f}, "
            f"mean wall {best.mean_wall_time:.2f}s"
        )
        print("")

    actual_top_k = min(top_k, len(ranked))
    print("=" * 65)
    print(f"🥇 Top {actual_top_k} Configurations (Best First):")
    print("=" * 65)
    for row in ranked[:actual_top_k]:
        steps = row.mean_steps_when_success
        steps_text = f"{steps:.1f}" if math.isfinite(steps) else "nan"
        print(
            f"  {row.tag}: success={row.n_success}/{row.n_samples} "
            f"({row.success_rate:.2f}), steps={steps_text}, "
            f"wall={row.mean_wall_time:.2f}s, errors={row.n_errors}"
        )
    print("")

    if len(ranked) > top_k:
        actual_bottom_k = min(top_k, len(ranked) - top_k)
        print("=" * 65)
        print(f"💀 Bottom {actual_bottom_k} Configurations (Worst First):")
        print("=" * 65)
        for row in reversed(ranked[-actual_bottom_k:]):
            steps = row.mean_steps_when_success
            steps_text = f"{steps:.1f}" if math.isfinite(steps) else "nan"
            print(
                f"  {row.tag}: success={row.n_success}/{row.n_samples} "
                f"({row.success_rate:.2f}), steps={steps_text}, "
                f"wall={row.mean_wall_time:.2f}s, errors={row.n_errors}"
            )
        print("")

    for key, rows in main_effects.items():
        print(f"--- Main effect: {key} ---")
        for row in rows:
            steps = row["mean_steps_when_success"]
            steps_text = f"{steps:.1f}" if math.isfinite(steps) else "nan"
            print(
                "  "
                f"{row['value']}: mean_success_rate={row['mean_success_rate']:.3f}, "
                f"mean_n_success={row['mean_n_success']:.2f}, "
                f"mean_steps={steps_text}, "
                f"mean_wall={row['mean_wall_time']:.2f}s"
            )
        print("")

    print("Interaction: mean success rate by max_atom_disp x tr_threshold")
    for row in mad_tr_table:
        parts = [f"mad={row['max_atom_disp']:g}"]
        for key in sorted(k for k in row if k.startswith("tr_")):
            val = row[key]
            val_text = f"{val:.2f}" if math.isfinite(val) else "nan"
            parts.append(f"{key.replace('tr_', 'tr=')}:{val_text}")
        print("  " + ", ".join(parts))
    print("")

    print("Interaction: plain vs mode_tracked at each (mad, tr)")
    for row in baseline_table:
        parts = [f"mad={row['max_atom_disp']:g}", f"tr={row['tr_threshold']:g}"]
        for key in sorted(k for k in row if k.startswith("success_rate_")):
            bl = key.replace("success_rate_", "")
            val = row[key]
            val_text = f"{val:.2f}" if math.isfinite(val) else "nan"
            parts.append(f"{bl}:{val_text}")
        print("  " + ", ".join(parts))
    print("")

    print("=" * 65)
    print("Final Morse index distribution across all runs:")
    print("=" * 65)
    for neg_vib, count in neg_vib_dist.items():
        label = {1: "→ TS (Morse index 1)", 0: "→ minimum (Morse index 0)"}.get(neg_vib, "other")
        print(f"  neg_vib={neg_vib}: {count} runs  {label}")
    print("")

    print("=" * 65)
    print("Hardest samples (lowest success rate across configs):")
    print("=" * 65)
    for row in sample_hardness[: min(10, len(sample_hardness))]:
        print(
            f"  sample_{row['sample_idx']:03d}: {row['n_success']}/{row['n_total']} "
            f"({row['success_rate']:.2f}), best_steps={row['best_steps_to_ts']}, "
            f"best_combo={row['best_combo_tag']}"
        )
    print("")

    print("=" * 65)
    print("📊 Cascade Evaluation Table (ts_eps × eval_threshold → success_rate)")
    print("=" * 65)
    print_cascade_table(cross_table)

    print("=" * 65)
    print("🔬 Negative Eigenvalue Gap at Convergence (per combo)")
    print("   gap = |λ₀| / |λ₁|:  ratio >> 1 → TS mode well-resolved")
    print("                        ratio ≈ 1  → TS mode buried in noise")
    print("=" * 65)
    print_neg_eval_gap_report(neg_eval_gap)
    print("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GAD grid-search results")
    parser.add_argument(
        "--grid-dir",
        type=str,
        required=True,
        help="Grid directory containing mad*_tr*_bl*_pg* subdirectories",
    )
    parser.add_argument(
        "--result-glob",
        type=str,
        default="*/gad_*_parallel_*_results.json",
        help="Glob (relative to --grid-dir) for result files",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top/bottom rows to print",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory for JSON/CSV exports",
    )
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    if not grid_dir.exists():
        raise FileNotFoundError(f"--grid-dir does not exist: {grid_dir}")

    records = load_records(grid_dir, args.result_glob)
    if not records:
        raise FileNotFoundError(
            f"No result files found in {grid_dir} with glob '{args.result_glob}'"
        )

    ranked = rank_records(records)
    main_effects = {
        "max_atom_disp": summarize_main_effect(records, "max_atom_disp"),
        "tr_threshold": summarize_main_effect(records, "tr_threshold"),
        "ts_eps": summarize_main_effect(records, "ts_eps"),
        "baseline": summarize_main_effect(records, "baseline"),
        "project_gradient_and_v": summarize_main_effect(records, "project_gradient_and_v"),
    }
    # Drop constant axes
    main_effects = {k: v for k, v in main_effects.items() if len(v) > 1}

    mad_tr_table = summarize_mad_tr_interaction(records)
    baseline_table = summarize_baseline_interaction(records)
    sample_hardness = summarize_sample_hardness(records)
    neg_vib_dist = summarize_neg_vib_distribution(records)
    cross_table = build_cascade_cross_table(records)
    neg_eval_gap = summarize_neg_eval_gap(records)

    print_report(
        records=records,
        ranked=ranked,
        top_k=args.top_k,
        main_effects=main_effects,
        mad_tr_table=mad_tr_table,
        baseline_table=baseline_table,
        sample_hardness=sample_hardness,
        neg_vib_dist=neg_vib_dist,
        cross_table=cross_table,
        neg_eval_gap=neg_eval_gap,
    )

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rows_for_json = [asdict(r) for r in ranked]
        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "grid_dir": str(grid_dir),
            "n_configs": len(records),
            "best_config": asdict(ranked[0]),
            "main_effects": main_effects,
            "mad_tr_interaction": mad_tr_table,
            "baseline_interaction": baseline_table,
            "sample_hardness": sample_hardness,
            "neg_vib_distribution": neg_vib_dist,
            "cascade_cross_table": cross_table,
            "neg_eval_gap_per_combo": neg_eval_gap,
            "ranked_configs": rows_for_json,
        }
        with open(out_dir / "gad_grid_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        ranked_rows = [
            {
                "rank": idx + 1,
                "tag": r.tag,
                "max_atom_disp": r.max_atom_disp,
                "tr_threshold": r.tr_threshold,
                "ts_eps": r.ts_eps,
                "baseline": r.baseline,
                "project_gradient_and_v": r.project_gradient_and_v,
                "n_samples": r.n_samples,
                "n_success": r.n_success,
                "n_errors": r.n_errors,
                "success_rate": r.success_rate,
                "mean_steps_when_success": r.mean_steps_when_success,
                "mean_wall_time": r.mean_wall_time,
                "total_wall_time": r.total_wall_time,
                "path": r.path,
            }
            for idx, r in enumerate(ranked)
        ]
        write_csv(
            out_dir / "gad_grid_ranked.csv",
            ranked_rows,
            [
                "rank", "tag", "max_atom_disp", "tr_threshold", "ts_eps",
                "baseline", "project_gradient_and_v",
                "n_samples", "n_success", "n_errors", "success_rate",
                "mean_steps_when_success", "mean_wall_time", "total_wall_time", "path",
            ],
        )
        write_csv(
            out_dir / "gad_grid_sample_hardness.csv",
            sample_hardness,
            [
                "sample_idx", "n_success", "n_total", "success_rate",
                "best_steps_to_ts", "best_combo_tag",
            ],
        )
        write_cascade_csv(out_dir / "gad_grid_cascade_table.csv", cross_table)
        write_csv(
            out_dir / "gad_grid_neg_eval_gap.csv",
            neg_eval_gap,
            [
                "tag", "ts_eps", "max_atom_disp", "baseline", "success_rate",
                "n_success", "n_samples", "mean_gap_ratio", "mean_abs_lambda0", "mean_abs_lambda1", "n_with_gap_data",
            ],
        )

        print("")
        print(f"Wrote analysis artifacts to: {out_dir}")
        print("  gad_grid_summary.json         — full summary (JSON)")
        print("  gad_grid_ranked.csv           — ranked configurations")
        print("  gad_grid_sample_hardness.csv  — per-sample success rates")
        print("  gad_grid_cascade_table.csv    — 2D cascade diagnostic table")
        print("  gad_grid_neg_eval_gap.csv     — negative eigenvalue gap per combo")


if __name__ == "__main__":
    main()
