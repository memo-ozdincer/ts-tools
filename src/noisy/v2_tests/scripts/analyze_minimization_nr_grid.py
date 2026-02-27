#!/usr/bin/env python3
"""Analyze Newton-Raphson minimization grid-search outputs.

Expected directory layout:
  <grid_dir>/mad*_nrt*_pg*_ph*/minimization_newton_raphson_*_results.json

(Legacy layout mad*_tr*_pg*_ph* is also accepted for backwards compatibility.)

New in v2:
  - Cascade evaluation table: rows = nr_threshold (optimizer param), columns =
    eval_threshold (how strictly we count n_neg). Values = convergence rate.
    Diagnoses whether failures are optimizer failures or evaluation strictness.
  - LM damping analysis: when --lm-mu > 0 the folder tag contains lmmu* instead
    of nrt*; detected automatically.
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
# Folder-name regex — accept legacy (tr), v2 (nrt / lmmu), and v3 (se) tags
# ---------------------------------------------------------------------------
# Hard filter: mad<v>_nrt<v>_pg<bool>_ph<bool>[_af<v>][_ct<v>]
# LM damping:  mad<v>_lmmu<v>_pg<bool>_ph<bool>[_af<v>][_sw<v>_ea<v>][_diag*]
# Shifted NR:  mad<v>_se<v>_pg<bool>_ph<bool>[_sw<v>_ea<v>[_ls]]
# Legacy:      mad<v>_tr<v>_pg<bool>_ph<bool>
COMBO_RE_NRT = re.compile(
    r"mad(?P<mad>[^_]+)_nrt(?P<nrt>[^_]+)_pg(?P<pg>true|false)_ph(?P<ph>true|false)"
    r"(?:_ec(?P<ec>[^_]+))?(?:_af(?P<af>[^_]+))?(?:_ct(?P<ct>[^_]+))?$"
)
COMBO_RE_LM = re.compile(
    r"mad(?P<mad>[^_]+)_lmmu(?P<lmmu>[^_]+)"
    r"(?:_af(?P<af>[^_]+))?"
    r"(?:_sw(?P<sw>[^_]+)_ea(?P<ea>[^_]+))?"
    r"(?:_(?P<diag>diag[^_]*))?"
    r"_pg(?P<pg>true|false)_ph(?P<ph>true|false)$"
)
# v4 shifted Newton with optional v4 features
# mad<v>_se<v>[_ntf<v>][_bmt<v>_bca<v>][_atr][_sw<v>_ea<v>[_ebd]][_ls][_mft<v>][_mfa<v>]_pg<bool>_ph<bool>
COMBO_RE_SHIFTED_V4 = re.compile(
    r"mad(?P<mad>[^_]+)_se(?P<se>[^_]+)"
    r"(?:_ntf(?P<ntf>[^_]+))?"
    r"(?:_bmt(?P<bmt>[^_]+)_bca(?P<bca>[^_]+))?"
    r"(?:_(?P<atr>atr))?"
    r"(?:_sw(?P<sw>[^_]+)_ea(?P<ea>[^_]+))?"
    r"(?:_(?P<ebd>ebd))?"
    r"(?:_(?P<ls>ls))?"
    r"(?:_mft(?P<mft>[^_]+))?"
    r"(?:_mfa(?P<mfa>[^_]+))?"
    r"_pg(?P<pg>true|false)_ph(?P<ph>true|false)$"
)
# v3 shifted Newton (simpler tag format, tried after v4)
COMBO_RE_SHIFTED = re.compile(
    r"mad(?P<mad>[^_]+)_se(?P<se>[^_]+)"
    r"(?:_sw(?P<sw>[^_]+)_ea(?P<ea>[^_]+))?"
    r"(?:_(?P<ls>ls))?"
    r"_pg(?P<pg>true|false)_ph(?P<ph>true|false)$"
)
COMBO_RE_LEGACY = re.compile(
    r"mad(?P<mad>[^_]+)_tr(?P<tr>[^_]+)_pg(?P<pg>true|false)_ph(?P<ph>true|false)$"
)

# Cascade thresholds must match CASCADE_THRESHOLDS in minimization.py
CASCADE_THRESHOLDS: List[float] = [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2]


@dataclass
class ComboRecord:
    tag: str
    path: str
    max_atom_disp: float
    nr_threshold: float          # 0.0 for LM-damping / shifted NR runs
    lm_mu: float                 # 0.0 for hard-filter / shifted NR runs
    shift_epsilon: float         # 0.0 for non-shifted runs (v3)
    anneal_force_threshold: float
    stagnation_window: int       # 0 for no stagnation escape (v3)
    escape_alpha: float          # v3
    lm_mu_anneal_factor: float   # 0.0 for no annealing (v3)
    neg_mode_line_search: bool   # v3
    project_gradient_and_v: bool
    purify_hessian: bool
    n_samples: int
    n_converged: int
    n_errors: int
    convergence_rate: float
    mean_steps_when_converged: float
    mean_wall_time: float
    total_wall_time: float
    total_escapes: int = 0
    total_line_searches: int = 0
    # v4 fields
    neg_trust_floor: float = 0.0
    blind_mode_threshold: float = 0.0
    blind_correction_alpha: float = 0.02
    aggressive_trust_recovery: bool = False
    escape_bidirectional: bool = False
    mode_follow_eval_threshold: float = 0.0
    total_mode_follows: int = 0
    # cascade_table from the results JSON (may be empty for old runs)
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


def _same_sample_outcomes(a: ComboRecord, b: ComboRecord) -> bool:
    def _map(results: List[Dict[str, Any]]) -> Dict[int, Tuple[bool, Optional[int]]]:
        out: Dict[int, Tuple[bool, Optional[int]]] = {}
        for row in results:
            idx = row.get("sample_idx")
            if idx is None:
                continue
            out[int(idx)] = (bool(row.get("converged")), row.get("converged_step"))
        return out

    return _map(a.results) == _map(b.results)


# ---------------------------------------------------------------------------
# Parsing folder names
# ---------------------------------------------------------------------------

def _parse_combo_tag(combo_tag: str) -> Optional[Dict[str, Any]]:
    """Parse a combo folder name into a dict of hyperparameter values.

    Returns None if the tag can't be matched.
    """
    # Default v3 fields
    _v3_defaults = {
        "shift_epsilon": 0.0,
        "stagnation_window": 0,
        "escape_alpha": 0.0,
        "lm_mu_anneal_factor": 0.0,
        "neg_mode_line_search": False,
    }
    # Default v4 fields
    _v4_defaults = {
        "neg_trust_floor": 0.0,
        "blind_mode_threshold": 0.0,
        "blind_correction_alpha": 0.02,
        "aggressive_trust_recovery": False,
        "escape_bidirectional": False,
        "mode_follow_eval_threshold": 0.0,
    }

    m = COMBO_RE_NRT.fullmatch(combo_tag)
    if m:
        return {
            "mad": _safe_float(m.group("mad")),
            "nr_threshold": _safe_float(m.group("nrt")),
            "lm_mu": 0.0,
            "anneal_force_threshold": _safe_float(m.group("af") or "0.0", 0.0),
            "project_gradient_and_v": m.group("pg") == "true",
            "purify_hessian": m.group("ph") == "true",
            **_v3_defaults,
            **_v4_defaults,
        }

    m = COMBO_RE_LM.fullmatch(combo_tag)
    if m:
        return {
            "mad": _safe_float(m.group("mad")),
            "nr_threshold": 0.0,
            "lm_mu": _safe_float(m.group("lmmu")),
            "anneal_force_threshold": 0.0,
            "lm_mu_anneal_factor": _safe_float(m.group("af") or "0.0", 0.0),
            "stagnation_window": int(_safe_float(m.group("sw") or "0", 0.0)),
            "escape_alpha": _safe_float(m.group("ea") or "0.0", 0.0),
            "shift_epsilon": 0.0,
            "neg_mode_line_search": False,
            "project_gradient_and_v": m.group("pg") == "true",
            "purify_hessian": m.group("ph") == "true",
            **_v4_defaults,
        }

    # Try v4 shifted newton first (more specific regex)
    m = COMBO_RE_SHIFTED_V4.fullmatch(combo_tag)
    if m:
        return {
            "mad": _safe_float(m.group("mad")),
            "nr_threshold": 0.0,
            "lm_mu": 0.0,
            "shift_epsilon": _safe_float(m.group("se")),
            "anneal_force_threshold": 0.0,
            "stagnation_window": int(_safe_float(m.group("sw") or "0", 0.0)),
            "escape_alpha": _safe_float(m.group("ea") or "0.0", 0.0),
            "lm_mu_anneal_factor": 0.0,
            "neg_mode_line_search": m.group("ls") is not None,
            "project_gradient_and_v": m.group("pg") == "true",
            "purify_hessian": m.group("ph") == "true",
            # v4 fields
            "neg_trust_floor": _safe_float(m.group("ntf") or "0.0", 0.0),
            "blind_mode_threshold": _safe_float(m.group("bmt") or "0.0", 0.0),
            "blind_correction_alpha": _safe_float(m.group("bca") or "0.02", 0.02),
            "aggressive_trust_recovery": m.group("atr") is not None,
            "escape_bidirectional": m.group("ebd") is not None,
            "mode_follow_eval_threshold": _safe_float(m.group("mft") or "0.0", 0.0),
        }

    # v3 shifted newton (simpler tag, fallback)
    m = COMBO_RE_SHIFTED.fullmatch(combo_tag)
    if m:
        return {
            "mad": _safe_float(m.group("mad")),
            "nr_threshold": 0.0,
            "lm_mu": 0.0,
            "shift_epsilon": _safe_float(m.group("se")),
            "anneal_force_threshold": 0.0,
            "stagnation_window": int(_safe_float(m.group("sw") or "0", 0.0)),
            "escape_alpha": _safe_float(m.group("ea") or "0.0", 0.0),
            "lm_mu_anneal_factor": 0.0,
            "neg_mode_line_search": m.group("ls") is not None,
            "project_gradient_and_v": m.group("pg") == "true",
            "purify_hessian": m.group("ph") == "true",
            **_v4_defaults,
        }

    m = COMBO_RE_LEGACY.fullmatch(combo_tag)
    if m:
        return {
            "mad": _safe_float(m.group("mad")),
            "nr_threshold": _safe_float(m.group("tr")),
            "lm_mu": 0.0,
            "anneal_force_threshold": 0.0,
            "project_gradient_and_v": m.group("pg") == "true",
            "purify_hessian": m.group("ph") == "true",
            **_v3_defaults,
            **_v4_defaults,
        }

    return None


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
            # Try treating the whole tag as unparseable but include with defaults
            print(f"  [warn] Cannot parse combo tag: {combo_tag} — skipping")
            continue

        with open(result_path) as f:
            payload = json.load(f)

        metrics = payload.get("metrics", {})
        # Aggregate escape/line-search/mode-follow counts from per-sample results
        per_sample_results = list(metrics.get("results", []))
        total_escapes = sum(r.get("total_escapes", 0) for r in per_sample_results)
        total_line_searches = sum(r.get("total_line_searches", 0) for r in per_sample_results)
        total_mode_follows = sum(r.get("total_mode_follows", 0) for r in per_sample_results)

        records.append(
            ComboRecord(
                tag=combo_tag,
                path=str(result_path),
                max_atom_disp=parsed["mad"],
                nr_threshold=parsed["nr_threshold"],
                lm_mu=parsed.get("lm_mu", 0.0),
                shift_epsilon=parsed.get("shift_epsilon", 0.0),
                anneal_force_threshold=parsed.get("anneal_force_threshold", 0.0),
                stagnation_window=int(parsed.get("stagnation_window", 0)),
                escape_alpha=parsed.get("escape_alpha", 0.0),
                lm_mu_anneal_factor=parsed.get("lm_mu_anneal_factor", 0.0),
                neg_mode_line_search=bool(parsed.get("neg_mode_line_search", False)),
                project_gradient_and_v=parsed["project_gradient_and_v"],
                purify_hessian=parsed["purify_hessian"],
                n_samples=int(metrics.get("n_samples", 0)),
                n_converged=int(metrics.get("n_converged", 0)),
                n_errors=int(metrics.get("n_errors", 0)),
                convergence_rate=_safe_float(metrics.get("convergence_rate"), 0.0),
                mean_steps_when_converged=_safe_float(metrics.get("mean_steps_when_converged")),
                mean_wall_time=_safe_float(metrics.get("mean_wall_time")),
                total_wall_time=_safe_float(metrics.get("total_wall_time")),
                total_escapes=total_escapes,
                total_line_searches=total_line_searches,
                # v4 fields
                neg_trust_floor=parsed.get("neg_trust_floor", 0.0),
                blind_mode_threshold=parsed.get("blind_mode_threshold", 0.0),
                blind_correction_alpha=parsed.get("blind_correction_alpha", 0.02),
                aggressive_trust_recovery=bool(parsed.get("aggressive_trust_recovery", False)),
                escape_bidirectional=bool(parsed.get("escape_bidirectional", False)),
                mode_follow_eval_threshold=parsed.get("mode_follow_eval_threshold", 0.0),
                total_mode_follows=total_mode_follows,
                cascade_table=metrics.get("cascade_table", {}),
                results=per_sample_results,
            )
        )

    return records


# ---------------------------------------------------------------------------
# Ranking and main-effect summaries  (unchanged from v1)
# ---------------------------------------------------------------------------

def rank_records(records: List[ComboRecord]) -> List[ComboRecord]:
    def _key(r: ComboRecord) -> Tuple[float, int, float, float]:
        steps = r.mean_steps_when_converged
        steps_sort = steps if math.isfinite(steps) else float("inf")
        wall = r.mean_wall_time if math.isfinite(r.mean_wall_time) else float("inf")
        return (-r.convergence_rate, -r.n_converged, steps_sort, wall)

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
                "mean_convergence_rate": _mean(r.convergence_rate for r in bucket),
                "mean_n_converged": _mean(float(r.n_converged) for r in bucket),
                "mean_steps_when_converged": _mean(r.mean_steps_when_converged for r in bucket),
                "mean_wall_time": _mean(r.mean_wall_time for r in bucket),
            }
        )
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
            if bool(row.get("converged")):
                success_count[idx] += 1
                step = row.get("converged_step")
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
                "n_converged": successes,
                "n_total": total,
                "convergence_rate": successes / max(total, 1),
                "best_converged_step": best_step.get(idx),
                "best_combo_tag": best_tag.get(idx),
            }
        )

    summary.sort(key=lambda x: (x["convergence_rate"], x["n_converged"], x["sample_idx"]))
    return summary


# ---------------------------------------------------------------------------
# NEW: cascade analysis — the 2D diagnostic table
# ---------------------------------------------------------------------------

def build_cascade_cross_table(records: List[ComboRecord]) -> Dict[str, Any]:
    """Build a 2D table: rows = nr_threshold (optimizer filter), columns =
    eval_threshold (how strictly we count n_neg at the final geometry).
    Values = convergence rate.

    Interpretation:
      - If rate@eval_thr=0.0 is low but rate@eval_thr=2e-3 is high:
        the OPTIMIZER IS WORKING. The gap is false rejection — tiny residual
        negative eigenvalues at evaluation.
      - If rate is low at ALL eval thresholds:
        the optimizer genuinely failed to find a stationary-point geometry.

    The table is built from the "cascade_table" field in each results JSON
    (populated by run_minimization_parallel.py v2). Old runs without this field
    are silently skipped.
    """
    # Collect unique parameter values for each optimizer mode
    nrt_values = sorted({r.nr_threshold for r in records if r.nr_threshold > 0})
    lm_mu_values = sorted({r.lm_mu for r in records if r.lm_mu > 0})
    se_values = sorted({r.shift_epsilon for r in records if r.shift_epsilon > 0})

    def _fill_eval_columns(bucket: List[ComboRecord], row: Dict[str, Any]) -> None:
        for thr in CASCADE_THRESHOLDS:
            rates = []
            for rec in bucket:
                ct = rec.cascade_table
                rate = ct.get("rate_at_thr", {}).get(str(thr))
                if rate is not None:
                    rates.append(float(rate))
            row[f"eval_{thr}"] = _mean(rates)
        row["optimizer_strict_rate"] = _mean(r.convergence_rate for r in bucket)

    # ---- Hard-filter runs (grouped by nr_threshold) ----
    rows_nrt: List[Dict[str, Any]] = []
    for nrt in nrt_values:
        bucket = [r for r in records if math.isclose(r.nr_threshold, nrt, rel_tol=1e-6)
                  and r.lm_mu == 0.0 and r.shift_epsilon == 0.0 and r.cascade_table]
        if not bucket:
            continue
        row: Dict[str, Any] = {"optimizer_mode": "hard_filter", "nr_threshold": nrt, "lm_mu": 0.0, "shift_epsilon": 0.0}
        _fill_eval_columns(bucket, row)
        rows_nrt.append(row)

    # ---- LM-damping runs (grouped by lm_mu) ----
    rows_lm: List[Dict[str, Any]] = []
    for mu in lm_mu_values:
        bucket = [r for r in records if math.isclose(r.lm_mu, mu, rel_tol=1e-6)
                  and r.shift_epsilon == 0.0 and r.cascade_table]
        if not bucket:
            continue
        row = {"optimizer_mode": "lm_damping", "nr_threshold": 0.0, "lm_mu": mu, "shift_epsilon": 0.0}
        _fill_eval_columns(bucket, row)
        rows_lm.append(row)

    # ---- Shifted Newton runs (grouped by shift_epsilon) ----
    rows_se: List[Dict[str, Any]] = []
    for se in se_values:
        bucket = [r for r in records if math.isclose(r.shift_epsilon, se, rel_tol=1e-6)
                  and r.cascade_table]
        if not bucket:
            continue
        row = {"optimizer_mode": "shifted_newton", "nr_threshold": 0.0, "lm_mu": 0.0, "shift_epsilon": se}
        _fill_eval_columns(bucket, row)
        rows_se.append(row)

    all_rows = rows_nrt + rows_lm + rows_se

    return {
        "eval_thresholds": CASCADE_THRESHOLDS,
        "nr_thresholds_tested": nrt_values,
        "lm_mu_tested": lm_mu_values,
        "shift_epsilon_tested": se_values,
        "table": all_rows,
    }


def print_cascade_table(cross: Dict[str, Any]) -> None:
    table = cross.get("table", [])
    thresholds = cross.get("eval_thresholds", CASCADE_THRESHOLDS)
    if not table:
        print("  (no cascade data — need results from run_minimization_parallel.py v2)")
        return

    # Header
    col_w = 10
    opt_w = 32
    header = f"{'optimizer':<{opt_w}}"
    for thr in thresholds:
        header += f"{'eval≥-'+str(thr):>{col_w}}"
    header += f"{'strict':>{col_w}}"
    print(header)
    print("-" * len(header))

    for row in table:
        mode = row.get("optimizer_mode", "?")
        nrt = row.get("nr_threshold", 0.0)
        mu = row.get("lm_mu", 0.0)
        se = row.get("shift_epsilon", 0.0)
        if mode == "shifted_newton":
            label = f"SN  ε={se:g}"
        elif mode == "lm_damping":
            label = f"LM  μ={mu:g}"
        else:
            label = f"HF  nrt={nrt:g}"
        line = f"{label:<{opt_w}}"
        for thr in thresholds:
            v = row.get(f"eval_{thr}", float("nan"))
            line += f"{v:>{col_w}.3f}" if math.isfinite(v) else f"{'nan':>{col_w}}"
        v_strict = row.get("optimizer_strict_rate", float("nan"))
        line += f"{v_strict:>{col_w}.3f}" if math.isfinite(v_strict) else f"{'nan':>{col_w}}"
        print(line)

    print("")
    print("Columns = eval threshold (how strict we count n_neg at final geometry).")
    print("'strict' = optimizer's own convergence flag (n_neg==0).")
    print("Gap between eval_0.0 and eval_2e-3 → false-rejection problem.")
    print("Flat across all eval thresholds → optimizer genuinely failing.")


# ---------------------------------------------------------------------------
# Printing the main report
# ---------------------------------------------------------------------------

def print_report(
    records: List[ComboRecord],
    ranked: List[ComboRecord],
    top_k: int,
    main_effects: Dict[str, List[Dict[str, Any]]],
    sample_hardness: List[Dict[str, Any]],
    cross_table: Dict[str, Any],
) -> None:
    print(f"Loaded {len(records)} configurations.")
    print("Ranking metric: Convergence Rate (desc) → Total Converged (desc) → Mean Steps (asc) → Wall Time (asc)")
    print("")

    if ranked:
        best = ranked[0]
        print(
            "🏆 Best overall configuration by convergence, then speed:\n"
            f"  {best.tag} | converged {best.n_converged}/{best.n_samples} "
            f"({best.convergence_rate:.2f}), mean steps {best.mean_steps_when_converged:.1f}, "
            f"mean wall {best.mean_wall_time:.2f}s"
        )
        print("")

    actual_top_k = min(top_k, len(ranked))
    print("=" * 65)
    print(f"🥇 Top {actual_top_k} Configurations (Best First):")
    print("=" * 65)
    for row in ranked[:actual_top_k]:
        steps = row.mean_steps_when_converged
        steps_text = f"{steps:.1f}" if math.isfinite(steps) else "nan"
        extras = []
        if row.lm_mu > 0:
            extras.append(f"lm_mu={row.lm_mu:g}")
        if row.shift_epsilon > 0:
            extras.append(f"se={row.shift_epsilon:g}")
        if row.stagnation_window > 0:
            extras.append(f"sw={row.stagnation_window}")
        if row.total_escapes > 0:
            extras.append(f"escapes={row.total_escapes}")
        if row.lm_mu_anneal_factor > 0:
            extras.append(f"anneal={row.lm_mu_anneal_factor:g}")
        if row.neg_trust_floor > 0:
            extras.append(f"ntf={row.neg_trust_floor:g}")
        if row.blind_mode_threshold > 0:
            extras.append(f"bmt={row.blind_mode_threshold:g}")
        if row.aggressive_trust_recovery:
            extras.append("atr")
        if row.escape_bidirectional:
            extras.append("ebd")
        if row.mode_follow_eval_threshold > 0:
            extras.append(f"mft={row.mode_follow_eval_threshold:g}")
        if row.total_mode_follows > 0:
            extras.append(f"mode_follows={row.total_mode_follows}")
        extra_str = " " + " ".join(extras) if extras else ""
        print(
            f"  {row.tag}: conv={row.n_converged}/{row.n_samples} "
            f"({row.convergence_rate:.2f}), steps={steps_text}, "
            f"wall={row.mean_wall_time:.2f}s, errors={row.n_errors}"
            f"{extra_str}"
        )
    print("")

    if len(ranked) > top_k:
        actual_bottom_k = min(top_k, len(ranked) - top_k)
        print("=" * 65)
        print(f"💀 Bottom {actual_bottom_k} Configurations (Worst First):")
        print("=" * 65)
        for row in reversed(ranked[-actual_bottom_k:]):
            steps = row.mean_steps_when_converged
            steps_text = f"{steps:.1f}" if math.isfinite(steps) else "nan"
            print(
                f"  {row.tag}: conv={row.n_converged}/{row.n_samples} "
                f"({row.convergence_rate:.2f}), steps={steps_text}, "
                f"wall={row.mean_wall_time:.2f}s, errors={row.n_errors}"
            )
        print("")

    for key, rows in main_effects.items():
        print(f"--- Main effect: {key} ---")
        for row in rows:
            print(
                "  "
                f"{row['value']}: mean_conv_rate={row['mean_convergence_rate']:.3f}, "
                f"mean_n_converged={row['mean_n_converged']:.2f}, "
                f"mean_steps={row['mean_steps_when_converged']:.1f}, "
                f"mean_wall={row['mean_wall_time']:.2f}s"
            )
        print("")

    print("=" * 65)
    print("Hardest samples (lowest convergence across configs):")
    print("=" * 65)
    for row in sample_hardness[: min(10, len(sample_hardness))]:
        print(
            f"  sample_{row['sample_idx']:03d}: {row['n_converged']}/{row['n_total']} "
            f"({row['convergence_rate']:.2f}), best_step={row['best_converged_step']}, "
            f"best_combo={row['best_combo_tag']}"
        )
    print("")

    print("=" * 65)
    print("📊 Cascade Evaluation Table (optimizer filter × eval threshold)")
    print("=" * 65)
    print_cascade_table(cross_table)


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
    fieldnames = ["optimizer_mode", "nr_threshold", "lm_mu", "shift_epsilon"]
    fieldnames += [f"eval_{t}" for t in thresholds]
    fieldnames += ["optimizer_strict_rate"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in table:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze minimization NR grid-search results")
    parser.add_argument(
        "--grid-dir",
        type=str,
        required=True,
        help="Grid directory containing combo subdirectories",
    )
    parser.add_argument(
        "--result-glob",
        type=str,
        default="*/minimization_newton_raphson_*_results.json",
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
        "nr_threshold": summarize_main_effect(records, "nr_threshold"),
        "lm_mu": summarize_main_effect(records, "lm_mu"),
        "shift_epsilon": summarize_main_effect(records, "shift_epsilon"),
        "stagnation_window": summarize_main_effect(records, "stagnation_window"),
        "lm_mu_anneal_factor": summarize_main_effect(records, "lm_mu_anneal_factor"),
        "neg_trust_floor": summarize_main_effect(records, "neg_trust_floor"),
        "blind_mode_threshold": summarize_main_effect(records, "blind_mode_threshold"),
        "aggressive_trust_recovery": summarize_main_effect(records, "aggressive_trust_recovery"),
        "escape_bidirectional": summarize_main_effect(records, "escape_bidirectional"),
        "mode_follow_eval_threshold": summarize_main_effect(records, "mode_follow_eval_threshold"),
        "project_gradient_and_v": summarize_main_effect(records, "project_gradient_and_v"),
        "purify_hessian": summarize_main_effect(records, "purify_hessian"),
    }
    # Drop main effects that are constant (only one unique value) to reduce noise
    main_effects = {k: v for k, v in main_effects.items() if len(v) > 1}

    sample_hardness = summarize_sample_hardness(records)
    cross_table = build_cascade_cross_table(records)

    print_report(
        records=records,
        ranked=ranked,
        top_k=args.top_k,
        main_effects=main_effects,
        sample_hardness=sample_hardness,
        cross_table=cross_table,
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
            "sample_hardness": sample_hardness,
            "cascade_cross_table": cross_table,
            "ranked_configs": rows_for_json,
        }
        with open(out_dir / "nr_grid_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        ranked_rows = [
            {
                "rank": idx + 1,
                "tag": r.tag,
                "max_atom_disp": r.max_atom_disp,
                "nr_threshold": r.nr_threshold,
                "lm_mu": r.lm_mu,
                "shift_epsilon": r.shift_epsilon,
                "anneal_force_threshold": r.anneal_force_threshold,
                "stagnation_window": r.stagnation_window,
                "escape_alpha": r.escape_alpha,
                "lm_mu_anneal_factor": r.lm_mu_anneal_factor,
                "neg_mode_line_search": r.neg_mode_line_search,
                "neg_trust_floor": r.neg_trust_floor,
                "blind_mode_threshold": r.blind_mode_threshold,
                "blind_correction_alpha": r.blind_correction_alpha,
                "aggressive_trust_recovery": r.aggressive_trust_recovery,
                "escape_bidirectional": r.escape_bidirectional,
                "mode_follow_eval_threshold": r.mode_follow_eval_threshold,
                "project_gradient_and_v": r.project_gradient_and_v,
                "purify_hessian": r.purify_hessian,
                "n_samples": r.n_samples,
                "n_converged": r.n_converged,
                "n_errors": r.n_errors,
                "convergence_rate": r.convergence_rate,
                "mean_steps_when_converged": r.mean_steps_when_converged,
                "mean_wall_time": r.mean_wall_time,
                "total_wall_time": r.total_wall_time,
                "total_escapes": r.total_escapes,
                "total_line_searches": r.total_line_searches,
                "total_mode_follows": r.total_mode_follows,
                "path": r.path,
            }
            for idx, r in enumerate(ranked)
        ]
        write_csv(
            out_dir / "nr_grid_ranked.csv",
            ranked_rows,
            [
                "rank", "tag", "max_atom_disp", "nr_threshold", "lm_mu",
                "shift_epsilon", "anneal_force_threshold",
                "stagnation_window", "escape_alpha", "lm_mu_anneal_factor",
                "neg_mode_line_search",
                "neg_trust_floor", "blind_mode_threshold",
                "blind_correction_alpha", "aggressive_trust_recovery",
                "escape_bidirectional", "mode_follow_eval_threshold",
                "project_gradient_and_v", "purify_hessian",
                "n_samples", "n_converged", "n_errors", "convergence_rate",
                "mean_steps_when_converged", "mean_wall_time", "total_wall_time",
                "total_escapes", "total_line_searches",
                "total_mode_follows", "path",
            ],
        )
        write_csv(
            out_dir / "nr_grid_sample_hardness.csv",
            sample_hardness,
            [
                "sample_idx", "n_converged", "n_total", "convergence_rate",
                "best_converged_step", "best_combo_tag",
            ],
        )
        write_cascade_csv(out_dir / "nr_grid_cascade_table.csv", cross_table)

        print("")
        print(f"Wrote analysis artifacts to: {out_dir}")
        print("  nr_grid_summary.json         — full summary (JSON)")
        print("  nr_grid_ranked.csv           — ranked configurations")
        print("  nr_grid_sample_hardness.csv  — per-sample convergence rates")
        print("  nr_grid_cascade_table.csv    — 2D cascade diagnostic table")


if __name__ == "__main__":
    main()
