#!/usr/bin/env python3
"""Analyze GAD grid-search outputs.

Expected directory layout:
  <grid_dir>/mad*_tr*_bl*_pg*/gad_*_parallel_*_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


COMBO_RE = re.compile(
    r"mad(?P<mad>[^_]+)_tr(?P<tr>[^_]+)_bl(?P<bl>[^_]+)_pg(?P<pg>true|false)$"
)


@dataclass
class ComboRecord:
    tag: str
    path: str
    max_atom_disp: float
    tr_threshold: float
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
    results: List[Dict[str, Any]]


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


def load_records(grid_dir: Path, result_glob: str) -> List[ComboRecord]:
    records: List[ComboRecord] = []

    for result_path in sorted(grid_dir.glob(result_glob)):
        if not result_path.is_file():
            continue

        combo_tag = result_path.parent.name
        match = COMBO_RE.fullmatch(combo_tag)
        if not match:
            raise ValueError(
                f"Cannot parse combo tag from folder name: {combo_tag} "
                f"(path: {result_path})"
            )

        with open(result_path) as f:
            payload = json.load(f)

        metrics = payload.get("metrics", {})
        neg_vib_raw = metrics.get("neg_vib_counts", {})
        neg_vib_counts = {int(k): int(v) for k, v in neg_vib_raw.items()}

        records.append(
            ComboRecord(
                tag=combo_tag,
                path=str(result_path),
                max_atom_disp=_safe_float(match.group("mad")),
                tr_threshold=_safe_float(match.group("tr")),
                baseline=match.group("bl"),
                project_gradient_and_v=(match.group("pg") == "true"),
                n_samples=int(metrics.get("n_samples", 0)),
                n_success=int(metrics.get("n_success", 0)),
                n_errors=int(metrics.get("n_errors", 0)),
                success_rate=_safe_float(metrics.get("success_rate"), 0.0),
                mean_steps_when_success=_safe_float(metrics.get("mean_steps_when_success")),
                mean_wall_time=_safe_float(metrics.get("mean_wall_time")),
                total_wall_time=_safe_float(metrics.get("total_wall_time")),
                neg_vib_counts=neg_vib_counts,
                results=list(metrics.get("results", [])),
            )
        )

    return records


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


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_report(
    records: List[ComboRecord],
    ranked: List[ComboRecord],
    top_k: int,
    main_effects: Dict[str, List[Dict[str, Any]]],
    mad_tr_table: List[Dict[str, Any]],
    baseline_table: List[Dict[str, Any]],
    sample_hardness: List[Dict[str, Any]],
    neg_vib_dist: Dict[str, Any],
) -> None:
    print(f"Loaded {len(records)} configurations.")
    print("Ranking metric: Success Rate (desc) -> Total Success (desc) -> Mean Steps (asc) -> Mean Wall Time (asc)")
    print("")

    if ranked:
        best = ranked[0]
        print(
            "Best overall configuration by success rate, then speed:\n"
            f"  {best.tag} | succeeded {best.n_success}/{best.n_samples} "
            f"({best.success_rate:.2f}), mean steps {best.mean_steps_when_success:.1f}, "
            f"mean wall {best.mean_wall_time:.2f}s"
        )
        print("")

    actual_top_k = min(top_k, len(ranked))
    print("=========================================================")
    print(f"Top {actual_top_k} Configurations (Best First):")
    print("=========================================================")
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
        print("=========================================================")
        print(f"Bottom {actual_bottom_k} Configurations (Worst First):")
        print("=========================================================")
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

    print("=========================================================")
    print("Final Morse index distribution across all runs:")
    print("=========================================================")
    for neg_vib, count in neg_vib_dist.items():
        label = {1: "-> TS (index-1)", 0: "-> minimum (index-0)"}.get(neg_vib, "other")
        print(f"  neg_vib={neg_vib}: {count} runs  {label}")
    print("")

    print("=========================================================")
    print("Hardest samples (lowest success rate across configs):")
    print("=========================================================")
    for row in sample_hardness[: min(10, len(sample_hardness))]:
        print(
            f"  sample_{row['sample_idx']:03d}: {row['n_success']}/{row['n_total']} "
            f"({row['success_rate']:.2f}), best_steps={row['best_steps_to_ts']}, "
            f"best_combo={row['best_combo_tag']}"
        )
    print("")


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
        "baseline": summarize_main_effect(records, "baseline"),
        "project_gradient_and_v": summarize_main_effect(records, "project_gradient_and_v"),
    }
    mad_tr_table = summarize_mad_tr_interaction(records)
    baseline_table = summarize_baseline_interaction(records)
    sample_hardness = summarize_sample_hardness(records)
    neg_vib_dist = summarize_neg_vib_distribution(records)

    print_report(
        records=records,
        ranked=ranked,
        top_k=args.top_k,
        main_effects=main_effects,
        mad_tr_table=mad_tr_table,
        baseline_table=baseline_table,
        sample_hardness=sample_hardness,
        neg_vib_dist=neg_vib_dist,
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
                "rank",
                "tag",
                "max_atom_disp",
                "tr_threshold",
                "baseline",
                "project_gradient_and_v",
                "n_samples",
                "n_success",
                "n_errors",
                "success_rate",
                "mean_steps_when_success",
                "mean_wall_time",
                "total_wall_time",
                "path",
            ],
        )
        write_csv(
            out_dir / "gad_grid_sample_hardness.csv",
            sample_hardness,
            [
                "sample_idx",
                "n_success",
                "n_total",
                "success_rate",
                "best_steps_to_ts",
                "best_combo_tag",
            ],
        )
        print("")
        print(f"Wrote analysis artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
