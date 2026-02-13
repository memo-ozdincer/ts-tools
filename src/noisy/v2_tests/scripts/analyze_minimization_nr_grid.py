#!/usr/bin/env python3
"""Analyze Newton-Raphson minimization grid-search outputs.

Expected directory layout:
  <grid_dir>/mad*_tr*_pg*_ph*/minimization_newton_raphson_*_results.json
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
    r"mad(?P<mad>[^_]+)_tr(?P<tr>[^_]+)_pg(?P<pg>true|false)_ph(?P<ph>true|false)$"
)


@dataclass
class ComboRecord:
    tag: str
    path: str
    max_atom_disp: float
    tr_threshold: float
    project_gradient_and_v: bool
    purify_hessian: bool
    n_samples: int
    n_converged: int
    n_errors: int
    convergence_rate: float
    mean_steps_when_converged: float
    mean_wall_time: float
    total_wall_time: float
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
        records.append(
            ComboRecord(
                tag=combo_tag,
                path=str(result_path),
                max_atom_disp=_safe_float(match.group("mad")),
                tr_threshold=_safe_float(match.group("tr")),
                project_gradient_and_v=(match.group("pg") == "true"),
                purify_hessian=(match.group("ph") == "true"),
                n_samples=int(metrics.get("n_samples", 0)),
                n_converged=int(metrics.get("n_converged", 0)),
                n_errors=int(metrics.get("n_errors", 0)),
                convergence_rate=_safe_float(metrics.get("convergence_rate"), 0.0),
                mean_steps_when_converged=_safe_float(metrics.get("mean_steps_when_converged")),
                mean_wall_time=_safe_float(metrics.get("mean_wall_time")),
                total_wall_time=_safe_float(metrics.get("total_wall_time")),
                results=list(metrics.get("results", [])),
            )
        )

    return records


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


def summarize_mad_tr_interaction(records: List[ComboRecord]) -> List[Dict[str, Any]]:
    mads = sorted({r.max_atom_disp for r in records})
    trs = sorted({r.tr_threshold for r in records})

    table: List[Dict[str, Any]] = []
    for mad in mads:
        row: Dict[str, Any] = {"max_atom_disp": mad}
        for tr in trs:
            bucket = [r for r in records if r.max_atom_disp == mad and r.tr_threshold == tr]
            row[f"tr_{tr:g}"] = _mean(r.convergence_rate for r in bucket)
        table.append(row)
    return table


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


def summarize_toggle_effects(records: List[ComboRecord]) -> Dict[str, Any]:
    by_key: Dict[Tuple[float, float, bool, bool], ComboRecord] = {}
    for r in records:
        by_key[(r.max_atom_disp, r.tr_threshold, r.project_gradient_and_v, r.purify_hessian)] = r

    ph_total = 0
    ph_identical = 0
    for mad in sorted({r.max_atom_disp for r in records}):
        for tr in sorted({r.tr_threshold for r in records}):
            for pg in (False, True):
                a = by_key.get((mad, tr, pg, False))
                b = by_key.get((mad, tr, pg, True))
                if a is None or b is None:
                    continue
                ph_total += 1
                if _same_sample_outcomes(a, b):
                    ph_identical += 1

    pg_deltas: List[Dict[str, Any]] = []
    for mad in sorted({r.max_atom_disp for r in records}):
        for tr in sorted({r.tr_threshold for r in records}):
            for ph in (False, True):
                off = by_key.get((mad, tr, False, ph))
                on = by_key.get((mad, tr, True, ph))
                if off is None or on is None:
                    continue
                off_steps = off.mean_steps_when_converged
                on_steps = on.mean_steps_when_converged
                delta_steps = (
                    on_steps - off_steps
                    if math.isfinite(off_steps) and math.isfinite(on_steps)
                    else float("nan")
                )
                pg_deltas.append(
                    {
                        "max_atom_disp": mad,
                        "tr_threshold": tr,
                        "purify_hessian": ph,
                        "delta_convergence_rate_pg_true_minus_false": (
                            on.convergence_rate - off.convergence_rate
                        ),
                        "delta_mean_steps_pg_true_minus_false": delta_steps,
                    }
                )

    return {
        "purify_hessian_identical_pairs": ph_identical,
        "purify_hessian_total_pairs": ph_total,
        "project_gradient_deltas": pg_deltas,
    }


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
    sample_hardness: List[Dict[str, Any]],
    toggle_effects: Dict[str, Any],
) -> None:
    print(f"Loaded {len(records)} configurations.")
    print("")

    if ranked:
        best = ranked[0]
        print(
            "Best overall by convergence then speed: "
            f"{best.tag} | converged {best.n_converged}/{best.n_samples} "
            f"({best.convergence_rate:.2f}), mean steps {best.mean_steps_when_converged:.1f}, "
            f"mean wall {best.mean_wall_time:.2f}s"
        )
        print("")

    print(f"Top {top_k}:")
    for row in ranked[:top_k]:
        steps = row.mean_steps_when_converged
        steps_text = f"{steps:.1f}" if math.isfinite(steps) else "nan"
        print(
            f"  {row.tag}: conv={row.n_converged}/{row.n_samples} "
            f"({row.convergence_rate:.2f}), steps={steps_text}, "
            f"wall={row.mean_wall_time:.2f}s, errors={row.n_errors}"
        )
    print("")

    print(f"Bottom {top_k}:")
    for row in ranked[-top_k:]:
        steps = row.mean_steps_when_converged
        steps_text = f"{steps:.1f}" if math.isfinite(steps) else "nan"
        print(
            f"  {row.tag}: conv={row.n_converged}/{row.n_samples} "
            f"({row.convergence_rate:.2f}), steps={steps_text}, "
            f"wall={row.mean_wall_time:.2f}s, errors={row.n_errors}"
        )
    print("")

    for key, rows in main_effects.items():
        print(f"Main effect: {key}")
        for row in rows:
            print(
                "  "
                f"{row['value']}: mean_conv_rate={row['mean_convergence_rate']:.3f}, "
                f"mean_n_converged={row['mean_n_converged']:.2f}, "
                f"mean_steps={row['mean_steps_when_converged']:.1f}, "
                f"mean_wall={row['mean_wall_time']:.2f}s"
            )
        print("")

    print("Interaction: mean convergence rate by max_atom_disp x tr_threshold")
    for row in mad_tr_table:
        parts = [f"mad={row['max_atom_disp']:g}"]
        for key in sorted(k for k in row if k.startswith("tr_")):
            parts.append(f"{key.replace('tr_', 'tr=')}:{row[key]:.2f}")
        print("  " + ", ".join(parts))
    print("")

    print("Hardest samples (lowest convergence across configs):")
    for row in sample_hardness[: min(10, len(sample_hardness))]:
        print(
            f"  sample_{row['sample_idx']:03d}: {row['n_converged']}/{row['n_total']} "
            f"({row['convergence_rate']:.2f}), best_step={row['best_converged_step']}, "
            f"best_combo={row['best_combo_tag']}"
        )
    print("")

    ph_identical = toggle_effects["purify_hessian_identical_pairs"]
    ph_total = toggle_effects["purify_hessian_total_pairs"]
    print(
        "Purify-Hessian toggle identical sample outcomes "
        f"(converged + converged_step): {ph_identical}/{ph_total} pairs"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze minimization NR grid-search results")
    parser.add_argument(
        "--grid-dir",
        type=str,
        required=True,
        help="Grid directory containing mad*_tr*_pg*_ph* subdirectories",
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
        "tr_threshold": summarize_main_effect(records, "tr_threshold"),
        "project_gradient_and_v": summarize_main_effect(records, "project_gradient_and_v"),
        "purify_hessian": summarize_main_effect(records, "purify_hessian"),
    }
    mad_tr_table = summarize_mad_tr_interaction(records)
    sample_hardness = summarize_sample_hardness(records)
    toggle_effects = summarize_toggle_effects(records)

    print_report(
        records=records,
        ranked=ranked,
        top_k=args.top_k,
        main_effects=main_effects,
        mad_tr_table=mad_tr_table,
        sample_hardness=sample_hardness,
        toggle_effects=toggle_effects,
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
            "sample_hardness": sample_hardness,
            "toggle_effects": toggle_effects,
            "ranked_configs": rows_for_json,
        }
        with open(out_dir / "nr_grid_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        ranked_rows = [
            {
                "rank": idx + 1,
                "tag": r.tag,
                "max_atom_disp": r.max_atom_disp,
                "tr_threshold": r.tr_threshold,
                "project_gradient_and_v": r.project_gradient_and_v,
                "purify_hessian": r.purify_hessian,
                "n_samples": r.n_samples,
                "n_converged": r.n_converged,
                "n_errors": r.n_errors,
                "convergence_rate": r.convergence_rate,
                "mean_steps_when_converged": r.mean_steps_when_converged,
                "mean_wall_time": r.mean_wall_time,
                "total_wall_time": r.total_wall_time,
                "path": r.path,
            }
            for idx, r in enumerate(ranked)
        ]
        write_csv(
            out_dir / "nr_grid_ranked.csv",
            ranked_rows,
            [
                "rank",
                "tag",
                "max_atom_disp",
                "tr_threshold",
                "project_gradient_and_v",
                "purify_hessian",
                "n_samples",
                "n_converged",
                "n_errors",
                "convergence_rate",
                "mean_steps_when_converged",
                "mean_wall_time",
                "total_wall_time",
                "path",
            ],
        )
        write_csv(
            out_dir / "nr_grid_sample_hardness.csv",
            sample_hardness,
            [
                "sample_idx",
                "n_converged",
                "n_total",
                "convergence_rate",
                "best_converged_step",
                "best_combo_tag",
            ],
        )
        print("")
        print(f"Wrote analysis artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
