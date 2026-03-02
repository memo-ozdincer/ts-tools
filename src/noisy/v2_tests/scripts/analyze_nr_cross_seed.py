#!/usr/bin/env python3
"""Cross-seed aggregation analysis for multi-seed NR minimization experiments.

Discovers results across noise levels and seeds from the v5 directory structure:

    <grid_dir>/noise_<X>A/seed_<Y>/<combo_tag>/minimization_*_results.json

Produces:
  - Mean ± std convergence rates per optimizer config across seeds
  - Per-noise-level difficulty curves
  - Identification of universally-hard vs seed-dependent samples
  - Per-sample convergence consistency across seeds
  - CSV and text report

Usage:
  python analyze_nr_cross_seed.py --grid-dir <path> --output-dir <path>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(vals: List[float]) -> float:
    f = [v for v in vals if math.isfinite(v)]
    return sum(f) / len(f) if f else float("nan")


def _std(vals: List[float]) -> float:
    f = [v for v in vals if math.isfinite(v)]
    if len(f) < 2:
        return float("nan")
    m = sum(f) / len(f)
    return (sum((v - m) ** 2 for v in f) / (len(f) - 1)) ** 0.5


def _fmt(v: float, fmt: str = ".3f") -> str:
    if math.isfinite(v):
        return format(v, fmt)
    return "N/A"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_structure(grid_dir: Path) -> Dict[str, Dict[str, Dict[str, Path]]]:
    """Discover the noise_level → seed → combo_tag → result_file structure.

    Returns nested dict: {noise_level: {seed: {combo_tag: result_path}}}.
    """
    structure: Dict[str, Dict[str, Dict[str, Path]]] = {}

    for noise_dir in sorted(grid_dir.iterdir()):
        if not noise_dir.is_dir() or not noise_dir.name.startswith("noise_"):
            continue
        noise_level = noise_dir.name  # e.g., "noise_0.5A"

        for seed_dir in sorted(noise_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed = seed_dir.name  # e.g., "seed_42"

            for combo_dir in sorted(seed_dir.iterdir()):
                if not combo_dir.is_dir():
                    continue
                combo_tag = combo_dir.name

                # Find the result JSON
                result_files = list(combo_dir.glob("minimization_newton_raphson_*_results.json"))
                if not result_files:
                    continue
                result_path = result_files[0]

                structure.setdefault(noise_level, {}).setdefault(seed, {})[combo_tag] = result_path

    return structure


def load_result(path: Path) -> Optional[Dict[str, Any]]:
    """Load a single result JSON and extract key metrics."""
    try:
        with open(path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    metrics = payload.get("metrics", {})
    per_sample = list(metrics.get("results", []))
    return {
        "n_samples": int(metrics.get("n_samples", 0)),
        "n_converged": int(metrics.get("n_converged", 0)),
        "n_errors": int(metrics.get("n_errors", 0)),
        "convergence_rate": _safe_float(metrics.get("convergence_rate"), 0.0),
        "mean_steps_when_converged": _safe_float(metrics.get("mean_steps_when_converged")),
        "mean_wall_time": _safe_float(metrics.get("mean_wall_time")),
        "per_sample": per_sample,
        "cascade_table": metrics.get("cascade_table", {}),
    }


# ---------------------------------------------------------------------------
# Analysis modules
# ---------------------------------------------------------------------------

def analyze_convergence_by_noise_and_optimizer(
    structure: Dict[str, Dict[str, Dict[str, Path]]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Main analysis: convergence rate by noise level × optimizer, averaged across seeds."""
    lines: List[str] = []
    csv_rows: List[Dict[str, Any]] = []

    lines.append("=" * 70)
    lines.append("CONVERGENCE BY NOISE LEVEL × OPTIMIZER (averaged across seeds)")
    lines.append("=" * 70)

    # Collect all combo tags across all noise/seed pairs
    all_combos: set = set()
    for noise_level in structure:
        for seed in structure[noise_level]:
            all_combos.update(structure[noise_level][seed].keys())
    combo_list = sorted(all_combos)

    for noise_level in sorted(structure.keys()):
        seeds_data = structure[noise_level]
        seed_list = sorted(seeds_data.keys())
        n_seeds = len(seed_list)

        lines.append(f"\n--- {noise_level} ({n_seeds} seeds) ---")

        for combo_tag in combo_list:
            rates: List[float] = []
            steps_list: List[float] = []
            n_errors_list: List[int] = []
            n_samples_list: List[int] = []
            n_converged_list: List[int] = []

            for seed in seed_list:
                path = seeds_data.get(seed, {}).get(combo_tag)
                if path is None:
                    continue
                result = load_result(path)
                if result is None:
                    continue
                rates.append(result["convergence_rate"])
                steps_list.append(result["mean_steps_when_converged"])
                n_errors_list.append(result["n_errors"])
                n_samples_list.append(result["n_samples"])
                n_converged_list.append(result["n_converged"])

            if not rates:
                continue

            mean_rate = _mean(rates)
            std_rate = _std(rates)
            mean_steps = _mean(steps_list)
            total_conv = sum(n_converged_list)
            total_samp = sum(n_samples_list)
            total_err = sum(n_errors_list)

            lines.append(
                f"  {combo_tag:<50s}  "
                f"conv={_fmt(mean_rate)} ± {_fmt(std_rate)}  "
                f"({total_conv}/{total_samp})  "
                f"steps={_fmt(mean_steps, '.0f')}  "
                f"errors={total_err}"
            )

            csv_rows.append({
                "noise_level": noise_level,
                "combo_tag": combo_tag,
                "n_seeds": len(rates),
                "mean_conv_rate": mean_rate,
                "std_conv_rate": std_rate,
                "min_conv_rate": min(rates) if rates else float("nan"),
                "max_conv_rate": max(rates) if rates else float("nan"),
                "total_converged": total_conv,
                "total_samples": total_samp,
                "total_errors": total_err,
                "mean_steps": mean_steps,
            })

    return "\n".join(lines), csv_rows


def analyze_difficulty_curve(
    structure: Dict[str, Dict[str, Dict[str, Path]]],
) -> str:
    """Convergence rate vs noise magnitude — the difficulty curve."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("DIFFICULTY CURVE (convergence rate vs noise magnitude)")
    lines.append("=" * 70)

    all_combos: set = set()
    for noise_level in structure:
        for seed in structure[noise_level]:
            all_combos.update(structure[noise_level][seed].keys())
    combo_list = sorted(all_combos)

    # Header
    noise_levels = sorted(structure.keys())
    header = f"{'optimizer':<50s}"
    for nl in noise_levels:
        header += f"  {nl:>12s}"
    lines.append(header)
    lines.append("-" * len(header))

    for combo_tag in combo_list:
        row = f"{combo_tag:<50s}"
        for noise_level in noise_levels:
            seeds_data = structure.get(noise_level, {})
            rates: List[float] = []
            for seed in seeds_data:
                path = seeds_data[seed].get(combo_tag)
                if path is None:
                    continue
                result = load_result(path)
                if result is None:
                    continue
                rates.append(result["convergence_rate"])
            if rates:
                m = _mean(rates)
                s = _std(rates)
                row += f"  {_fmt(m)}±{_fmt(s):>5s}"
            else:
                row += f"  {'N/A':>12s}"
        lines.append(row)

    return "\n".join(lines)


def analyze_sample_consistency(
    structure: Dict[str, Dict[str, Dict[str, Path]]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Which samples are universally hard vs seed-dependent?

    For each sample index, count how often it converges across all
    noise_level × seed × combo combinations.
    """
    lines: List[str] = []
    csv_rows: List[Dict[str, Any]] = []

    lines.append("=" * 70)
    lines.append("SAMPLE CONSISTENCY ACROSS SEEDS")
    lines.append("=" * 70)

    # Per noise level, per sample: track convergence across seeds×combos
    for noise_level in sorted(structure.keys()):
        seeds_data = structure[noise_level]
        sample_conv_counts: Dict[int, int] = defaultdict(int)
        sample_total_counts: Dict[int, int] = defaultdict(int)
        sample_seed_conv: Dict[int, Dict[str, bool]] = defaultdict(dict)

        for seed in sorted(seeds_data.keys()):
            for combo_tag, path in seeds_data[seed].items():
                result = load_result(path)
                if result is None:
                    continue
                for r in result["per_sample"]:
                    idx = r.get("sample_idx")
                    if idx is None:
                        continue
                    idx = int(idx)
                    converged = bool(r.get("converged"))
                    sample_total_counts[idx] += 1
                    if converged:
                        sample_conv_counts[idx] += 1
                    # Track per-seed convergence (any combo)
                    key = f"{seed}_{combo_tag}"
                    sample_seed_conv[idx][key] = converged

        if not sample_total_counts:
            continue

        lines.append(f"\n--- {noise_level} ---")

        # Classify samples
        n_seeds = len(seeds_data)
        all_combos = set()
        for seed in seeds_data:
            all_combos.update(seeds_data[seed].keys())
        n_combos = len(all_combos)
        max_trials = n_seeds * n_combos

        never_converge: List[int] = []
        always_converge: List[int] = []
        seed_dependent: List[Tuple[int, float]] = []

        for idx in sorted(sample_total_counts.keys()):
            conv = sample_conv_counts[idx]
            total = sample_total_counts[idx]
            rate = conv / max(total, 1)

            if conv == 0:
                never_converge.append(idx)
            elif conv == total:
                always_converge.append(idx)
            else:
                seed_dependent.append((idx, rate))

            csv_rows.append({
                "noise_level": noise_level,
                "sample_idx": idx,
                "n_converged": conv,
                "n_total": total,
                "convergence_rate": rate,
                "category": "never" if conv == 0 else ("always" if conv == total else "variable"),
            })

        lines.append(f"  Total samples seen: {len(sample_total_counts)}")
        lines.append(f"  Max trials per sample: {max_trials} ({n_seeds} seeds × {n_combos} combos)")
        lines.append(f"  Always converge (100%): {len(always_converge)} samples")
        lines.append(f"  Never converge (0%):    {len(never_converge)} samples")
        lines.append(f"  Seed-dependent:         {len(seed_dependent)} samples")

        if never_converge:
            lines.append(f"    Never: {', '.join(f'sample_{i:03d}' for i in never_converge[:15])}"
                        + (" ..." if len(never_converge) > 15 else ""))
        if seed_dependent:
            seed_dependent.sort(key=lambda x: x[1])
            lines.append("    Seed-dependent (hardest first):")
            for idx, rate in seed_dependent[:10]:
                lines.append(f"      sample_{idx:03d}: {rate:.1%}")

    return "\n".join(lines), csv_rows


def analyze_seed_variance(
    structure: Dict[str, Dict[str, Dict[str, Path]]],
) -> str:
    """How much does convergence rate vary across seeds for each optimizer?

    High variance → optimizer is sensitive to starting geometry.
    Low variance → optimizer is robust.
    """
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("SEED VARIANCE ANALYSIS (optimizer robustness)")
    lines.append("=" * 70)

    all_combos: set = set()
    for noise_level in structure:
        for seed in structure[noise_level]:
            all_combos.update(structure[noise_level][seed].keys())
    combo_list = sorted(all_combos)

    for noise_level in sorted(structure.keys()):
        seeds_data = structure[noise_level]
        lines.append(f"\n--- {noise_level} ---")

        combo_stats: List[Tuple[str, float, float, List[float]]] = []
        for combo_tag in combo_list:
            rates: List[float] = []
            for seed in sorted(seeds_data.keys()):
                path = seeds_data.get(seed, {}).get(combo_tag)
                if path is None:
                    continue
                result = load_result(path)
                if result is None:
                    continue
                rates.append(result["convergence_rate"])
            if rates:
                combo_stats.append((combo_tag, _mean(rates), _std(rates), rates))

        # Sort by mean rate descending
        combo_stats.sort(key=lambda x: -x[1])
        for combo_tag, mean_r, std_r, rates in combo_stats:
            per_seed = "  ".join(f"{r:.2f}" for r in rates)
            lines.append(
                f"  {combo_tag:<50s}  "
                f"mean={_fmt(mean_r)}  std={_fmt(std_r)}  "
                f"[{per_seed}]"
            )

    return "\n".join(lines)


def analyze_cascade_across_seeds(
    structure: Dict[str, Dict[str, Dict[str, Path]]],
) -> str:
    """Aggregate cascade table across seeds — shows false-rejection problem at scale."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("CASCADE TABLE (aggregated across seeds)")
    lines.append("=" * 70)

    cascade_thresholds = [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2]

    all_combos: set = set()
    for noise_level in structure:
        for seed in structure[noise_level]:
            all_combos.update(structure[noise_level][seed].keys())
    combo_list = sorted(all_combos)

    for noise_level in sorted(structure.keys()):
        seeds_data = structure[noise_level]
        lines.append(f"\n--- {noise_level} ---")

        # Header
        col_w = 10
        opt_w = 52
        header = f"{'optimizer':<{opt_w}}"
        for thr in cascade_thresholds:
            header += f"{'eval≥-'+str(thr):>{col_w}}"
        header += f"{'strict':>{col_w}}"
        lines.append(header)
        lines.append("-" * len(header))

        for combo_tag in combo_list:
            # Aggregate cascade rates across seeds
            thr_rates: Dict[float, List[float]] = {t: [] for t in cascade_thresholds}
            strict_rates: List[float] = []

            for seed in sorted(seeds_data.keys()):
                path = seeds_data.get(seed, {}).get(combo_tag)
                if path is None:
                    continue
                result = load_result(path)
                if result is None:
                    continue

                ct = result.get("cascade_table", {})
                rate_at_thr = ct.get("rate_at_thr", {})
                for thr in cascade_thresholds:
                    r = rate_at_thr.get(str(thr))
                    if r is not None:
                        thr_rates[thr].append(float(r))
                strict_rates.append(result["convergence_rate"])

            if not strict_rates:
                continue

            line = f"{combo_tag:<{opt_w}}"
            for thr in cascade_thresholds:
                vals = thr_rates[thr]
                if vals:
                    line += f"{_mean(vals):>{col_w}.3f}"
                else:
                    line += f"{'N/A':>{col_w}}"
            line += f"{_mean(strict_rates):>{col_w}.3f}"
            lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-seed aggregation analysis for multi-seed NR experiments"
    )
    parser.add_argument(
        "--grid-dir", required=True,
        help="Top-level grid directory containing noise_*/seed_*/combo_tag/ structure",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write analysis outputs",
    )
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning directory structure ...")
    structure = discover_structure(grid_dir)

    if not structure:
        print(f"No noise_*/seed_*/combo/ structure found in {grid_dir}")
        print("Expected: <grid_dir>/noise_<X>A/seed_<Y>/<combo_tag>/minimization_*_results.json")
        return

    # Summary of what was found
    n_noise = len(structure)
    n_seeds_per_noise = {nl: len(seeds) for nl, seeds in structure.items()}
    n_combos_per_seed: Dict[str, set] = {}
    for nl, seeds in structure.items():
        for seed, combos in seeds.items():
            n_combos_per_seed.setdefault(nl, set()).update(combos.keys())

    total_results = sum(
        len(combos) for seeds in structure.values() for combos in seeds.values()
    )

    print(f"Found {n_noise} noise levels, {total_results} result files total")
    for nl in sorted(structure.keys()):
        n_s = n_seeds_per_noise[nl]
        n_c = len(n_combos_per_seed.get(nl, set()))
        print(f"  {nl}: {n_s} seeds × {n_c} combos = {n_s * n_c} expected results")
    print()

    # ===== Run analyses =====
    print("=" * 70)
    print("CROSS-SEED AGGREGATION ANALYSIS REPORT")
    print("=" * 70)
    print()

    # 1. Convergence by noise × optimizer
    conv_report, conv_csv = analyze_convergence_by_noise_and_optimizer(structure)
    print(conv_report)
    print()

    # 2. Difficulty curve
    diff_report = analyze_difficulty_curve(structure)
    print(diff_report)
    print()

    # 3. Sample consistency
    sample_report, sample_csv = analyze_sample_consistency(structure)
    print(sample_report)
    print()

    # 4. Seed variance
    var_report = analyze_seed_variance(structure)
    print(var_report)
    print()

    # 5. Cascade across seeds
    cascade_report = analyze_cascade_across_seeds(structure)
    print(cascade_report)
    print()

    # ===== Summary =====
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Find best config per noise level
    for nl in sorted(structure.keys()):
        best_tag = None
        best_mean = -1.0
        best_std = float("inf")
        seeds_data = structure[nl]

        all_combos: set = set()
        for seed in seeds_data:
            all_combos.update(seeds_data[seed].keys())

        for combo_tag in all_combos:
            rates: List[float] = []
            for seed in sorted(seeds_data.keys()):
                path = seeds_data.get(seed, {}).get(combo_tag)
                if path is None:
                    continue
                result = load_result(path)
                if result is None:
                    continue
                rates.append(result["convergence_rate"])
            if rates and _mean(rates) > best_mean:
                best_mean = _mean(rates)
                best_std = _std(rates)
                best_tag = combo_tag

        if best_tag:
            print(f"  {nl}: best={best_tag}, rate={_fmt(best_mean)} ± {_fmt(best_std)}")

    print()

    # ===== Write outputs =====
    # Convergence CSV
    if conv_csv:
        conv_path = out_dir / "convergence_by_noise_optimizer.csv"
        fieldnames = list(conv_csv[0].keys())
        with open(conv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in conv_csv:
                writer.writerow(row)
        print(f"Wrote: {conv_path}")

    # Sample consistency CSV
    if sample_csv:
        sample_path = out_dir / "sample_consistency.csv"
        fieldnames = list(sample_csv[0].keys())
        with open(sample_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sample_csv:
                writer.writerow(row)
        print(f"Wrote: {sample_path}")

    # Full JSON summary
    json_path = out_dir / "cross_seed_summary.json"
    summary = {
        "noise_levels": sorted(structure.keys()),
        "seeds_per_noise": n_seeds_per_noise,
        "total_results": total_results,
        "convergence_table": conv_csv,
        "sample_consistency": sample_csv,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote: {json_path}")

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
