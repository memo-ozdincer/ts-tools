#!/usr/bin/env python3
"""Verify Transition1x DFT-labelled minima on the DFTB0 surface.

For each sample's reactant/product geometry, this script evaluates DFTB0
energy/forces/Hessian, computes vibrational eigenvalues, and reports whether
the point is a true minimum under DFTB0 (n_neg == 0) or a saddle (n_neg > 0).

Optionally, if --results-dir is provided, it cross-references optimizer runs
and measures correlation between n_neg at the DFT-labelled geometry and
optimizer failure rate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - dependency optional for --help path
    np = None

try:
    import torch
    from torch_geometric.loader import DataLoader
except Exception:  # pragma: no cover - dependency optional for --help path
    torch = None
    DataLoader = None

try:
    from src.dependencies.common_utils import Transition1xDataset, UsePos
    from src.noisy.multi_mode_eckartmw import _atomic_nums_to_symbols, get_vib_evals_evecs
    from src.noisy.v2_tests.baselines.minimization import (
        _bottom_k_spectrum,
        _eigenvalue_band_populations,
        _force_mean,
    )
    from src.parallel.scine_parallel import ParallelSCINEProcessor
    from src.parallel.utils import run_batch_parallel
except Exception:  # pragma: no cover - dependency optional for --help path
    Transition1xDataset = None
    UsePos = None
    _atomic_nums_to_symbols = None
    get_vib_evals_evecs = None
    _bottom_k_spectrum = None
    _eigenvalue_band_populations = None
    _force_mean = None
    ParallelSCINEProcessor = None
    run_batch_parallel = None


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "n": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
        }
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def _rank_with_ties(values: List[float]) -> List[float]:
    pairs = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[pairs[k][1]] = avg_rank
        i = j
    return ranks


def _pearson(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _spearman(x: List[float], y: List[float]) -> float:
    return _pearson(_rank_with_ties(x), _rank_with_ties(y))


def evaluate_geometry(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    atomsymbols: List[str],
    *,
    log_spectrum_k: int,
) -> Dict[str, Any]:
    coords_flat = coords.reshape(1, -1).detach().requires_grad_(True)
    out = predict_fn(coords_flat, atomic_nums, do_hessian=True, require_grad=False)

    energy = float(out["energy"].item())
    forces = out["forces"].detach().to("cpu")
    hessian = out["hessian"].detach().to("cpu")
    force_norm = _force_mean(forces)

    evals_vib, _evecs_vib_3n, _q_vib = get_vib_evals_evecs(
        hessian, coords.reshape(-1, 3).detach().to("cpu"), atomsymbols, purify_hessian=False,
    )

    evals_sorted = torch.sort(evals_vib).values
    n_neg = int((evals_sorted < 0.0).sum().item())
    min_eval = float(evals_sorted[0].item()) if evals_sorted.numel() > 0 else float("nan")
    max_eval = float(evals_sorted[-1].item()) if evals_sorted.numel() > 0 else float("nan")
    abs_vals = torch.abs(evals_sorted)
    cond = float(abs_vals.max().item() / max(abs_vals.min().item(), 1e-30)) if abs_vals.numel() > 0 else float("nan")

    return {
        "energy": energy,
        "force_norm": force_norm,
        "n_neg": n_neg,
        "is_dftb0_minimum": bool(n_neg == 0),
        "min_eval": min_eval,
        "max_eval": max_eval,
        "condition_number": cond,
        "log10_condition_number": math.log10(cond) if cond > 0 and math.isfinite(cond) else float("nan"),
        "bottom_spectrum": _bottom_k_spectrum(evals_sorted, log_spectrum_k),
        "eigenvalues": [float(v) for v in evals_sorted.tolist()],
        **_eigenvalue_band_populations(evals_sorted),
    }


def verify_worker_sample(predict_fn, payload) -> Dict[str, Any]:
    sample_idx, batch, log_spectrum_k = payload
    t0 = time.time()
    batch = batch.to("cpu")

    atomic_nums = batch.z.detach().to("cpu")
    atomsymbols = _atomic_nums_to_symbols(atomic_nums)
    formula = str(getattr(batch, "formula", ""))

    reactant_coords = getattr(batch, "pos_reactant", None)
    if reactant_coords is not None:
        reactant_coords = reactant_coords.detach().to("cpu")

    product_coords = None
    has_product = getattr(batch, "has_product", None)
    if has_product is not None and bool(has_product.item()):
        product_coords = getattr(batch, "pos_product", None)
        if product_coords is not None:
            product_coords = product_coords.detach().to("cpu")

    result: Dict[str, Any] = {
        "sample_idx": int(sample_idx),
        "formula": formula,
        "n_atoms": int(atomic_nums.numel()),
        "reactant": None,
        "product": None,
        "error": None,
    }

    try:
        if reactant_coords is not None:
            result["reactant"] = evaluate_geometry(
                predict_fn, reactant_coords, atomic_nums, atomsymbols, log_spectrum_k=log_spectrum_k,
            )
        if product_coords is not None:
            result["product"] = evaluate_geometry(
                predict_fn, product_coords, atomic_nums, atomsymbols, log_spectrum_k=log_spectrum_k,
            )
    except Exception as exc:  # pragma: no cover - defensive for cluster-side failures
        result["error"] = str(exc)

    result["wall_time"] = time.time() - t0
    return result


def _summarize_geometry(results: List[Dict[str, Any]], geom_key: str) -> Dict[str, Any]:
    rows = [r for r in results if r.get(geom_key) is not None and r.get("error") is None]
    entries = [r[geom_key] for r in rows]
    if not entries:
        return {
            "n_samples": 0,
            "n_minima": 0,
            "n_saddles": 0,
            "fraction_minima": float("nan"),
            "n_neg_distribution": {},
            "n_neg_stats": _summary_stats([]),
            "force_norm_stats": _summary_stats([]),
            "min_eval_stats": _summary_stats([]),
            "all_eigenvalue_stats": _summary_stats([]),
            "genuine_minima_samples": [],
            "dftb0_saddle_samples": [],
        }

    n_neg_vals = [int(e["n_neg"]) for e in entries]
    force_vals = [float(e["force_norm"]) for e in entries]
    min_eval_vals = [float(e["min_eval"]) for e in entries]
    all_evals = [float(ev) for e in entries for ev in e.get("eigenvalues", [])]

    minima_samples = [int(r["sample_idx"]) for r in rows if int(r[geom_key]["n_neg"]) == 0]
    saddle_samples = [int(r["sample_idx"]) for r in rows if int(r[geom_key]["n_neg"]) > 0]
    n_min = len(minima_samples)
    n_total = len(entries)

    return {
        "n_samples": n_total,
        "n_minima": n_min,
        "n_saddles": n_total - n_min,
        "fraction_minima": n_min / max(n_total, 1),
        "n_neg_distribution": {str(k): int(v) for k, v in sorted(Counter(n_neg_vals).items())},
        "n_neg_stats": _summary_stats([float(v) for v in n_neg_vals]),
        "force_norm_stats": _summary_stats(force_vals),
        "min_eval_stats": _summary_stats(min_eval_vals),
        "all_eigenvalue_stats": _summary_stats(all_evals),
        "genuine_minima_samples": minima_samples,
        "dftb0_saddle_samples": saddle_samples,
    }


NOISE_TAG_RE = re.compile(r"^n(?P<noise>[0-9.]+)_")


def _parse_noise_from_tag(tag: str) -> Optional[str]:
    m = NOISE_TAG_RE.match(tag)
    return m.group("noise") if m else None


def load_optimizer_outcomes(results_dir: Path) -> Dict[str, Any]:
    files = sorted(results_dir.rglob("minimization_newton_raphson_*_results.json"))
    by_sample: Dict[int, Dict[str, Any]] = {}
    seen_files = 0

    for path in files:
        if not path.is_file():
            continue
        try:
            with open(path) as f:
                payload = json.load(f)
        except Exception:
            continue

        seen_files += 1
        combo_tag = path.parent.name
        noise = _parse_noise_from_tag(combo_tag)
        rows = payload.get("metrics", {}).get("results", [])
        if not isinstance(rows, list):
            continue

        for row in rows:
            idx = row.get("sample_idx")
            if idx is None:
                continue
            idx = int(idx)
            rec = by_sample.setdefault(
                idx,
                {
                    "n_total": 0,
                    "n_converged": 0,
                    "by_noise": defaultdict(lambda: {"n_total": 0, "n_converged": 0}),
                },
            )
            rec["n_total"] += 1
            if bool(row.get("converged")):
                rec["n_converged"] += 1
            if noise is not None:
                rec["by_noise"][noise]["n_total"] += 1
                if bool(row.get("converged")):
                    rec["by_noise"][noise]["n_converged"] += 1

    for rec in by_sample.values():
        rec["failure_rate"] = 1.0 - (rec["n_converged"] / max(rec["n_total"], 1))
        rec["convergence_rate"] = rec["n_converged"] / max(rec["n_total"], 1)
        rec["by_noise"] = {
            noise: {
                "n_total": int(vals["n_total"]),
                "n_converged": int(vals["n_converged"]),
                "failure_rate": 1.0 - (vals["n_converged"] / max(vals["n_total"], 1)),
                "convergence_rate": vals["n_converged"] / max(vals["n_total"], 1),
            }
            for noise, vals in sorted(rec["by_noise"].items(), key=lambda x: float(x[0]))
        }

    return {
        "results_dir": str(results_dir),
        "n_result_files": seen_files,
        "n_samples_with_outcomes": len(by_sample),
        "by_sample": by_sample,
    }


def _correlate_nneg_with_failure(
    all_results: List[Dict[str, Any]],
    outcomes_by_sample: Dict[int, Dict[str, Any]],
    geom_key: str,
) -> Dict[str, Any]:
    paired: List[Tuple[float, float, int]] = []
    by_nneg: Dict[int, List[float]] = defaultdict(list)

    for row in all_results:
        if row.get("error") is not None:
            continue
        geom = row.get(geom_key)
        if not geom:
            continue
        idx = int(row["sample_idx"])
        out = outcomes_by_sample.get(idx)
        if not out or out.get("n_total", 0) <= 0:
            continue
        n_neg = float(geom.get("n_neg", float("nan")))
        fail = float(out.get("failure_rate", float("nan")))
        if not (math.isfinite(n_neg) and math.isfinite(fail)):
            continue
        paired.append((n_neg, fail, idx))
        by_nneg[int(n_neg)].append(fail)

    if len(paired) < 2:
        return {
            "n_pairs": len(paired),
            "pearson_r": float("nan"),
            "spearman_r": float("nan"),
            "point_biserial_minimum_vs_failure": float("nan"),
            "failure_rate_by_n_neg": {},
            "example_pairs": [],
        }

    x = [p[0] for p in paired]
    y = [p[1] for p in paired]
    is_min = [1.0 if p[0] == 0.0 else 0.0 for p in paired]

    return {
        "n_pairs": len(paired),
        "pearson_r": _pearson(x, y),
        "spearman_r": _spearman(x, y),
        "point_biserial_minimum_vs_failure": _pearson(is_min, y),
        "failure_rate_by_n_neg": {
            str(k): {
                "n_samples": len(v),
                "mean_failure_rate": float(np.mean(v)),
                "median_failure_rate": float(np.median(v)),
            }
            for k, v in sorted(by_nneg.items())
        },
        "example_pairs": [
            {"sample_idx": idx, "n_neg": n_neg, "failure_rate": fail}
            for (n_neg, fail, idx) in paired[:25]
        ],
    }


def write_human_report(summary: Dict[str, Any], out_path: Path) -> None:
    react = summary["geometry_summary"]["reactant"]
    prod = summary["geometry_summary"]["product"]
    cross = summary.get("optimizer_cross_reference")

    lines: List[str] = []
    lines.append("Transition1x Frequency Verification (DFTB0)")
    lines.append("=" * 72)
    lines.append(f"Generated at: {summary['generated_at']}")
    lines.append(f"Samples processed: {summary['n_samples']}")
    lines.append(f"Samples with worker errors: {summary['n_errors']}")
    lines.append("")

    def _fmt_geom(name: str, payload: Dict[str, Any]) -> None:
        lines.append(f"{name}")
        lines.append("-" * len(name))
        lines.append(
            f"n={payload['n_samples']} | minima(n_neg=0)={payload['n_minima']} "
            f"({payload['fraction_minima']:.3f}) | saddles={payload['n_saddles']}"
        )
        lines.append(f"n_neg distribution: {payload['n_neg_distribution']}")
        lines.append(
            "force_norm stats: "
            f"median={payload['force_norm_stats']['median']:.4e}, "
            f"mean={payload['force_norm_stats']['mean']:.4e}, "
            f"max={payload['force_norm_stats']['max']:.4e}"
        )
        lines.append(
            "min_eval stats: "
            f"median={payload['min_eval_stats']['median']:.4e}, "
            f"mean={payload['min_eval_stats']['mean']:.4e}, "
            f"min={payload['min_eval_stats']['min']:.4e}"
        )
        lines.append(
            "all-eigenvalue stats: "
            f"median={payload['all_eigenvalue_stats']['median']:.4e}, "
            f"p10={payload['all_eigenvalue_stats']['p10']:.4e}, "
            f"p90={payload['all_eigenvalue_stats']['p90']:.4e}"
        )
        lines.append("")

    _fmt_geom("Reactant minima under DFTB0", react)
    _fmt_geom("Product minima under DFTB0", prod)

    if cross is None:
        lines.append("Optimizer cross-reference: not requested (--results-dir omitted).")
        lines.append("")
    else:
        lines.append("Optimizer cross-reference")
        lines.append("-" * 28)
        lines.append(f"Result files read: {cross['n_result_files']}")
        lines.append(f"Samples with optimizer outcomes: {cross['n_samples_with_outcomes']}")
        for geom_key in ("reactant", "product"):
            corr = cross["correlation"].get(geom_key, {})
            lines.append(
                f"{geom_key}: n={corr.get('n_pairs', 0)}, "
                f"pearson(n_neg, failure)={corr.get('pearson_r', float('nan')):.4f}, "
                f"spearman={corr.get('spearman_r', float('nan')):.4f}, "
                f"point_biserial(is_minimum, failure)={corr.get('point_biserial_minimum_vs_failure', float('nan')):.4f}"
            )
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Transition1x DFT-labelled minima on DFTB0 via Hessian frequencies.",
    )
    parser.add_argument("--h5-path", type=str, required=True, help="Path to transition1x.h5")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--scine-functional", type=str, default="DFTB0")
    parser.add_argument("--results-dir", type=str, default=None, help="Optional optimizer results directory")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--log-spectrum-k", type=int, default=10)
    args = parser.parse_args()

    missing: List[str] = []
    if np is None:
        missing.append("numpy")
    if torch is None or DataLoader is None:
        missing.append("torch/torch_geometric")
    if Transition1xDataset is None or UsePos is None:
        missing.append("project dataset utilities")
    if _atomic_nums_to_symbols is None or get_vib_evals_evecs is None:
        missing.append("vibrational analysis utilities")
    if _bottom_k_spectrum is None or _eigenvalue_band_populations is None or _force_mean is None:
        missing.append("minimization diagnostics helpers")
    if ParallelSCINEProcessor is None or run_batch_parallel is None:
        missing.append("parallel SCINE helpers")
    if missing:
        raise RuntimeError(
            "Missing dependencies required to run verify_transition1x_minima.py: "
            + ", ".join(missing)
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_samples = None if args.max_samples <= 0 else args.max_samples
    dataset = Transition1xDataset(
        h5_path=args.h5_path,
        split=args.split,
        max_samples=max_samples,
        transform=UsePos("pos_transition"),
    )
    if len(dataset) == 0:
        raise RuntimeError("No Transition1x samples loaded. Check --h5-path and --split.")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    samples: List[Tuple[int, Tuple[int, Any, int]]] = []
    for i, batch in enumerate(dataloader):
        samples.append((i, (i, batch, args.log_spectrum_k)))

    print("=" * 72)
    print("Transition1x minima verification on DFTB0")
    print("=" * 72)
    print(f"Samples: {len(samples)}")
    print(f"SCINE functional: {args.scine_functional}")
    print(f"Workers: {args.n_workers}  |  Threads/worker: {args.threads_per_worker}")
    if args.results_dir:
        print(f"Cross-reference results dir: {args.results_dir}")
    print("=" * 72)

    t0 = time.time()
    processor = ParallelSCINEProcessor(
        functional=args.scine_functional,
        threads_per_worker=args.threads_per_worker,
        n_workers=args.n_workers,
        worker_fn=verify_worker_sample,
    )
    processor.start()
    try:
        all_results = run_batch_parallel(samples, processor)
    finally:
        processor.close()
    wall = time.time() - t0

    n_errors = sum(1 for r in all_results if r.get("error") is not None)
    react_summary = _summarize_geometry(all_results, "reactant")
    prod_summary = _summarize_geometry(all_results, "product")

    summary: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "wall_time_seconds": wall,
        "h5_path": args.h5_path,
        "split": args.split,
        "scine_functional": args.scine_functional,
        "n_samples": len(all_results),
        "n_errors": n_errors,
        "geometry_summary": {
            "reactant": react_summary,
            "product": prod_summary,
        },
        "results": all_results,
    }

    cross_ref_payload = None
    if args.results_dir:
        outcomes = load_optimizer_outcomes(Path(args.results_dir))
        by_sample = outcomes["by_sample"]
        correlation = {
            "reactant": _correlate_nneg_with_failure(all_results, by_sample, "reactant"),
            "product": _correlate_nneg_with_failure(all_results, by_sample, "product"),
        }
        cross_ref_payload = {
            "results_dir": outcomes["results_dir"],
            "n_result_files": outcomes["n_result_files"],
            "n_samples_with_outcomes": outcomes["n_samples_with_outcomes"],
            "correlation": correlation,
        }
        summary["optimizer_cross_reference"] = cross_ref_payload

    json_path = out_dir / "verify_transition1x_minima.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    report_path = out_dir / "verify_transition1x_minima_report.txt"
    write_human_report(summary, report_path)

    # Compact CSV for quick filtering in notebooks/spreadsheets.
    csv_rows: List[Dict[str, Any]] = []
    for row in all_results:
        for geom_key in ("reactant", "product"):
            geom = row.get(geom_key)
            if not geom:
                continue
            csv_rows.append(
                {
                    "sample_idx": row["sample_idx"],
                    "formula": row.get("formula", ""),
                    "geometry": geom_key,
                    "n_atoms": row.get("n_atoms", 0),
                    "n_neg": geom.get("n_neg"),
                    "is_dftb0_minimum": geom.get("is_dftb0_minimum"),
                    "force_norm": geom.get("force_norm"),
                    "min_eval": geom.get("min_eval"),
                    "condition_number": geom.get("condition_number"),
                }
            )
    if csv_rows:
        import csv

        csv_path = out_dir / "verify_transition1x_minima_samples.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)

    print("")
    print(f"Done in {wall:.1f}s")
    print(f"JSON:   {json_path}")
    print(f"Report: {report_path}")
    if args.results_dir and cross_ref_payload is not None:
        c_react = cross_ref_payload["correlation"]["reactant"]
        c_prod = cross_ref_payload["correlation"]["product"]
        print(
            "Correlation (n_neg vs failure_rate): "
            f"reactant pearson={c_react['pearson_r']:.4f}, product pearson={c_prod['pearson_r']:.4f}"
        )


if __name__ == "__main__":
    main()
