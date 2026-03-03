#!/usr/bin/env python3
"""Diagnostic analysis from labelled minima (reactant/product).

Uses ground-truth reactant and product geometries from Transition1x to
answer three questions:

1. Does DFTB0 predict n_neg=0 at the DFT-computed reactant minimum?
   (If not, it's a fundamental DFTB0 accuracy issue, not an optimizer bug.)

2. At the noisy starting point, does the gradient (SD direction) or the
   Newton step point closer to the true minimum?

3. Along the interpolation path from noisy start -> true minimum, where
   does n_neg first reach 0?  Where does the quadratic model become valid?

IMPORTANT: This script uses labelled ground-truth data that the optimizer
does NOT have access to. Results here are for analysis/debugging only —
using this information in the optimizer would be data leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.dependencies.common_utils import (
    Transition1xDataset,
    UsePos,
    parse_starting_geometry,
)
from src.noisy.multi_mode_eckartmw import (
    _atomic_nums_to_symbols,
    get_vib_evals_evecs,
)
from src.noisy.v2_tests.baselines.minimization import (
    _bottom_k_spectrum,
    _cascade_n_neg,
    _eigenvalue_band_populations,
    _force_mean,
    _nr_step_shifted_newton,
)
from src.parallel.scine_parallel import ParallelSCINEProcessor
from src.parallel.utils import run_batch_parallel


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def evaluate_geometry(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,  # noqa: ARG001 — kept for API consistency
    atomsymbols: List[str],
    log_spectrum_k: int = 10,
) -> Dict[str, Any]:
    """Evaluate a single geometry: energy, forces, Hessian, eigenspectrum.

    Returns a dict with energy, force_norm, n_neg, condition_number,
    eigenvalue band populations, cascade counts, bottom spectrum, etc.
    """
    coords_flat = coords.reshape(1, -1).detach().requires_grad_(True)
    result = predict_fn(coords_flat, do_hessian=True)

    energy = float(result["energy"].item())
    forces = result["forces"].detach().to("cpu")
    hessian = result["hessian"].detach().to("cpu")
    force_norm = _force_mean(forces)

    coords_2d = coords.reshape(-1, 3).detach().to("cpu")
    evals_vib, evecs_vib_3N, _Q_vib = get_vib_evals_evecs(
        hessian, coords_2d, atomsymbols, purify_hessian=False,
    )

    n_neg = int((evals_vib < 0.0).sum().item())
    cascade = _cascade_n_neg(evals_vib)
    band_pops = _eigenvalue_band_populations(evals_vib)
    bottom_spec = _bottom_k_spectrum(evals_vib, log_spectrum_k)

    # Condition number
    if evals_vib.numel() > 0:
        abs_evals = torch.abs(evals_vib)
        min_abs = float(abs_evals.min().item())
        max_abs = float(abs_evals.max().item())
        cond_num = max_abs / min_abs if min_abs > 0 else float("inf")
    else:
        min_abs = float("nan")
        max_abs = float("nan")
        cond_num = float("nan")

    min_vib = float(evals_vib.min().item()) if evals_vib.numel() > 0 else float("nan")
    max_vib = float(evals_vib.max().item()) if evals_vib.numel() > 0 else float("nan")

    return {
        "energy": energy,
        "force_norm": force_norm,
        "n_neg": n_neg,
        "min_vib_eval": min_vib,
        "max_vib_eval": max_vib,
        "min_abs_vib_eval": min_abs,
        "max_abs_vib_eval": max_abs,
        "condition_number": cond_num,
        "log10_condition_number": math.log10(cond_num) if cond_num > 0 and math.isfinite(cond_num) else float("nan"),
        "bottom_spectrum": bottom_spec,
        **cascade,
        **band_pops,
        # Keep raw eigenvalues for downstream analysis (not serialised to JSON)
        "_evals_vib": evals_vib,
        "_evecs_vib_3N": evecs_vib_3N,
        "_hessian": hessian,
        "_forces": forces,
        "_grad": -forces.reshape(-1),  # gradient = -forces
    }


def analyze_interpolation_path(
    predict_fn,
    start_coords: torch.Tensor,
    end_coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    atomsymbols: List[str],
    n_points: int = 20,
    log_spectrum_k: int = 10,
) -> List[Dict[str, Any]]:
    """Linearly interpolate between start and end, evaluating at each point.

    Returns a list of dicts (one per interpolation point) showing how
    eigenvalues, forces, condition number, n_neg evolve along the path.
    """
    start = start_coords.reshape(-1).to(torch.float64)
    end = end_coords.reshape(-1).to(torch.float64)

    path_results = []
    for i in range(n_points + 1):
        t = i / n_points
        coords = start * (1.0 - t) + end * t

        info = evaluate_geometry(
            predict_fn,
            coords.reshape(-1, 3),
            atomic_nums,
            atomsymbols,
            log_spectrum_k=log_spectrum_k,
        )
        # Remove non-serialisable fields
        info_clean = {k: v for k, v in info.items() if not k.startswith("_")}
        info_clean["t"] = t
        info_clean["dist_to_end"] = float((coords - end).norm().item())
        info_clean["dist_to_start"] = float((coords - start).norm().item())
        path_results.append(info_clean)

    return path_results


def analyze_direction_quality(
    predict_fn,
    noisy_coords: torch.Tensor,
    true_min_coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    atomsymbols: List[str],
    shift_epsilon: float = 1e-3,
) -> Dict[str, Any]:
    """At the noisy start, compare SD, Newton, and true directions.

    Computes:
    - SD direction: normalized(-gradient)
    - Newton direction: normalized(shifted Newton step)
    - True direction: normalized(true_min - noisy_start)
    - Dot products: sd_dot_true, newton_dot_true
    """
    info = evaluate_geometry(
        predict_fn, noisy_coords, atomic_nums, atomsymbols,
    )

    grad = info["_grad"].to(torch.float64)
    evals_vib = info["_evals_vib"]
    evecs_vib_3N = info["_evecs_vib_3N"]

    # True direction
    noisy_flat = noisy_coords.reshape(-1).to(torch.float64)
    true_flat = true_min_coords.reshape(-1).to(torch.float64)
    true_dir = true_flat - noisy_flat
    true_dir_norm = true_dir / (true_dir.norm() + 1e-30)

    # SD direction = -grad (normalised)
    sd_dir = -grad
    sd_dir_norm = sd_dir / (sd_dir.norm() + 1e-30)

    # Newton direction (shifted)
    delta_x, _, _ = _nr_step_shifted_newton(
        grad, evecs_vib_3N.to(torch.float64), evals_vib.to(torch.float64),
        shift_epsilon,
    )
    newton_dir_norm = delta_x / (delta_x.norm() + 1e-30)

    sd_dot_true = float(sd_dir_norm.dot(true_dir_norm).item())
    newton_dot_true = float(newton_dir_norm.dot(true_dir_norm).item())
    sd_dot_newton = float(sd_dir_norm.dot(newton_dir_norm).item())

    return {
        "sd_dot_true": sd_dot_true,
        "newton_dot_true": newton_dot_true,
        "sd_dot_newton": sd_dot_newton,
        "grad_norm": float(grad.norm().item()),
        "newton_step_norm": float(delta_x.norm().item()),
        "true_displacement_norm": float(true_dir.norm().item()),
        "force_norm": info["force_norm"],
        "n_neg": info["n_neg"],
        "condition_number": info["condition_number"],
        "log10_condition_number": info.get("log10_condition_number", float("nan")),
    }


# ---------------------------------------------------------------------------
# Worker function for parallel execution
# ---------------------------------------------------------------------------

def diagnostic_worker_sample(predict_fn, payload) -> Dict[str, Any]:
    """Parallel worker: evaluate diagnostics for one Transition1x sample.

    payload = (sample_idx, batch, noise_levels, noise_seed,
               n_interpolation_points, shift_epsilon, log_spectrum_k)
    """
    (
        sample_idx, batch, noise_levels, noise_seed,
        n_interpolation_points, shift_epsilon, log_spectrum_k,
    ) = payload

    t0 = time.time()
    batch = batch.to("cpu")
    atomic_nums = batch.z.detach().to("cpu")
    atomsymbols = _atomic_nums_to_symbols(atomic_nums)

    # Ground-truth geometries
    reactant_coords = getattr(batch, "pos_reactant", None)
    if reactant_coords is not None:
        reactant_coords = reactant_coords.detach().to("cpu")

    product_coords = None
    has_product = getattr(batch, "has_product", None)
    if has_product is not None and bool(has_product.item()):
        product_coords = getattr(batch, "pos_product", None)
        if product_coords is not None:
            product_coords = product_coords.detach().to("cpu")

    ts_coords = getattr(batch, "pos_transition", None)
    if ts_coords is not None:
        ts_coords = ts_coords.detach().to("cpu")

    formula = str(getattr(batch, "formula", ""))
    n_atoms = int(atomic_nums.numel())

    result: Dict[str, Any] = {
        "sample_idx": sample_idx,
        "formula": formula,
        "n_atoms": n_atoms,
        "has_product": product_coords is not None,
    }

    try:
        # -------------------------------------------------------------------
        # 1. Evaluate at true reactant minimum
        # -------------------------------------------------------------------
        if reactant_coords is not None:
            react_info = evaluate_geometry(
                predict_fn, reactant_coords, atomic_nums, atomsymbols,
                log_spectrum_k=log_spectrum_k,
            )
            result["reactant"] = {
                k: v for k, v in react_info.items() if not k.startswith("_")
            }
        else:
            result["reactant"] = None

        # -------------------------------------------------------------------
        # 2. Evaluate at true product minimum (if available)
        # -------------------------------------------------------------------
        if product_coords is not None:
            prod_info = evaluate_geometry(
                predict_fn, product_coords, atomic_nums, atomsymbols,
                log_spectrum_k=log_spectrum_k,
            )
            result["product"] = {
                k: v for k, v in prod_info.items() if not k.startswith("_")
            }
        else:
            result["product"] = None

        # -------------------------------------------------------------------
        # 3. Evaluate at true TS
        # -------------------------------------------------------------------
        if ts_coords is not None:
            ts_info = evaluate_geometry(
                predict_fn, ts_coords, atomic_nums, atomsymbols,
                log_spectrum_k=log_spectrum_k,
            )
            result["transition_state"] = {
                k: v for k, v in ts_info.items() if not k.startswith("_")
            }
        else:
            result["transition_state"] = None

        # -------------------------------------------------------------------
        # 4. Per noise level: noisy start diagnostics
        # -------------------------------------------------------------------
        noise_results = {}
        for noise_level in noise_levels:
            start_from = f"midpoint_rt_noise{noise_level}A"
            noisy_coords = parse_starting_geometry(
                start_from, batch,
                noise_seed=noise_seed,
                sample_index=sample_idx,
            ).detach().to("cpu")

            # Evaluate at noisy start
            noisy_info = evaluate_geometry(
                predict_fn, noisy_coords, atomic_nums, atomsymbols,
                log_spectrum_k=log_spectrum_k,
            )
            noisy_clean = {
                k: v for k, v in noisy_info.items() if not k.startswith("_")
            }

            # Direction quality (noisy start -> reactant)
            dir_quality = None
            if reactant_coords is not None:
                dir_quality = analyze_direction_quality(
                    predict_fn, noisy_coords, reactant_coords,
                    atomic_nums, atomsymbols, shift_epsilon,
                )

            # Direction quality (noisy start -> product)
            dir_quality_product = None
            if product_coords is not None:
                dir_quality_product = analyze_direction_quality(
                    predict_fn, noisy_coords, product_coords,
                    atomic_nums, atomsymbols, shift_epsilon,
                )

            # Interpolation path (noisy start -> closer minimum)
            interp_path = None
            interp_target = None
            if reactant_coords is not None and product_coords is not None:
                # Pick the closer minimum as interpolation target
                dist_react = float((noisy_coords - reactant_coords).norm().item())
                dist_prod = float((noisy_coords - product_coords).norm().item())
                if dist_react <= dist_prod:
                    interp_target = "reactant"
                    target_coords = reactant_coords
                else:
                    interp_target = "product"
                    target_coords = product_coords
                interp_path = analyze_interpolation_path(
                    predict_fn, noisy_coords, target_coords,
                    atomic_nums, atomsymbols,
                    n_points=n_interpolation_points,
                    log_spectrum_k=log_spectrum_k,
                )
            elif reactant_coords is not None:
                interp_target = "reactant"
                interp_path = analyze_interpolation_path(
                    predict_fn, noisy_coords, reactant_coords,
                    atomic_nums, atomsymbols,
                    n_points=n_interpolation_points,
                    log_spectrum_k=log_spectrum_k,
                )

            noise_results[str(noise_level)] = {
                "noisy_start": noisy_clean,
                "direction_quality_to_reactant": dir_quality,
                "direction_quality_to_product": dir_quality_product,
                "interpolation_target": interp_target,
                "interpolation_path": interp_path,
            }

        result["noise_levels"] = noise_results
        result["error"] = None

    except Exception as e:
        result["error"] = str(e)

    result["wall_time"] = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(all_results: List[Dict[str, Any]], noise_levels: List[float]) -> None:
    """Print a diagnostic summary to stdout."""
    n_total = len(all_results)
    n_errors = sum(1 for r in all_results if r.get("error") is not None)
    print(f"Samples: {n_total}  (errors: {n_errors})")
    print()

    # Q1: Does DFTB0 predict n_neg=0 at true minima?
    print("=" * 70)
    print("Q1: n_neg at ground-truth minima (DFTB0 Hessian)")
    print("=" * 70)

    for geom_type in ["reactant", "product", "transition_state"]:
        valid = [r for r in all_results if r.get(geom_type) is not None]
        if not valid:
            continue
        n_neg_vals = [r[geom_type]["n_neg"] for r in valid]
        cond_vals = [r[geom_type]["condition_number"] for r in valid
                     if math.isfinite(r[geom_type]["condition_number"])]
        force_vals = [r[geom_type]["force_norm"] for r in valid]
        n_correct = sum(1 for v in n_neg_vals if v == (0 if geom_type != "transition_state" else 1))
        expected = 0 if geom_type != "transition_state" else 1

        print(f"\n  {geom_type} ({len(valid)} samples, expected n_neg={expected}):")
        print(f"    n_neg=={expected}: {n_correct}/{len(valid)} ({100*n_correct/max(len(valid),1):.1f}%)")
        if n_neg_vals:
            print(f"    n_neg distribution: min={min(n_neg_vals)}, max={max(n_neg_vals)}, "
                  f"mean={np.mean(n_neg_vals):.2f}, median={np.median(n_neg_vals):.0f}")
        if cond_vals:
            print(f"    condition_number: min={min(cond_vals):.2e}, max={max(cond_vals):.2e}, "
                  f"median={np.median(cond_vals):.2e}")
            log_conds = [math.log10(c) for c in cond_vals if c > 0]
            if log_conds:
                print(f"    log10(cond): min={min(log_conds):.1f}, max={max(log_conds):.1f}, "
                      f"median={np.median(log_conds):.1f}")
        if force_vals:
            print(f"    force_norm: min={min(force_vals):.4e}, max={max(force_vals):.4e}, "
                  f"median={np.median(force_vals):.4e}")

        # Band population summary
        band_keys = [k for k in valid[0][geom_type] if k.startswith("n_eval_")]
        if band_keys:
            print(f"    eigenvalue band populations (mean across samples):")
            for bk in sorted(band_keys):
                vals = [r[geom_type][bk] for r in valid]
                print(f"      {bk}: {np.mean(vals):.2f}")

    # Q2: Direction quality at noisy starting points
    print()
    print("=" * 70)
    print("Q2: Direction quality at noisy starting points")
    print("    (dot product: 1.0 = perfect alignment, 0.0 = orthogonal, -1.0 = opposite)")
    print("=" * 70)

    for nl in noise_levels:
        nl_str = str(nl)
        valid = [r for r in all_results
                 if r.get("noise_levels", {}).get(nl_str, {}).get("direction_quality_to_reactant") is not None]
        if not valid:
            continue

        sd_dots = [r["noise_levels"][nl_str]["direction_quality_to_reactant"]["sd_dot_true"] for r in valid]
        newton_dots = [r["noise_levels"][nl_str]["direction_quality_to_reactant"]["newton_dot_true"] for r in valid]
        conds = [r["noise_levels"][nl_str]["direction_quality_to_reactant"]["condition_number"] for r in valid
                 if math.isfinite(r["noise_levels"][nl_str]["direction_quality_to_reactant"]["condition_number"])]
        n_negs = [r["noise_levels"][nl_str]["noisy_start"]["n_neg"] for r in valid]

        print(f"\n  Noise {nl} A ({len(valid)} samples):")
        print(f"    SD · true:     mean={np.mean(sd_dots):.4f}, median={np.median(sd_dots):.4f}, "
              f"std={np.std(sd_dots):.4f}")
        print(f"    Newton · true: mean={np.mean(newton_dots):.4f}, median={np.median(newton_dots):.4f}, "
              f"std={np.std(newton_dots):.4f}")
        sd_better = sum(1 for s, n in zip(sd_dots, newton_dots) if s > n)
        print(f"    SD better than Newton: {sd_better}/{len(valid)} ({100*sd_better/max(len(valid),1):.1f}%)")
        if conds:
            log_conds = [math.log10(c) for c in conds if c > 0]
            print(f"    condition_number at start: median={np.median(conds):.2e}, "
                  f"log10 median={np.median(log_conds):.1f}")
        print(f"    n_neg at start: mean={np.mean(n_negs):.1f}, median={np.median(n_negs):.0f}, "
              f"max={max(n_negs)}")

    # Q3: Interpolation path analysis
    print()
    print("=" * 70)
    print("Q3: Along interpolation path (noisy start -> closest minimum)")
    print("    Where does n_neg first reach 0? Where does quadratic model become valid?")
    print("=" * 70)

    for nl in noise_levels:
        nl_str = str(nl)
        valid = [r for r in all_results
                 if r.get("noise_levels", {}).get(nl_str, {}).get("interpolation_path") is not None]
        if not valid:
            continue

        first_zero_ts = []  # interpolation parameter t where n_neg first = 0
        cond_at_halfway = []
        cond_at_minimum = []
        n_never_zero = 0

        for r in valid:
            path = r["noise_levels"][nl_str]["interpolation_path"]
            first_zero_t = None
            for pt in path:
                if pt["n_neg"] == 0:
                    first_zero_t = pt["t"]
                    break
            if first_zero_t is not None:
                first_zero_ts.append(first_zero_t)
            else:
                n_never_zero += 1

            # Condition number at midpoint and end
            mid_idx = len(path) // 2
            if mid_idx < len(path):
                c = path[mid_idx]["condition_number"]
                if math.isfinite(c):
                    cond_at_halfway.append(c)
            if path:
                c = path[-1]["condition_number"]
                if math.isfinite(c):
                    cond_at_minimum.append(c)

        print(f"\n  Noise {nl} A ({len(valid)} samples):")
        if first_zero_ts:
            print(f"    t where n_neg first = 0: mean={np.mean(first_zero_ts):.3f}, "
                  f"median={np.median(first_zero_ts):.3f}, min={min(first_zero_ts):.3f}, "
                  f"max={max(first_zero_ts):.3f}")
        print(f"    n_neg never reaches 0 along path: {n_never_zero}/{len(valid)} "
              f"({100*n_never_zero/max(len(valid),1):.1f}%)")
        if cond_at_halfway:
            log_ch = [math.log10(c) for c in cond_at_halfway if c > 0]
            print(f"    condition at path midpoint: median={np.median(cond_at_halfway):.2e}, "
                  f"log10 median={np.median(log_ch):.1f}")
        if cond_at_minimum:
            log_cm = [math.log10(c) for c in cond_at_minimum if c > 0]
            print(f"    condition at true minimum:  median={np.median(cond_at_minimum):.2e}, "
                  f"log10 median={np.median(log_cm):.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic analysis from labelled minima (Transition1x)",
    )
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--scine-functional", type=str, default="DFTB0")
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--noise-levels", type=str, default="0.5,1.0,1.5,2.0",
                        help="Comma-separated noise levels in Angstroms")
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--n-interpolation-points", type=int, default=20)
    parser.add_argument("--shift-epsilon", type=float, default=1e-3)
    parser.add_argument("--log-spectrum-k", type=int, default=10)
    args = parser.parse_args()

    noise_levels = [float(x.strip()) for x in args.noise_levels.split(",")]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = Transition1xDataset(
        h5_path=args.h5_path,
        split=args.split,
        max_samples=args.max_samples,
        transform=UsePos("pos_transition"),
    )
    if len(dataset) == 0:
        raise RuntimeError("No Transition1x samples loaded. Check h5 path and split.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("=" * 70)
    print("Diagnostic from Labelled Minima")
    print("=" * 70)
    print(f"  Samples:              {min(args.max_samples, len(dataset))}")
    print(f"  Noise levels:         {noise_levels}")
    print(f"  Noise seed:           {args.noise_seed}")
    print(f"  Interpolation points: {args.n_interpolation_points}")
    print(f"  Shift epsilon:        {args.shift_epsilon}")
    print(f"  Workers:              {args.n_workers}")
    print(f"  Threads/worker:       {args.threads_per_worker}")
    print("=" * 70)

    # Build payloads
    samples = []
    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break
        payload = (
            i, batch, noise_levels, args.noise_seed,
            args.n_interpolation_points, args.shift_epsilon, args.log_spectrum_k,
        )
        samples.append((i, payload))

    # Run parallel
    processor = ParallelSCINEProcessor(
        functional=args.scine_functional,
        threads_per_worker=args.threads_per_worker,
        n_workers=args.n_workers,
        worker_fn=diagnostic_worker_sample,
    )
    processor.start()
    try:
        all_results = run_batch_parallel(samples, processor)
    finally:
        processor.close()

    # Print summary
    print()
    print_summary(all_results, noise_levels)

    # Save full results
    output_path = out_dir / "diagnostic_from_minima_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "noise_levels": noise_levels,
                "noise_seed": args.noise_seed,
                "shift_epsilon": args.shift_epsilon,
                "n_interpolation_points": args.n_interpolation_points,
                "n_samples": len(all_results),
                "n_errors": sum(1 for r in all_results if r.get("error") is not None),
                "results": all_results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nFull results saved to: {output_path}")

    # Save per-noise-level summaries as CSV for easy plotting
    for nl in noise_levels:
        nl_str = str(nl)
        csv_rows = []
        for r in all_results:
            nl_data = r.get("noise_levels", {}).get(nl_str, {})
            if not nl_data:
                continue
            row = {
                "sample_idx": r["sample_idx"],
                "formula": r.get("formula", ""),
                "n_atoms": r.get("n_atoms", 0),
            }
            # Noisy start info
            ns = nl_data.get("noisy_start", {})
            row["noisy_n_neg"] = ns.get("n_neg")
            row["noisy_force_norm"] = ns.get("force_norm")
            row["noisy_condition_number"] = ns.get("condition_number")
            row["noisy_log10_cond"] = ns.get("log10_condition_number")
            # Direction quality
            dq = nl_data.get("direction_quality_to_reactant", {})
            if dq:
                row["sd_dot_true"] = dq.get("sd_dot_true")
                row["newton_dot_true"] = dq.get("newton_dot_true")
                row["sd_dot_newton"] = dq.get("sd_dot_newton")
            # Reactant info
            react = r.get("reactant", {})
            if react:
                row["reactant_n_neg"] = react.get("n_neg")
                row["reactant_force_norm"] = react.get("force_norm")
                row["reactant_condition_number"] = react.get("condition_number")
            csv_rows.append(row)

        if csv_rows:
            import csv
            csv_path = out_dir / f"diagnostic_noise{nl}A.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                for row in csv_rows:
                    writer.writerow(row)
            print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
