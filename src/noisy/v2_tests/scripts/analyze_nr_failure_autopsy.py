#!/usr/bin/env python3
"""Failure autopsy for Newton-Raphson minimization trajectories.

Reads trajectory JSONs from a grid-search output directory and classifies
each failed trajectory into failure modes. For each failure, reports:

1. Final state: bottom eigenvalues, n_neg, force norm, energy
2. Eigenvalue trajectory: are negative eigenvalues shrinking? oscillating? stuck?
3. Energy trajectory: is energy still decreasing or plateaued?
4. Gradient-mode overlap: can the optimizer even address the negative modes?
5. Stagnation: how many steps was n_neg unchanged at the end?
6. Geometry cycling: is the geometry oscillating between two configurations?
7. Classification: "almost_converged", "genuinely_stuck", "oscillating",
   "drifting", "energy_plateau"

Usage:
  python analyze_nr_failure_autopsy.py --grid-dir <path> --output-dir <path>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def classify_trajectory(traj_data: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a single trajectory into a failure mode.

    Returns a dict with classification info.
    """
    trajectory = traj_data.get("trajectory", [])
    final_neg_vib = traj_data.get("final_neg_vib")
    sample_id = traj_data.get("sample_id", "unknown")

    if not trajectory:
        return {
            "sample_id": sample_id,
            "classification": "empty_trajectory",
            "n_steps": 0,
        }

    converged = final_neg_vib == 0 if final_neg_vib is not None else False
    if converged:
        return {
            "sample_id": sample_id,
            "classification": "converged",
            "converged_step": len(trajectory),
        }

    n_steps = len(trajectory)
    last = trajectory[-1]

    # --- Final state ---
    final_n_neg = last.get("n_neg_evals", -1)
    final_min_eval = last.get("min_vib_eval", float("nan"))
    final_force_norm = last.get("force_norm", float("nan"))
    final_energy = last.get("energy", float("nan"))
    bottom_spectrum = last.get("bottom_spectrum", [])

    # --- Eigenvalue trajectory ---
    min_eval_history = [s.get("min_vib_eval", float("nan")) for s in trajectory]
    n_neg_history = [s.get("n_neg_evals", 0) for s in trajectory]
    energy_history = [s.get("energy", float("nan")) for s in trajectory]

    # Is min_eval monotonically increasing (toward zero) in the last 100 steps?
    tail = min_eval_history[-100:]
    valid_tail = [v for v in tail if math.isfinite(v)]
    eval_improving = False
    eval_oscillating = False
    if len(valid_tail) >= 10:
        diffs = [valid_tail[i+1] - valid_tail[i] for i in range(len(valid_tail)-1)]
        pos_diffs = sum(1 for d in diffs if d > 0)
        eval_improving = pos_diffs > 0.7 * len(diffs)
        eval_oscillating = 0.3 < pos_diffs / max(len(diffs), 1) < 0.7

    # --- Stagnation ---
    stagnation_count = 0
    if n_neg_history:
        final_n_neg_val = n_neg_history[-1]
        for i in range(len(n_neg_history) - 2, -1, -1):
            if n_neg_history[i] == final_n_neg_val:
                stagnation_count += 1
            else:
                break

    # --- Energy plateau ---
    energy_tail = energy_history[-100:]
    valid_energy_tail = [v for v in energy_tail if math.isfinite(v)]
    energy_plateau = False
    energy_range = float("nan")
    if len(valid_energy_tail) >= 10:
        energy_range = max(valid_energy_tail) - min(valid_energy_tail)
        energy_plateau = energy_range < 1e-6

    # --- Energy still decreasing? ---
    energy_decreasing = False
    if len(valid_energy_tail) >= 10:
        first_half = valid_energy_tail[:len(valid_energy_tail)//2]
        second_half = valid_energy_tail[len(valid_energy_tail)//2:]
        if first_half and second_half:
            energy_decreasing = sum(second_half)/len(second_half) < sum(first_half)/len(first_half) - 1e-8

    # --- Gradient-mode overlap (from v3 diagnostics) ---
    neg_diag = last.get("neg_mode_diag", {})
    neg_grad_overlaps = neg_diag.get("neg_mode_grad_overlaps", [])
    neg_eigenvalues = neg_diag.get("neg_mode_eigenvalues", [])
    min_grad_overlap = neg_diag.get("min_neg_grad_overlap", float("nan"))
    step_neg_frac = neg_diag.get("step_along_neg_frac", float("nan"))

    # --- Escape info ---
    total_escapes = traj_data.get("total_escapes", 0)
    total_line_searches = traj_data.get("total_line_searches", 0)
    escape_steps = [s["step"] for s in trajectory if s.get("escape_triggered")]

    # --- Trust radius history ---
    tr_history = [s.get("trust_radius", float("nan")) for s in trajectory]
    valid_tr = [v for v in tr_history if math.isfinite(v)]
    final_tr = valid_tr[-1] if valid_tr else float("nan")
    min_tr = min(valid_tr) if valid_tr else float("nan")

    # --- v4 diagnostics ---
    total_mode_follows = traj_data.get("total_mode_follows", 0)

    # Blind correction info from final step
    blind_info = last.get("blind_correction", {})
    n_blind_corrections = blind_info.get("n_blind_modes", 0)

    # Neg-mode trust radius from final step
    final_neg_tr = last.get("neg_trust_radius")
    if final_neg_tr is None:
        final_neg_tr = float("nan")

    # Count escape accepted/rejected events
    escape_accepted_count = sum(1 for s in trajectory if s.get("escape_accepted"))
    escape_rejected_count = sum(1 for s in trajectory if s.get("escape_rejected"))

    # Mode-follow events
    mode_follow_steps = [s.get("step", i) for i, s in enumerate(trajectory)
                         if s.get("mode_follow_triggered")]

    # --- Classification ---
    if final_n_neg >= 0 and abs(final_min_eval) < 2e-3 and final_n_neg <= 3:
        classification = "almost_converged"
    elif eval_oscillating:
        classification = "oscillating"
    elif energy_plateau and not eval_improving:
        classification = "energy_plateau"
    elif stagnation_count > n_steps * 0.5:
        classification = "genuinely_stuck"
    elif eval_improving:
        classification = "slow_convergence"
    else:
        classification = "drifting"

    return {
        "sample_id": sample_id,
        "classification": classification,
        "n_steps": n_steps,
        "final_n_neg": final_n_neg,
        "final_min_eval": final_min_eval,
        "final_force_norm": final_force_norm,
        "final_energy": final_energy,
        "bottom_spectrum_final": bottom_spectrum[:5],
        "stagnation_count": stagnation_count,
        "stagnation_frac": stagnation_count / max(n_steps, 1),
        "eval_improving": eval_improving,
        "eval_oscillating": eval_oscillating,
        "energy_plateau": energy_plateau,
        "energy_range_last100": energy_range,
        "energy_decreasing": energy_decreasing,
        "neg_grad_overlaps": neg_grad_overlaps,
        "neg_eigenvalues": neg_eigenvalues,
        "min_grad_overlap": min_grad_overlap,
        "step_along_neg_frac": step_neg_frac,
        "total_escapes": total_escapes,
        "total_line_searches": total_line_searches,
        "n_escape_events": len(escape_steps),
        "final_trust_radius": final_tr,
        "min_trust_radius": min_tr,
        # v4 diagnostics
        "total_mode_follows": total_mode_follows,
        "n_blind_corrections": n_blind_corrections,
        "final_neg_trust_radius": final_neg_tr,
        "escape_accepted_count": escape_accepted_count,
        "escape_rejected_count": escape_rejected_count,
        "n_mode_follow_events": len(mode_follow_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Failure autopsy for NR minimization trajectories")
    parser.add_argument("--grid-dir", type=str, required=True,
                        help="Grid directory containing combo subdirectories")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for autopsy results")
    parser.add_argument("--traj-glob", type=str,
                        default="*/diagnostics/*_trajectory.json",
                        help="Glob for trajectory files relative to grid-dir")
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all trajectory files
    traj_files = sorted(grid_dir.glob(args.traj_glob))
    if not traj_files:
        print(f"No trajectory files found with glob '{args.traj_glob}' in {grid_dir}")
        return

    print(f"Found {len(traj_files)} trajectory files")

    all_results: List[Dict[str, Any]] = []
    per_combo: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for traj_path in traj_files:
        combo_tag = traj_path.parent.parent.name
        with open(traj_path) as f:
            traj_data = json.load(f)

        result = classify_trajectory(traj_data)
        result["combo_tag"] = combo_tag
        result["traj_file"] = str(traj_path.relative_to(grid_dir))
        all_results.append(result)
        per_combo[combo_tag].append(result)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("FAILURE AUTOPSY SUMMARY")
    print("=" * 70)

    # Overall classification counts
    class_counts: Dict[str, int] = defaultdict(int)
    for r in all_results:
        class_counts[r["classification"]] += 1

    print("\nOverall classification distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(len(all_results), 1)
        print(f"  {cls:<25s}: {count:4d} ({pct:5.1f}%)")

    # --- Per-combo breakdown ---
    print("\n" + "=" * 70)
    print("PER-COMBO FAILURE MODES")
    print("=" * 70)
    for combo_tag in sorted(per_combo.keys()):
        results = per_combo[combo_tag]
        n_total = len(results)
        n_converged = sum(1 for r in results if r["classification"] == "converged")
        n_failed = n_total - n_converged

        print(f"\n--- {combo_tag} ({n_converged}/{n_total} converged) ---")
        if n_failed == 0:
            print("  All converged!")
            continue

        failed = [r for r in results if r["classification"] != "converged"]
        combo_classes: Dict[str, int] = defaultdict(int)
        for r in failed:
            combo_classes[r["classification"]] += 1

        for cls, count in sorted(combo_classes.items(), key=lambda x: -x[1]):
            print(f"  {cls}: {count}")

    # --- Detailed failed trajectory reports ---
    print("\n" + "=" * 70)
    print("DETAILED FAILURE REPORTS (failed trajectories)")
    print("=" * 70)

    failed_results = [r for r in all_results if r["classification"] != "converged"]
    # Group by sample_id to see which samples fail across combos
    per_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in failed_results:
        per_sample[r["sample_id"]].append(r)

    # Sort by number of failures (most failures first = hardest samples)
    sorted_samples = sorted(per_sample.items(), key=lambda x: -len(x[1]))

    for sample_id, failures in sorted_samples[:15]:  # top 15 hardest
        n_fail = len(failures)
        classifications = [f["classification"] for f in failures]
        class_set = set(classifications)
        print(f"\n  {sample_id}: {n_fail} failures across combos")
        print(f"    Failure modes: {', '.join(sorted(class_set))}")

        # Show the "best" failure (closest to convergence)
        best = min(failures, key=lambda r: abs(r.get("final_min_eval", float("inf"))))
        print(f"    Best attempt: {best['combo_tag']}")
        print(f"      final_n_neg={best['final_n_neg']}, "
              f"min_eval={best.get('final_min_eval', 'nan'):.6f}, "
              f"force_norm={best.get('final_force_norm', 'nan'):.6f}")
        if best.get("bottom_spectrum_final"):
            spec = [f"{v:.5f}" for v in best["bottom_spectrum_final"]]
            print(f"      bottom_spectrum: [{', '.join(spec)}]")
        if best.get("neg_grad_overlaps"):
            overlaps = [f"{v:.4f}" for v in best["neg_grad_overlaps"]]
            evals = [f"{v:.5f}" for v in best.get("neg_eigenvalues", [])]
            print(f"      neg_mode_eigenvalues: [{', '.join(evals)}]")
            print(f"      grad_overlaps:        [{', '.join(overlaps)}]")
        print(f"      stagnation: {best['stagnation_count']} steps "
              f"({best['stagnation_frac']:.1%}), "
              f"eval_improving={best['eval_improving']}, "
              f"energy_plateau={best['energy_plateau']}")
        print(f"      trust_radius: final={best.get('final_trust_radius', 'nan'):.4f}, "
              f"min={best.get('min_trust_radius', 'nan'):.4f}")
        if best.get("total_escapes", 0) > 0:
            print(f"      escapes={best['total_escapes']}, "
                  f"line_searches={best.get('total_line_searches', 0)}")

    # --- CSV output ---
    csv_path = out_dir / "failure_autopsy.csv"
    fieldnames = [
        "sample_id", "combo_tag", "classification", "n_steps",
        "final_n_neg", "final_min_eval", "final_force_norm", "final_energy",
        "stagnation_count", "stagnation_frac",
        "eval_improving", "eval_oscillating",
        "energy_plateau", "energy_range_last100", "energy_decreasing",
        "min_grad_overlap", "step_along_neg_frac",
        "total_escapes", "total_line_searches",
        "final_trust_radius", "min_trust_radius",
        # v4
        "total_mode_follows", "n_blind_corrections", "final_neg_trust_radius",
        "escape_accepted_count", "escape_rejected_count", "n_mode_follow_events",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    # --- JSON output ---
    json_path = out_dir / "failure_autopsy.json"
    with open(json_path, "w") as f:
        json.dump({
            "n_trajectories": len(all_results),
            "n_converged": class_counts.get("converged", 0),
            "n_failed": len(failed_results),
            "classification_counts": dict(class_counts),
            "per_combo_summary": {
                combo: {
                    "n_total": len(results),
                    "n_converged": sum(1 for r in results if r["classification"] == "converged"),
                    "classifications": dict(defaultdict(int, ((r["classification"], 1) for r in results))),
                }
                for combo, results in per_combo.items()
            },
            "hardest_samples": [
                {
                    "sample_id": sample_id,
                    "n_failures": len(failures),
                    "classifications": list(set(f["classification"] for f in failures)),
                    "best_min_eval": min(abs(f.get("final_min_eval", float("inf"))) for f in failures),
                }
                for sample_id, failures in sorted_samples[:20]
            ],
        }, f, indent=2)

    print(f"\nWrote autopsy artifacts to: {out_dir}")
    print(f"  failure_autopsy.csv  — one row per trajectory")
    print(f"  failure_autopsy.json — structured summary")


if __name__ == "__main__":
    main()
