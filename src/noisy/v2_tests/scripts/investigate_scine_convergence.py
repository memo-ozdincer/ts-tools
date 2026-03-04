#!/usr/bin/env python3
"""Investigate whether tighter SCINE settings reduce Hessian noise floor.

Workflow:
1. Enumerate available SCINE/Sparrow calculator settings.
2. Build a few "tighter" setting profiles if convergence-like keys exist.
3. Evaluate DFTB0 Hessians at DFT-labelled minima (reactants) for a small sample.
4. Compare eigenvalue spectra/noise proxies across profiles.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - optional for --help path
    np = None

try:
    import torch
except Exception:  # pragma: no cover - optional for --help path
    torch = None


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "n": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
        }
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def _jsonable(x: Any) -> Any:
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return str(x)


def _load_scine_runtime() -> Tuple[Any, Any, Any]:
    import scine_sparrow
    import scine_utilities

    manager = scine_utilities.core.ModuleManager.get_instance()
    sparrow_module = Path(scine_sparrow.__file__).parent / "sparrow.module.so"
    if not sparrow_module.exists():
        raise RuntimeError(f"Sparrow module not found: {sparrow_module}")
    try:
        manager.load(str(sparrow_module))
    except Exception:
        # Safe to continue if already loaded in this process.
        pass
    scine_utilities.core.Log.silent()
    return scine_sparrow, scine_utilities, manager


def _make_calculator(manager: Any, functional: str) -> Any:
    calc = manager.get("calculator", functional)
    if calc is None:
        raise RuntimeError(f"SCINE calculator '{functional}' is unavailable.")
    return calc


def _extract_settings(calc: Any) -> Dict[str, Any]:
    settings_obj = getattr(calc, "settings", None)
    if settings_obj is None:
        return {}

    # Try common APIs first.
    if hasattr(settings_obj, "as_dict"):
        try:
            out = settings_obj.as_dict()
            if isinstance(out, dict):
                return {str(k): _jsonable(v) for k, v in out.items()}
        except Exception:
            pass

    keys: List[str] = []
    if hasattr(settings_obj, "keys"):
        try:
            keys = [str(k) for k in list(settings_obj.keys())]
        except Exception:
            keys = []
    if not keys and hasattr(settings_obj, "__iter__"):
        try:
            keys = [str(k) for k in list(settings_obj)]
        except Exception:
            keys = []

    out: Dict[str, Any] = {}
    for key in keys:
        val = None
        got = False
        try:
            val = settings_obj[key]
            got = True
        except Exception:
            pass
        if not got and hasattr(settings_obj, "get"):
            try:
                val = settings_obj.get(key)
                got = True
            except Exception:
                pass
        if got:
            out[str(key)] = _jsonable(val)
    return out


def _set_setting(settings_obj: Any, key: str, value: Any) -> bool:
    # Try dict-style assignment.
    try:
        settings_obj[key] = value
        return True
    except Exception:
        pass

    # Try dedicated setter-style APIs.
    for method_name in ("set", "set_value", "modify", "update"):
        method = getattr(settings_obj, method_name, None)
        if method is None:
            continue
        try:
            if method_name == "update":
                method({key: value})
            else:
                method(key, value)
            return True
        except Exception:
            continue
    return False


def _build_tighter_profiles(settings: Dict[str, Any], max_profiles: int = 4) -> List[Dict[str, Any]]:
    tweaks: Dict[str, Any] = {}
    for key, val in settings.items():
        k = key.lower()
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue

        # Tighten convergence-like thresholds.
        if any(tok in k for tok in ("scf", "convergence", "criterion", "threshold", "tolerance", "accuracy")):
            if float(val) > 0:
                if any(tok in k for tok in ("max_iter", "max_iteration", "maxiterations", "iterations")):
                    tweaks[key] = int(max(1, round(float(val) * 2.0)))
                else:
                    tweaks[key] = float(val) * 0.1

    if not tweaks:
        return []

    profiles: List[Dict[str, Any]] = [{"name": "tight_combined", "overrides": tweaks}]
    for key, new_val in tweaks.items():
        profiles.append({"name": f"tight_{key}", "overrides": {key: new_val}})
        if len(profiles) >= max_profiles + 1:
            break
    return profiles[: max_profiles + 1]


def _evaluate_with_calculator(
    calc: Any,
    scine_utilities: Any,
    atomic_nums: torch.Tensor,
    coords_angstrom: torch.Tensor,
    atomsymbols: List[str],
) -> Dict[str, Any]:
    from src.dependencies.scine_calculator import suppress_output
    from src.dependencies.scine_masses import Z_TO_SCINE_ELEMENT as Z_TO_ELEMENT_TYPE
    from src.noisy.multi_mode_eckartmw import get_vib_evals_evecs
    from src.noisy.v2_tests.baselines.minimization import _force_mean

    required_props = [
        scine_utilities.Property.Energy,
        scine_utilities.Property.Gradients,
        scine_utilities.Property.Hessian,
    ]

    z_np = atomic_nums.detach().cpu().numpy().tolist()
    elements = [Z_TO_ELEMENT_TYPE[int(z)] for z in z_np]
    pos_bohr = coords_angstrom.detach().cpu().numpy() * scine_utilities.BOHR_PER_ANGSTROM
    structure = scine_utilities.AtomCollection(elements, pos_bohr)

    calc.structure = structure
    calc.set_required_properties(required_props)
    with suppress_output():
        res = calc.calculate()

    hartree_to_ev = 27.211386245988
    bohr_to_ang = 0.529177210903

    energy = float(res.energy * hartree_to_ev)
    forces = torch.tensor(
        res.gradients.reshape(-1, 3) * (-hartree_to_ev / bohr_to_ang),
        dtype=torch.float32,
    )
    hessian = torch.tensor(
        res.hessian * (hartree_to_ev / (bohr_to_ang ** 2)),
        dtype=torch.float32,
    )

    evals_vib, _evecs_vib, _q_vib = get_vib_evals_evecs(
        hessian, coords_angstrom.reshape(-1, 3).detach().cpu(), atomsymbols, purify_hessian=False,
    )
    evals = torch.sort(evals_vib).values
    abs_sorted = torch.sort(torch.abs(evals)).values

    k = min(5, int(abs_sorted.numel()))
    low_abs_mean = float(abs_sorted[:k].mean().item()) if k > 0 else float("nan")

    return {
        "energy": energy,
        "force_norm": _force_mean(forces),
        "n_neg": int((evals < 0.0).sum().item()),
        "min_eval": float(evals[0].item()) if evals.numel() > 0 else float("nan"),
        "min_abs_eval": float(abs_sorted[0].item()) if abs_sorted.numel() > 0 else float("nan"),
        "low_abs_eval_mean_k5": low_abs_mean,
        "eigenvalues": [float(v) for v in evals.tolist()],
    }


def write_report(payload: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    lines.append("SCINE Convergence Investigation")
    lines.append("=" * 72)
    lines.append(f"Generated at: {payload['generated_at']}")
    lines.append(f"Samples: {payload['n_samples']}")
    lines.append(f"Functional: {payload['scine_functional']}")
    lines.append("")

    enum_data = payload["settings_enumeration"]
    lines.append("Settings Enumeration")
    lines.append("-" * 22)
    lines.append(f"settings_detected: {enum_data['settings_detected']}")
    lines.append(f"n_settings: {enum_data['n_settings']}")
    if enum_data.get("candidate_profiles"):
        lines.append("candidate tighter profiles:")
        for prof in enum_data["candidate_profiles"]:
            lines.append(f"  - {prof['name']}: {prof['overrides']}")
    else:
        lines.append("candidate tighter profiles: none")
    lines.append("")

    lines.append("Profile Summary")
    lines.append("-" * 15)
    for profile_name, stats in payload["profile_summary"].items():
        lines.append(
            f"{profile_name}: n_ok={stats['n_ok']}, "
            f"n_neg_median={stats['n_neg']['median']:.3f}, "
            f"min_abs_eval_median={stats['min_abs_eval']['median']:.4e}, "
            f"low_abs_k5_median={stats['low_abs_eval_mean_k5']['median']:.4e}"
        )
    lines.append("")

    comparisons = payload.get("comparisons_vs_default", {})
    if comparisons:
        lines.append("Comparison vs default")
        lines.append("-" * 22)
        for profile_name, comp in comparisons.items():
            lines.append(
                f"{profile_name}: Δmin_abs_eval median={comp['delta_min_abs_eval']['median']:.4e}, "
                f"Δlow_abs_k5 median={comp['delta_low_abs_eval_mean_k5']['median']:.4e}, "
                f"Δn_neg median={comp['delta_n_neg']['median']:.3f}"
            )
    else:
        lines.append("No valid tighter-profile comparisons were produced.")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Investigate SCINE convergence settings and Hessian spectra.",
    )
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--scine-functional", type=str, default="DFTB0")
    parser.add_argument("--max-profiles", type=int, default=4, help="Max tighter profiles to test")
    args = parser.parse_args()

    missing: List[str] = []
    if np is None:
        missing.append("numpy")
    if torch is None:
        missing.append("torch")
    if missing:
        raise RuntimeError(
            "Missing dependencies required to run investigate_scine_convergence.py: "
            + ", ".join(missing)
        )

    from src.dependencies.common_utils import Transition1xDataset, UsePos
    from src.noisy.multi_mode_eckartmw import _atomic_nums_to_symbols

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = Transition1xDataset(
        h5_path=args.h5_path,
        split=args.split,
        max_samples=max(1, args.max_samples),
        transform=UsePos("pos_reactant"),
    )
    if len(dataset) == 0:
        raise RuntimeError("No samples loaded. Check --h5-path and --split.")

    scine_sparrow, scine_utilities, manager = _load_scine_runtime()
    base_calc = _make_calculator(manager, args.scine_functional)
    base_settings = _extract_settings(base_calc)

    candidate_profiles = _build_tighter_profiles(base_settings, max_profiles=args.max_profiles)
    profiles: List[Dict[str, Any]] = [{"name": "default", "overrides": {}}] + candidate_profiles

    print("=" * 72)
    print("SCINE convergence investigation")
    print("=" * 72)
    print(f"Samples: {len(dataset)}")
    print(f"Functional: {args.scine_functional}")
    print(f"Detected settings: {len(base_settings)}")
    print(f"Profiles: {[p['name'] for p in profiles]}")
    print("=" * 72)

    per_profile_rows: Dict[str, List[Dict[str, Any]]] = {p["name"]: [] for p in profiles}
    per_profile_errors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    t0 = time.time()
    for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]
        atomic_nums = sample.z.detach().to("cpu")
        coords = sample.pos_reactant.detach().to("cpu")
        atomsymbols = _atomic_nums_to_symbols(atomic_nums)

        for profile in profiles:
            name = profile["name"]
            overrides = profile["overrides"]
            try:
                calc = _make_calculator(manager, args.scine_functional)
                settings_obj = getattr(calc, "settings", None)
                applied = {}
                failed = {}
                if settings_obj is not None:
                    for key, value in overrides.items():
                        if _set_setting(settings_obj, key, value):
                            applied[key] = value
                        else:
                            failed[key] = value

                eval_data = _evaluate_with_calculator(
                    calc, scine_utilities, atomic_nums, coords, atomsymbols,
                )
                eval_data.update(
                    {
                        "sample_idx": sample_idx,
                        "formula": str(getattr(sample, "formula", "")),
                        "applied_settings": applied,
                        "failed_settings": failed,
                    }
                )
                per_profile_rows[name].append(eval_data)
            except Exception as exc:  # pragma: no cover - defensive for cluster environment
                per_profile_errors[name].append({"sample_idx": sample_idx, "error": str(exc)})

    wall = time.time() - t0

    profile_summary: Dict[str, Any] = {}
    for name, rows in per_profile_rows.items():
        profile_summary[name] = {
            "n_ok": len(rows),
            "n_errors": len(per_profile_errors.get(name, [])),
            "n_neg": _summary([float(r["n_neg"]) for r in rows]),
            "min_eval": _summary([float(r["min_eval"]) for r in rows]),
            "min_abs_eval": _summary([float(r["min_abs_eval"]) for r in rows]),
            "low_abs_eval_mean_k5": _summary([float(r["low_abs_eval_mean_k5"]) for r in rows]),
        }

    comparisons_vs_default: Dict[str, Any] = {}
    default_rows = {r["sample_idx"]: r for r in per_profile_rows.get("default", [])}
    for name, rows in per_profile_rows.items():
        if name == "default":
            continue
        deltas_min_abs: List[float] = []
        deltas_low_abs: List[float] = []
        deltas_n_neg: List[float] = []
        for row in rows:
            base = default_rows.get(row["sample_idx"])
            if base is None:
                continue
            deltas_min_abs.append(float(row["min_abs_eval"]) - float(base["min_abs_eval"]))
            deltas_low_abs.append(float(row["low_abs_eval_mean_k5"]) - float(base["low_abs_eval_mean_k5"]))
            deltas_n_neg.append(float(row["n_neg"]) - float(base["n_neg"]))
        comparisons_vs_default[name] = {
            "n_compared": len(deltas_min_abs),
            "delta_min_abs_eval": _summary(deltas_min_abs),
            "delta_low_abs_eval_mean_k5": _summary(deltas_low_abs),
            "delta_n_neg": _summary(deltas_n_neg),
        }

    payload: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "wall_time_seconds": wall,
        "h5_path": args.h5_path,
        "split": args.split,
        "scine_functional": args.scine_functional,
        "n_samples": len(dataset),
        "settings_enumeration": {
            "scine_sparrow_module": str(getattr(scine_sparrow, "__file__", "")),
            "settings_detected": bool(base_settings),
            "n_settings": len(base_settings),
            "settings": base_settings,
            "candidate_profiles": candidate_profiles,
        },
        "profile_summary": profile_summary,
        "comparisons_vs_default": comparisons_vs_default,
        "results_by_profile": per_profile_rows,
        "errors_by_profile": per_profile_errors,
    }

    json_path = out_dir / "investigate_scine_convergence.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    report_path = out_dir / "investigate_scine_convergence_report.txt"
    write_report(payload, report_path)

    print("")
    print(f"Done in {wall:.1f}s")
    print(f"JSON:   {json_path}")
    print(f"Report: {report_path}")
    if not candidate_profiles:
        print("No convergence-like SCINE settings were detected; this is itself a useful finding.")


if __name__ == "__main__":
    main()
