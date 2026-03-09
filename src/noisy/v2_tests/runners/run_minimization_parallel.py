#!/usr/bin/env python
"""Parallel SCINE runner for minimization baselines.

Methods:
- fixed_step_gd: Fixed-step gradient descent (no line search)
- newton_raphson: Newton-Raphson with Hessian-preconditioned steps

v2 flags:
- --lm-mu         Levenberg-Marquardt damping coefficient (0 = hard filter, default)
- --anneal-force-threshold
                  Force norm at which two-phase annealing kicks in (0 = off)
- --cleanup-nr-threshold
                  NR threshold used in cleanup phase (0 = full pseudoinverse)
- --cleanup-max-steps
                  Maximum extra steps in cleanup phase (default 50)
- --log-spectrum-k
                  Number of bottom vibrational eigenvalues to log per step (default 10)

v3 flags:
- --shift-epsilon  Shifted Newton: σ = max(0,-λ_min) + shift_epsilon (0 = off)
- --stagnation-window
                  Trigger escape perturbation after this many stagnant steps (0 = off)
- --escape-alpha   Max atom displacement for escape perturbation (default 0.1)
- --lm-mu-anneal-factor
                  Multiply μ by this when close to convergence (0 = off)
- --neg-mode-line-search
                  Enable line search along negative eigenvector during escape
- --trust-radius-floor
                  Minimum trust radius (default 0.01)

v7 flags:
- --step-control   Step control: 'trust_region' (default) or 'line_search' (Armijo backtracking)
- --max-nr-weight  Cap shifted Newton weight (0 = no cap)

v10 flags:
- --optimizer-mode  'arc' for ARC, 'rfo' for Rational Function Optimization
- --arc-sigma-init  ARC: initial cubic regularization σ (default 1.0)
- --arc-gamma1      ARC: σ increase factor on bad step (default 2.0)
- --gdiis-buffer-size
                  ARC: GDIIS buffer for oscillation damping (0 = off)
- --gdiis-every   ARC: attempt GDIIS every N steps (default 5)
- --gdiis-late-force-threshold
                  Attempt GDIIS when force_norm < this (0 = off, default)
- --schlegel-trust-update
                  Use Schlegel trust radius rules (boundary-check growth, step-anchored shrink)
- --polynomial-linesearch
                  v10c: cubic interpolation refinement on accepted trust-region steps

Cascade evaluation:
  Every trajectory step now contains "n_neg_at_<threshold>" fields for
  thresholds [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2].
  The summary JSON includes a "cascade_table" whose rows are keyed by
  nr_threshold and columns by eval_threshold, with values = convergence rate.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.dependencies.common_utils import Transition1xDataset, UsePos, parse_starting_geometry
from src.noisy.multi_mode_eckartmw import get_vib_evals_evecs, _atomic_nums_to_symbols
from src.noisy.v2_tests.baselines.minimization import (
    CASCADE_THRESHOLDS,
    run_fixed_step_gd,
    run_newton_raphson,
)
from src.noisy.v2_tests.baselines.pic_arc import run_pic_arc
from src.parallel.scine_parallel import ParallelSCINEProcessor
from src.parallel.utils import run_batch_parallel


def create_dataloader(h5_path: str, split: str, max_samples: int):
    dataset = Transition1xDataset(
        h5_path=h5_path,
        split=split,
        max_samples=max_samples,
        transform=UsePos("pos_transition"),
    )
    if len(dataset) == 0:
        raise RuntimeError("No Transition1x samples loaded. Check h5 path and split.")
    return DataLoader(dataset, batch_size=1, shuffle=False)


def run_single_sample(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    params: Dict[str, Any],
    n_steps: int,
    *,
    sample_id: str,
    formula: str,
    known_ts_coords: Optional[torch.Tensor] = None,
    known_reactant_coords: Optional[torch.Tensor] = None,
    known_product_coords: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    method = params["method"]
    project_gradient_and_v = params.get("project_gradient_and_v", False)
    # Atom symbols are always needed: reduced-basis eigendecomposition requires them.
    all_atomsymbols = _atomic_nums_to_symbols(atomic_nums)

    def _count_neg_vib(coords_local: torch.Tensor) -> Optional[int]:
        try:
            out = predict_fn(coords_local, atomic_nums, do_hessian=True, require_grad=False)
            evals_vib, _, _ = get_vib_evals_evecs(out["hessian"], coords_local, all_atomsymbols)
            return int((evals_vib < 0.0).sum().item())
        except Exception:
            return None

    def _cascade_neg_vib(coords_local: torch.Tensor) -> Optional[Dict[str, int]]:
        """Count n_neg at every cascade threshold for the final geometry."""
        try:
            out = predict_fn(coords_local, atomic_nums, do_hessian=True, require_grad=False)
            evals_vib, _, _ = get_vib_evals_evecs(out["hessian"], coords_local, all_atomsymbols)
            result: Dict[str, int] = {}
            for thr in CASCADE_THRESHOLDS:
                result[f"n_neg_at_{thr}"] = int((evals_vib < -thr).sum().item())
            result["min_vib_eval"] = float(evals_vib.min().item()) if evals_vib.numel() > 0 else float("nan")
            return result
        except Exception:
            return None

    t0 = time.time()
    try:
        start_neg_vib = _count_neg_vib(coords)

        if method == "fixed_step_gd":
            result, trajectory = run_fixed_step_gd(
                predict_fn,
                coords,
                atomic_nums,
                all_atomsymbols,
                n_steps=n_steps,
                step_size=params["step_size"],
                max_atom_disp=params["max_atom_disp"],
                force_converged=params["force_converged"],
                min_interatomic_dist=params["min_interatomic_dist"],
                project_gradient_and_v=project_gradient_and_v,
                purify_hessian=params.get("purify_hessian", False),
                log_spectrum_k=params.get("log_spectrum_k", 10),
            )
        elif method == "newton_raphson":
            result, trajectory = run_newton_raphson(
                predict_fn,
                coords,
                atomic_nums,
                all_atomsymbols,
                n_steps=n_steps,
                max_atom_disp=params["max_atom_disp"],
                force_converged=params["force_converged"],
                min_interatomic_dist=params["min_interatomic_dist"],
                nr_threshold=params.get("nr_threshold", 8e-3),
                project_gradient_and_v=project_gradient_and_v,
                purify_hessian=params.get("purify_hessian", False),
                known_ts_coords=known_ts_coords,
                known_reactant_coords=known_reactant_coords,
                known_product_coords=known_product_coords,
                # v2
                lm_mu=params.get("lm_mu", 0.0),
                anneal_force_threshold=params.get("anneal_force_threshold", 0.0),
                cleanup_nr_threshold=params.get("cleanup_nr_threshold", 0.0),
                cleanup_max_steps=params.get("cleanup_max_steps", 50),
                log_spectrum_k=params.get("log_spectrum_k", 10),
                # v3
                shift_epsilon=params.get("shift_epsilon", 0.0),
                stagnation_window=params.get("stagnation_window", 0),
                escape_alpha=params.get("escape_alpha", 0.1),
                lm_mu_anneal_factor=params.get("lm_mu_anneal_factor", 0.0),
                lm_mu_anneal_n_neg_leq=params.get("lm_mu_anneal_n_neg_leq", 2),
                lm_mu_anneal_eval_leq=params.get("lm_mu_anneal_eval_leq", 5e-3),
                neg_mode_line_search=params.get("neg_mode_line_search", False),
                trust_radius_floor=params.get("trust_radius_floor", 0.01),
                # v4
                neg_trust_floor=params.get("neg_trust_floor", 0.0),
                blind_mode_threshold=params.get("blind_mode_threshold", 0.0),
                blind_correction_alpha=params.get("blind_correction_alpha", 0.02),
                aggressive_trust_recovery=params.get("aggressive_trust_recovery", False),
                escape_bidirectional=params.get("escape_bidirectional", False),
                mode_follow_eval_threshold=params.get("mode_follow_eval_threshold", 0.0),
                mode_follow_alpha=params.get("mode_follow_alpha", 0.15),
                mode_follow_after_steps=params.get("mode_follow_after_steps", 2000),
                # v5 SPDN
                optimizer_mode=params.get("optimizer_mode", ""),
                spdn_tau_hard=params.get("spdn_tau_hard", 0.01),
                spdn_tau_soft=params.get("spdn_tau_soft", 1e-4),
                spdn_diis_size=params.get("spdn_diis_size", 8),
                spdn_diis_every=params.get("spdn_diis_every", 5),
                spdn_momentum=params.get("spdn_momentum", 0.0),
                # v7
                step_control=params.get("step_control", "trust_region"),
                max_nr_weight=params.get("max_nr_weight", 0.0),
                # v8 crossover
                crossover_mu_max=params.get("crossover_mu_max", 0.0),
                crossover_n_neg_ref=params.get("crossover_n_neg_ref", 3.0),
                crossover_force_ref=params.get("crossover_force_ref", 0.1),
                # v9 relaxed convergence
                relaxed_eval_threshold=params.get("relaxed_eval_threshold", 0.0),
                accept_relaxed=params.get("accept_relaxed", False),
                # v10 ARC
                arc_sigma_init=params.get("arc_sigma_init", 1.0),
                arc_sigma_min=params.get("arc_sigma_min", 1e-4),
                arc_sigma_max=params.get("arc_sigma_max", 1e4),
                arc_eta1=params.get("arc_eta1", 0.1),
                arc_eta2=params.get("arc_eta2", 0.9),
                arc_gamma1=params.get("arc_gamma1", 2.0),
                arc_gamma2=params.get("arc_gamma2", 0.5),
                gdiis_buffer_size=params.get("gdiis_buffer_size", 0),
                gdiis_every=params.get("gdiis_every", 5),
                gdiis_late_force_threshold=params.get("gdiis_late_force_threshold", 0.0),
                schlegel_trust_update=params.get("schlegel_trust_update", False),
                polynomial_linesearch=params.get("polynomial_linesearch", False),
                # v12 kicks
                osc_kick=params.get("osc_kick", False),
                osc_kick_scale=params.get("osc_kick_scale", 0.1),
                osc_kick_patience=params.get("osc_kick_patience", 3),
                osc_kick_cooldown=params.get("osc_kick_cooldown", 50),
                blind_kick=params.get("blind_kick", False),
                blind_kick_scale=params.get("blind_kick_scale", 0.5),
                blind_kick_overlap_thresh=params.get("blind_kick_overlap_thresh", 0.1),
                blind_kick_force_thresh=params.get("blind_kick_force_thresh", 0.1),
                blind_kick_patience=params.get("blind_kick_patience", 100),
                kick_eigvec_index=params.get("kick_eigvec_index", 0),
                # v12b
                adaptive_kick_scale=params.get("adaptive_kick_scale", False),
                adaptive_kick_C=params.get("adaptive_kick_C", 0.1),
                blind_kick_probe=params.get("blind_kick_probe", False),
                late_escape=params.get("late_escape", False),
                late_escape_after=params.get("late_escape_after", 15000),
                late_escape_alpha=params.get("late_escape_alpha", 0.1),
                late_escape_cooldown=params.get("late_escape_cooldown", 500),
            )
        elif method == "pic_arc":
            result, trajectory = run_pic_arc(
                predict_fn,
                coords,
                atomic_nums,
                all_atomsymbols,
                n_steps=n_steps,
                max_atom_disp=params["max_atom_disp"],
                force_converged=params["force_converged"],
                min_interatomic_dist=params["min_interatomic_dist"],
                project_gradient_and_v=project_gradient_and_v,
                purify_hessian=params.get("purify_hessian", False),
                log_spectrum_k=params.get("log_spectrum_k", 10),
                # Trust region
                trust_radius_init=params.get("trust_radius_init", 0.5),
                trust_radius_floor=params.get("trust_radius_floor", 0.01),
                # Metric
                k_bond=params.get("k_bond", 0.45),
                bond_threshold_factor=params.get("bond_threshold_factor", 1.3),
                metric_regularization=params.get("metric_regularization", 1e-3),
                metric_refresh_every=params.get("metric_refresh_every", 0),
                # ARC
                sigma_init=params.get("sigma_init", 1.0),
                sigma_min=params.get("sigma_min", 0.01),
                sigma_max=params.get("sigma_max", 100.0),
                max_neg_modes_in_subspace=params.get("max_neg_modes_in_subspace", 5),
                # State machine
                kappa_threshold=params.get("kappa_threshold", 1e6),
                n_neg_max_for_arc=params.get("n_neg_max_for_arc", 5),
                force_max_for_arc=params.get("force_max_for_arc", 1.0),
                stability_window=params.get("stability_window", 3),
                max_consecutive_arc_rejects=params.get("max_consecutive_arc_rejects", 3),
                # Termination
                relaxed_eval_threshold=params.get("relaxed_eval_threshold", 0.01),
                accept_relaxed=params.get("accept_relaxed", False),
                # Reference geometries
                known_ts_coords=known_ts_coords,
                known_reactant_coords=known_reactant_coords,
                known_product_coords=known_product_coords,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        final_coords = result.get("final_coords")
        final_cascade = _cascade_neg_vib(
            final_coords.to(coords.device) if isinstance(final_coords, torch.Tensor) else coords
        )
        final_neg_vib = final_cascade.get("n_neg_at_0.0") if final_cascade else None

        if params.get("log_dir"):
            log_dir = Path(params["log_dir"])
            log_dir.mkdir(parents=True, exist_ok=True)
            traj_path = log_dir / f"{sample_id}_{method}_trajectory.json"
            with open(traj_path, "w") as f:
                json.dump(
                    {
                        "sample_id": sample_id,
                        "formula": formula,
                        "method": method,
                        "start_neg_vib": start_neg_vib,
                        "final_neg_vib": final_neg_vib,
                        "final_cascade": final_cascade,
                        "cascade_at_convergence": result.get("cascade_at_convergence", {}),
                        "bottom_spectrum_at_convergence": result.get("bottom_spectrum_at_convergence", []),
                        "final_min_vib_eval": result.get("final_min_vib_eval"),
                        "cleanup_steps_taken": result.get("cleanup_steps_taken", 0),
                        "total_escapes": result.get("total_escapes", 0),
                        "total_line_searches": result.get("total_line_searches", 0),
                        "total_mode_follows": result.get("total_mode_follows", 0),
                        "total_diis_attempts": result.get("total_diis_attempts", 0),
                        "total_diis_accepts": result.get("total_diis_accepts", 0),
                        "total_diis_energy_accepts": result.get("total_diis_energy_accepts", 0),
                        "optimizer_mode": result.get("optimizer_mode", ""),
                        "final_arc_sigma": result.get("final_arc_sigma"),
                        "arc_gdiis_attempts": result.get("arc_gdiis_attempts", 0),
                        "arc_gdiis_accepts": result.get("arc_gdiis_accepts", 0),
                        "total_osc_kicks": result.get("total_osc_kicks", 0),
                        "total_blind_kicks": result.get("total_blind_kicks", 0),
                        "total_late_escapes": result.get("total_late_escapes", 0),
                        "trajectory": trajectory,
                    },
                    f,
                    indent=2,
                )

        wall_time = time.time() - t0
        return {
            "converged": bool(result.get("converged")),
            "converged_step": result.get("converged_step"),
            "final_energy": result.get("final_energy"),
            "final_force_norm": result.get("final_force_norm"),
            "total_steps": result.get("total_steps", n_steps),
            "start_neg_vib": start_neg_vib,
            "final_neg_vib": final_neg_vib,
            "final_cascade": final_cascade,
            "final_min_vib_eval": result.get("final_min_vib_eval"),
            "cascade_at_convergence": result.get("cascade_at_convergence", {}),
            "bottom_spectrum_at_convergence": result.get("bottom_spectrum_at_convergence", []),
            "cleanup_steps_taken": result.get("cleanup_steps_taken", 0),
            "total_escapes": result.get("total_escapes", 0),
            "total_line_searches": result.get("total_line_searches", 0),
            "total_mode_follows": result.get("total_mode_follows", 0),
            "wall_time": wall_time,
            "error": None,
        }
    except Exception as e:
        wall_time = time.time() - t0
        return {
            "converged": False,
            "converged_step": None,
            "final_energy": None,
            "final_force_norm": None,
            "total_steps": 0,
            "start_neg_vib": None,
            "final_neg_vib": None,
            "final_cascade": None,
            "final_min_vib_eval": None,
            "cascade_at_convergence": {},
            "bottom_spectrum_at_convergence": [],
            "cleanup_steps_taken": 0,
            "wall_time": wall_time,
            "error": str(e),
        }


def scine_worker_sample(predict_fn, payload) -> Dict[str, Any]:
    sample_idx, batch, params, n_steps, start_from, noise_seed = payload
    batch = batch.to("cpu")
    atomic_nums = batch.z.detach().to("cpu")
    start_coords = parse_starting_geometry(
        start_from,
        batch,
        noise_seed=noise_seed,
        sample_index=sample_idx,
    ).detach().to("cpu")

    known_ts_coords = getattr(batch, "pos_transition", None)
    if known_ts_coords is None:
        known_ts_coords = getattr(batch, "pos", None)
    if known_ts_coords is not None:
        known_ts_coords = known_ts_coords.detach().to("cpu")

    known_reactant_coords = getattr(batch, "pos_reactant", None)
    if known_reactant_coords is not None:
        known_reactant_coords = known_reactant_coords.detach().to("cpu")

    known_product_coords = None
    has_product = getattr(batch, "has_product", None)
    if has_product is not None and bool(has_product.item()):
        known_product_coords = getattr(batch, "pos_product", None)
        if known_product_coords is not None:
            known_product_coords = known_product_coords.detach().to("cpu")

    formula = getattr(batch, "formula", "")
    result = run_single_sample(
        predict_fn,
        start_coords,
        atomic_nums,
        params,
        n_steps,
        sample_id=f"sample_{sample_idx:03d}",
        formula=str(formula),
        known_ts_coords=known_ts_coords,
        known_reactant_coords=known_reactant_coords,
        known_product_coords=known_product_coords,
    )
    result["sample_idx"] = sample_idx
    result["formula"] = getattr(batch, "formula", "")
    return result


def _build_cascade_table(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a 2D convergence-rate table over cascade evaluation thresholds.

    For each eval_threshold in CASCADE_THRESHOLDS, a sample is "converged at
    that eval threshold" if its final_cascade["n_neg_at_<thresh>"] == 0.

    This lets the analysis script answer: "If we relaxed the evaluation to
    accept eigenvalues down to -<thresh>, what would our convergence rate be?"
    WITHOUT changing the optimizer at all. The optimizer's nr_threshold is
    the row key (passed in as a param label); columns are eval thresholds.

    Returns a dict:
      {
        "eval_thresholds": [0.0, 1e-4, ...],
        "n_samples": <int>,
        "n_converged_at_thr": {"0.0": <int>, "0.0001": <int>, ...},
        "rate_at_thr":        {"0.0": <float>, ...},
      }
    """
    n = len(results)
    n_converged_at: Dict[str, int] = {}
    rate_at: Dict[str, float] = {}
    for thr in CASCADE_THRESHOLDS:
        key = f"n_neg_at_{thr}"
        count = sum(
            1 for r in results
            if r.get("final_cascade") and r["final_cascade"].get(key, 1) == 0
        )
        n_converged_at[str(thr)] = count
        rate_at[str(thr)] = count / max(n, 1)
    return {
        "eval_thresholds": CASCADE_THRESHOLDS,
        "n_samples": n,
        "n_converged_at_thr": n_converged_at,
        "rate_at_thr": rate_at,
    }


def run_batch(
    processor: ParallelSCINEProcessor,
    dataloader,
    params: Dict[str, Any],
    n_steps: int,
    max_samples: int,
    start_from: str,
    noise_seed: Optional[int],
) -> Dict[str, Any]:
    samples = []
    for i, batch in enumerate(dataloader):
        if i >= max_samples:
            break
        payload = (i, batch, params, n_steps, start_from, noise_seed)
        samples.append((i, payload))

    results = run_batch_parallel(samples, processor)

    n_samples = len(results)
    n_converged = sum(1 for r in results if r.get("converged"))
    n_errors = sum(1 for r in results if r.get("error") is not None)

    steps_when_success = [r["converged_step"] for r in results if r.get("converged_step") is not None]
    wall_times = [r["wall_time"] for r in results]

    cascade_table = _build_cascade_table(results)

    return {
        "n_samples": n_samples,
        "n_converged": n_converged,
        "n_errors": n_errors,
        "convergence_rate": n_converged / max(n_samples, 1),
        "mean_steps_when_converged": float(np.mean(steps_when_success)) if steps_when_success else float("nan"),
        "mean_wall_time": float(np.mean(wall_times)) if wall_times else float("nan"),
        "total_wall_time": float(sum(wall_times)),
        "cascade_table": cascade_table,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimization baselines (parallel, SCINE)")
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--scine-functional", type=str, default="DFTB0")
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--start-from", type=str, default="midpoint_rt_noise1.0A")
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=4)

    # Method selection
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["fixed_step_gd", "newton_raphson", "pic_arc"],
        help="Minimization method: 'fixed_step_gd' or 'newton_raphson'",
    )

    # Shared parameters
    parser.add_argument("--max-atom-disp", type=float, default=1.3)
    parser.add_argument("--force-converged", type=float, default=1e-4)
    parser.add_argument("--min-interatomic-dist", type=float, default=0.5)
    parser.add_argument(
        "--nr-threshold", type=float, default=8e-3,
        help="Eigenvalue cutoff for NR step filtering (|λ| < threshold → excluded from "
             "pseudoinverse).",
    )

    # --- New v2 flags ---
    parser.add_argument(
        "--lm-mu", type=float, default=0.0,
        help="Levenberg-Marquardt damping coefficient μ. 0 (default) = hard filter mode. "
             ">0 = LM mode: step_i = (g·v_i)*|λ_i|/(λ_i²+μ²). "
             "Sweep this with the same grid as --nr-threshold.",
    )
    parser.add_argument(
        "--anneal-force-threshold", type=float, default=0.0,
        help="Two-phase annealing: once force_norm < this value, switch from nr-threshold "
             "to cleanup-nr-threshold for cleanup-max-steps steps. 0 = off (default).",
    )
    parser.add_argument(
        "--cleanup-nr-threshold", type=float, default=0.0,
        help="NR threshold used in the cleanup phase of two-phase annealing. "
             "0 (default) = full pseudoinverse (no filtering). "
             "Only used when --anneal-force-threshold > 0.",
    )
    parser.add_argument(
        "--cleanup-max-steps", type=int, default=50,
        help="Maximum extra steps in cleanup phase (default 50).",
    )
    parser.add_argument(
        "--log-spectrum-k", type=int, default=10,
        help="Number of bottom vibrational eigenvalues to log per step (default 10, 0 = none).",
    )

    # --- v3 flags ---
    parser.add_argument(
        "--shift-epsilon", type=float, default=0.0,
        help="Shifted Newton: σ = max(0,-λ_min) + shift_epsilon. "
             ">0 activates shifted Newton mode (takes priority over LM and HF). "
             "Negative modes get larger weights → more aggressive escape.",
    )
    parser.add_argument(
        "--stagnation-window", type=int, default=0,
        help="Trigger escape perturbation when n_neg is unchanged for this many "
             "consecutive steps. 0 = off (default).",
    )
    parser.add_argument(
        "--escape-alpha", type=float, default=0.1,
        help="Max atom displacement (Å) for stagnation escape perturbation (default 0.1).",
    )
    parser.add_argument(
        "--lm-mu-anneal-factor", type=float, default=0.0,
        help="When close to convergence (few small neg evals), multiply μ by this factor. "
             "0 = off (default). E.g. 0.1 → μ becomes 10x smaller near convergence.",
    )
    parser.add_argument(
        "--lm-mu-anneal-n-neg-leq", type=int, default=2,
        help="Anneal μ only when n_neg <= this value (default 2).",
    )
    parser.add_argument(
        "--lm-mu-anneal-eval-leq", type=float, default=5e-3,
        help="Anneal μ only when |min_vib_eval| < this value (default 5e-3).",
    )
    parser.add_argument(
        "--neg-mode-line-search", action="store_true", default=False,
        help="During stagnation escape, also do a line search along the most negative "
             "eigenvector to find where curvature changes sign.",
    )
    parser.add_argument(
        "--trust-radius-floor", type=float, default=0.01,
        help="Minimum trust radius (Å). Prevents optimizer from shrinking to tiny "
             "displacements (default 0.01).",
    )

    # --- v4 flags ---
    parser.add_argument(
        "--neg-trust-floor", type=float, default=0.0,
        help="Separate trust-radius floor for negative-mode subspace (Å). "
             "0 = off (use single trust radius). >0 enables split trust regions "
             "with independent neg/pos capping (default 0.0).",
    )
    parser.add_argument(
        "--blind-mode-threshold", type=float, default=0.0,
        help="Gradient overlap threshold for blind-mode correction. Negative modes "
             "with |g·v_i|/|g| < this get a fixed perturbation. 0 = off (default).",
    )
    parser.add_argument(
        "--blind-correction-alpha", type=float, default=0.02,
        help="Max atom displacement (Å) for blind-mode correction (default 0.02).",
    )
    parser.add_argument(
        "--aggressive-trust-recovery", action="store_true", default=False,
        help="Softer trust-radius shrink (×0.5 instead of ×0.25) near convergence, "
             "plus automatic recovery when eigenvalues improve.",
    )
    parser.add_argument(
        "--escape-bidirectional", action="store_true", default=False,
        help="Use v4 bidirectional stagnation escape: probe ±v_0 and pick direction "
             "with less negative eigenvalue. Conditional acceptance.",
    )
    parser.add_argument(
        "--mode-follow-eval-threshold", type=float, default=0.0,
        help="Trigger mode-following when |min_vib_eval| > this (true saddle). "
             "0 = off (default). Typical: 0.01-0.05.",
    )
    parser.add_argument(
        "--mode-follow-alpha", type=float, default=0.15,
        help="Displacement magnitude (Å) for mode-following probes (default 0.15).",
    )
    parser.add_argument(
        "--mode-follow-after-steps", type=int, default=2000,
        help="Only trigger mode-following after this many steps (default 2000).",
    )

    # --- v5 SPDN flags ---
    parser.add_argument(
        "--optimizer-mode", type=str, default="",
        help="Optimizer mode: '' (default, use v1-v4 logic), 'spdn' (Spectrally-Partitioned "
             "DIIS-Newton), 'arc' (v10: full-spectrum ARC with adaptive σ), "
             "'rfo' (v10b: Rational Function Optimization with augmented Hessian).",
    )
    parser.add_argument(
        "--spdn-tau-hard", type=float, default=0.01,
        help="SPDN: hard-mode threshold. Eigenvalues with |λ|>tau_hard get full Newton "
             "weight; others are capped at 1/tau_hard (default 0.01).",
    )
    parser.add_argument(
        "--spdn-tau-soft", type=float, default=1e-4,
        help="SPDN: ghost-mode threshold. Eigenvalues with |λ|<=tau_soft are treated as "
             "effectively zero for convergence. Default 1e-4.",
    )
    parser.add_argument(
        "--spdn-diis-size", type=int, default=8,
        help="SPDN: GDIIS buffer size (number of stored geometry/gradient pairs). "
             "Should match or exceed the oscillation period (default 8).",
    )
    parser.add_argument(
        "--spdn-diis-every", type=int, default=5,
        help="SPDN: attempt GDIIS extrapolation every N steps (default 5).",
    )
    parser.add_argument(
        "--spdn-momentum", type=float, default=0.0,
        help="SPDN: Polyak heavy-ball momentum coefficient. 0 = off (default). "
             "Typical: 0.2-0.5.",
    )

    # v7 options
    parser.add_argument(
        "--step-control", type=str, default="trust_region",
        choices=["trust_region", "line_search"],
        help="Step control strategy: 'trust_region' (adaptive trust radius, v1-v6 default) "
             "or 'line_search' (Armijo backtracking, stateless). Default: trust_region.",
    )
    parser.add_argument(
        "--max-nr-weight", type=float, default=0.0,
        help="Cap shifted Newton weight at this value. 0 = no cap (default). "
             "E.g. 200 caps mode amplification at 200x instead of 1/epsilon.",
    )

    # v8 crossover options
    parser.add_argument(
        "--crossover-mu-max", type=float, default=0.0,
        help="iHiSD crossover: max additive damping mu. 0 = off (default). "
             "When active, forces Armijo line search. Typical: 0.1-2.0.",
    )
    parser.add_argument(
        "--crossover-n-neg-ref", type=float, default=3.0,
        help="iHiSD crossover: Morse index reference for alpha. "
             "alpha_morse = max(0, 1 - n_neg/ref). Default: 3.0.",
    )
    parser.add_argument(
        "--crossover-force-ref", type=float, default=0.1,
        help="iHiSD crossover: force norm reference for alpha. "
             "alpha_force = 1/(1 + force_norm/ref). Default: 0.1 eV/A.",
    )

    # --- v10 ARC flags ---
    parser.add_argument(
        "--arc-sigma-init", type=float, default=1.0,
        help="ARC: initial cubic regularization parameter σ (default 1.0).",
    )
    parser.add_argument(
        "--arc-sigma-min", type=float, default=1e-4,
        help="ARC: minimum σ clamp (default 1e-4).",
    )
    parser.add_argument(
        "--arc-sigma-max", type=float, default=1e4,
        help="ARC: maximum σ clamp (default 1e4).",
    )
    parser.add_argument(
        "--arc-eta1", type=float, default=0.1,
        help="ARC: successful step threshold (ρ ≥ η₁ → accept). Default 0.1.",
    )
    parser.add_argument(
        "--arc-eta2", type=float, default=0.9,
        help="ARC: very successful step threshold (ρ ≥ η₂ → decrease σ). Default 0.9.",
    )
    parser.add_argument(
        "--arc-gamma1", type=float, default=2.0,
        help="ARC: σ increase factor on unsuccessful step. Default 2.0.",
    )
    parser.add_argument(
        "--arc-gamma2", type=float, default=0.5,
        help="ARC: σ decrease factor on very successful step. Default 0.5.",
    )
    parser.add_argument(
        "--gdiis-buffer-size", type=int, default=0,
        help="ARC: GDIIS buffer size for oscillation damping. 0 = off (default).",
    )
    parser.add_argument(
        "--gdiis-every", type=int, default=5,
        help="ARC: attempt GDIIS every N steps when oscillation detected. Default 5.",
    )
    parser.add_argument(
        "--gdiis-late-force-threshold", type=float, default=0.0,
        help="GDIIS late-stage: attempt GDIIS when force_norm < this. 0 = off (default).",
    )
    parser.add_argument(
        "--schlegel-trust-update", action="store_true", default=False,
        help="Use Schlegel trust radius rules: boundary-check growth + step-anchored shrink.",
    )
    parser.add_argument(
        "--polynomial-linesearch", action="store_true", default=False,
        help="v10c: cubic interpolation refinement on accepted trust-region steps.",
    )

    # --- v12 kick flags ---
    parser.add_argument(
        "--osc-kick", action="store_true", default=False,
        help="v12: enable oscillation kick. Perturbs along longest-stuck negative mode "
             "when oscillation persists at trust-radius floor.",
    )
    parser.add_argument(
        "--osc-kick-scale", type=float, default=0.1,
        help="v12: oscillation kick magnitude as fraction of trust_radius_floor (default 0.1).",
    )
    parser.add_argument(
        "--osc-kick-patience", type=int, default=3,
        help="v12: consecutive oscillation detections before kicking (default 3).",
    )
    parser.add_argument(
        "--osc-kick-cooldown", type=int, default=50,
        help="v12: steps between oscillation kicks (default 50).",
    )
    parser.add_argument(
        "--blind-kick", action="store_true", default=False,
        help="v12: enable blind-mode kick. Perturbs along gradient-orthogonal negative mode "
             "when force is low and blind-mode persists.",
    )
    parser.add_argument(
        "--blind-kick-scale", type=float, default=0.5,
        help="v12: blind-mode kick magnitude as fraction of trust_radius_floor (default 0.5).",
    )
    parser.add_argument(
        "--blind-kick-overlap-thresh", type=float, default=0.1,
        help="v12: gradient overlap threshold for blind mode detection (default 0.1).",
    )
    parser.add_argument(
        "--blind-kick-force-thresh", type=float, default=0.1,
        help="v12: force norm threshold to enable blind-mode kick (default 0.1 eV/A).",
    )
    parser.add_argument(
        "--blind-kick-patience", type=int, default=100,
        help="v12: consecutive steps with blind mode before kicking (default 100).",
    )
    parser.add_argument(
        "--kick-eigvec-index", type=int, default=0,
        help="v12: eigenvector to kick along. 0=longest-stuck/blind (default), "
             "1=second-longest (ablation).",
    )

    # --- v12b improvement flags ---
    parser.add_argument(
        "--adaptive-kick-scale", action="store_true", default=False,
        help="v12b: scale kick magnitude with |lambda_min|^{1/2} instead of fixed fraction.",
    )
    parser.add_argument(
        "--adaptive-kick-C", type=float, default=0.1,
        help="v12b: multiplier for adaptive kick: mag = C * |eval|^{1/2} (default 0.1).",
    )
    parser.add_argument(
        "--blind-kick-probe", action="store_true", default=False,
        help="v12b: line-probe along blind eigvec at increasing magnitudes, "
             "accept first point where n_neg decreases.",
    )
    parser.add_argument(
        "--late-escape", action="store_true", default=False,
        help="v12b: aggressive displacement along most negative eigvec after "
             "late-escape-after steps, accept if n_neg decreases.",
    )
    parser.add_argument(
        "--late-escape-after", type=int, default=15000,
        help="v12b: step threshold for late escape (default 15000).",
    )
    parser.add_argument(
        "--late-escape-alpha", type=float, default=0.1,
        help="v12b: displacement magnitude for late escape in Angstrom (default 0.1).",
    )
    parser.add_argument(
        "--late-escape-cooldown", type=int, default=500,
        help="v12b: steps between late escapes (default 500).",
    )

    # --- PIC-ARC flags ---
    parser.add_argument(
        "--trust-radius-init", type=float, default=0.5,
        help="PIC-ARC: initial trust radius (default 0.5 A).",
    )
    parser.add_argument(
        "--sigma-init", type=float, default=1.0,
        help="PIC-ARC: initial cubic regularization parameter (default 1.0).",
    )
    parser.add_argument(
        "--sigma-min", type=float, default=0.01,
        help="PIC-ARC: minimum sigma (default 0.01).",
    )
    parser.add_argument(
        "--sigma-max", type=float, default=100.0,
        help="PIC-ARC: maximum sigma (default 100.0).",
    )
    parser.add_argument(
        "--kappa-threshold", type=float, default=1e6,
        help="PIC-ARC: condition number threshold for FLOW/ARC switch (default 1e6).",
    )
    parser.add_argument(
        "--n-neg-max-for-arc", type=int, default=5,
        help="PIC-ARC: max negative modes to allow ARC phase (default 5).",
    )
    parser.add_argument(
        "--force-max-for-arc", type=float, default=1.0,
        help="PIC-ARC: max force norm to allow ARC phase (default 1.0 eV/A).",
    )
    parser.add_argument(
        "--stability-window", type=int, default=3,
        help="PIC-ARC: stable steps required before entering ARC (default 3).",
    )
    parser.add_argument(
        "--max-consecutive-arc-rejects", type=int, default=3,
        help="PIC-ARC: rejected ARC steps before fallback to FLOW (default 3).",
    )
    parser.add_argument(
        "--relaxed-eval-threshold", type=float, default=0.0,
        help="Relaxed convergence: accept if min_vib_eval >= -threshold AND "
             "force < converged. 0 = strict only (n_neg==0). Typical: 0.001-0.01.",
    )
    parser.add_argument(
        "--accept-relaxed", action="store_true", default=False,
        help="Accept RELAXED convergence (min_eval >= -threshold) as success.",
    )
    parser.add_argument(
        "--k-bond", type=float, default=0.45,
        help="PIC-ARC: Lindh stretch force constant for metric (default 0.45).",
    )
    parser.add_argument(
        "--bond-threshold-factor", type=float, default=1.3,
        help="PIC-ARC: bond detection factor on covalent radii (default 1.3).",
    )
    parser.add_argument(
        "--metric-regularization", type=float, default=1e-3,
        help="PIC-ARC: diagonal regularization for metric (default 1e-3).",
    )
    parser.add_argument(
        "--metric-refresh-every", type=int, default=0,
        help="PIC-ARC: refresh metric every N steps (0 = never, default 0).",
    )
    parser.add_argument(
        "--max-neg-modes-in-subspace", type=int, default=5,
        help="PIC-ARC: max negative modes in ARC subspace (default 5).",
    )

    # Gradient descent parameters
    parser.add_argument("--step-size", type=float, default=0.01,
                        help="Fixed step size for gradient descent")

    # Projection options
    parser.add_argument(
        "--project-gradient-and-v",
        action="store_true",
        default=True,
        help="Eckart-project the gradient to prevent TR drift",
    )
    parser.add_argument(
        "--purify-hessian",
        action="store_true",
        default=False,
        help="Enforce translational sum rules on the Hessian (Newton-Raphson only)",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    diag_dir = Path(args.out_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "method": args.method,
        "step_size": args.step_size,
        "max_atom_disp": args.max_atom_disp,
        "force_converged": args.force_converged,
        "min_interatomic_dist": args.min_interatomic_dist,
        "nr_threshold": args.nr_threshold,
        "project_gradient_and_v": args.project_gradient_and_v,
        "purify_hessian": args.purify_hessian,
        "log_dir": str(diag_dir),
        # v2 additions
        "lm_mu": args.lm_mu,
        "anneal_force_threshold": args.anneal_force_threshold,
        "cleanup_nr_threshold": args.cleanup_nr_threshold,
        "cleanup_max_steps": args.cleanup_max_steps,
        "log_spectrum_k": args.log_spectrum_k,
        # v3 additions
        "shift_epsilon": args.shift_epsilon,
        "stagnation_window": args.stagnation_window,
        "escape_alpha": args.escape_alpha,
        "lm_mu_anneal_factor": args.lm_mu_anneal_factor,
        "lm_mu_anneal_n_neg_leq": args.lm_mu_anneal_n_neg_leq,
        "lm_mu_anneal_eval_leq": args.lm_mu_anneal_eval_leq,
        "neg_mode_line_search": args.neg_mode_line_search,
        "trust_radius_floor": args.trust_radius_floor,
        # v4 additions
        "neg_trust_floor": args.neg_trust_floor,
        "blind_mode_threshold": args.blind_mode_threshold,
        "blind_correction_alpha": args.blind_correction_alpha,
        "aggressive_trust_recovery": args.aggressive_trust_recovery,
        "escape_bidirectional": args.escape_bidirectional,
        "mode_follow_eval_threshold": args.mode_follow_eval_threshold,
        "mode_follow_alpha": args.mode_follow_alpha,
        "mode_follow_after_steps": args.mode_follow_after_steps,
        # v5 SPDN additions
        "optimizer_mode": args.optimizer_mode,
        "spdn_tau_hard": args.spdn_tau_hard,
        "spdn_tau_soft": args.spdn_tau_soft,
        "spdn_diis_size": args.spdn_diis_size,
        "spdn_diis_every": args.spdn_diis_every,
        "spdn_momentum": args.spdn_momentum,
        # v7 additions
        "step_control": args.step_control,
        "max_nr_weight": args.max_nr_weight,
        # v8 crossover additions
        "crossover_mu_max": args.crossover_mu_max,
        "crossover_n_neg_ref": args.crossover_n_neg_ref,
        "crossover_force_ref": args.crossover_force_ref,
        # v10 ARC additions
        "arc_sigma_init": args.arc_sigma_init,
        "arc_sigma_min": args.arc_sigma_min,
        "arc_sigma_max": args.arc_sigma_max,
        "arc_eta1": args.arc_eta1,
        "arc_eta2": args.arc_eta2,
        "arc_gamma1": args.arc_gamma1,
        "arc_gamma2": args.arc_gamma2,
        "gdiis_buffer_size": args.gdiis_buffer_size,
        "gdiis_every": args.gdiis_every,
        "gdiis_late_force_threshold": args.gdiis_late_force_threshold,
        "schlegel_trust_update": args.schlegel_trust_update,
        "polynomial_linesearch": args.polynomial_linesearch,
        # v12 kick additions
        "osc_kick": args.osc_kick,
        "osc_kick_scale": args.osc_kick_scale,
        "osc_kick_patience": args.osc_kick_patience,
        "osc_kick_cooldown": args.osc_kick_cooldown,
        "blind_kick": args.blind_kick,
        "blind_kick_scale": args.blind_kick_scale,
        "blind_kick_overlap_thresh": args.blind_kick_overlap_thresh,
        "blind_kick_force_thresh": args.blind_kick_force_thresh,
        "blind_kick_patience": args.blind_kick_patience,
        "kick_eigvec_index": args.kick_eigvec_index,
        # v12b additions
        "adaptive_kick_scale": args.adaptive_kick_scale,
        "adaptive_kick_C": args.adaptive_kick_C,
        "blind_kick_probe": args.blind_kick_probe,
        "late_escape": args.late_escape,
        "late_escape_after": args.late_escape_after,
        "late_escape_alpha": args.late_escape_alpha,
        "late_escape_cooldown": args.late_escape_cooldown,
        # PIC-ARC additions
        "trust_radius_init": args.trust_radius_init,
        "sigma_init": args.sigma_init,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "kappa_threshold": args.kappa_threshold,
        "n_neg_max_for_arc": args.n_neg_max_for_arc,
        "force_max_for_arc": args.force_max_for_arc,
        "stability_window": args.stability_window,
        "max_consecutive_arc_rejects": args.max_consecutive_arc_rejects,
        "relaxed_eval_threshold": args.relaxed_eval_threshold,
        "accept_relaxed": args.accept_relaxed,
        "k_bond": args.k_bond,
        "bond_threshold_factor": args.bond_threshold_factor,
        "metric_regularization": args.metric_regularization,
        "metric_refresh_every": args.metric_refresh_every,
        "max_neg_modes_in_subspace": args.max_neg_modes_in_subspace,
    }

    processor = ParallelSCINEProcessor(
        functional=args.scine_functional,
        threads_per_worker=args.threads_per_worker,
        n_workers=args.n_workers,
        worker_fn=scine_worker_sample,
    )
    processor.start()

    try:
        dataloader = create_dataloader(args.h5_path, args.split, args.max_samples)
        metrics = run_batch(
            processor,
            dataloader,
            params,
            args.n_steps,
            args.max_samples,
            args.start_from,
            args.noise_seed,
        )

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        results_path = Path(args.out_dir) / f"minimization_{args.method}_{job_id}_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "method": args.method,
                    "params": params,
                    "config": {
                        "n_steps": args.n_steps,
                        "max_samples": args.max_samples,
                        "start_from": args.start_from,
                        "noise_seed": args.noise_seed,
                        "scine_functional": args.scine_functional,
                        "n_workers": args.n_workers,
                        "threads_per_worker": args.threads_per_worker,
                        "split": args.split,
                    },
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        print(f"Results saved to: {results_path}")

        # Print cascade table to stdout for quick inspection
        ct = metrics.get("cascade_table", {})
        print("\n--- Cascade Evaluation Table ---")
        print(f"{'eval_threshold':<20} {'n_converged':<15} {'rate':<10}")
        print("-" * 46)
        for thr in CASCADE_THRESHOLDS:
            key = str(thr)
            n_conv = ct.get("n_converged_at_thr", {}).get(key, "?")
            rate = ct.get("rate_at_thr", {}).get(key, float("nan"))
            rate_str = f"{rate:.3f}" if isinstance(rate, float) else str(rate)
            print(f"  {thr:<18} {n_conv:<15} {rate_str}")
        print("")

    finally:
        processor.close()


if __name__ == "__main__":
    main()
