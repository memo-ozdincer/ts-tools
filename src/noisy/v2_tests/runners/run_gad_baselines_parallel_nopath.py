#!/usr/bin/env python
"""Parallel SCINE runner for plain GAD baselines (no kicking).

Baselines:
- plain: no mode tracking (always lowest eigenvector)
- mode_tracked: track v1 across steps

Adaptive dt strategies (state-based only, no path information):

Basic strategies (these don't work well in practice):
- none: fixed timestep
- gradient: dt_eff = clamp(dt_base * scale_factor / (grad_norm + eps), dt_min, dt_max)
- eigenvalue: dt_eff = clamp(dt_base * scale_factor / (|eig_0| + eps), dt_min, dt_max)
- eigenvalue_clamped: dt_eff = clamp(dt_base * scale_factor / clamp(|eig_0|, 1e-2, 1e2), dt_min, dt_max)
- trust_region: dt_eff = clamp(dt_scale_factor * max_atom_disp / (max(gad_vec_norm) + eps), dt_min, dt_max)

Advanced strategies (better alternatives):
- harmonic: Natural timescale from inverse square root of curvature: dt ~ 1/sqrt(|λ₀|)
           Mimics harmonic oscillator period, gives physically meaningful step sizes.
- spectral_gap: Use eigenvalue gap (λ₁ - λ₀) as stability measure.
           Larger gap = more distinct saddle direction = larger safe steps.
- gad_angle: Use angle between force and GAD vector.
           cos(θ) ≈ 1: near minimum, take larger steps
           cos(θ) ≈ -1: near maximum along v, take careful steps
           cos(θ) ≈ 0: balanced saddle region, moderate steps
- composite: Weighted combination of curvature and GAD vector magnitude.
           Balances local geometry info with step magnitude directly.
- force_gad_ratio: Ratio of |F| to |GAD| as a geometry indicator.
           Near TS, |F| → 0 but |GAD| may stay nonzero (if eigenvalue significant).
           This ratio naturally decreases near saddles.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
import math

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.core_algos.gad import pick_tracked_mode
from src.dependencies.common_utils import Transition1xDataset, UsePos, parse_starting_geometry
from src.dependencies.differentiable_projection import gad_dynamics_projected_torch
from src.dependencies.hessian import get_scine_elements_from_predict_output, prepare_hessian
from src.noisy.multi_mode_eckartmw import get_projected_hessian, _min_interatomic_distance, _atomic_nums_to_symbols
from src.noisy.v2_tests.logging.trajectory_logger_nopath import TrajectoryLogger
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


def _vib_mask_from_evals(evals: torch.Tensor, tr_threshold: float) -> torch.Tensor:
    return evals.abs() > float(tr_threshold)


def compute_state_based_dt(
    dt_base: float,
    dt_min: float,
    dt_max: float,
    dt_adaptation: str,
    grad_norm: float,
    eig_0: float,
    gad_vec_norm_max: float = 1.0,
    max_atom_disp: float = 0.1,
    dt_scale_factor: float = 1.0,
    eps: float = 1e-8,
    # Advanced strategy inputs
    eig_1: float = 0.0,
    force_dot_gad: float = 0.0,
    force_norm_sq: float = 0.0,
    gad_norm_sq: float = 0.0,
) -> float:
    """Compute adaptive dt using only state-based information (no path history).

    Args:
        dt_base: Base timestep
        dt_min: Minimum allowed timestep
        dt_max: Maximum allowed timestep
        dt_adaptation: Adaptation method (see module docstring for list)
        grad_norm: Current gradient norm (||-∇E|| = ||F||)
        eig_0: Lowest vibrational eigenvalue
        gad_vec_norm_max: Max norm of GAD vector on any atom
        max_atom_disp: Maximum allowed atomic displacement
        dt_scale_factor: Scaling factor for adaptation
        eps: Small constant to prevent division by zero
        eig_1: Second lowest vibrational eigenvalue (for spectral_gap)
        force_dot_gad: Dot product of force and GAD vectors (for gad_angle)
        force_norm_sq: Squared norm of force vector (for gad_angle)
        gad_norm_sq: Squared norm of GAD vector (for gad_angle)

    Returns:
        Effective timestep
    """
    if dt_adaptation == "none":
        return dt_base

    # =========================================================================
    # Basic strategies (existing, but don't work well)
    # =========================================================================
    if dt_adaptation == "gradient":
        # Smaller steps when gradient is large
        dt_eff = dt_base * dt_scale_factor / (grad_norm + eps)

    elif dt_adaptation == "eigenvalue":
        # Smaller steps when curvature (|λ₀|) is large
        dt_eff = dt_base * dt_scale_factor / (abs(eig_0) + eps)

    elif dt_adaptation == "eigenvalue_clamped":
        # Same as 'eigenvalue', but with a hard clamp on |λ₀| to prevent extreme dt.
        # Requested form: dt ~ 1 / clamp(lambda, 1e-2, 1e2)
        lam = abs(eig_0)
        lam_clamped = min(max(lam, 1e-2), 1e2)
        dt_eff = dt_base * dt_scale_factor / (lam_clamped + eps)

    elif dt_adaptation == "trust_region":
        # Limit step size such that max atomic displacement is approximately max_atom_disp
        dt_eff = dt_scale_factor * max_atom_disp / (gad_vec_norm_max + eps)

    # =========================================================================
    # Advanced strategies (new, potentially better)
    # =========================================================================
    elif dt_adaptation == "harmonic":
        # Use natural timescale from harmonic oscillator: τ ~ 1/ω ~ 1/sqrt(k/m) ~ 1/sqrt(|λ|)
        # For transition state search, the relevant "stiffness" is |λ₀|
        # This gives physically meaningful step sizes that adapt to local curvature
        # When |λ₀| is small (flat region), take larger steps
        # When |λ₀| is large (curved region), take smaller steps
        omega = math.sqrt(abs(eig_0) + eps)  # Effective angular frequency
        dt_eff = dt_base * dt_scale_factor / omega

    elif dt_adaptation == "spectral_gap":
        # Use the eigenvalue gap (λ₁ - λ₀) as a stability indicator
        # Larger gap = clearer separation between TS mode and others = safer to take larger steps
        # Small gap = potential mode crossing = be careful, use smaller steps
        # This captures mode stability without needing path history
        gap = eig_1 - eig_0  # Always positive since eig_0 is lowest
        # Normalize gap by typical eigenvalue scale
        gap_scale = gap / (abs(eig_0) + abs(eig_1) + eps)
        # Larger gap_scale → larger dt, but tempered by sqrt to avoid extremes
        dt_eff = dt_base * dt_scale_factor * math.sqrt(gap_scale + 0.1)

    elif dt_adaptation == "gad_angle":
        # Use the angle between force F and GAD vector as geometry indicator
        # GAD = F + 2(F·v)v, so the relationship between F and GAD tells us about
        # the local geometry:
        # - cos(θ) ≈ 1: Force and GAD aligned, minimum-like region, can step larger
        # - cos(θ) ≈ -1: Force and GAD opposite, near maximum along v direction
        # - cos(θ) ≈ 0: Balanced saddle region, intermediate steps
        #
        # Near TS: F → 0, so both norms vanish; this naturally gives moderate dt
        if force_norm_sq < eps or gad_norm_sq < eps:
            # Very small force = probably near critical point, use base dt
            cos_theta = 0.0
        else:
            cos_theta = force_dot_gad / (math.sqrt(force_norm_sq * gad_norm_sq) + eps)

        # Map cos(theta) to step size multiplier:
        # cos = 1 (aligned): multiplier ~ 1.5 (larger steps, minimum-like)
        # cos = 0 (orthogonal): multiplier ~ 1.0 (balanced)
        # cos = -1 (opposite): multiplier ~ 0.7 (careful steps)
        multiplier = 1.0 + 0.3 * cos_theta  # Range: [0.7, 1.3]
        dt_eff = dt_base * dt_scale_factor * multiplier

    elif dt_adaptation == "composite":
        # Weighted combination of multiple signals:
        # 1. Curvature contribution (harmonic-like)
        # 2. GAD magnitude contribution (trust region-like)
        # This balances "what the PES looks like" with "how big the step will be"
        omega = math.sqrt(abs(eig_0) + eps)
        curvature_factor = 1.0 / omega

        # GAD magnitude factor: smaller when GAD is large
        gad_factor = max_atom_disp / (gad_vec_norm_max + eps)

        # Geometric mean gives balanced contribution from both
        dt_eff = dt_base * dt_scale_factor * math.sqrt(curvature_factor * gad_factor)

    elif dt_adaptation == "force_gad_ratio":
        # Ratio of |F| to |GAD| as a geometry indicator
        # At minimum: F = 0, GAD = 0 → ratio undefined, use base
        # At saddle: F small, GAD can be nonzero if λ₀ significant → ratio small
        # Far from critical: F large, GAD ≈ F → ratio ≈ 1
        #
        # Use this ratio to slow down near saddles where GAD > F
        force_norm = math.sqrt(force_norm_sq + eps)
        gad_norm = math.sqrt(gad_norm_sq + eps)
        ratio = force_norm / (gad_norm + eps)

        # ratio < 1: GAD > F (saddle-like) → smaller steps
        # ratio ≈ 1: GAD ≈ F → base steps
        # ratio > 1: GAD < F (shouldn't happen often) → larger steps
        # Cap the multiplier to avoid extremes
        multiplier = min(max(ratio, 0.3), 2.0)
        dt_eff = dt_base * dt_scale_factor * multiplier

    else:
        dt_eff = dt_base

    return float(np.clip(dt_eff, dt_min, dt_max))


def run_gad_baseline(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    n_steps: int,
    dt: float,
    dt_adaptation: str,
    dt_min: float,
    dt_max: float,
    dt_scale_factor: float,
    max_atom_disp: float,
    ts_eps: float,
    stop_at_ts: bool,
    min_interatomic_dist: float,
    tr_threshold: float,
    track_mode: bool,
    project_gradient_and_v: bool,
    hessian_projection: str,
    log_dir: Optional[str],
    sample_id: str,
    formula: str,
) -> Dict[str, Any]:
    """Run GAD baseline with state-based adaptive dt (no path information).

    The adaptive dt uses only current-state information:
    - 'none': fixed timestep
    - 'gradient': scale by 1/grad_norm
    - 'eigenvalue': scale by 1/|λ₀|
    """
    coords = coords0.detach().clone().to(torch.float32)
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)

    v_prev = None
    logger = TrajectoryLogger(sample_id=sample_id, formula=formula) if log_dir else None
    last_evals_vib: Optional[torch.Tensor] = None

    for step in range(n_steps):
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        forces = out["forces"]
        hessian = out["hessian"]

        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)
        num_atoms = int(forces.shape[0])

        if hessian_projection == "none":
            hess_proj = prepare_hessian(hessian, num_atoms)
        else:
            scine_elements = get_scine_elements_from_predict_output(out)
            hess_proj = get_projected_hessian(hessian, coords, atomic_nums, scine_elements)
            if hess_proj.dim() != 2 or hess_proj.shape[0] != 3 * num_atoms:
                hess_proj = prepare_hessian(hess_proj, num_atoms)

        evals, evecs = torch.linalg.eigh(hess_proj)
        vib_mask = _vib_mask_from_evals(evals, tr_threshold)
        vib_indices = torch.where(vib_mask)[0]

        if int(vib_indices.numel()) == 0:
            evals_vib = evals
            candidate_indices = torch.arange(min(8, evecs.shape[1]), device=evecs.device)
        else:
            evals_vib = evals[vib_mask]
            candidate_indices = vib_indices[: min(8, int(vib_indices.numel()))]

        last_evals_vib = evals_vib.detach()

        V = evecs[:, candidate_indices].to(device=forces.device, dtype=forces.dtype)
        v_prev_local = v_prev.to(device=forces.device, dtype=forces.dtype).reshape(-1) if (track_mode and v_prev is not None) else None
        v_new, mode_index, _overlap = pick_tracked_mode(V, v_prev_local, k=int(V.shape[1]))
        v = v_new

        # Compute GAD direction with optional vector projection
        f_flat = forces.reshape(-1)
        if project_gradient_and_v:
            atomsymbols = _atomic_nums_to_symbols(atomic_nums)
            gad_vec, v_proj, _proj_info = gad_dynamics_projected_torch(
                coords=coords,
                forces=forces,
                v=v,
                atomsymbols=atomsymbols,
            )
            v = v_proj.reshape(-1)  # Update v with projected version
        else:
            gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
            gad_vec = gad_flat.view(num_atoms, 3)

        if track_mode:
            v_prev = v.detach().clone().reshape(-1)
        else:
            v_prev = None

        # State-based quantities for adaptive dt and convergence check
        grad_norm = float(f_flat.norm().item())
        eig_0 = float(evals_vib[0].item()) if int(evals_vib.numel()) > 0 else 0.0
        gad_vec_norm_max = float(gad_vec.norm(dim=1).max().item())

        if int(evals_vib.numel()) >= 2:
            eig_1 = float(evals_vib[1].item())
            eig_product = eig_0 * eig_1
        else:
            eig_1 = 0.0
            eig_product = float("inf")

        # Additional quantities for advanced dt strategies
        gad_flat = gad_vec.reshape(-1)
        force_dot_gad = float(torch.dot(f_flat, gad_flat).item())
        force_norm_sq = float(torch.dot(f_flat, f_flat).item())
        gad_norm_sq = float(torch.dot(gad_flat, gad_flat).item())

        # Compute state-based adaptive dt (no path history)
        dt_eff = compute_state_based_dt(
            dt_base=dt,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_adaptation=dt_adaptation,
            grad_norm=grad_norm,
            eig_0=eig_0,
            gad_vec_norm_max=gad_vec_norm_max,
            max_atom_disp=max_atom_disp,
            dt_scale_factor=dt_scale_factor,
            # Advanced strategy inputs
            eig_1=eig_1,
            force_dot_gad=force_dot_gad,
            force_norm_sq=force_norm_sq,
            gad_norm_sq=gad_norm_sq,
        )

        if logger is not None:
            energy = float(out["energy"].detach().reshape(-1)[0].item()) if isinstance(out["energy"], torch.Tensor) else float(out["energy"])
            logger.log_step(
                step=step,
                coords=coords,
                energy=energy,
                forces=forces,
                hessian_proj=hess_proj,
                gad_vec=gad_vec,
                dt_eff=dt_eff,
                mode_index=mode_index,
            )

        if stop_at_ts and np.isfinite(eig_product) and eig_product < -abs(ts_eps):
            final_morse_index = int((evals_vib < -float(tr_threshold)).sum().item()) if int(evals_vib.numel()) > 0 else -1
            result = {
                "converged": True,
                "converged_step": step,
                "final_morse_index": final_morse_index,
                "total_steps": step + 1,
            }
            if logger is not None:
                logger.finalize(
                    final_coords=coords,
                    final_morse_index=final_morse_index,
                    converged_to_ts=True,
                )
                logger.save(log_dir)
            return result

        # Take step with displacement capping (state-based, no dt history update)
        step_disp = dt_eff * gad_vec
        max_disp = float(step_disp.norm(dim=1).max().item())
        if max_disp > max_atom_disp and max_disp > 0:
            scale = max_atom_disp / max_disp
            step_disp = scale * step_disp

        new_coords = coords + step_disp
        if _min_interatomic_distance(new_coords) < min_interatomic_dist:
            step_disp = step_disp * 0.5
            new_coords = coords + step_disp

        coords = new_coords.detach()

    if last_evals_vib is not None and int(last_evals_vib.numel()) > 0:
        final_morse_index = int((last_evals_vib < -float(tr_threshold)).sum().item())
    else:
        final_morse_index = -1

    result = {
        "converged": False,
        "converged_step": None,
        "final_morse_index": final_morse_index,
        "total_steps": n_steps,
    }
    if logger is not None:
        logger.finalize(
            final_coords=coords,
            final_morse_index=result["final_morse_index"],
            converged_to_ts=bool(result.get("converged")),
        )
        logger.save(log_dir)
    return result


def run_single_sample(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    params: Dict[str, Any],
    n_steps: int,
    *,
    sample_id: str,
    formula: str,
) -> Dict[str, Any]:
    t0 = time.time()
    try:
        result = run_gad_baseline(
            predict_fn,
            coords,
            atomic_nums,
            n_steps=n_steps,
            dt=params["dt"],
            dt_adaptation=params["dt_adaptation"],
            dt_min=params["dt_min"],
            dt_max=params["dt_max"],
            dt_scale_factor=params["dt_scale_factor"],
            max_atom_disp=params["max_atom_disp"],
            ts_eps=params["ts_eps"],
            stop_at_ts=params["stop_at_ts"],
            min_interatomic_dist=params["min_interatomic_dist"],
            tr_threshold=params["tr_threshold"],
            track_mode=params["track_mode"],
            project_gradient_and_v=params["project_gradient_and_v"],
            hessian_projection=params["hessian_projection"],
            log_dir=params.get("log_dir"),
            sample_id=sample_id,
            formula=formula,
        )
        wall_time = time.time() - t0
        return {
            "final_neg_vib": result.get("final_morse_index", -1),
            "steps_taken": result.get("total_steps", n_steps),
            "steps_to_ts": result.get("converged_step"),
            "success": bool(result.get("converged")),
            "wall_time": wall_time,
            "error": None,
        }
    except Exception as e:
        wall_time = time.time() - t0
        return {
            "final_neg_vib": -1,
            "steps_taken": 0,
            "steps_to_ts": None,
            "success": False,
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
    formula = getattr(batch, "formula", "")
    result = run_single_sample(
        predict_fn,
        start_coords,
        atomic_nums,
        params,
        n_steps,
        sample_id=f"sample_{sample_idx:03d}",
        formula=str(formula),
    )
    result["sample_idx"] = sample_idx
    result["formula"] = getattr(batch, "formula", "")
    return result


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
    n_success = sum(1 for r in results if r.get("success"))
    n_errors = sum(1 for r in results if r.get("error") is not None)

    steps_when_success = [r["steps_to_ts"] for r in results if r.get("steps_to_ts") is not None]
    wall_times = [r["wall_time"] for r in results]

    final_neg_vibs = [r["final_neg_vib"] for r in results if r.get("error") is None]
    neg_vib_counts: Dict[int, int] = {}
    for v in final_neg_vibs:
        neg_vib_counts[v] = neg_vib_counts.get(v, 0) + 1

    nonconverged_neg_vibs = [
        r["final_neg_vib"]
        for r in results
        if (r.get("error") is None and not r.get("success"))
    ]
    nonconverged_neg_vib_counts: Dict[int, int] = {}
    for v in nonconverged_neg_vibs:
        nonconverged_neg_vib_counts[v] = nonconverged_neg_vib_counts.get(v, 0) + 1

    return {
        "n_samples": n_samples,
        "n_success": n_success,
        "n_errors": n_errors,
        "success_rate": n_success / max(n_samples, 1),
        "mean_steps_when_success": float(np.mean(steps_when_success)) if steps_when_success else float("nan"),
        "mean_wall_time": float(np.mean(wall_times)) if wall_times else float("nan"),
        "total_wall_time": float(sum(wall_times)),
        "neg_vib_counts": neg_vib_counts,
        "nonconverged_neg_vib_counts": nonconverged_neg_vib_counts,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plain GAD baselines (parallel, SCINE)")
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

    parser.add_argument("--baseline", type=str, default="plain", choices=["plain", "mode_tracked"])

    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument(
        "--dt-adaptation",
        type=str,
        default="composite",
        choices=[
            "none",  # Fixed timestep
            # Basic (don't work well):
            "gradient",  # dt ~ 1/grad_norm
            "eigenvalue",  # dt ~ 1/|λ₀|
            "eigenvalue_clamped",  # dt ~ 1/clamp(|λ₀|, 1e-2, 1e2)
            "trust_region",  # dt = scale * max_disp / |gad_vec|
            # Advanced (recommended):
            "harmonic",  # dt ~ 1/sqrt(|λ₀|), natural timescale from curvature
            "spectral_gap",  # dt ~ sqrt(λ₁ - λ₀), eigenvalue gap as stability indicator
            "gad_angle",  # Uses angle between F and GAD as geometry indicator
            "composite",  # Geometric mean of harmonic + trust_region
            "force_gad_ratio",  # dt ~ |F|/|GAD|, ratio indicates saddle proximity
        ],
        help="State-based adaptive dt method. Advanced strategies (harmonic, spectral_gap, "
             "gad_angle, composite, force_gad_ratio) are recommended over basic ones.",
    )
    parser.add_argument("--dt-min", type=float, default=1e-6)
    parser.add_argument("--dt-max", type=float, default=0.08)
    parser.add_argument(
        "--dt-scale-factor",
        type=float,
        default=1.0,
        help="Scaling factor for adaptive dt (dt_eff = dt_base * scale_factor / denominator)",
    )
    parser.add_argument("--max-atom-disp", type=float, default=0.35)
    parser.add_argument("--min-interatomic-dist", type=float, default=0.5)
    parser.add_argument("--ts-eps", type=float, default=1e-5)
    parser.add_argument("--tr-threshold", type=float, default=1e-6)

    parser.add_argument(
        "--project-gradient-and-v",
        action="store_true",
        default=False,
        help="Project gradient and guide vector to prevent TR leakage (recommended for stability)",
    )

    parser.add_argument(
        "--hessian-projection",
        type=str,
        default="eckart_mw",
        choices=["eckart_mw", "none"],
        help="Hessian eigenvector projection mode: 'eckart_mw' (Eckart-projected, mass-weighted) or 'none' (raw).",
    )

    parser.add_argument("--stop-at-ts", action="store_true")
    parser.add_argument("--no-stop-at-ts", dest="stop_at_ts", action="store_false")
    parser.set_defaults(stop_at_ts=True)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    diag_dir = Path(args.out_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    track_mode = args.baseline == "mode_tracked"

    params = {
        "dt": args.dt,
        "dt_adaptation": args.dt_adaptation,
        "dt_min": args.dt_min,
        "dt_max": args.dt_max,
        "dt_scale_factor": args.dt_scale_factor,
        "max_atom_disp": args.max_atom_disp,
        "min_interatomic_dist": args.min_interatomic_dist,
        "ts_eps": args.ts_eps,
        "tr_threshold": args.tr_threshold,
        "track_mode": track_mode,
        "project_gradient_and_v": args.project_gradient_and_v,
        "hessian_projection": args.hessian_projection,
        "stop_at_ts": args.stop_at_ts,
        "log_dir": str(diag_dir),
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
        results_path = Path(args.out_dir) / f"gad_{args.baseline}_parallel_{job_id}_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "baseline": args.baseline,
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
    finally:
        processor.close()


if __name__ == "__main__":
    main()
