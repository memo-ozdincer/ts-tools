#!/usr/bin/env python
"""Parallel SCINE runner for plain GAD baselines (no kicking).

Baselines:
- plain: no mode tracking (always lowest eigenvector)
- mode_tracked: track v1 across steps
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.core_algos.gad import pick_tracked_mode
from src.dependencies.common_utils import Transition1xDataset, UsePos, parse_starting_geometry
from src.dependencies.differentiable_projection import (
    gad_dynamics_projected_torch,
    gad_dynamics_reduced_basis_torch,
    eckart_frame_align_torch,
    get_mass_weights_torch,
)
from src.dependencies.hessian import get_scine_elements_from_predict_output, prepare_hessian
from src.noisy.multi_mode_eckartmw import get_projected_hessian, get_vib_evals_evecs, _min_interatomic_distance, _atomic_nums_to_symbols
from src.noisy.v2_tests.logging import TrajectoryLogger
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



def _cap_displacement(step_disp: torch.Tensor, max_atom_disp: float) -> torch.Tensor:
    disp_3d = step_disp.reshape(-1, 3)
    max_disp = float(disp_3d.norm(dim=1).max().item())
    if max_disp > max_atom_disp and max_disp > 0:
        disp_3d = disp_3d * (max_atom_disp / max_disp)
    return disp_3d.reshape(step_disp.shape)

def run_gad_baseline(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    n_steps: int,
    dt: float,
    dt_control: str,
    dt_min: float,
    dt_max: float,
    max_atom_disp: float,
    ts_eps: float,
    stop_at_ts: bool,
    min_interatomic_dist: float,
    tr_threshold: float,
    track_mode: bool,
    project_gradient_and_v: bool,
    log_dir: Optional[str],
    sample_id: str,
    formula: str,
    projection_mode: str = "eckart_full",
    purify_hessian: bool = False,
    frame_tracking: bool = False,
) -> Dict[str, Any]:
    coords = coords0.detach().clone().to(torch.float32)
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)

    v_prev = None
    dt_eff = float(dt)
    logger = TrajectoryLogger(sample_id=sample_id, formula=formula) if log_dir else None
    disp_history: List[float] = []
    prev_pos = coords.clone()
    prev_energy: Optional[float] = None

    # Always compute atom symbols: needed for reduced-basis eigendecomposition on
    # every step, regardless of projection_mode or project_gradient_and_v.
    atomsymbols = _atomic_nums_to_symbols(atomic_nums)

    # Frame tracking reference
    ref_coords = coords.clone() if frame_tracking else None

    for step in range(n_steps):
        # Frame tracking: align to reference geometry
        if frame_tracking and ref_coords is not None and atomsymbols is not None:
            masses_ft, _, _, _ = get_mass_weights_torch(atomsymbols, device=coords.device)
            coords, _, _ = eckart_frame_align_torch(coords, ref_coords, masses_ft)

        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        forces = out["forces"]
        hessian = out["hessian"]

        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)
        num_atoms = int(forces.shape[0])

        scine_elements = get_scine_elements_from_predict_output(out)

        # ---- Reduced-basis path (Solution A) ----
        if projection_mode == "reduced_basis" and scine_elements is None and atomsymbols is not None:
            gad_vec, v_next, rb_info = gad_dynamics_reduced_basis_torch(
                coords=coords,
                forces=forces,
                hessian=hessian,
                atomsymbols=atomsymbols,
                v_prev_full=v_prev if track_mode else None,
                purify=purify_hessian,
                k_track=8,
            )
            evals_vib_0 = rb_info["eig0"]
            evals_vib_1 = rb_info["eig1"]
            eig_product = rb_info["eig_product"]
            neg_vib_count = rb_info["neg_vib"]
            mode_index = rb_info["mode_index"]
            if track_mode:
                v_prev = v_next.detach().clone().reshape(-1)
            else:
                v_prev = None

        # ---- Original eckart_full path ----
        else:
            # Projected Hessian (3N×3N) is still needed for the trust-radius
            # quadratic-model prediction (dE_pred = g·dx + ½ dx^T H dx).
            hess_proj = get_projected_hessian(hessian, coords, atomic_nums, scine_elements,
                                              purify_hessian=purify_hessian)
            if hess_proj.dim() != 2 or hess_proj.shape[0] != 3 * num_atoms:
                hess_proj = prepare_hessian(hess_proj, num_atoms)

            # Vibrational eigenvalues via reduced basis — exactly 3N-k values,
            # no threshold, no silent exclusion of soft or near-zero modes.
            evals_vib, evecs_vib_3N, _Q_vib = get_vib_evals_evecs(
                hessian, coords, atomsymbols, purify_hessian=purify_hessian,
            )
            evals_vib = evals_vib.to(device=forces.device, dtype=forces.dtype)
            evecs_vib_3N = evecs_vib_3N.to(device=forces.device, dtype=forces.dtype)

            n_candidates = min(8, int(evals_vib.numel()))
            V = evecs_vib_3N[:, :n_candidates]
            v_prev_local = v_prev.to(device=forces.device, dtype=forces.dtype).reshape(-1) if (track_mode and v_prev is not None) else None
            v_new, mode_index, _overlap = pick_tracked_mode(V, v_prev_local, k=int(V.shape[1]))
            v = v_new

            # Compute GAD direction with optional vector projection
            if project_gradient_and_v:
                gad_vec, v_proj, _proj_info = gad_dynamics_projected_torch(
                    coords=coords,
                    forces=forces,
                    v=v,
                    atomsymbols=atomsymbols,
                )
                v = v_proj.reshape(-1)
                gad_flat = gad_vec.reshape(-1)
            else:
                f_flat = forces.reshape(-1)
                gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
                gad_vec = gad_flat.view(num_atoms, 3)

            if track_mode:
                v_prev = v.detach().clone().reshape(-1)
            else:
                v_prev = None

            if int(evals_vib.numel()) >= 2:
                evals_vib_0 = float(evals_vib[0].item())
                evals_vib_1 = float(evals_vib[1].item())
                eig_product = evals_vib_0 * evals_vib_1
            else:
                eig_product = float("inf")
                evals_vib_0 = float("nan")
                evals_vib_1 = float("nan")
            neg_vib_count = int((evals_vib < 0.0).sum().item()) if int(evals_vib.numel()) > 0 else -1

        disp_from_last = float((coords - prev_pos).norm(dim=1).mean().item()) if step > 0 else 0.0
        if step > 0:
            disp_history.append(disp_from_last)
        x_disp_window = float(np.mean(disp_history[-10:])) if disp_history else float("nan")

        # Initialize trust radius on the very first step before logging
        if step == 0:
            dt_eff = max_atom_disp

        if logger is not None and projection_mode != "reduced_basis":
            logger.log_step(
                step=step,
                coords=coords,
                energy=float(out["energy"].detach().reshape(-1)[0].item()) if isinstance(out["energy"], torch.Tensor) else float(out["energy"]),
                forces=forces,
                hessian_proj=hess_proj,
                gad_vec=gad_vec,
                dt_eff=dt_eff,
                coords_prev=prev_pos if step > 0 else None,
                energy_prev=prev_energy,
                mode_index=mode_index,
                x_disp_window=x_disp_window,
                tr_threshold=tr_threshold,
                vib_evals=evals_vib,
                vib_evecs_full=evecs_vib_3N,
            )

        if stop_at_ts and np.isfinite(eig_product) and eig_product < -abs(ts_eps):
            final_morse_index = neg_vib_count
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

        # GAD step direction: normalized gad_flat (unit per-atom displacement).
        # We deliberately do NOT apply Newton preconditioning (1/|λ|) here.
        # Newton preconditioning is correct for NR minimization (all eigenvalues
        # positive near a minimum) but harmful for GAD: at Morse index > 1 the
        # second-most-negative mode λ_1 has |λ_1| < |λ_0|, so 1/|λ_1| > 1/|λ_0|,
        # making the un-flipped mode 1 dominate the step and driving the algorithm
        # uphill along BOTH modes simultaneously.  The trust radius controls magnitude;
        # the GAD force vector controls direction.
        gad_dir = gad_vec.reshape(-1, 3) if projection_mode != "reduced_basis" else gad_vec.clone()
        gad_dir_norm = float(gad_dir.norm().item())
        if gad_dir_norm > 1e-10:
            step_disp_raw = gad_dir / gad_dir_norm   # unit-vector; trust radius sets magnitude
        else:
            step_disp_raw = gad_dir

        # Trust region: scale by dt_eff, compare quadratic-model prediction against
        # actual energy change, and adapt the radius.
        accepted = False
        max_retries = 10
        retries = 0
        rho = 1.0
        current_energy = float(out["energy"].detach().reshape(-1)[0].item()) if isinstance(out["energy"], torch.Tensor) else float(out["energy"])

        while not accepted and retries < max_retries:
            radius_used_for_step = dt_eff
            capped_disp = _cap_displacement(step_disp_raw, radius_used_for_step)
            dx_flat = capped_disp.reshape(-1)

            # Quadratic model of the TRUE energy change (Taylor): dE ≈ ∇E·dx + ½ dx^T H dx
            # This is purely for quality-of-fit assessment; we do not require it to be negative.
            grad_true = -forces.reshape(-1)
            if projection_mode != "reduced_basis" and "hess_proj" in locals():
                pred_dE = float((grad_true.dot(dx_flat) + 0.5 * dx_flat.dot(hess_proj @ dx_flat)).item())
            else:
                pred_dE = float("nan")

            new_coords = coords + capped_disp

            if _min_interatomic_distance(new_coords) < min_interatomic_dist:
                dt_eff *= 0.5
                retries += 1
                continue

            out_new = predict_fn(new_coords, atomic_nums, do_hessian=False, require_grad=False)
            energy_new = float(out_new["energy"].detach().reshape(-1)[0].item()) if isinstance(out_new["energy"], torch.Tensor) else float(out_new["energy"])
            actual_dE = energy_new - current_energy

            # ρ = actual / predicted energy change — measures quadratic model quality.
            # For GAD the energy can go up OR down, so we only reject when the quadratic
            # model and the actual step strongly disagree in magnitude (|ρ| < 0.1),
            # implying the step is too large for the local approximation to hold.
            if not math.isfinite(pred_dE) or abs(pred_dE) < 1e-8:
                rho = 1.0
                accepted = True
            else:
                rho = actual_dE / pred_dE
                if abs(rho) > 0.1 or radius_used_for_step < 1e-3:
                    accepted = True
                else:
                    dt_eff *= 0.25
                    retries += 1

        # Adapt trust radius based on model quality
        if accepted:
            if abs(rho) > 0.75:
                dt_eff = min(dt_eff * 1.5, max_atom_disp)
            elif abs(rho) < 0.25:
                dt_eff = max(dt_eff * 0.5, 0.001)
        else:
            dt_eff = max(dt_eff * 0.5, 0.001)

        prev_pos = coords.clone()
        prev_energy = current_energy
        coords = new_coords.detach()

    result = {
        "converged": False,
        "converged_step": None,
        "final_morse_index": -1,
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
            dt_control=params["dt_control"],
            dt_min=params["dt_min"],
            dt_max=params["dt_max"],
            max_atom_disp=params["max_atom_disp"],
            ts_eps=params["ts_eps"],
            stop_at_ts=params["stop_at_ts"],
            min_interatomic_dist=params["min_interatomic_dist"],
            tr_threshold=params["tr_threshold"],
            track_mode=params["track_mode"],
            project_gradient_and_v=params["project_gradient_and_v"],
            log_dir=params.get("log_dir"),
            sample_id=sample_id,
            formula=formula,
            projection_mode=params.get("projection_mode", "eckart_full"),
            purify_hessian=params.get("purify_hessian", False),
            frame_tracking=params.get("frame_tracking", False),
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

    return {
        "n_samples": n_samples,
        "n_success": n_success,
        "n_errors": n_errors,
        "success_rate": n_success / max(n_samples, 1),
        "mean_steps_when_success": float(np.mean(steps_when_success)) if steps_when_success else float("nan"),
        "mean_wall_time": float(np.mean(wall_times)) if wall_times else float("nan"),
        "total_wall_time": float(sum(wall_times)),
        "neg_vib_counts": neg_vib_counts,
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
    parser.add_argument("--dt-control", type=str, default="adaptive", choices=["adaptive", "fixed"])
    parser.add_argument("--dt-min", type=float, default=1e-6)
    parser.add_argument("--dt-max", type=float, default=0.08)
    parser.add_argument("--max-atom-disp", type=float, default=1.3)
    parser.add_argument("--min-interatomic-dist", type=float, default=0.5)
    parser.add_argument("--ts-eps", type=float, default=1e-5)
    parser.add_argument("--tr-threshold", type=float, default=8e-3)

    parser.add_argument(
        "--project-gradient-and-v",
        action="store_true",
        default=False,
        help="Project gradient and guide vector to prevent TR leakage (recommended for stability)",
    )

    parser.add_argument("--stop-at-ts", action="store_true")
    parser.add_argument("--no-stop-at-ts", dest="stop_at_ts", action="store_false")
    parser.set_defaults(stop_at_ts=True)

    # Projection experiments
    parser.add_argument(
        "--projection-mode",
        type=str,
        default="eckart_full",
        choices=["eckart_full", "reduced_basis"],
        help="Hessian projection mode: 'eckart_full' (P H P) or 'reduced_basis' (QR complement)",
    )
    parser.add_argument(
        "--purify-hessian",
        action="store_true",
        default=False,
        help="Enforce translational sum rules on the Hessian before projection",
    )
    parser.add_argument(
        "--frame-tracking",
        action="store_true",
        default=False,
        help="Kabsch-align coordinates to reference frame each step to prevent rigid-body drift",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    diag_dir = Path(args.out_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    track_mode = args.baseline == "mode_tracked"

    params = {
        "dt": args.dt,
        "dt_control": args.dt_control,
        "dt_min": args.dt_min,
        "dt_max": args.dt_max,
        "max_atom_disp": args.max_atom_disp,
        "min_interatomic_dist": args.min_interatomic_dist,
        "ts_eps": args.ts_eps,
        "tr_threshold": args.tr_threshold,
        "track_mode": track_mode,
        "project_gradient_and_v": args.project_gradient_and_v,
        "stop_at_ts": args.stop_at_ts,
        "log_dir": str(diag_dir),
        "projection_mode": args.projection_mode,
        "purify_hessian": args.purify_hessian,
        "frame_tracking": args.frame_tracking,
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
                        "projection_mode": args.projection_mode,
                        "purify_hessian": args.purify_hessian,
                        "frame_tracking": args.frame_tracking,
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