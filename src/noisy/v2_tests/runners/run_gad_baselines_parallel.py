#!/usr/bin/env python
"""Parallel SCINE runner for plain GAD baselines (no kicking).

Baselines:
- plain: no mode tracking (always lowest eigenvector)
- mode_tracked: track v1 across steps

New in v2:
- Cascading evaluation: at every convergence check the runner records
  n_neg_at_<thr> for the SET of evaluation thresholds
  [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2].

  GAD convergence criterion: n_neg == 1 (Morse index 1 = true TS).
  The cascade asks  "if we accepted n_neg_at_threshold <= 1, would this
  sample count as converged?"  Values < 1 would mean we've overshot into
  a minimum; values > 1 mean we're still above index-1.

  Separately, the existing ts_eps criterion
  (eig_product = λ_0 * λ_1 < -ts_eps)
  is the actual algorithmic gate and is unchanged.

- Negative eigenvalue magnitude logging: at convergence (or failure) the
  runner records:
    lambda_0        — the most-negative eigenvalue (climbing mode)
    lambda_1        — the second eigenvalue (first of the positive ladder)
    abs_lambda_0    — |lambda_0| (magnitude of the climbing mode)
    lambda_gap_ratio — |lambda_0| / |lambda_1|  (how separated is the TS
                       mode from the noise floor?)
    bottom_spectrum — bottom-K sorted vibrational eigenvalues
  This directly tests the user's hypothesis: "is λ_0 meaningfully different
  from the rest of the eigenvalues that we call noise?"  A large gap_ratio
  means it is; a gap_ratio ≈ 1 means the TS mode is buried in noise and the
  convergence claim is suspect.

- --log-spectrum-k: number of bottom eigenvalues to record per step (default 10).
- cascade_table in results JSON: rows=tr_threshold, cols=eval_threshold → success_rate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from src.noisy.multi_mode_eckartmw import (
    get_projected_hessian,
    get_vib_evals_evecs,
    _min_interatomic_distance,
    _atomic_nums_to_symbols,
)
from src.noisy.v2_tests.logging import TrajectoryLogger
from src.parallel.scine_parallel import ParallelSCINEProcessor
from src.parallel.utils import run_batch_parallel


# ---------------------------------------------------------------------------
# Cascade evaluation thresholds
# ---------------------------------------------------------------------------
# These are applied to the NEGATIVE eigenvalue count at final geometry.
# GAD target is n_neg == 1 (Morse index 1 = transition state).
# For the cascade we ask: "for eval threshold T, n_neg_at_T <= 1?"
# (i.e. all eigenvalues > -T except possibly one).
CASCADE_THRESHOLDS: List[float] = [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2]


def _cascade_n_neg(evals_vib: torch.Tensor) -> Dict[str, int]:
    """Count eigenvalues < -threshold for each cascade threshold.

    n_neg_at_0.0 is the strict count (eigenvalues strictly < 0).
    GAD convergence at eval threshold T means n_neg_at_T <= 1.
    """
    result: Dict[str, int] = {}
    for thr in CASCADE_THRESHOLDS:
        result[f"n_neg_at_{thr}"] = int((evals_vib < -thr).sum().item())
    return result


def _neg_eval_info(evals_vib: torch.Tensor, label: str = "") -> Dict[str, float]:
    """Extract magnitude info about the most-negative eigenvalue(s).

    Returns (with optional label prefix for distinguishing filtered vs unfiltered):
      lambda_0        — most negative eigenvalue (the TS climbing mode)
      lambda_1        — second eigenvalue (first of the positive ladder, or
                        the second-most-negative if Morse index > 1)
      abs_lambda_0    — |lambda_0|
      lambda_gap_ratio — |lambda_0| / max(|lambda_1|, 1e-10)
                         Large ratio → TS mode well-separated from noise floor.
                         Ratio ≈ 1   → TS mode buried in noise.

    NOTE: lambda_0 and lambda_1 are the TRUE two smallest eigenvalues of the
    passed tensor — no thresholding is applied here. Pass a pre-filtered tensor
    to get filtered results, and the raw tensor for unfiltered results.
    """
    prefix = f"{label}_" if label else ""
    if evals_vib.numel() == 0:
        return {
            f"{prefix}lambda_0": float("nan"),
            f"{prefix}lambda_1": float("nan"),
            f"{prefix}abs_lambda_0": float("nan"),
            f"{prefix}lambda_gap_ratio": float("nan"),
        }
    sorted_evals, _ = torch.sort(evals_vib)
    lam0 = float(sorted_evals[0].item())
    lam1 = float(sorted_evals[1].item()) if sorted_evals.numel() > 1 else float("nan")
    abs_lam0 = abs(lam0)
    gap_ratio = abs_lam0 / max(abs(lam1), 1e-10) if math.isfinite(lam1) else float("nan")
    return {
        f"{prefix}lambda_0": lam0,
        f"{prefix}lambda_1": lam1,
        f"{prefix}abs_lambda_0": abs_lam0,
        f"{prefix}lambda_gap_ratio": gap_ratio,
    }


def _bottom_k_spectrum(evals_vib: torch.Tensor, k: int = 10) -> List[float]:
    """Return the k smallest vibrational eigenvalues as a sorted Python list."""
    vals = evals_vib.detach().cpu()
    sorted_vals, _ = torch.sort(vals)
    return [float(v) for v in sorted_vals[:k].tolist()]


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


def _vib_mask_from_evals(evals: torch.Tensor, tr_threshold: float) -> torch.Tensor:
    """Return a boolean mask keeping only eigenvalues with |λ| > tr_threshold.

    This matches the old implementation's filtering logic and excludes near-zero
    (soft/translation/rotation-leaked) modes from convergence decisions.
    """
    return evals.abs() > float(tr_threshold)


def _newton_gad_step(
    forces: torch.Tensor,
    evals_vib: torch.Tensor,
    evecs_vib_3N: torch.Tensor,
    tracked_mode_index: int,
    step_filter_threshold: float,
    atomsymbols: list,
    device: torch.device,
) -> torch.Tensor:
    """Compute Newton-preconditioned GAD step in mass-weighted vibrational eigenbasis.

    For the tracked TS mode (λ < 0):  standard Newton → drives toward saddle.
    For positive modes (λ > 0):       standard Newton → drives toward minimum.
    For non-tracked negative modes:   absolute-value pseudoinverse → gradient
                                      descent away from maximum.
    For near-zero modes (|λ| < thr):  filtered out (noise).

    Returns:
        step_cart: (N, 3) Cartesian displacement with meaningful Newton magnitude.
    """
    dtype = torch.float64
    _, _, sqrt_m, sqrt_m_inv = get_mass_weights_torch(atomsymbols, device=device, dtype=dtype)

    # Mass-weighted gradient: g_mw = M^{-1/2} * (-F)
    g_mw = sqrt_m_inv * (-forces.reshape(-1).to(dtype))

    evecs_f64 = evecs_vib_3N.to(dtype)
    evals_f64 = evals_vib.to(dtype)

    # Project gradient onto vibrational eigenmodes
    g_vib = evecs_f64.T @ g_mw  # (3N-k,)

    # Mode-decomposed Newton step with filtering
    step_vib = torch.zeros_like(g_vib)
    for i in range(len(evals_f64)):
        lam = float(evals_f64[i].item())
        if abs(lam) < step_filter_threshold:
            continue  # skip noise modes
        if i == tracked_mode_index:
            # Tracked TS mode: standard Newton toward saddle point
            # Since λ < 0, -g/λ = +g/|λ| → moves toward the maximum (saddle)
            step_vib[i] = -g_vib[i] / lam
        elif lam > 0:
            # Positive mode: standard Newton minimization
            step_vib[i] = -g_vib[i] / lam
        else:
            # Non-tracked negative mode: absolute-value pseudoinverse
            # = gradient descent away from maximum, scaled by curvature
            step_vib[i] = -g_vib[i] / abs(lam)

    # Reconstruct in MW space → Cartesian
    step_mw = evecs_f64 @ step_vib
    step_cart = (sqrt_m_inv * step_mw).to(forces.dtype)  # Δq_cart = M^{-1/2} × Δq_mw
    return step_cart.reshape(-1, 3)


def _cleanup_step_orthogonal(
    forces: torch.Tensor,
    evals_vib: torch.Tensor,
    evecs_vib_3N: torch.Tensor,
    tracked_mode_index: int,
    step_filter_threshold: float,
    atomsymbols: list,
    device: torch.device,
    max_disp: float = 0.1,
) -> torch.Tensor:
    """NR minimization step orthogonal to the TS mode (for post-convergence cleanup).

    Minimizes energy in the (3N-k-1)-dimensional subspace orthogonal to the
    tracked TS eigenvector.  This pushes residual small negative modes positive
    without disturbing the saddle geometry along the TS mode.

    Returns:
        step_cart: (N, 3) capped Cartesian displacement.
    """
    dtype = torch.float64
    _, _, sqrt_m, sqrt_m_inv = get_mass_weights_torch(atomsymbols, device=device, dtype=dtype)

    g_mw = sqrt_m_inv * (-forces.reshape(-1).to(dtype))
    evecs_f64 = evecs_vib_3N.to(dtype)
    evals_f64 = evals_vib.to(dtype)
    g_vib = evecs_f64.T @ g_mw

    step_vib = torch.zeros_like(g_vib)
    for i in range(len(evals_f64)):
        if i == tracked_mode_index:
            continue  # don't move along TS mode
        lam = float(evals_f64[i].item())
        if abs(lam) < step_filter_threshold:
            continue
        if lam > 0:
            step_vib[i] = -g_vib[i] / lam
        else:
            step_vib[i] = -g_vib[i] / abs(lam)

    step_mw = evecs_f64 @ step_vib
    step_cart = (sqrt_m_inv * step_mw).to(forces.dtype)
    return _cap_displacement(step_cart.reshape(-1, 3), max_disp)


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
    log_spectrum_k: int = 10,
    tr_filter_eig: bool = False,
    # ── v3 Newton-GAD improvements ──────────────────────────────
    step_mode: str = "first_order",
    step_filter_threshold: float = 8e-3,
    converge_on_filtered: bool = False,
    anti_overshoot: bool = False,
    cleanup_steps: int = 0,
) -> Dict[str, Any]:
    """Run one GAD trajectory.

    tr_filter_eig: if True, use the old tr_threshold masking for picking eig0/eig1
        and for the convergence gate.  Near-zero modes (|λ| < tr_threshold) are
        excluded before computing eig_product = evals_vib[0] * evals_vib[1], which
        prevents soft modes leaking through Eckart projection from suppressing the
        product below the -ts_eps gate.

        If False (new default), the raw vibrational eigenvalues from get_vib_evals_evecs
        are used unfiltered.  Small residual modes may cause eig_product ≈ 0 even at a
        true TS, which inflates the gap ratio and suppresses the strict success rate.

    v3 improvements (Newton-GAD):
        step_mode: "first_order" (classic unit-vector GAD) or "newton_gad"
            (mode-decomposed Newton step in vibrational eigenbasis).  Newton mode
            applies 1/|λ| preconditioning per eigenmode and filters noise modes,
            giving second-order convergence while preserving the GAD direction logic.

        step_filter_threshold: eigenvalue magnitude threshold for Newton step.
            Modes with |λ| < this are zeroed out in the step (noise filtering).
            Also used for the convergence gate when converge_on_filtered=True.

        converge_on_filtered: if True, the eig_product convergence gate uses
            eigenvalues filtered by step_filter_threshold (prevents overshooting
            past the TS due to near-zero λ₁ suppressing the product).  The cascade
            table always reports unfiltered n_neg for honest evaluation.

        anti_overshoot: if True, monitors the Morse index each step and reduces
            the trust radius when approaching Morse-1, preventing the optimizer
            from passing through the TS into a minimum.

        cleanup_steps: number of post-convergence NR minimization steps in the
            subspace orthogonal to the TS mode.  Pushes residual small negative
            eigenvalues positive without disturbing the saddle geometry.
    """
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

    # Anti-overshoot state
    prev_n_neg_soft: Optional[int] = None  # Morse index at T=0.002
    in_ts_region: bool = False

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
            # For cascade/spectrum we need the full eigenvalue tensor
            evals_vib_full = rb_info.get("evals_vib", None)
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

            # Vibrational eigenvalues via reduced basis — exactly 3N-k values.
            evals_vib, evecs_vib_3N, _Q_vib = get_vib_evals_evecs(
                hessian, coords, atomsymbols, purify_hessian=purify_hessian,
            )
            evals_vib = evals_vib.to(device=forces.device, dtype=forces.dtype)
            evecs_vib_3N = evecs_vib_3N.to(device=forces.device, dtype=forces.dtype)
            # Always keep the full unfiltered eigenvalue tensor for cascade/spectrum logging.
            evals_vib_full = evals_vib

            if tr_filter_eig:
                # Legacy mode: filter out near-zero modes before picking climbing mode
                # and computing eig_product (matches old implementation exactly).
                vib_mask = _vib_mask_from_evals(evals_vib, tr_threshold)
                vib_indices = torch.where(vib_mask)[0]
                if int(vib_indices.numel()) == 0:
                    evals_vib_conv = evals_vib
                    candidate_indices = torch.arange(min(8, evecs_vib_3N.shape[1]), device=evecs_vib_3N.device)
                else:
                    evals_vib_conv = evals_vib[vib_mask]
                    candidate_indices = vib_indices[:min(8, int(vib_indices.numel()))]
                V = evecs_vib_3N[:, candidate_indices]
            else:
                # New mode: raw vibrational eigenvalues, no threshold filtering.
                evals_vib_conv = evals_vib
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

            # eig_product uses the (possibly filtered) eigenvalues for the convergence gate.
            # converge_on_filtered: use step_filter_threshold to remove noise modes from
            # the eig_product check *independently* of the old tr_filter_eig pathway.
            if converge_on_filtered and not tr_filter_eig:
                _conv_mask = evals_vib.abs() > float(step_filter_threshold)
                _conv_idx = torch.where(_conv_mask)[0]
                evals_vib_conv = evals_vib[_conv_mask] if int(_conv_idx.numel()) >= 2 else evals_vib

            if int(evals_vib_conv.numel()) >= 2:
                evals_vib_0 = float(evals_vib_conv[0].item())
                evals_vib_1 = float(evals_vib_conv[1].item())
                eig_product = evals_vib_0 * evals_vib_1
            else:
                eig_product = float("inf")
                evals_vib_0 = float("nan")
                evals_vib_1 = float("nan")
            # Morse index always counts from the raw unfiltered spectrum.
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

        # Convergence check: eig_product < -ts_eps means λ_0 < 0 and λ_1 > 0,
        # i.e. we have exactly one negative mode (Morse index = 1 = transition state).
        if stop_at_ts and np.isfinite(eig_product) and eig_product < -abs(ts_eps):
            final_morse_index = neg_vib_count

            # --- Cascade evaluation and eigenvalue diagnostics at convergence ---
            ev_full = evals_vib_full
            cascade = _cascade_n_neg(ev_full) if ev_full is not None else {}
            # Unfiltered diagnostics (true two smallest eigenvalues)
            neg_info = _neg_eval_info(ev_full) if ev_full is not None else {}
            # Also log filtered diagnostics when tr_filter_eig is active
            if tr_filter_eig and ev_full is not None:
                ev_filtered = ev_full[_vib_mask_from_evals(ev_full, tr_threshold)]
                neg_info_filt = _neg_eval_info(ev_filtered, label="filt")
            else:
                neg_info_filt = {}
            spectrum = _bottom_k_spectrum(ev_full, log_spectrum_k) if ev_full is not None and log_spectrum_k > 0 else []

            # ── Post-convergence cleanup ──────────────────────────────
            # NR minimization orthogonal to TS mode to push residual small
            # negative eigenvalues positive.
            cleanup_done = 0
            if cleanup_steps > 0 and ev_full is not None:
                for _cs in range(cleanup_steps):
                    out_cl = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
                    f_cl = out_cl["forces"]
                    if f_cl.dim() == 3 and f_cl.shape[0] == 1:
                        f_cl = f_cl[0]
                    f_cl = f_cl.reshape(-1, 3)
                    h_cl = out_cl["hessian"]
                    ev_cl, evc_cl, _ = get_vib_evals_evecs(
                        h_cl, coords, atomsymbols, purify_hessian=purify_hessian,
                    )
                    ev_cl = ev_cl.to(device=forces.device, dtype=forces.dtype)
                    evc_cl = evc_cl.to(device=forces.device, dtype=forces.dtype)
                    cleanup_disp = _cleanup_step_orthogonal(
                        forces=f_cl,
                        evals_vib=ev_cl,
                        evecs_vib_3N=evc_cl,
                        tracked_mode_index=mode_index,
                        step_filter_threshold=step_filter_threshold,
                        atomsymbols=atomsymbols,
                        device=forces.device,
                        max_disp=0.05,
                    )
                    new_cl = coords + cleanup_disp
                    if _min_interatomic_distance(new_cl) >= min_interatomic_dist:
                        coords = new_cl.detach()
                    cleanup_done += 1
                # Re-evaluate eigenvalues after cleanup
                out_post = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
                h_post = out_post["hessian"]
                ev_post, _, _ = get_vib_evals_evecs(
                    h_post, coords, atomsymbols, purify_hessian=purify_hessian,
                )
                ev_post = ev_post.to(device=forces.device, dtype=forces.dtype)
                ev_full = ev_post
                cascade = _cascade_n_neg(ev_full)
                neg_info = _neg_eval_info(ev_full)
                spectrum = _bottom_k_spectrum(ev_full, log_spectrum_k) if log_spectrum_k > 0 else []
                final_morse_index = int((ev_full < 0.0).sum().item())
                neg_info_filt = {}

            result = {
                "converged": True,
                "converged_step": step,
                "final_morse_index": final_morse_index,
                "total_steps": step + 1,
                "cleanup_steps_taken": cleanup_done,
                "tr_filter_eig": tr_filter_eig,
                "step_mode": step_mode,
                # Cascade: n_neg at every eval threshold
                **cascade,
                # Unfiltered negative eigenvalue magnitude diagnostics
                **neg_info,
                # Filtered diagnostics (only present when tr_filter_eig=True)
                **neg_info_filt,
                "bottom_spectrum_at_convergence": spectrum,
                "eig_product_at_convergence": float(eig_product),
            }
            if logger is not None:
                logger.finalize(
                    final_coords=coords,
                    final_morse_index=final_morse_index,
                    converged_to_ts=True,
                )
                logger.save(log_dir)
            return result

        # ── Anti-overshoot mechanism ─────────────────────────────────────
        # Track Morse index at a soft threshold (T=0.002) to detect when we
        # enter the TS region.  Reduce trust radius to prevent passing through.
        if anti_overshoot and evals_vib_full is not None:
            current_n_neg_soft = int((evals_vib_full < -0.002).sum().item())
            if prev_n_neg_soft is not None:
                # Entering TS region: Morse index dropped to ≤1
                if prev_n_neg_soft > 1 and current_n_neg_soft <= 1:
                    dt_eff = min(dt_eff, max_atom_disp * 0.25)
                    in_ts_region = True
                # Overshot: went from 1 to 0
                elif prev_n_neg_soft >= 1 and current_n_neg_soft == 0 and in_ts_region:
                    dt_eff = min(dt_eff, max_atom_disp * 0.1)
            # Once in TS region, cap trust radius growth
            if in_ts_region:
                max_tr_for_step = max_atom_disp * 0.5
            else:
                max_tr_for_step = max_atom_disp
            prev_n_neg_soft = current_n_neg_soft
        else:
            max_tr_for_step = max_atom_disp

        # ── Step direction ─────────────────────────────────────────────────
        if step_mode == "newton_gad" and projection_mode != "reduced_basis":
            # Mode-decomposed Newton step: second-order convergence with
            # noise-mode filtering.  Handles non-tracked negative modes via
            # absolute-value pseudoinverse (gradient descent from maxima).
            step_disp_raw = _newton_gad_step(
                forces=forces,
                evals_vib=evals_vib,
                evecs_vib_3N=evecs_vib_3N,
                tracked_mode_index=mode_index,
                step_filter_threshold=step_filter_threshold,
                atomsymbols=atomsymbols,
                device=forces.device,
            )
        else:
            # First-order GAD: unit-normalized direction + trust radius magnitude.
            # (Original behaviour — no Newton preconditioning.)
            gad_dir = gad_vec.reshape(-1, 3) if projection_mode != "reduced_basis" else gad_vec.clone()
            gad_dir_norm = float(gad_dir.norm().item())
            if gad_dir_norm > 1e-10:
                step_disp_raw = gad_dir / gad_dir_norm
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
                dt_eff = min(dt_eff * 1.5, max_tr_for_step)
            elif abs(rho) < 0.25:
                dt_eff = max(dt_eff * 0.5, 0.001)
        else:
            dt_eff = max(dt_eff * 0.5, 0.001)

        prev_pos = coords.clone()
        prev_energy = current_energy
        coords = new_coords.detach()

    # --- Failure path: log cascade and spectrum at final geometry ---
    ev_full_final = evals_vib_full if "evals_vib_full" in locals() and evals_vib_full is not None else None
    cascade_final = _cascade_n_neg(ev_full_final) if ev_full_final is not None else {}
    # Unfiltered diagnostics at failure
    neg_info_final = _neg_eval_info(ev_full_final) if ev_full_final is not None else {}
    # Filtered diagnostics at failure when tr_filter_eig is active
    if tr_filter_eig and ev_full_final is not None:
        ev_filt_final = ev_full_final[_vib_mask_from_evals(ev_full_final, tr_threshold)]
        neg_info_filt_final = _neg_eval_info(ev_filt_final, label="filt")
    else:
        neg_info_filt_final = {}
    spectrum_final = _bottom_k_spectrum(ev_full_final, log_spectrum_k) if ev_full_final is not None and log_spectrum_k > 0 else []
    neg_vib_final = int((ev_full_final < 0.0).sum().item()) if ev_full_final is not None else -1

    result = {
        "converged": False,
        "converged_step": None,
        "final_morse_index": neg_vib_final,
        "total_steps": n_steps,
        "cleanup_steps_taken": 0,
        "tr_filter_eig": tr_filter_eig,
        "step_mode": step_mode,
        **cascade_final,
        **neg_info_final,
        **neg_info_filt_final,
        "bottom_spectrum_at_convergence": spectrum_final,
        "eig_product_at_convergence": float(eig_product) if "eig_product" in locals() and np.isfinite(eig_product) else float("nan"),
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
            log_spectrum_k=params.get("log_spectrum_k", 10),
            tr_filter_eig=params.get("tr_filter_eig", False),
            # v3 Newton-GAD improvements
            step_mode=params.get("step_mode", "first_order"),
            step_filter_threshold=params.get("step_filter_threshold", 8e-3),
            converge_on_filtered=params.get("converge_on_filtered", False),
            anti_overshoot=params.get("anti_overshoot", False),
            cleanup_steps=params.get("cleanup_steps", 0),
        )
        wall_time = time.time() - t0

        # Extract cascade fields from result
        cascade_fields = {k: v for k, v in result.items() if k.startswith("n_neg_at_")}
        # Unfiltered eigenvalue diagnostics (always present)
        neg_info_fields = {
            k: result.get(k)
            for k in ["lambda_0", "lambda_1", "abs_lambda_0", "lambda_gap_ratio"]
        }
        # Filtered eigenvalue diagnostics (only when tr_filter_eig=True)
        neg_info_filt_fields = {
            k: result.get(k)
            for k in ["filt_lambda_0", "filt_lambda_1", "filt_abs_lambda_0", "filt_lambda_gap_ratio"]
            if k in result
        }

        return {
            "final_neg_vib": result.get("final_morse_index", -1),
            "steps_taken": result.get("total_steps", n_steps),
            "steps_to_ts": result.get("converged_step"),
            "success": bool(result.get("converged")),
            "wall_time": wall_time,
            "error": None,
            "tr_filter_eig": result.get("tr_filter_eig", False),
            "cascade": cascade_fields,
            "neg_eval_info": neg_info_fields,
            "neg_eval_info_filt": neg_info_filt_fields,
            "bottom_spectrum_at_convergence": result.get("bottom_spectrum_at_convergence", []),
            "eig_product_at_convergence": result.get("eig_product_at_convergence", float("nan")),
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
            "tr_filter_eig": params.get("tr_filter_eig", False),
            "cascade": {},
            "neg_eval_info": {},
            "neg_eval_info_filt": {},
            "bottom_spectrum_at_convergence": [],
            "eig_product_at_convergence": float("nan"),
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


def _build_gad_cascade_table(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a 2D success-rate table over cascade evaluation thresholds.

    GAD success at eval_threshold T means  n_neg_at_T <= 1  (Morse index ≤ 1
    at the final geometry, evaluated as if any eigenvalue more positive than
    -T counts as zero).

    For each T in CASCADE_THRESHOLDS, count successes = samples where
    n_neg_at_T <= 1 (in practice 0 or 1 — a strict minimum would be 0,
    which shouldn't count as a TS success).

    Actually we split into two sub-counts:
      success_at_thr_eq1  — n_neg_at_T == 1  (exactly one mode below -T)
      success_at_thr_le1  — n_neg_at_T <= 1  (at most one mode below -T)

    The difference between these tells us how many samples ended up at
    a minimum (n_neg==0) rather than a TS (n_neg==1).
    """
    n = len(results)
    eq1: Dict[str, int] = {}
    le1: Dict[str, int] = {}
    rate_eq1: Dict[str, float] = {}
    rate_le1: Dict[str, float] = {}

    for thr in CASCADE_THRESHOLDS:
        key = f"n_neg_at_{thr}"
        count_eq1 = sum(
            1 for r in results
            if r.get("cascade", {}).get(key, -1) == 1
        )
        count_le1 = sum(
            1 for r in results
            if r.get("cascade", {}).get(key, -1) <= 1 and r.get("cascade", {}).get(key, -1) >= 0
        )
        eq1[str(thr)] = count_eq1
        le1[str(thr)] = count_le1
        rate_eq1[str(thr)] = count_eq1 / max(n, 1)
        rate_le1[str(thr)] = count_le1 / max(n, 1)

    # --- Eigenvalue diagnostics (unfiltered + filtered, success + failure) ---
    def _ni(sample_list: List[Dict[str, Any]], key: str, info_field: str = "neg_eval_info") -> List[float]:
        out = []
        for r in sample_list:
            ni = r.get(info_field, {})
            v = ni.get(key) if ni else None
            if v is not None and math.isfinite(float(v)):
                out.append(float(v))
        return out

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success") and not r.get("error")]

    # Unfiltered — success
    gap_succ   = _ni(successful, "lambda_gap_ratio")
    lam0_succ  = _ni(successful, "abs_lambda_0")
    lam1_succ  = [abs(v) for v in _ni(successful, "lambda_1")]
    # Unfiltered — failure
    gap_fail   = _ni(failed, "lambda_gap_ratio")
    lam0_fail  = _ni(failed, "abs_lambda_0")
    lam1_fail  = [abs(v) for v in _ni(failed, "lambda_1")]
    # Filtered — success
    fgap_succ  = _ni(successful, "filt_lambda_gap_ratio", "neg_eval_info_filt")
    flam0_succ = _ni(successful, "filt_abs_lambda_0", "neg_eval_info_filt")
    flam1_succ = [abs(v) for v in _ni(successful, "filt_lambda_1", "neg_eval_info_filt")]
    # Filtered — failure
    fgap_fail  = _ni(failed, "filt_lambda_gap_ratio", "neg_eval_info_filt")
    flam0_fail = _ni(failed, "filt_abs_lambda_0", "neg_eval_info_filt")
    flam1_fail = [abs(v) for v in _ni(failed, "filt_lambda_1", "neg_eval_info_filt")]

    return {
        "eval_thresholds": CASCADE_THRESHOLDS,
        "n_samples": n,
        "n_success_strict": sum(1 for r in results if r.get("success")),
        "n_neg_eq1_at_thr": eq1,
        "n_neg_le1_at_thr": le1,
        "rate_eq1_at_thr": rate_eq1,
        "rate_le1_at_thr": rate_le1,
        # Unfiltered (true two smallest eigenvalues) — at success
        "mean_gap_ratio_at_success":    float(np.mean(gap_succ))   if gap_succ   else float("nan"),
        "mean_abs_lambda0_at_success":  float(np.mean(lam0_succ))  if lam0_succ  else float("nan"),
        "mean_abs_lambda1_at_success":  float(np.mean(lam1_succ))  if lam1_succ  else float("nan"),
        # Unfiltered — at failure (diagnostic: where is the final geometry?)
        "mean_gap_ratio_at_failure":    float(np.mean(gap_fail))   if gap_fail   else float("nan"),
        "mean_abs_lambda0_at_failure":  float(np.mean(lam0_fail))  if lam0_fail  else float("nan"),
        "mean_abs_lambda1_at_failure":  float(np.mean(lam1_fail))  if lam1_fail  else float("nan"),
        # Filtered (tr_threshold mask applied) — at success
        "mean_filt_gap_ratio_at_success":   float(np.mean(fgap_succ))   if fgap_succ   else float("nan"),
        "mean_filt_abs_lambda0_at_success": float(np.mean(flam0_succ))  if flam0_succ  else float("nan"),
        "mean_filt_abs_lambda1_at_success": float(np.mean(flam1_succ))  if flam1_succ  else float("nan"),
        # Filtered — at failure
        "mean_filt_gap_ratio_at_failure":   float(np.mean(fgap_fail))   if fgap_fail   else float("nan"),
        "mean_filt_abs_lambda0_at_failure": float(np.mean(flam0_fail))  if flam0_fail  else float("nan"),
        "mean_filt_abs_lambda1_at_failure": float(np.mean(flam1_fail))  if flam1_fail  else float("nan"),
        "n_successful_with_gap_data": len(gap_succ),
        "n_failed_with_gap_data":     len(gap_fail),
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
    n_success = sum(1 for r in results if r.get("success"))
    n_errors = sum(1 for r in results if r.get("error") is not None)

    steps_when_success = [r["steps_to_ts"] for r in results if r.get("steps_to_ts") is not None]
    wall_times = [r["wall_time"] for r in results]

    final_neg_vibs = [r["final_neg_vib"] for r in results if r.get("error") is None]
    neg_vib_counts: Dict[int, int] = {}
    for v in final_neg_vibs:
        neg_vib_counts[v] = neg_vib_counts.get(v, 0) + 1

    cascade_table = _build_gad_cascade_table(results)

    return {
        "n_samples": n_samples,
        "n_success": n_success,
        "n_errors": n_errors,
        "success_rate": n_success / max(n_samples, 1),
        "mean_steps_when_success": float(np.mean(steps_when_success)) if steps_when_success else float("nan"),
        "mean_wall_time": float(np.mean(wall_times)) if wall_times else float("nan"),
        "total_wall_time": float(sum(wall_times)),
        "neg_vib_counts": neg_vib_counts,
        "cascade_table": cascade_table,
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
    parser.add_argument("--ts-eps", type=float, default=1e-5,
                        help="Convergence threshold: eig_product = λ_0*λ_1 < -ts_eps declares a TS. "
                             "Smaller → stricter (requires larger gap between λ_0 < 0 and λ_1 > 0). "
                             "Larger → more permissive (accepts smaller sign separation).")
    parser.add_argument("--tr-threshold", type=float, default=8e-3,
                        help="Eigenvalue magnitude threshold. When --tr-filter-eig is enabled, "
                             "modes with |λ| < tr_threshold are excluded from eig0/eig1 and "
                             "eig_product computation (legacy behaviour). Always used as diagnostic "
                             "threshold for TR residual monitoring.")
    parser.add_argument(
        "--tr-filter-eig",
        action="store_true",
        default=False,
        help="Legacy mode: filter near-zero modes (|λ| < tr_threshold) before computing "
             "eig_product for the convergence gate. Matches old implementation. "
             "When off (default), raw vibrational eigenvalues are used unfiltered.",
    )

    parser.add_argument(
        "--log-spectrum-k", type=int, default=10,
        help="Number of bottom vibrational eigenvalues to record at convergence/failure (default 10, 0 = none).",
    )

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

    # ── v3 Newton-GAD improvements ──────────────────────────────────────────
    parser.add_argument(
        "--step-mode",
        type=str,
        default="first_order",
        choices=["first_order", "newton_gad"],
        help="Step computation: 'first_order' (unit-vector GAD + trust radius) or "
             "'newton_gad' (mode-decomposed Newton step with curvature preconditioning "
             "and noise-mode filtering). Newton mode gives second-order convergence.",
    )
    parser.add_argument(
        "--step-filter-threshold",
        type=float,
        default=8e-3,
        help="Eigenvalue magnitude threshold for the Newton step: modes with |λ| < this "
             "are zeroed out (noise filtering). Also used for the convergence gate when "
             "--converge-on-filtered is set. Default 8e-3 (proven optimal for NR minimization).",
    )
    parser.add_argument(
        "--converge-on-filtered",
        action="store_true",
        default=False,
        help="Use filtered eigenvalues (|λ| > step_filter_threshold) for the eig_product "
             "convergence gate, preventing near-zero λ₁ from suppressing the product and "
             "causing overshoot.  Cascade table always reports unfiltered n_neg.",
    )
    parser.add_argument(
        "--anti-overshoot",
        action="store_true",
        default=False,
        help="Monitor Morse index per step; reduce trust radius when entering Morse-1 region "
             "to prevent passing through the TS into a minimum.",
    )
    parser.add_argument(
        "--cleanup-steps",
        type=int,
        default=0,
        help="Post-convergence NR minimization steps orthogonal to the TS mode. "
             "Pushes residual small negative eigenvalues positive without disturbing "
             "the saddle geometry. 0 = disabled (default).",
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
        "tr_filter_eig": args.tr_filter_eig,
        "track_mode": track_mode,
        "project_gradient_and_v": args.project_gradient_and_v,
        "stop_at_ts": args.stop_at_ts,
        "log_dir": str(diag_dir),
        "projection_mode": args.projection_mode,
        "purify_hessian": args.purify_hessian,
        "frame_tracking": args.frame_tracking,
        "log_spectrum_k": args.log_spectrum_k,
        # v3 Newton-GAD improvements
        "step_mode": args.step_mode,
        "step_filter_threshold": args.step_filter_threshold,
        "converge_on_filtered": args.converge_on_filtered,
        "anti_overshoot": args.anti_overshoot,
        "cleanup_steps": args.cleanup_steps,
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

        # Print cascade table to stdout for quick inspection
        ct = metrics.get("cascade_table", {})
        tr_filt = params.get("tr_filter_eig", False)
        print(f"\n--- GAD Cascade Evaluation Table  [tr_filter_eig={tr_filt}] ---")
        print(f"{'eval_threshold':<20} {'n_neg==1':>10} {'rate_eq1':>10} {'n_neg<=1':>10} {'rate_le1':>10}")
        print("-" * 62)
        for thr in CASCADE_THRESHOLDS:
            key = str(thr)
            n_eq1 = ct.get("n_neg_eq1_at_thr", {}).get(key, "?")
            r_eq1 = ct.get("rate_eq1_at_thr", {}).get(key, float("nan"))
            n_le1 = ct.get("n_neg_le1_at_thr", {}).get(key, "?")
            r_le1 = ct.get("rate_le1_at_thr", {}).get(key, float("nan"))
            print(f"  {thr:<18} {n_eq1:>10} {r_eq1:>10.3f} {n_le1:>10} {r_le1:>10.3f}")
        print(f"\n  Strict success rate (eig_product criterion): {metrics['success_rate']:.3f}")

        def _fmt(v: float) -> str:
            return f"{v:.5f}" if math.isfinite(v) else "nan"

        def _fmtg(v: float) -> str:
            return f"{v:.3f}" if math.isfinite(v) else "nan"

        # Unfiltered eigenvalue diagnostics — success and failure
        mg_s  = ct.get("mean_gap_ratio_at_success",   float("nan"))
        ml0_s = ct.get("mean_abs_lambda0_at_success", float("nan"))
        ml1_s = ct.get("mean_abs_lambda1_at_success", float("nan"))
        mg_fa = ct.get("mean_gap_ratio_at_failure",   float("nan"))
        ml0_fa = ct.get("mean_abs_lambda0_at_failure", float("nan"))
        ml1_fa = ct.get("mean_abs_lambda1_at_failure", float("nan"))

        print(f"  --- Unfiltered eigenvalues (true two smallest) ---")
        print(f"  At SUCCESS: |λ_0|={_fmt(ml0_s)}  |λ_1|={_fmt(ml1_s)}  gap={_fmtg(mg_s)}")
        print(f"  At FAILURE: |λ_0|={_fmt(ml0_fa)}  |λ_1|={_fmt(ml1_fa)}  gap={_fmtg(mg_fa)}")
        print(f"  (gap=|λ_0|/|λ_1|; if |λ_1|≈0 at success, soft mode contaminates eig_product)")

        if tr_filt:
            mg_sf  = ct.get("mean_filt_gap_ratio_at_success",   float("nan"))
            ml0_sf = ct.get("mean_filt_abs_lambda0_at_success", float("nan"))
            ml1_sf = ct.get("mean_filt_abs_lambda1_at_success", float("nan"))
            mg_ff  = ct.get("mean_filt_gap_ratio_at_failure",   float("nan"))
            ml0_ff = ct.get("mean_filt_abs_lambda0_at_failure", float("nan"))
            ml1_ff = ct.get("mean_filt_abs_lambda1_at_failure", float("nan"))
            print(f"  --- Filtered eigenvalues (|λ|>tr_threshold mask applied) ---")
            print(f"  At SUCCESS: |λ_0|={_fmt(ml0_sf)}  |λ_1|={_fmt(ml1_sf)}  gap={_fmtg(mg_sf)}")
            print(f"  At FAILURE: |λ_0|={_fmt(ml0_ff)}  |λ_1|={_fmt(ml1_ff)}  gap={_fmtg(mg_ff)}")
        print("")

    finally:
        processor.close()


if __name__ == "__main__":
    main()
