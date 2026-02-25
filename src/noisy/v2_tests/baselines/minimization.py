"""Minimization baselines: fixed-step gradient descent and Newton-Raphson.

Both methods find energy MINIMA on the potential energy surface.

1. Fixed-step gradient descent:
   x_{k+1} = x_k - alpha * grad E(x_k)
   = x_k + alpha * forces(x_k)

2. Newton-Raphson:
   x_{k+1} = x_k - H(x_k)^{-1} * grad E(x_k)
   = x_k + H(x_k)^{-1} * forces(x_k)

   The inverse Hessian is computed via pseudoinverse in the vibrational
   subspace to avoid singularities from translation/rotation modes.

Both support optional Eckart projection of the gradient to prevent
translation/rotation drift.

Algorithmic improvements (v2):
  - Cascading evaluation: n_neg counted at 8 thresholds so analysis can
    distinguish "optimizer found good geometry but evaluation too strict"
    from "optimizer genuinely failed". Logged in every trajectory step.
  - Levenberg-Marquardt (LM) damping: smooth alternative to hard filtering.
    step_i = (g·v_i) * |λ_i| / (λ_i² + μ²)
    Activated when lm_mu > 0; otherwise falls back to hard filter.
  - Two-phase threshold annealing: bulk optimization uses nr_threshold;
    once force_norm < anneal_force_threshold the threshold is dropped to
    cleanup_nr_threshold for a capped number of cleanup steps. The idea:
    near a stationary point the Hessian is better-conditioned, so lower
    thresholds no longer cause explosive steps.
  - Relaxed convergence criterion: instead of n_neg == 0 (exact), accept
    evals_vib.min() > -eps_conv. eps_conv=0 reproduces the old behavior.
    Useful when the cascade shows convergence failures are tiny λ ≈ -1e-4.
  - Full bottom-K vibrational spectrum logged at every step (sorted, first K
    entries). Invaluable for post-hoc debugging without re-running.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from src.dependencies.differentiable_projection import (
    project_vector_to_vibrational_torch,
)
from src.noisy.multi_mode_eckartmw import get_vib_evals_evecs


# ---------------------------------------------------------------------------
# Cascade evaluation thresholds (never change the optimizer; pure diagnostics)
# ---------------------------------------------------------------------------
CASCADE_THRESHOLDS: List[float] = [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2]


def _cascade_n_neg(evals_vib: torch.Tensor) -> Dict[str, int]:
    """Return n_neg counted at each cascade threshold.

    Key format: "n_neg_at_<threshold>" where threshold is stringified as
    the repr of the float (e.g. "n_neg_at_0.0", "n_neg_at_0.001").
    This is safe for JSON serialization and unambiguous in CSVs.
    """
    result: Dict[str, int] = {}
    for thr in CASCADE_THRESHOLDS:
        result[f"n_neg_at_{thr}"] = int((evals_vib < -thr).sum().item())
    return result


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().reshape(-1)[0].item())
    return float(x)


def _force_mean(forces: torch.Tensor) -> float:
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    f = forces.reshape(-1, 3)
    return float(f.norm(dim=1).mean().item())


def _min_interatomic_distance(coords: torch.Tensor) -> float:
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    n = coords.shape[0]
    if n < 2:
        return float("inf")
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    dist = diff.norm(dim=2)
    dist = dist + torch.eye(n, device=coords.device, dtype=coords.dtype) * 1e10
    return float(dist.min().item())


def _cap_displacement(step_disp: torch.Tensor, max_atom_disp: float) -> torch.Tensor:
    """Cap per-atom displacement to max_atom_disp."""
    disp_3d = step_disp.reshape(-1, 3)
    max_disp = float(disp_3d.norm(dim=1).max().item())
    if max_disp > max_atom_disp and max_disp > 0:
        disp_3d = disp_3d * (max_atom_disp / max_disp)
    return disp_3d.reshape(step_disp.shape)


def _bottom_k_spectrum(evals_vib: torch.Tensor, k: int = 10) -> List[float]:
    """Return the k smallest vibrational eigenvalues as a sorted Python list."""
    vals = evals_vib.detach().cpu()
    sorted_vals, _ = torch.sort(vals)
    return [float(v) for v in sorted_vals[:k].tolist()]


# ---------------------------------------------------------------------------
# NR step builders
# ---------------------------------------------------------------------------

def _nr_step_hard_filter(
    grad: torch.Tensor,
    V_all: torch.Tensor,
    lam_all: torch.Tensor,
    nr_threshold: float,
    forces: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard-filter NR step: exclude |λ| < nr_threshold from pseudoinverse.

    Returns (delta_x, V_used, lam_used) where V_used/lam_used are the filtered
    subsets used to compute the quadratic model for the trust-region ratio.
    """
    step_mask = torch.abs(lam_all) >= nr_threshold
    V = V_all[:, step_mask]
    lam = lam_all[step_mask]
    if V.shape[1] > 0:
        coeffs = V.T @ grad
        nr_step = V @ (coeffs / torch.abs(lam))
        delta_x = -nr_step
    else:
        delta_x = forces.reshape(-1) * 0.001
    return delta_x, V, lam


def _nr_step_lm_damping(
    grad: torch.Tensor,
    V_all: torch.Tensor,
    lam_all: torch.Tensor,
    lm_mu: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Levenberg-Marquardt damped NR step.

    step_i = (g·v_i) * |λ_i| / (λ_i² + μ²)

    Regimes:
      |λ| >> μ  →  1/|λ|   (pure Newton)
      |λ| =  μ  →  1/(2μ)  (bounded transition)
      |λ| << μ  →  |λ|/μ²  (flat modes → zero contribution)

    All modes contribute; no hard cutoff. The step magnitude along flat modes
    vanishes smoothly as μ → ∞.
    """
    mu2 = lm_mu ** 2
    abs_lam = torch.abs(lam_all)
    # LM weight: |λ| / (λ² + μ²)
    lm_weights = abs_lam / (lam_all ** 2 + mu2)
    coeffs = V_all.T @ grad
    nr_step = V_all @ (coeffs * lm_weights)
    delta_x = -nr_step
    return delta_x, V_all, lam_all


# ---------------------------------------------------------------------------
# Fixed-step gradient descent
# ---------------------------------------------------------------------------

def run_fixed_step_gd(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    atomsymbols: list,
    *,
    n_steps: int = 5000,
    step_size: float = 0.01,
    max_atom_disp: float = 0.3,
    force_converged: float = 1e-4,
    min_interatomic_dist: float = 0.5,
    project_gradient_and_v: bool = False,
    purify_hessian: bool = False,
    eps_conv: float = 0.0,
    log_spectrum_k: int = 10,
) -> Tuple[Dict[str, Any], list]:
    """Fixed-step gradient descent to find energy minimum.

    Update rule: x_{k+1} = x_k + alpha * forces(x_k)

    No line search, no adaptive step sizing. Pure fixed-step descent.
    Convergence: force norm < threshold AND evals_vib.min() > -eps_conv
    (eps_conv=0 → exact zero-negative-eigenvalue criterion, unchanged behavior).

    Args:
        predict_fn: Energy/force prediction function.
        coords0: Starting coordinates.
        atomic_nums: Atomic numbers.
        atomsymbols: Atom symbols (required for Eckart projection).
        n_steps: Maximum number of steps.
        step_size: Fixed step size alpha.
        max_atom_disp: Maximum per-atom displacement per step.
        force_converged: Force convergence threshold (eV/A).
        min_interatomic_dist: Minimum allowed interatomic distance.
        project_gradient_and_v: If True, Eckart-project the gradient.
        purify_hessian: If True, enforce translational sum rules on Hessian.
        eps_conv: Relaxed convergence tolerance. Accept evals_vib.min() > -eps_conv.
                  0.0 = strict (original behavior).
        log_spectrum_k: Number of bottom eigenvalues to log per step (0 = none).

    Returns:
        result: Summary dictionary.
        trajectory: Per-step data.
    """
    coords = coords0.detach().clone().to(torch.float32)
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)

    trajectory = []

    for step in range(n_steps):
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        energy = _to_float(out["energy"])
        forces = out["forces"]
        hessian = out["hessian"]
        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)

        force_norm = _force_mean(forces)

        # Vibrational eigenvalues via reduced basis — no threshold filtering.
        evals_vib, _, _ = get_vib_evals_evecs(hessian, coords, atomsymbols,
                                              purify_hessian=purify_hessian)

        # Cascading evaluation diagnostic
        cascade = _cascade_n_neg(evals_vib)
        # n_neg at strict threshold (for convergence check)
        n_neg = cascade["n_neg_at_0.0"]

        step_record: Dict[str, Any] = {
            "step": step,
            "energy": energy,
            "force_norm": force_norm,
            "step_size": step_size,
            "n_neg_evals": n_neg,
            "min_vib_eval": float(evals_vib.min().item()) if evals_vib.numel() > 0 else float("nan"),
            "min_dist": _min_interatomic_distance(coords),
            **cascade,
        }
        if log_spectrum_k > 0:
            step_record["bottom_spectrum"] = _bottom_k_spectrum(evals_vib, log_spectrum_k)

        trajectory.append(step_record)

        # Relaxed convergence: forces small AND min eigenvalue > -eps_conv
        min_eval = float(evals_vib.min().item()) if evals_vib.numel() > 0 else float("nan")
        converged_now = force_norm < force_converged and min_eval > -eps_conv
        if converged_now:
            return {
                "converged": True,
                "converged_step": step,
                "final_energy": energy,
                "final_force_norm": force_norm,
                "final_coords": coords.detach().cpu(),
                "total_steps": step + 1,
                "final_n_neg_evals": n_neg,
                "final_min_vib_eval": min_eval,
                "cascade_at_convergence": cascade,
                "bottom_spectrum_at_convergence": _bottom_k_spectrum(evals_vib, log_spectrum_k),
            }, trajectory

        # Optionally project gradient to remove TR components
        forces_flat = forces.reshape(-1)
        if project_gradient_and_v:
            forces_flat = project_vector_to_vibrational_torch(
                forces_flat, coords, atomsymbols,
            )

        # Fixed-step update: x_{k+1} = x_k + alpha * forces
        step_disp = step_size * forces_flat.reshape(-1, 3)
        step_disp = _cap_displacement(step_disp, max_atom_disp)

        new_coords = coords + step_disp
        if _min_interatomic_distance(new_coords) < min_interatomic_dist:
            # Halve displacement until geometry is valid
            for scale in [0.5, 0.25, 0.1, 0.05]:
                new_coords = coords + step_disp * scale
                if _min_interatomic_distance(new_coords) >= min_interatomic_dist:
                    break
            else:
                continue  # skip step entirely if nothing works

        coords = new_coords.detach()

    return {
        "converged": False,
        "converged_step": None,
        "final_energy": trajectory[-1]["energy"] if trajectory else float("nan"),
        "final_force_norm": trajectory[-1]["force_norm"] if trajectory else float("nan"),
        "final_coords": coords.detach().cpu(),
        "total_steps": len(trajectory),
        "final_n_neg_evals": trajectory[-1]["n_neg_evals"] if trajectory else -1,
        "final_min_vib_eval": trajectory[-1].get("min_vib_eval", float("nan")) if trajectory else float("nan"),
        "cascade_at_convergence": trajectory[-1] if trajectory else {},
        "bottom_spectrum_at_convergence": trajectory[-1].get("bottom_spectrum", []) if trajectory else [],
    }, trajectory


# ---------------------------------------------------------------------------
# Newton-Raphson
# ---------------------------------------------------------------------------

def run_newton_raphson(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    atomsymbols: list,
    *,
    n_steps: int = 5000,
    max_atom_disp: float = 0.3,
    force_converged: float = 1e-4,
    min_interatomic_dist: float = 0.5,
    nr_threshold: float = 8e-3,
    project_gradient_and_v: bool = True,
    purify_hessian: bool = False,
    known_ts_coords: Optional[torch.Tensor] = None,
    # --- New options ---
    lm_mu: float = 0.0,
    anneal_force_threshold: float = 0.0,
    cleanup_nr_threshold: float = 0.0,
    cleanup_max_steps: int = 50,
    eps_conv: float = 0.0,
    log_spectrum_k: int = 10,
) -> Tuple[Dict[str, Any], list]:
    """Newton-Raphson optimization to find energy minimum.

    Update rule: x_{k+1} = x_k - H(x_k)^{-1} * grad E(x_k)

    The inverse Hessian step is computed via pseudoinverse in the
    vibrational subspace, using absolute values of eigenvalues to ensure
    it is always a descent direction.

    Three step-building modes (mutually exclusive; selected at runtime):
      1. Hard filter (lm_mu == 0, anneal_force_threshold == 0):
         Exclude |λ| < nr_threshold from pseudoinverse. Proven best on DFTB0
         at 2 Å noise (99% convergence). Default mode.
      2. LM damping (lm_mu > 0):
         step_i = (g·v_i) * |λ_i| / (λ_i² + μ²). No hard cutoff; flat modes
         contribute a vanishingly small amount. Sweep with mu = NR_THRESHOLD_GRID.
      3. Two-phase annealing (anneal_force_threshold > 0):
         Phase 1: hard filter with nr_threshold (bulk optimization).
         Phase 2: once force_norm < anneal_force_threshold, switch to
         cleanup_nr_threshold (default 0 → full pseudoinverse) for up to
         cleanup_max_steps steps. Near a minimum the Hessian is better-
         conditioned, so the lower threshold is safe.

    Cascading evaluation:
      n_neg is computed at thresholds [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3,
      8e-3, 1e-2] and stored in every trajectory step. The analysis script
      builds a 2D table (optimizer_threshold × eval_threshold → conv_rate)
      to reveal whether failures are optimizer failures or evaluation strictness.

    Relaxed convergence:
      eps_conv > 0 accepts evals_vib.min() > -eps_conv instead of n_neg == 0.
      This directly addresses the "false rejection" problem for samples where
      the optimizer produces a geometry with λ_min ≈ -0.001 (numerically a
      minimum but failing the strict n_neg == 0 gate).

    Full spectrum logging:
      The bottom log_spectrum_k sorted vibrational eigenvalues are stored in
      every trajectory step under "bottom_spectrum". At convergence/failure the
      field "bottom_spectrum_at_convergence" is added to the result dict.

    Adaptive step scaling (Trust Region):
      Predicted energy change dE_pred = g^T dx + 0.5 dx^T H dx
      Actual energy change dE_actual = E_new - E_old
      rho = dE_actual / dE_pred
      rho > 0.75 → grow trust radius (up to max_atom_disp)
      rho < 0.25 → shrink trust radius
      Steps with dE_actual > 0 → rejected, retried with smaller radius.
    """
    coords = coords0.detach().clone().to(torch.float32)
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)

    coords0_reshaped = coords.clone()

    if known_ts_coords is not None:
        known_ts_coords = known_ts_coords.detach().clone().to(torch.float32).to(coords.device)
        if known_ts_coords.dim() == 3 and known_ts_coords.shape[0] == 1:
            known_ts_coords = known_ts_coords[0]
        known_ts_coords = known_ts_coords.reshape(-1, 3)

    # --- Mode selection ---
    use_lm = lm_mu > 0.0
    use_anneal = (not use_lm) and anneal_force_threshold > 0.0
    in_cleanup_phase = False  # activated in two-phase annealing

    trajectory = []
    current_trust_radius = max_atom_disp

    # Evaluate initial state
    out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)

    step = 0
    cleanup_steps_taken = 0

    while step < n_steps:
        energy = _to_float(out["energy"])
        forces = out["forces"]
        hessian = out["hessian"]

        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)

        force_norm = _force_mean(forces)

        # Vibrational eigenvalues via reduced basis — exactly 3N-k values, no threshold.
        evals_vib, evecs_vib_3N, _ = get_vib_evals_evecs(
            hessian, coords, atomsymbols, purify_hessian=purify_hessian,
        )

        # Cascading evaluation: n_neg at each threshold (pure diagnostics, no optimization effect)
        cascade = _cascade_n_neg(evals_vib)
        n_neg = cascade["n_neg_at_0.0"]  # strict count used for logging

        # Spectrum statistics for logging
        min_vib_eval = float(evals_vib.min().item()) if evals_vib.numel() > 0 else float("nan")
        max_vib_eval = float(evals_vib.max().item()) if evals_vib.numel() > 0 else float("nan")

        if evals_vib.numel() > 0:
            abs_evals = torch.abs(evals_vib)
            min_abs_vib = float(abs_evals.min().item())
            max_abs_vib = float(abs_evals.max().item())
            cond_num = max_abs_vib / min_abs_vib if min_abs_vib > 0 else float("inf")
        else:
            cond_num = float("nan")

        vib_pos = evals_vib[evals_vib > 0]
        eff_step = float(1.0 / vib_pos.min().item()) if vib_pos.numel() > 0 else float("nan")

        disp_from_start_max = float((coords - coords0_reshaped).norm(dim=1).max().item())
        dist_to_ts_max = (
            float((coords - known_ts_coords).norm(dim=1).max().item())
            if known_ts_coords is not None else None
        )

        step_record: Dict[str, Any] = {
            "step": step,
            "energy": energy,
            "force_norm": force_norm,
            "n_neg_evals": n_neg,
            "min_vib_eval": min_vib_eval,
            "max_vib_eval": max_vib_eval,
            "cond_num": cond_num,
            "eff_step_size": eff_step,
            "min_dist": _min_interatomic_distance(coords),
            "trust_radius": current_trust_radius,
            "disp_from_start_max": disp_from_start_max,
            "dist_to_ts_max": dist_to_ts_max,
            "phase": "cleanup" if in_cleanup_phase else "bulk",
            **cascade,
        }
        if log_spectrum_k > 0:
            step_record["bottom_spectrum"] = _bottom_k_spectrum(evals_vib, log_spectrum_k)

        trajectory.append(step_record)

        # ---------------------------------------------------------------
        # Convergence check: relaxed or strict
        # ---------------------------------------------------------------
        converged_now = min_vib_eval > -eps_conv  # NOTE: eps_conv=0 → n_neg==0 equivalent
        # Two-phase annealing gate: when annealing is active we only declare victory
        # after at least one cleanup step, so the cleanup phase always gets a chance
        # to run.  In all other cases (annealing off, or already in cleanup) we exit
        # as soon as the geometry satisfies the criterion.
        anneal_gate_passed = (not use_anneal) or in_cleanup_phase
        if converged_now and anneal_gate_passed:
            return {
                "converged": True,
                "converged_step": step,
                "final_energy": energy,
                "final_force_norm": force_norm,
                "final_coords": coords.detach().cpu(),
                "total_steps": step + 1,
                "final_n_neg_evals": n_neg,
                "final_min_vib_eval": min_vib_eval,
                "cascade_at_convergence": cascade,
                "bottom_spectrum_at_convergence": _bottom_k_spectrum(evals_vib, log_spectrum_k),
                "cleanup_steps_taken": cleanup_steps_taken,
            }, trajectory

        # ---------------------------------------------------------------
        # Two-phase annealing: transition to cleanup phase
        # ---------------------------------------------------------------
        if use_anneal and not in_cleanup_phase and force_norm < anneal_force_threshold:
            in_cleanup_phase = True

        # ---------------------------------------------------------------
        # Determine effective nr_threshold for this step
        # ---------------------------------------------------------------
        if in_cleanup_phase:
            effective_threshold = cleanup_nr_threshold
        else:
            effective_threshold = nr_threshold

        # ---------------------------------------------------------------
        # Build gradient
        # ---------------------------------------------------------------
        grad = -forces.reshape(-1)
        if project_gradient_and_v:
            grad = -project_vector_to_vibrational_torch(
                forces.reshape(-1), coords, atomsymbols,
            )

        work_dtype = grad.dtype
        V_all = evecs_vib_3N.to(device=grad.device, dtype=work_dtype)
        lam_all = evals_vib.to(device=grad.device, dtype=work_dtype)

        # ---------------------------------------------------------------
        # Compute NR step (three modes)
        # ---------------------------------------------------------------
        if use_lm:
            delta_x, V, lam = _nr_step_lm_damping(grad, V_all, lam_all, lm_mu)
        else:
            delta_x, V, lam = _nr_step_hard_filter(grad, V_all, lam_all, effective_threshold, forces)

        step_disp = delta_x.reshape(-1, 3)

        # ---------------------------------------------------------------
        # Adaptive Trust Region
        # ---------------------------------------------------------------
        accepted = False
        max_retries = 10
        retries = 0

        while not accepted and retries < max_retries:
            radius_used_for_step = current_trust_radius
            capped_disp = _cap_displacement(step_disp, radius_used_for_step)

            # Predict energy change using spectral form over the modes used for the step
            dx_flat = capped_disp.reshape(-1).to(work_dtype)
            dx_red = V.T @ dx_flat
            pred_dE = float((grad.dot(dx_flat) + 0.5 * (lam * dx_red * dx_red).sum()).item())

            new_coords = coords + capped_disp

            if _min_interatomic_distance(new_coords) < min_interatomic_dist:
                current_trust_radius *= 0.5
                retries += 1
                continue

            # Evaluate new energy
            out_new = predict_fn(new_coords, atomic_nums, do_hessian=True, require_grad=False)
            energy_new = _to_float(out_new["energy"])
            actual_dE = energy_new - energy

            # Accept if energy decreased (allow tiny numerical noise)
            if actual_dE <= 1e-5:
                accepted = True

                # Update trust radius based on rho
                rho = actual_dE / pred_dE if pred_dE < -1e-8 else 0.0

                if rho > 0.75:
                    current_trust_radius = min(current_trust_radius * 1.5, max_atom_disp)
                elif rho < 0.25:
                    current_trust_radius = max(current_trust_radius * 0.5, 0.001)
                elif rho < 0.0:
                    current_trust_radius = max(current_trust_radius * 0.25, 0.001)

                coords = new_coords.detach()
                out = out_new
            else:
                current_trust_radius *= 0.25
                retries += 1

        if not accepted:
            # If all retries failed, take the smallest step anyway and continue
            coords = new_coords.detach()
            out = out_new

        actual_step_disp = float(capped_disp.reshape(-1, 3).norm(dim=1).max().item())
        trajectory[-1]["actual_step_disp"] = actual_step_disp
        trajectory[-1]["hit_trust_radius"] = bool(actual_step_disp >= radius_used_for_step * 0.99)
        trajectory[-1]["retries"] = retries

        step += 1

        # In cleanup phase, cap extra steps taken
        if in_cleanup_phase:
            cleanup_steps_taken += 1
            if cleanup_steps_taken >= cleanup_max_steps:
                break

    last = trajectory[-1] if trajectory else {}
    return {
        "converged": False,
        "converged_step": None,
        "final_energy": last.get("energy", float("nan")),
        "final_force_norm": last.get("force_norm", float("nan")),
        "final_coords": coords.detach().cpu(),
        "total_steps": len(trajectory),
        "final_n_neg_evals": last.get("n_neg_evals", -1),
        "final_min_vib_eval": last.get("min_vib_eval", float("nan")),
        "cascade_at_convergence": {k: last.get(k) for k in last if k.startswith("n_neg_at_")},
        "bottom_spectrum_at_convergence": last.get("bottom_spectrum", []),
        "cleanup_steps_taken": cleanup_steps_taken,
    }, trajectory
