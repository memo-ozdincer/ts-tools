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
    cleanup_nr_threshold for a capped number of cleanup steps.
  - Spectral gap diagnostic: |λ_min| / |λ_second_min| on the vibrational
    eigenvalues, logged every step.
  - Full bottom-K vibrational spectrum logged at every step.

Algorithmic improvements (v3):
  - Shifted Newton step: step_i = (g·v_i) / (λ_i + σ), where
    σ = max(0, -λ_min) + shift_epsilon. Standard Levenberg shift that
    makes all effective eigenvalues positive. Negative modes get LARGER
    weights than equally-sized positive modes → more aggressive along
    the problematic directions.
  - Stagnation-triggered negative-mode perturbation: when n_neg is
    unchanged for stagnation_window steps and negative eigenvalues are
    small, apply a targeted displacement along the negative eigenvectors.
  - Adaptive LM μ annealing: when close to convergence (few small negative
    eigenvalues), reduce μ to let the optimizer take more aggressive steps.
  - Negative-mode line search: when stagnated with persistent negative
    modes, scan along the negative eigenvector to find where curvature
    changes sign.
  - Trust-region fixes: higher floor, reset after escape, gradient-norm
    based minimum step size to prevent optimizer from getting stuck with
    tiny displacements.
  - Per-step diagnostic logging: gradient-mode overlap for negative modes,
    step-mode decomposition, stagnation counter, energy plateau detection.
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


def _spectral_gap_info(evals_vib: torch.Tensor) -> Dict[str, Any]:
    """Spectral gap diagnostic for the vibrational eigenvalue spectrum.

    Computes |λ_min| / |λ_second_min| to detect an isolated dominant negative
    mode.  A large ratio signals transition-state-like character (one mode
    dominates the negative part of the spectrum).

    Returns:
        spectral_gap_ratio: |λ₀|/|λ₁| where λ₀ < λ₁ are the two most
            negative vibrational eigenvalues.  inf if exactly 1 negative
            eigenvalue; nan if 0 negative eigenvalues.
        dominant_neg_mode: True if exactly 1 negative eigenvalue, or
            the gap ratio exceeds 10.
    """
    if evals_vib.numel() < 2:
        return {"spectral_gap_ratio": float("nan"), "dominant_neg_mode": False}

    sorted_evals = torch.sort(evals_vib).values  # ascending
    lam0 = float(sorted_evals[0].item())
    lam1 = float(sorted_evals[1].item())

    if lam0 >= 0:
        # No negative eigenvalues
        return {"spectral_gap_ratio": float("nan"), "dominant_neg_mode": False}

    if lam1 >= 0:
        # Exactly one negative eigenvalue — perfectly isolated
        return {"spectral_gap_ratio": float("inf"), "dominant_neg_mode": True}

    # Both negative
    ratio = abs(lam0) / abs(lam1)
    return {"spectral_gap_ratio": ratio, "dominant_neg_mode": ratio > 10.0}


def _total_hessian_n_neg(hessian: torch.Tensor) -> int:
    """Count negative eigenvalues of the full (unprojected) Hessian."""
    H = hessian.detach()
    if H.dim() == 4:
        n = H.shape[0]
        H = H.reshape(3 * n, 3 * n)
    elif H.dim() == 3 and H.shape[0] == 1:
        H = H.squeeze(0)
    H = 0.5 * (H + H.T)  # symmetrise for numerical safety
    evals = torch.linalg.eigvalsh(H)
    return int((evals < 0.0).sum().item())


# ---------------------------------------------------------------------------
# v3 diagnostic: gradient-mode overlap for negative modes
# ---------------------------------------------------------------------------

def _neg_mode_diagnostics(
    grad: torch.Tensor,
    delta_x: torch.Tensor,
    evals_vib: torch.Tensor,
    evecs_vib_3N: torch.Tensor,
) -> Dict[str, Any]:
    """Compute per-step diagnostics for negative vibrational modes.

    Returns:
        neg_mode_grad_overlaps: list of |g·v_i|/|g| for each negative mode i,
            sorted by eigenvalue (most negative first). If near zero, the NR
            step cannot address that mode.
        neg_mode_eigenvalues: the negative eigenvalues themselves.
        step_along_neg_frac: fraction of ||delta_x||² that lies in the
            negative-eigenvalue subspace.
        step_along_pos_frac: fraction in the positive-eigenvalue subspace.
        max_neg_grad_overlap: the largest gradient-mode overlap among negatives.
        min_neg_grad_overlap: the smallest (bottleneck mode).
    """
    neg_mask = evals_vib < 0.0
    n_neg = int(neg_mask.sum().item())
    if n_neg == 0:
        return {
            "neg_mode_grad_overlaps": [],
            "neg_mode_eigenvalues": [],
            "step_along_neg_frac": 0.0,
            "step_along_pos_frac": 1.0,
            "max_neg_grad_overlap": float("nan"),
            "min_neg_grad_overlap": float("nan"),
        }

    grad_norm = float(grad.norm().item())
    if grad_norm < 1e-30:
        grad_norm = 1e-30

    neg_evals = evals_vib[neg_mask]
    neg_evecs = evecs_vib_3N[:, neg_mask]  # (3N, n_neg)

    # Sort by eigenvalue ascending (most negative first)
    sort_idx = torch.argsort(neg_evals)
    neg_evals = neg_evals[sort_idx]
    neg_evecs = neg_evecs[:, sort_idx]

    # Gradient overlap with each negative mode
    grad_projs = neg_evecs.T @ grad  # (n_neg,)
    overlaps = (grad_projs.abs() / grad_norm).tolist()
    neg_eval_list = neg_evals.tolist()

    # Step decomposition: what fraction of the step is along negative modes?
    dx_norm_sq = float((delta_x ** 2).sum().item())
    if dx_norm_sq < 1e-30:
        return {
            "neg_mode_grad_overlaps": overlaps,
            "neg_mode_eigenvalues": neg_eval_list,
            "step_along_neg_frac": 0.0,
            "step_along_pos_frac": 0.0,
            "max_neg_grad_overlap": max(overlaps) if overlaps else float("nan"),
            "min_neg_grad_overlap": min(overlaps) if overlaps else float("nan"),
        }

    dx_in_neg = neg_evecs.T @ delta_x  # (n_neg,)
    neg_frac = float((dx_in_neg ** 2).sum().item()) / dx_norm_sq

    pos_mask = evals_vib > 0.0
    if pos_mask.any():
        pos_evecs = evecs_vib_3N[:, pos_mask]
        dx_in_pos = pos_evecs.T @ delta_x
        pos_frac = float((dx_in_pos ** 2).sum().item()) / dx_norm_sq
    else:
        pos_frac = 0.0

    return {
        "neg_mode_grad_overlaps": overlaps,
        "neg_mode_eigenvalues": neg_eval_list,
        "step_along_neg_frac": neg_frac,
        "step_along_pos_frac": pos_frac,
        "max_neg_grad_overlap": max(overlaps),
        "min_neg_grad_overlap": min(overlaps),
    }


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


def _nr_step_shifted_newton(
    grad: torch.Tensor,
    V_all: torch.Tensor,
    lam_all: torch.Tensor,
    shift_epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shifted Newton step (Levenberg shift).

    step_i = (g·v_i) / (λ_i + σ)
    where σ = max(0, -λ_min) + shift_epsilon

    This makes all effective eigenvalues (λ_i + σ) positive. Crucially,
    negative modes get LARGER weights than equally-sized positive modes:
      λ=-0.01, σ=0.011 → weight = 1/0.001 = 1000
      λ=+0.01, σ=0.011 → weight = 1/0.021 ≈ 48
    So the step is more aggressive along the problematic directions.

    As σ→0 near a true minimum, this recovers the pure Newton step.
    """
    lam_min = float(lam_all.min().item())
    sigma = max(0.0, -lam_min) + shift_epsilon
    shifted_lam = lam_all + sigma  # all positive
    coeffs = V_all.T @ grad
    nr_step = V_all @ (coeffs / shifted_lam)
    delta_x = -nr_step
    return delta_x, V_all, lam_all


# ---------------------------------------------------------------------------
# v3: Stagnation escape — targeted perturbation along negative modes
# ---------------------------------------------------------------------------

def _stagnation_escape_perturbation(
    coords: torch.Tensor,
    grad: torch.Tensor,
    evals_vib: torch.Tensor,
    evecs_vib_3N: torch.Tensor,
    escape_alpha: float,
) -> torch.Tensor:
    """Build a targeted perturbation along negative eigenvectors.

    The displacement is:
        perturbation = alpha * sum_i( sign(g·v_i) * v_i )  for negative modes i
    normalized so max atom displacement = escape_alpha.

    sign(g·v_i) ensures we move downhill along each negative mode.
    If the gradient projection is zero, we pick an arbitrary sign (+1).

    Returns new coordinates after the perturbation.
    """
    neg_mask = evals_vib < 0.0
    if not neg_mask.any():
        return coords

    neg_evecs = evecs_vib_3N[:, neg_mask]  # (3N, n_neg)
    grad_projs = neg_evecs.T @ grad  # (n_neg,)

    # Build perturbation: sum of negative eigenvectors with gradient-aligned signs
    signs = torch.sign(grad_projs)
    signs[signs == 0] = 1.0  # arbitrary positive if exactly zero overlap
    perturbation = neg_evecs @ signs  # (3N,)

    # Normalize to max atom displacement = escape_alpha
    pert_3d = perturbation.reshape(-1, 3)
    max_disp = float(pert_3d.norm(dim=1).max().item())
    if max_disp > 1e-10:
        pert_3d = pert_3d * (escape_alpha / max_disp)

    return coords + pert_3d


# ---------------------------------------------------------------------------
# v3: Negative-mode line search
# ---------------------------------------------------------------------------

def _neg_mode_line_search(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    atomsymbols: list,
    evals_vib: torch.Tensor,
    evecs_vib_3N: torch.Tensor,
    grad: torch.Tensor,
    *,
    purify_hessian: bool = False,
    alphas: Tuple[float, ...] = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3),
    min_interatomic_dist: float = 0.5,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """Line search along the most negative eigenvector.

    For each alpha in alphas, evaluate the geometry at coords ± alpha * v_neg
    (sign chosen to move downhill). Pick the trial point whose most-negative
    eigenvalue is closest to zero (or positive).

    Returns:
        best_coords: new coordinates, or None if no improvement found.
        info: dict with line search diagnostics.
    """
    neg_mask = evals_vib < 0.0
    if not neg_mask.any():
        return None, {"line_search_skipped": True, "reason": "no_neg_modes"}

    # Target the most negative mode
    sorted_evals, sorted_idx = torch.sort(evals_vib)
    most_neg_idx = sorted_idx[0]
    v_neg = evecs_vib_3N[:, most_neg_idx]  # (3N,)
    lam_neg = float(sorted_evals[0].item())

    # Determine direction: move downhill
    g_proj = float((grad @ v_neg).item())
    direction = v_neg if g_proj > 0 else -v_neg  # step = -direction * alpha (downhill)

    best_min_eval = lam_neg
    best_coords = None
    best_alpha = None
    trials = []

    for alpha in alphas:
        disp = direction.reshape(-1, 3) * alpha
        trial_coords = coords + disp
        if _min_interatomic_distance(trial_coords) < min_interatomic_dist:
            continue
        try:
            trial_out = predict_fn(trial_coords, atomic_nums, do_hessian=True, require_grad=False)
            trial_evals, _, _ = get_vib_evals_evecs(
                trial_out["hessian"], trial_coords, atomsymbols,
                purify_hessian=purify_hessian,
            )
            trial_min_eval = float(trial_evals.min().item())
            trial_n_neg = int((trial_evals < 0.0).sum().item())
            trials.append({"alpha": alpha, "min_eval": trial_min_eval, "n_neg": trial_n_neg})

            if trial_min_eval > best_min_eval:
                best_min_eval = trial_min_eval
                best_coords = trial_coords
                best_alpha = alpha
        except Exception:
            continue

    info = {
        "line_search_skipped": False,
        "original_min_eval": lam_neg,
        "best_min_eval": best_min_eval,
        "best_alpha": best_alpha,
        "n_trials": len(trials),
        "trials": trials,
        "improved": best_coords is not None,
    }
    return best_coords, info


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
    log_spectrum_k: int = 10,
) -> Tuple[Dict[str, Any], list]:
    """Fixed-step gradient descent to find energy minimum.

    Update rule: x_{k+1} = x_k + alpha * forces(x_k)

    No line search, no adaptive step sizing. Pure fixed-step descent.
    Convergence: force norm < threshold AND n_neg == 0 (no negative vibrational eigenvalues).

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

        # Spectral gap + total Hessian diagnostics
        gap_info = _spectral_gap_info(evals_vib)
        n_neg_total = _total_hessian_n_neg(hessian)

        step_record: Dict[str, Any] = {
            "step": step,
            "energy": energy,
            "force_norm": force_norm,
            "step_size": step_size,
            "n_neg_evals": n_neg,
            "n_neg_total_hessian": n_neg_total,
            "min_vib_eval": float(evals_vib.min().item()) if evals_vib.numel() > 0 else float("nan"),
            "min_dist": _min_interatomic_distance(coords),
            **gap_info,
            **cascade,
        }
        if log_spectrum_k > 0:
            step_record["bottom_spectrum"] = _bottom_k_spectrum(evals_vib, log_spectrum_k)

        trajectory.append(step_record)

        # Convergence: forces small AND no negative vibrational eigenvalues
        converged_now = force_norm < force_converged and n_neg == 0
        if converged_now:
            return {
                "converged": True,
                "converged_step": step,
                "final_energy": energy,
                "final_force_norm": force_norm,
                "final_coords": coords.detach().cpu(),
                "total_steps": step + 1,
                "final_n_neg_evals": n_neg,
                "final_n_neg_total_hessian": n_neg_total,
                "final_min_vib_eval": float(evals_vib.min().item()) if evals_vib.numel() > 0 else float("nan"),
                "final_spectral_gap_ratio": gap_info["spectral_gap_ratio"],
                "final_dominant_neg_mode": gap_info["dominant_neg_mode"],
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
        "final_n_neg_total_hessian": trajectory[-1].get("n_neg_total_hessian", -1) if trajectory else -1,
        "final_min_vib_eval": trajectory[-1].get("min_vib_eval", float("nan")) if trajectory else float("nan"),
        "final_spectral_gap_ratio": trajectory[-1].get("spectral_gap_ratio", float("nan")) if trajectory else float("nan"),
        "final_dominant_neg_mode": trajectory[-1].get("dominant_neg_mode", False) if trajectory else False,
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
    # --- v2 options ---
    lm_mu: float = 0.0,
    anneal_force_threshold: float = 0.0,
    cleanup_nr_threshold: float = 0.0,
    cleanup_max_steps: int = 50,
    log_spectrum_k: int = 10,
    # --- v3 options ---
    shift_epsilon: float = 0.0,
    stagnation_window: int = 0,
    escape_alpha: float = 0.1,
    lm_mu_anneal_factor: float = 0.0,
    lm_mu_anneal_n_neg_leq: int = 2,
    lm_mu_anneal_eval_leq: float = 5e-3,
    neg_mode_line_search: bool = False,
    line_search_alphas: Optional[List[float]] = None,
    trust_radius_floor: float = 0.01,
) -> Tuple[Dict[str, Any], list]:
    """Newton-Raphson optimization to find energy minimum.

    Update rule: x_{k+1} = x_k - H(x_k)^{-1} * grad E(x_k)

    The inverse Hessian step is computed via pseudoinverse in the
    vibrational subspace, using absolute values of eigenvalues to ensure
    it is always a descent direction.

    Step-building modes (selected at runtime by which parameters are nonzero):
      1. Shifted Newton (shift_epsilon > 0):
         step_i = (g·v_i) / (λ_i + σ), σ = max(0, -λ_min) + shift_epsilon.
         Makes all effective eigenvalues positive. Negative modes get larger
         weights → more aggressive along problematic directions.
      2. LM damping (lm_mu > 0, shift_epsilon == 0):
         step_i = (g·v_i) * |λ_i| / (λ_i² + μ²). No hard cutoff; flat modes
         contribute a vanishingly small amount.
      3. Two-phase annealing (anneal_force_threshold > 0, lm_mu == 0):
         Phase 1: hard filter with nr_threshold. Phase 2: cleanup_nr_threshold.
      4. Hard filter (default, all above == 0):
         Exclude |λ| < nr_threshold from pseudoinverse.

    v3 features:
      - Stagnation escape (stagnation_window > 0): when n_neg is unchanged
        for stagnation_window consecutive steps, apply targeted perturbation
        along negative eigenvectors, then optionally a negative-mode line search.
      - Adaptive LM μ annealing (lm_mu_anneal_factor > 0): when n_neg <= threshold
        and |λ_min| < threshold, reduce μ by anneal_factor.
      - Per-step diagnostic logging: gradient-mode overlap, step decomposition.
      - Trust-region floor: prevents the trust radius from shrinking below
        trust_radius_floor, ensuring the optimizer can always take meaningful steps.
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

    # --- Mode selection (priority: shifted > LM > anneal > hard filter) ---
    use_shifted = shift_epsilon > 0.0
    use_lm = (not use_shifted) and lm_mu > 0.0
    use_anneal = (not use_shifted) and (not use_lm) and anneal_force_threshold > 0.0
    in_cleanup_phase = False

    # Effective LM mu (may be annealed)
    effective_lm_mu = lm_mu

    # Line search alphas
    ls_alphas = tuple(line_search_alphas) if line_search_alphas else (0.02, 0.05, 0.1, 0.15, 0.2, 0.3)

    trajectory: list = []
    current_trust_radius = max_atom_disp

    # Stagnation tracking
    prev_n_neg = -1
    stagnation_counter = 0
    total_escapes = 0
    total_line_searches = 0

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

        # Spectral gap + total Hessian diagnostics
        gap_info = _spectral_gap_info(evals_vib)
        n_neg_total = _total_hessian_n_neg(hessian)

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

        # ---------------------------------------------------------------
        # Stagnation tracking
        # ---------------------------------------------------------------
        if n_neg == prev_n_neg and n_neg > 0:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        prev_n_neg = n_neg

        # Energy plateau detection: compare to energy 10 steps ago
        energy_plateau = False
        if len(trajectory) >= 10:
            old_energy = trajectory[-10].get("energy", energy)
            energy_plateau = abs(energy - old_energy) < 1e-8

        step_record: Dict[str, Any] = {
            "step": step,
            "energy": energy,
            "force_norm": force_norm,
            "n_neg_evals": n_neg,
            "n_neg_total_hessian": n_neg_total,
            "min_vib_eval": min_vib_eval,
            "max_vib_eval": max_vib_eval,
            "cond_num": cond_num,
            "eff_step_size": eff_step,
            "min_dist": _min_interatomic_distance(coords),
            "trust_radius": current_trust_radius,
            "disp_from_start_max": disp_from_start_max,
            "dist_to_ts_max": dist_to_ts_max,
            "phase": "cleanup" if in_cleanup_phase else "bulk",
            "stagnation_counter": stagnation_counter,
            "energy_plateau": energy_plateau,
            "effective_lm_mu": effective_lm_mu if use_lm else None,
            **gap_info,
            **cascade,
        }
        if log_spectrum_k > 0:
            step_record["bottom_spectrum"] = _bottom_k_spectrum(evals_vib, log_spectrum_k)

        trajectory.append(step_record)

        # ---------------------------------------------------------------
        # Convergence check: strict n_neg == 0
        # ---------------------------------------------------------------
        converged_now = n_neg == 0
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
                "final_n_neg_total_hessian": n_neg_total,
                "final_min_vib_eval": min_vib_eval,
                "final_spectral_gap_ratio": gap_info["spectral_gap_ratio"],
                "final_dominant_neg_mode": gap_info["dominant_neg_mode"],
                "cascade_at_convergence": cascade,
                "bottom_spectrum_at_convergence": _bottom_k_spectrum(evals_vib, log_spectrum_k),
                "cleanup_steps_taken": cleanup_steps_taken,
                "total_escapes": total_escapes,
                "total_line_searches": total_line_searches,
            }, trajectory

        # ---------------------------------------------------------------
        # v3: Stagnation escape
        # ---------------------------------------------------------------
        did_escape = False
        if (stagnation_window > 0
                and stagnation_counter >= stagnation_window
                and n_neg > 0
                and abs(min_vib_eval) < 0.02):  # only escape from shallow saddles
            # Phase 1: targeted perturbation along negative modes
            grad_for_escape = -forces.reshape(-1)
            if project_gradient_and_v:
                grad_for_escape = -project_vector_to_vibrational_torch(
                    forces.reshape(-1), coords, atomsymbols,
                )

            new_coords = _stagnation_escape_perturbation(
                coords, grad_for_escape, evals_vib, evecs_vib_3N, escape_alpha,
            )

            # Phase 2: optional line search along the most negative mode
            if neg_mode_line_search:
                ls_coords, ls_info = _neg_mode_line_search(
                    predict_fn, coords, atomic_nums, atomsymbols,
                    evals_vib, evecs_vib_3N, grad_for_escape,
                    purify_hessian=purify_hessian,
                    alphas=ls_alphas,
                    min_interatomic_dist=min_interatomic_dist,
                )
                step_record["line_search_info"] = ls_info
                total_line_searches += 1
                if ls_coords is not None:
                    new_coords = ls_coords  # line search found something better

            if _min_interatomic_distance(new_coords) >= min_interatomic_dist:
                coords = new_coords.detach()
                out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
                # Reset trust radius after escape
                current_trust_radius = max_atom_disp
                stagnation_counter = 0
                total_escapes += 1
                did_escape = True
                step_record["escape_triggered"] = True
                step += 1
                continue

        # ---------------------------------------------------------------
        # v3: Adaptive LM μ annealing
        # ---------------------------------------------------------------
        if (use_lm
                and lm_mu_anneal_factor > 0.0
                and n_neg <= lm_mu_anneal_n_neg_leq
                and n_neg > 0
                and abs(min_vib_eval) < lm_mu_anneal_eval_leq):
            effective_lm_mu = lm_mu * lm_mu_anneal_factor
        elif use_lm:
            effective_lm_mu = lm_mu

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
        # Compute NR step (four modes, priority: shifted > LM > anneal > HF)
        # ---------------------------------------------------------------
        if use_shifted:
            delta_x, V, lam = _nr_step_shifted_newton(grad, V_all, lam_all, shift_epsilon)
            step_record["step_mode"] = "shifted_newton"
        elif use_lm:
            delta_x, V, lam = _nr_step_lm_damping(grad, V_all, lam_all, effective_lm_mu)
            step_record["step_mode"] = "lm_damping"
        else:
            delta_x, V, lam = _nr_step_hard_filter(grad, V_all, lam_all, effective_threshold, forces)
            step_record["step_mode"] = "hard_filter"

        # ---------------------------------------------------------------
        # v3: Per-step negative-mode diagnostics
        # ---------------------------------------------------------------
        if n_neg > 0:
            neg_diag = _neg_mode_diagnostics(grad, delta_x, evals_vib, evecs_vib_3N)
            step_record["neg_mode_diag"] = neg_diag

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
                current_trust_radius = max(current_trust_radius * 0.5, trust_radius_floor)
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
                    current_trust_radius = max(current_trust_radius * 0.5, trust_radius_floor)
                elif rho < 0.0:
                    current_trust_radius = max(current_trust_radius * 0.25, trust_radius_floor)

                coords = new_coords.detach()
                out = out_new
            else:
                current_trust_radius = max(current_trust_radius * 0.25, trust_radius_floor)
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
        "final_n_neg_total_hessian": last.get("n_neg_total_hessian", -1),
        "final_min_vib_eval": last.get("min_vib_eval", float("nan")),
        "final_spectral_gap_ratio": last.get("spectral_gap_ratio", float("nan")),
        "final_dominant_neg_mode": last.get("dominant_neg_mode", False),
        "cascade_at_convergence": {k: last.get(k) for k in last if k.startswith("n_neg_at_")},
        "bottom_spectrum_at_convergence": last.get("bottom_spectrum", []),
        "cleanup_steps_taken": cleanup_steps_taken,
        "total_escapes": total_escapes,
        "total_line_searches": total_line_searches,
    }, trajectory
