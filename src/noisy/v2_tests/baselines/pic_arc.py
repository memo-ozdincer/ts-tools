"""PIC-ARC: Preconditioned Internal-Coordinate Adaptive Regularized Cubics.

A clean-sheet two-phase molecular geometry optimizer for energy minimization
on the DFTB0 potential energy surface.

Design informed by v1-v8 experimental record:

  Phase A (FLOW):  Preconditioned steepest descent with trust-region acceptance.
      Purpose: safe, cheap, non-destructive small steps when the Hessian is
      unreliable (high noise, high condition number, many negative modes).
      Uses a molecule-aware positive-definite metric M (Lindh-type model
      Hessian) to scale steps by bonding environment.

  Phase B (ARC):   Subspace cubic regularization.
      Builds a low-dimensional subspace from the preconditioned gradient and
      the most negative eigenvectors, then solves a cubic-regularized model
      in that subspace.  Handles indefinite Hessians naturally without
      projection to PSD or reciprocal-weight explosion on near-zero modes.

  Switching:  Conservative state machine.  Starts in FLOW.  Enters ARC only
      when condition number, negative-mode count, and force norm are all
      stable for several consecutive steps.  Falls back to FLOW on any
      instability.

  Termination:  Multi-channel.
      STRICT:   force < threshold AND n_neg == 0
      RELAXED:  force < threshold AND no eigenvalue below -relaxed_threshold
      STALLED:  neither achieved at max_steps

Key design decisions (informed by critique of PIC-ARC proposal):
  - Trust region in BOTH phases (v7 proved TR > line search at all noise levels)
  - No escape mechanisms (v3/v4 showed all escape attempts hurt or had no effect)
  - Cartesian coordinates with PD metric only (TRIC unmotivated by data)
  - ARC in small subspace (m <= 6) — captures essential curvature cheaply
  - Chemistry-aware acceptance: also accept if force norm drops > 50%
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from src.dependencies.differentiable_projection import (
    project_vector_to_vibrational_torch,
)
from src.noisy.multi_mode_eckartmw import get_vib_evals_evecs
from src.noisy.v2_tests.baselines.minimization import (
    _bottom_k_spectrum,
    _cap_displacement,
    _cascade_n_neg,
    _eigenvalue_band_populations,
    _force_mean,
    _min_interatomic_distance,
    _spectral_gap_info,
    _to_float,
    _total_hessian_n_neg,
)


# ---------------------------------------------------------------------------
# Module 1: Molecule-aware metric builder
# ---------------------------------------------------------------------------

# Covalent radii (Angstroms) — standard Lindh/Cordero values
COVALENT_RADII: Dict[int, float] = {
    1: 0.32,    # H
    5: 0.82,    # B
    6: 0.77,    # C
    7: 0.75,    # N
    8: 0.73,    # O
    9: 0.72,    # F
    14: 1.17,   # Si
    15: 1.10,   # P
    16: 1.04,   # S
    17: 0.99,   # Cl
    35: 1.14,   # Br
    53: 1.33,   # I
}
DEFAULT_COV_RADIUS = 1.5


def build_metric(
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    k_bond: float = 0.45,
    bond_threshold_factor: float = 1.3,
    regularization: float = 1e-3,
) -> torch.Tensor:
    """Build molecule-aware positive-definite metric M (3N x 3N).

    Assembles a Lindh-type stretch Hessian from molecular connectivity:
      1. Detect bonds from interatomic distances + covalent radii.
      2. For each bond (i,j), add k_bond * (r_hat outer r_hat) to the
         diagonal blocks and subtract from off-diagonal blocks.
      3. Add regularization * I to guarantee positive-definiteness.

    Args:
        coords: (N, 3) atomic positions in Angstroms.
        atomic_nums: (N,) atomic numbers.
        k_bond: stretch force constant (Hartree/Bohr^2 scale).
        bond_threshold_factor: bond if d < factor * (r_cov_i + r_cov_j).
        regularization: diagonal regularization epsilon.

    Returns:
        M: (3N, 3N) positive-definite metric tensor (float64).
    """
    coords_3 = coords.reshape(-1, 3).to(torch.float64)
    n_atoms = coords_3.shape[0]
    dim = 3 * n_atoms
    M = torch.zeros(dim, dim, dtype=torch.float64)

    # Get covalent radii for each atom
    z_list = atomic_nums.reshape(-1).tolist()
    radii = [COVALENT_RADII.get(int(z), DEFAULT_COV_RADIUS) for z in z_list]

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            r_ij = coords_3[j] - coords_3[i]
            d_ij = float(r_ij.norm().item())
            threshold = bond_threshold_factor * (radii[i] + radii[j])

            if d_ij < threshold and d_ij > 1e-10:
                r_hat = r_ij / d_ij
                # Outer product: (3,) outer (3,) -> (3, 3)
                outer = torch.outer(r_hat, r_hat)
                contribution = k_bond * outer

                si, ei = 3 * i, 3 * i + 3
                sj, ej = 3 * j, 3 * j + 3

                # Standard Hessian assembly for stretch term
                M[si:ei, si:ei] += contribution
                M[sj:ej, sj:ej] += contribution
                M[si:ei, sj:ej] -= contribution
                M[sj:ej, si:ei] -= contribution

    # Regularize to ensure PD
    M += regularization * torch.eye(dim, dtype=torch.float64)

    return M


# ---------------------------------------------------------------------------
# Module 2: FLOW step (preconditioned gradient)
# ---------------------------------------------------------------------------

def flow_step(
    grad_flat: torch.Tensor,
    M_cholesky: torch.Tensor,
) -> torch.Tensor:
    """Preconditioned steepest descent step: p = -M^{-1} g.

    Uses pre-computed Cholesky factor L of M for efficiency.

    Args:
        grad_flat: (3N,) gradient vector.
        M_cholesky: (3N, 3N) lower-triangular Cholesky factor of M.

    Returns:
        step_flat: (3N,) raw step direction (capping done by caller).
    """
    g = grad_flat.to(torch.float64).unsqueeze(1)
    p = torch.cholesky_solve(-g, M_cholesky).squeeze(1)
    return p.to(grad_flat.dtype)


# ---------------------------------------------------------------------------
# Module 3: ARC subproblem solver
# ---------------------------------------------------------------------------

def _solve_secular_equation(
    evals_HS: torch.Tensor,
    g_rotated: torch.Tensor,
    sigma: float,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, float, int]:
    """Solve the ARC secular equation via safeguarded Newton.

    Finds lambda* > max(0, -lam_min(H_S)) such that:
        z(lambda) = -(evals_HS + lambda)^{-1} g_rotated
        sigma * ||z(lambda)|| = lambda

    Args:
        evals_HS: (m,) eigenvalues of the projected Hessian H_S.
        g_rotated: (m,) gradient projected into H_S eigenbasis.
        sigma: cubic regularization parameter > 0.
        max_iter: maximum Newton iterations.
        tol: convergence tolerance.

    Returns:
        z_rotated: (m,) solution in the H_S eigenbasis.
        lambda_star: the regularization multiplier found.
        n_iter: number of iterations used.
    """
    lam_min_HS = float(evals_HS[0].item())  # evals sorted ascending
    # Initial lambda: well above the boundary
    lambda_k = max(0.0, -lam_min_HS) + 0.1 * sigma + 1e-6

    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        shifted = evals_HS + lambda_k
        # Clamp for numerical safety
        shifted_safe = torch.clamp(shifted.abs(), min=1e-15) * shifted.sign()
        shifted_safe = torch.where(
            shifted_safe.abs() < 1e-15,
            torch.full_like(shifted_safe, 1e-15),
            shifted_safe,
        )

        z_rot = -g_rotated / shifted_safe
        z_norm = float(z_rot.norm().item())

        if z_norm < 1e-30:
            break

        # phi(lambda) = 1/||z|| - sigma/lambda
        phi = 1.0 / z_norm - sigma / lambda_k

        if abs(phi) < tol:
            break

        # dphi/dlambda = (sum g_i^2 / (lam_i + lambda)^3) / ||z||^3
        dphi_numer = float(((g_rotated ** 2) / (shifted_safe ** 3)).sum().item())
        dphi = dphi_numer / (z_norm ** 3)

        if abs(dphi) < 1e-30:
            break

        delta = phi / dphi
        lambda_new = lambda_k - delta
        # Stay above the boundary
        lambda_k = max(lambda_new, max(0.0, -lam_min_HS) + 1e-8)

    # Compute final z
    shifted_final = evals_HS + lambda_k
    shifted_final = torch.clamp(shifted_final.abs(), min=1e-15)
    z_rotated = -g_rotated / shifted_final

    return z_rotated, lambda_k, n_iter


def arc_subproblem(
    grad_flat: torch.Tensor,
    hessian_flat: torch.Tensor,
    evals_vib: torch.Tensor,
    evecs_vib_3N: torch.Tensor,
    M_cholesky: torch.Tensor,
    sigma: float,
    max_neg_modes_in_subspace: int = 5,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Solve the cubic regularization subproblem in a low-dimensional subspace.

    Subspace S = {M^{-1}g_hat, v_1^-, ..., v_r^-} where r <= max_neg_modes.

    The reduced cubic model:
        m(z) = g_S^T z + 0.5 z^T H_S z + (sigma/3) ||z||^3
    is solved via the secular equation approach.

    Args:
        grad_flat: (3N,) gradient in Cartesian space.
        hessian_flat: (3N, 3N) full Hessian matrix.
        evals_vib: vibrational eigenvalues (ascending).
        evecs_vib_3N: (3N, n_vib) vibrational eigenvectors.
        M_cholesky: (3N, 3N) Cholesky factor of metric M.
        sigma: cubic regularization parameter > 0.
        max_neg_modes_in_subspace: max negative modes to include.

    Returns:
        step_flat: (3N,) step in full Cartesian space.
        info: diagnostic dict.
    """
    work_dtype = torch.float64
    g = grad_flat.to(work_dtype)
    H = hessian_flat.to(work_dtype)

    # --- Build subspace ---
    # 1. Preconditioned gradient direction
    Minv_g = torch.cholesky_solve(-g.unsqueeze(1), M_cholesky).squeeze(1)
    Minv_g_norm = Minv_g.norm()
    if Minv_g_norm > 1e-30:
        g_hat = Minv_g / Minv_g_norm
    else:
        # Fallback: use raw gradient direction
        g_hat = g / max(g.norm().item(), 1e-30)

    # 2. Negative eigenvectors (most negative first)
    neg_mask = evals_vib < 0.0
    n_neg = int(neg_mask.sum().item())
    n_use = min(n_neg, max_neg_modes_in_subspace)

    columns = [g_hat.unsqueeze(1)]
    if n_use > 0:
        neg_evecs = evecs_vib_3N[:, neg_mask].to(work_dtype)
        neg_evals = evals_vib[neg_mask]
        # Sort: most negative first
        sort_idx = torch.argsort(neg_evals)
        columns.append(neg_evecs[:, sort_idx[:n_use]])

    S_raw = torch.cat(columns, dim=1)  # (3N, 1 + n_use)

    # Orthonormalize via QR
    S, _ = torch.linalg.qr(S_raw)  # (3N, m)
    m = S.shape[1]

    # --- Project into subspace ---
    g_S = S.T @ g          # (m,)
    H_S = S.T @ H @ S      # (m, m)
    H_S = 0.5 * (H_S + H_S.T)  # symmetrise

    # --- Eigendecompose H_S ---
    evals_HS, evecs_HS = torch.linalg.eigh(H_S)  # ascending
    g_rotated = evecs_HS.T @ g_S  # (m,)

    # --- Solve secular equation ---
    z_rotated, lambda_star, n_iter = _solve_secular_equation(
        evals_HS, g_rotated, sigma,
    )

    # Map from eigenbasis back to subspace, then to full space
    z = evecs_HS @ z_rotated  # (m,)
    step_full = S @ z         # (3N,)

    # --- Diagnostics ---
    z_norm = float(z.norm().item())
    step_norm = float(step_full.norm().item())

    # Model value at z: m(z) = g_S.z + 0.5 z^T H_S z + (sigma/3) ||z||^3
    model_value = float(
        (g_S @ z + 0.5 * z @ H_S @ z + (sigma / 3.0) * z_norm ** 3).item()
    )

    info = {
        "subspace_dim": m,
        "n_neg_in_subspace": n_use,
        "sigma": sigma,
        "lambda_star": lambda_star,
        "secular_iterations": n_iter,
        "z_norm": z_norm,
        "step_norm": step_norm,
        "g_S_norm": float(g_S.norm().item()),
        "H_S_evals": [float(v) for v in evals_HS.tolist()],
        "model_value": model_value,
    }

    return step_full.to(grad_flat.dtype), info


# ---------------------------------------------------------------------------
# Module 4: State machine controller
# ---------------------------------------------------------------------------

class PicArcState:
    FLOW = "FLOW"
    ARC = "ARC"


class PicArcController:
    """State machine managing FLOW/ARC phase transitions.

    Conservative policy: stays in FLOW unless ALL stability conditions
    are met for stability_window consecutive accepted steps.
    """

    def __init__(
        self,
        kappa_threshold: float = 1e6,
        n_neg_max_for_arc: int = 5,
        force_max_for_arc: float = 1.0,
        stability_window: int = 3,
        max_consecutive_arc_rejects: int = 3,
    ):
        self.kappa_threshold = kappa_threshold
        self.n_neg_max_for_arc = n_neg_max_for_arc
        self.force_max_for_arc = force_max_for_arc
        self.stability_window = stability_window
        self.max_consecutive_arc_rejects = max_consecutive_arc_rejects

        self.state = PicArcState.FLOW
        self.stability_counter = 0
        self.consecutive_arc_rejects = 0
        self.steps_in_current_phase = 0
        self.total_flow_steps = 0
        self.total_arc_steps = 0

    def update(
        self,
        cond_num: float,
        n_neg: int,
        force_norm: float,
        step_accepted: bool,
    ) -> str:
        """Update state machine after a step.  Returns phase for NEXT step."""
        self.steps_in_current_phase += 1

        if self.state == PicArcState.ARC:
            self.total_arc_steps += 1

            if not step_accepted:
                self.consecutive_arc_rejects += 1
            else:
                self.consecutive_arc_rejects = 0

            # Check exit conditions
            should_exit = (
                cond_num > self.kappa_threshold
                or n_neg > self.n_neg_max_for_arc
                or force_norm > self.force_max_for_arc
                or self.consecutive_arc_rejects >= self.max_consecutive_arc_rejects
            )
            if should_exit:
                self.state = PicArcState.FLOW
                self.stability_counter = 0
                self.consecutive_arc_rejects = 0
                self.steps_in_current_phase = 0

        else:  # FLOW
            self.total_flow_steps += 1

            stable_now = (
                cond_num < self.kappa_threshold
                and n_neg <= self.n_neg_max_for_arc
                and force_norm < self.force_max_for_arc
                and step_accepted
            )
            if stable_now:
                self.stability_counter += 1
            else:
                self.stability_counter = 0

            if self.stability_counter >= self.stability_window:
                self.state = PicArcState.ARC
                self.stability_counter = 0
                self.consecutive_arc_rejects = 0
                self.steps_in_current_phase = 0

        return self.state

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "phase": self.state,
            "stability_counter": self.stability_counter,
            "consecutive_arc_rejects": self.consecutive_arc_rejects,
            "steps_in_current_phase": self.steps_in_current_phase,
            "total_flow_steps": self.total_flow_steps,
            "total_arc_steps": self.total_arc_steps,
        }


# ---------------------------------------------------------------------------
# Module 5: Multi-channel termination
# ---------------------------------------------------------------------------

class ConvergenceClass:
    STRICT = "STRICT"
    RELAXED = "RELAXED"
    STALLED = "STALLED"
    RUNNING = "RUNNING"


def classify_convergence(
    force_norm: float,
    n_neg: int,
    evals_vib: torch.Tensor,
    force_converged: float = 1e-4,
    relaxed_eval_threshold: float = 0.01,
) -> str:
    """Classify the convergence state of the current geometry.

    STRICT:   force < threshold AND n_neg == 0
    RELAXED:  force < threshold AND min eigenvalue >= -relaxed_eval_threshold
    RUNNING:  neither condition met
    STALLED:  set externally when max_steps reached
    """
    if force_norm < force_converged and n_neg == 0:
        return ConvergenceClass.STRICT

    if force_norm < force_converged:
        min_eval = float(evals_vib.min().item()) if evals_vib.numel() > 0 else 0.0
        if min_eval >= -relaxed_eval_threshold:
            return ConvergenceClass.RELAXED

    return ConvergenceClass.RUNNING


# ---------------------------------------------------------------------------
# Module 6: Main optimizer loop
# ---------------------------------------------------------------------------

def _build_converged_result(
    coords: torch.Tensor,
    step: int,
    energy: float,
    force_norm: float,
    n_neg: int,
    n_neg_total: int,
    min_vib_eval: float,
    gap_info: Dict[str, Any],
    cascade: Dict[str, int],
    evals_vib: torch.Tensor,
    log_spectrum_k: int,
    convergence_class: str,
    controller: PicArcController,
) -> Dict[str, Any]:
    ctrl = controller.get_diagnostics()
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
        "convergence_class": convergence_class,
        "total_flow_steps": ctrl["total_flow_steps"],
        "total_arc_steps": ctrl["total_arc_steps"],
    }


def _build_timeout_result(
    coords: torch.Tensor,
    trajectory: list,
    controller: PicArcController,
    convergence_class: str,
) -> Dict[str, Any]:
    last = trajectory[-1] if trajectory else {}
    ctrl = controller.get_diagnostics()
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
        "cascade_at_convergence": {
            k: last.get(k) for k in last if k.startswith("n_neg_at_")
        },
        "bottom_spectrum_at_convergence": last.get("bottom_spectrum", []),
        "convergence_class": convergence_class,
        "total_flow_steps": ctrl["total_flow_steps"],
        "total_arc_steps": ctrl["total_arc_steps"],
    }


def run_pic_arc(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    atomsymbols: list,
    *,
    n_steps: int = 10000,
    max_atom_disp: float = 1.3,
    force_converged: float = 1e-4,
    min_interatomic_dist: float = 0.5,
    project_gradient_and_v: bool = True,
    purify_hessian: bool = False,
    log_spectrum_k: int = 10,
    # Trust region
    trust_radius_init: float = 0.5,
    trust_radius_floor: float = 0.01,
    # Metric
    k_bond: float = 0.45,
    bond_threshold_factor: float = 1.3,
    metric_regularization: float = 1e-3,
    metric_refresh_every: int = 0,
    # ARC
    sigma_init: float = 1.0,
    sigma_min: float = 0.01,
    sigma_max: float = 100.0,
    max_neg_modes_in_subspace: int = 5,
    # State machine
    kappa_threshold: float = 1e6,
    n_neg_max_for_arc: int = 5,
    force_max_for_arc: float = 1.0,
    stability_window: int = 3,
    max_consecutive_arc_rejects: int = 3,
    # Termination
    relaxed_eval_threshold: float = 0.01,
    accept_relaxed: bool = False,
    # Diagnostics
    known_ts_coords: Optional[torch.Tensor] = None,
    known_reactant_coords: Optional[torch.Tensor] = None,
    known_product_coords: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, Any], list]:
    """PIC-ARC optimizer: two-phase geometry minimizer.

    Phase A (FLOW): preconditioned steepest descent + trust region.
    Phase B (ARC):  subspace cubic regularization + trust region.

    Returns:
        result_dict: Summary with converged, final_energy, etc.
        trajectory: Per-step diagnostic list.
    """
    # --- Setup ---
    coords = coords0.detach().clone().to(torch.float32)
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    n_atoms = coords.shape[0]

    coords0_flat = coords.reshape(-1, 3).clone()

    # Reshape reference geometries
    def _reshape_ref(t):
        if t is None:
            return None
        t = t.detach().clone().to(torch.float32)
        if t.dim() == 3 and t.shape[0] == 1:
            t = t[0]
        return t.reshape(-1, 3)

    known_ts_coords = _reshape_ref(known_ts_coords)
    known_reactant_coords = _reshape_ref(known_reactant_coords)
    known_product_coords = _reshape_ref(known_product_coords)

    # Build molecule-aware metric
    M = build_metric(
        coords, atomic_nums,
        k_bond=k_bond,
        bond_threshold_factor=bond_threshold_factor,
        regularization=metric_regularization,
    )
    M_chol = torch.linalg.cholesky(M)

    # Initialize state machine
    controller = PicArcController(
        kappa_threshold=kappa_threshold,
        n_neg_max_for_arc=n_neg_max_for_arc,
        force_max_for_arc=force_max_for_arc,
        stability_window=stability_window,
        max_consecutive_arc_rejects=max_consecutive_arc_rejects,
    )

    # Trust region & ARC state
    current_trust_radius = trust_radius_init
    sigma = sigma_init

    trajectory: List[Dict[str, Any]] = []

    # Initial prediction
    out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)

    for step in range(n_steps):
        # ----- Extract predictions -----
        energy = _to_float(out["energy"])
        forces = out["forces"]
        hessian = out["hessian"]

        if isinstance(forces, torch.Tensor):
            if forces.dim() == 3 and forces.shape[0] == 1:
                forces = forces[0]
            forces = forces.reshape(-1, 3)

        force_norm = _force_mean(forces)

        # ----- Vibrational eigendecomposition -----
        evals_vib, evecs_vib_3N, _ = get_vib_evals_evecs(
            hessian, coords, atomsymbols, purify_hessian=purify_hessian,
        )

        # ----- Diagnostics -----
        cascade = _cascade_n_neg(evals_vib)
        n_neg = cascade["n_neg_at_0.0"]
        gap_info = _spectral_gap_info(evals_vib)
        n_neg_total = _total_hessian_n_neg(hessian)
        band_pops = _eigenvalue_band_populations(evals_vib)

        min_vib_eval = float(evals_vib.min().item()) if evals_vib.numel() > 0 else float("nan")
        max_vib_eval = float(evals_vib.max().item()) if evals_vib.numel() > 0 else float("nan")

        abs_evals = torch.abs(evals_vib) if evals_vib.numel() > 0 else torch.tensor([1.0])
        min_abs_vib = float(abs_evals.min().item())
        max_abs_vib = float(abs_evals.max().item())
        cond_num = max_abs_vib / max(min_abs_vib, 1e-30)

        # Displacement diagnostics
        disp_from_start = (coords - coords0_flat).norm(dim=1)
        disp_from_start_max = float(disp_from_start.max().item())
        disp_from_start_rmsd = float(disp_from_start.pow(2).mean().sqrt().item())

        # ----- Convergence classification -----
        convergence_class = classify_convergence(
            force_norm, n_neg, evals_vib,
            force_converged=force_converged,
            relaxed_eval_threshold=relaxed_eval_threshold,
        )

        # ----- Build step record -----
        phase = controller.state
        step_record: Dict[str, Any] = {
            "step": step,
            "energy": energy,
            "force_norm": force_norm,
            "n_neg_evals": n_neg,
            "n_neg_total_hessian": n_neg_total,
            "min_vib_eval": min_vib_eval,
            "max_vib_eval": max_vib_eval,
            "cond_num": cond_num,
            "min_dist": _min_interatomic_distance(coords),
            "trust_radius": current_trust_radius,
            "sigma": sigma,
            "phase": phase,
            "convergence_class": convergence_class,
            "disp_from_start_max": disp_from_start_max,
            "disp_from_start_rmsd": disp_from_start_rmsd,
            **gap_info,
            **cascade,
            **band_pops,
        }

        # Distance-to-reference diagnostics
        if known_ts_coords is not None:
            ts_disp = (coords - known_ts_coords).norm(dim=1)
            step_record["dist_to_ts_max"] = float(ts_disp.max().item())
            step_record["dist_to_ts_rmsd"] = float(ts_disp.pow(2).mean().sqrt().item())
        if known_reactant_coords is not None:
            r_disp = (coords - known_reactant_coords).norm(dim=1)
            step_record["dist_to_reactant_max"] = float(r_disp.max().item())
            step_record["dist_to_reactant_rmsd"] = float(r_disp.pow(2).mean().sqrt().item())

        if log_spectrum_k > 0:
            step_record["bottom_spectrum"] = _bottom_k_spectrum(evals_vib, log_spectrum_k)

        trajectory.append(step_record)

        # ----- Convergence check -----
        if convergence_class == ConvergenceClass.STRICT:
            return _build_converged_result(
                coords, step, energy, force_norm, n_neg, n_neg_total,
                min_vib_eval, gap_info, cascade, evals_vib, log_spectrum_k,
                convergence_class, controller,
            ), trajectory

        if accept_relaxed and convergence_class == ConvergenceClass.RELAXED:
            return _build_converged_result(
                coords, step, energy, force_norm, n_neg, n_neg_total,
                min_vib_eval, gap_info, cascade, evals_vib, log_spectrum_k,
                convergence_class, controller,
            ), trajectory

        # ----- Optional metric refresh -----
        if metric_refresh_every > 0 and step > 0 and step % metric_refresh_every == 0:
            M = build_metric(
                coords, atomic_nums,
                k_bond=k_bond,
                bond_threshold_factor=bond_threshold_factor,
                regularization=metric_regularization,
            )
            M_chol = torch.linalg.cholesky(M)
            step_record["metric_refreshed"] = True

        # ----- Build gradient -----
        if project_gradient_and_v:
            grad = -project_vector_to_vibrational_torch(
                forces.reshape(-1), coords, atomsymbols,
            )
        else:
            grad = -forces.reshape(-1)

        # ----- Compute step -----
        if phase == PicArcState.FLOW:
            raw_step = flow_step(grad, M_chol)
            step_record["step_mode"] = "flow"
        else:
            # ARC phase: need full Hessian in (3N, 3N)
            H_flat = hessian.detach().reshape(3 * n_atoms, 3 * n_atoms)
            H_flat = 0.5 * (H_flat + H_flat.T)

            raw_step, arc_info = arc_subproblem(
                grad, H_flat, evals_vib, evecs_vib_3N, M_chol, sigma,
                max_neg_modes_in_subspace=max_neg_modes_in_subspace,
            )
            step_record["step_mode"] = "arc"
            step_record["arc_info"] = arc_info

        # ----- Trust region acceptance -----
        accepted = False
        max_retries = 10
        retries = 0
        rho = 0.0
        accepted_by = "energy"

        # Pre-initialize for the possibly-zero-iteration case (should not happen
        # since max_retries >= 1, but keeps the type checker happy)
        capped_disp = _cap_displacement(raw_step.reshape(-1, 3), current_trust_radius)
        new_coords = coords + capped_disp
        out_new = out

        # Prepare Hessian for predicted dE computation
        H_flat_pred = hessian.detach().reshape(3 * n_atoms, 3 * n_atoms).to(torch.float64)
        H_flat_pred = 0.5 * (H_flat_pred + H_flat_pred.T)

        while not accepted and retries < max_retries:
            capped_disp = _cap_displacement(raw_step.reshape(-1, 3), current_trust_radius)

            # Predicted energy change (quadratic model)
            dx_flat = capped_disp.reshape(-1).to(torch.float64)
            grad_f64 = grad.to(torch.float64)
            pred_dE = float(
                (grad_f64.dot(dx_flat) + 0.5 * dx_flat @ H_flat_pred @ dx_flat).item()
            )

            new_coords = coords + capped_disp

            # Safety: check interatomic distances
            if _min_interatomic_distance(new_coords) < min_interatomic_dist:
                current_trust_radius = max(current_trust_radius * 0.5, trust_radius_floor)
                retries += 1
                continue

            # Evaluate
            out_new = predict_fn(new_coords, atomic_nums, do_hessian=True, require_grad=False)
            energy_new = _to_float(out_new["energy"])
            actual_dE = energy_new - energy

            # Primary acceptance: energy decreased
            if actual_dE <= 1e-5:
                accepted = True
                accepted_by = "energy"
            else:
                # Chemistry-aware acceptance: force norm dropped > 50%
                new_forces = out_new["forces"]
                if isinstance(new_forces, torch.Tensor):
                    if new_forces.dim() == 3 and new_forces.shape[0] == 1:
                        new_forces = new_forces[0]
                new_force_norm = _force_mean(new_forces.reshape(-1, 3))
                if new_force_norm < 0.5 * force_norm and force_norm > 0:
                    accepted = True
                    accepted_by = "force_drop"

            if accepted:
                rho = actual_dE / pred_dE if abs(pred_dE) > 1e-10 else 0.0

                # Update trust radius
                if rho > 0.75:
                    current_trust_radius = min(current_trust_radius * 1.5, max_atom_disp)
                elif rho < 0.0:
                    current_trust_radius = max(current_trust_radius * 0.25, trust_radius_floor)
                elif rho < 0.25:
                    current_trust_radius = max(current_trust_radius * 0.5, trust_radius_floor)

                # ARC: update sigma
                if phase == PicArcState.ARC:
                    if rho > 0.75:
                        sigma = max(sigma / 2.0, sigma_min)
                    elif rho < 0.25:
                        sigma = min(sigma * 2.0, sigma_max)

                coords = new_coords.detach()
                out = out_new
            else:
                current_trust_radius = max(current_trust_radius * 0.25, trust_radius_floor)
                retries += 1

        if not accepted:
            # Last resort: take the smallest step anyway
            coords = new_coords.detach()
            out = out_new

        # Log step outcome
        actual_disp = float(capped_disp.reshape(-1, 3).norm(dim=1).max().item())
        step_record["actual_step_disp"] = actual_disp
        step_record["step_accepted"] = accepted
        step_record["accepted_by"] = accepted_by if accepted else "forced"
        step_record["retries"] = retries
        step_record["rho"] = rho

        # ----- Update state machine -----
        controller.update(cond_num, n_neg, force_norm, accepted)

    # ----- Timeout -----
    last = trajectory[-1] if trajectory else {}
    final_class = last.get("convergence_class", ConvergenceClass.RUNNING)
    if final_class == ConvergenceClass.RUNNING:
        final_class = ConvergenceClass.STALLED

    return _build_timeout_result(coords, trajectory, controller, final_class), trajectory
