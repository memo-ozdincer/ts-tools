from __future__ import annotations

"""Multi-mode escape experiment for GAD with ECKART-PROJECTED MASS-WEIGHTED Hessian.

This is a variant of multi_mode.py that uses the Eckart-projected, mass-weighted
Hessian for ALL eigenvector computations. This stabilizes eigenvectors by:
1. Removing translation/rotation null modes via Eckart projection
2. Mass-weighting ensures physically meaningful vibrational modes

Key difference from multi_mode.py:
- GAD direction uses lowest eigenvector of PROJECTED Hessian (not raw)
- Escape v2 direction uses second eigenvector of PROJECTED Hessian
- Vibrational eigenvalue analysis uses projected Hessian (already the case)

For HIP: Uses project_hessian_remove_rigid_modes (3N x 3N, 6 near-zero eigenvalues)
For SCINE: Uses scine_project_hessian_full (3N x 3N, 6 near-zero eigenvalues)

Detection uses DISPLACEMENT-based criteria (not GAD norm), since:
- GAD norm ~ force_mean (0.2-3 eV/Å) stays moderate even at plateaus
- disp_from_last dropping to ~1µÅ is the true indicator (dt shrinks/caps)
- We trigger escape when: mean(disp[-window:]) < threshold AND neg_vib stable AND >1

Algorithm:
1. Run GAD using projected Hessian for eigenvector computation
2. Detect plateau: mean displacement < threshold over window, stable neg_vib, index > 1
3. If index = 1: Success - found TS
4. If plateau at index > 1: Apply perturbation along v2 (from projected Hessian)
5. Resume GAD from perturbed geometry
6. Repeat until index = 1 or max escape cycles reached
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Note: We use gad_euler_step_projected defined below, not from core_algos
from ..dependencies.common_utils import add_common_args, parse_starting_geometry, setup_experiment
from ..dependencies.experiment_logger import ExperimentLogger, RunResult, build_loss_type_flags
from ..dependencies.hessian import (
    vibrational_eigvals,
    get_scine_elements_from_predict_output,
    project_hessian_remove_rigid_modes,
    prepare_hessian,
)
from ..logging import finish_wandb, init_wandb_run, log_sample, log_summary
from ..logging.plotly_utils import plot_gad_trajectory_interactive
from ..runners._predict import make_predict_fn_from_calculator

# SCINE projection (may not be available)
try:
    from ..dependencies.scine_masses import (
        ScineFrequencyAnalyzer,
        get_scine_masses,
    )
    SCINE_PROJECTION_AVAILABLE = True
except ImportError:
    SCINE_PROJECTION_AVAILABLE = False


def _sanitize_wandb_name(s: str) -> str:
    s = str(s)
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s[:128] if len(s) > 128 else s


def _auto_wandb_name(*, script: str, loss_type_flags: str, args: argparse.Namespace) -> str:
    calculator = getattr(args, "calculator", "hip")
    start_from = getattr(args, "start_from", "unknown")
    method = getattr(args, "method", "euler")
    escape_delta = getattr(args, "escape_delta", None)
    n_steps = getattr(args, "n_steps", None)
    noise_seed = getattr(args, "noise_seed", None)
    job_id = os.environ.get("SLURM_JOB_ID")

    parts = [
        script,
        str(calculator),
        str(start_from),
        str(method),
        f"delta{escape_delta}" if escape_delta is not None else None,
        f"steps{n_steps}" if n_steps is not None else None,
        f"seed{noise_seed}" if noise_seed is not None else None,
        f"job{job_id}" if job_id else None,
        str(loss_type_flags),
    ]
    parts = [p for p in parts if p]
    return _sanitize_wandb_name("__".join(parts))


def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().reshape(-1)[0].item())
    return float(x)


def _force_mean(forces: torch.Tensor) -> float:
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    f = forces.reshape(-1, 3)
    return float(f.norm(dim=1).mean().item())


def _mean_atom_norm(x: torch.Tensor) -> float:
    if x.dim() == 3 and x.shape[0] == 1:
        x = x[0]
    return float(x.reshape(-1, 3).norm(dim=1).mean().item())


def _max_atom_norm(x: torch.Tensor) -> float:
    if x.dim() == 3 and x.shape[0] == 1:
        x = x[0]
    return float(x.reshape(-1, 3).norm(dim=1).max().item())


def _vib_mask_from_evals(evals: torch.Tensor, *, tr_threshold: float) -> torch.Tensor:
    """Mask out translation/rotation (near-zero) modes."""
    if tr_threshold <= 0:
        return torch.ones_like(evals, dtype=torch.bool)
    return evals.abs() > float(tr_threshold)


def _step_metrics_from_projected_hessian(
    *,
    forces: torch.Tensor,
    hessian_proj: torch.Tensor,
    tr_threshold: float,
    eigh_device: str,
) -> tuple[torch.Tensor, float, float, float, int]:
    """Compute GAD vec + eigen metrics from ONE eigendecomp of projected Hessian."""
    forces = forces[0] if forces.dim() == 3 and forces.shape[0] == 1 else forces
    forces = forces.reshape(-1, 3)
    num_atoms = int(forces.shape[0])

    hess = hessian_proj
    if hess.dim() != 2 or hess.shape[0] != 3 * num_atoms:
        hess = prepare_hessian(hess, num_atoms)

    if str(eigh_device).lower() == "cpu":
        hess_eigh = hess.detach().to(device=torch.device("cpu"))
    else:
        hess_eigh = hess

    evals, evecs = torch.linalg.eigh(hess_eigh)
    evals = evals.to(device=forces.device, dtype=torch.float32)

    vib_mask = _vib_mask_from_evals(evals, tr_threshold=tr_threshold)
    vib_indices = torch.where(vib_mask)[0]
    if int(vib_indices.numel()) == 0:
        first_vib_idx = 0
        v = evecs[:, 0]
        evals_vib = evals
    else:
        first_vib_idx = int(vib_indices[0].item())
        v = evecs[:, first_vib_idx]
        evals_vib = evals[vib_mask]

    v = v.to(device=forces.device, dtype=forces.dtype)
    v = v / (v.norm() + 1e-12)

    f_flat = forces.reshape(-1)
    gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
    gad_vec = gad_flat.view(num_atoms, 3)

    if int(evals_vib.numel()) >= 2:
        eig0 = float(evals_vib[0].item())
        eig1 = float(evals_vib[1].item())
        eig_product = float((evals_vib[0] * evals_vib[1]).item())
    else:
        eig0, eig1, eig_product = float("nan"), float("nan"), float("inf")

    neg_vib = int((evals_vib < -float(tr_threshold)).sum().item()) if int(evals_vib.numel()) > 0 else -1

    return gad_vec, eig0, eig1, eig_product, neg_vib


# ============================================================================
# Eckart-projected, mass-weighted Hessian functions
# ============================================================================

def get_projected_hessian(
    hessian_raw: torch.Tensor,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    scine_elements: Optional[list] = None,
) -> torch.Tensor:
    """Get Eckart-projected, mass-weighted Hessian (3N x 3N).

    For HIP: Uses project_hessian_remove_rigid_modes (differentiable)
    For SCINE: Uses full 3N x 3N projection (non-differentiable, NumPy backend)

    Both return 3N x 3N Hessians with 6 near-zero eigenvalues for TR modes.

    Args:
        hessian_raw: Raw Hessian tensor
        coords: Atomic coordinates (N, 3)
        atomic_nums: Atomic numbers
        scine_elements: If provided, uses SCINE-specific mass-weighting

    Returns:
        Projected Hessian (3N, 3N) with TR modes zeroed out
    """
    if scine_elements is not None:
        # SCINE path: compute full 3N x 3N projected Hessian
        if not SCINE_PROJECTION_AVAILABLE:
            raise RuntimeError("SCINE projection requested but scine_masses not available")
        return _scine_project_hessian_full(hessian_raw, coords, scine_elements)

    # HIP path: use existing projection
    return project_hessian_remove_rigid_modes(hessian_raw, coords, atomic_nums)


def _scine_project_hessian_full(
    hessian_raw: torch.Tensor,
    coords: torch.Tensor,
    elements: list,
) -> torch.Tensor:
    """SCINE Eckart projection returning full 3N x 3N Hessian.

    Unlike scine_project_hessian_remove_rigid_modes which returns (3N-6, 3N-6),
    this returns (3N, 3N) with 6 near-zero eigenvalues for TR modes.
    This allows eigenvectors to be used directly in 3N space (like HIP).
    """
    import numpy as np
    from scipy.linalg import eigh

    # Convert to numpy
    hess_np = hessian_raw.detach().cpu().numpy()
    coords_np = coords.detach().cpu().numpy().reshape(-1, 3)
    n_atoms = len(elements)

    # Get masses
    masses_amu = get_scine_masses(elements)
    m_sqrt = np.sqrt(masses_amu)
    m_sqrt_3n = np.repeat(m_sqrt, 3)

    # Mass-weight Hessian: M^{-1/2} H M^{-1/2}
    inv_m_sqrt_mat = np.outer(1.0 / m_sqrt_3n, 1.0 / m_sqrt_3n)
    hess_np_2d = hess_np.reshape(3 * n_atoms, 3 * n_atoms)
    H_mw = hess_np_2d * inv_m_sqrt_mat

    # Build full 3N x 3N projector using ScineFrequencyAnalyzer's method
    analyzer = ScineFrequencyAnalyzer()
    P_reduced = analyzer._get_vibrational_projector(coords_np, masses_amu)  # (3N-k, 3N)

    # P_full = I - P_reduced.T @ P_reduced would give us the TR space projector
    # We want the vibrational projector: P_vib = P_reduced.T @ P_reduced
    # Then H_proj = P_vib @ H_mw @ P_vib
    P_full = P_reduced.T @ P_reduced  # (3N, 3N)

    # Project
    H_proj = P_full @ H_mw @ P_full
    H_proj = 0.5 * (H_proj + H_proj.T)  # Symmetrize

    return torch.from_numpy(H_proj).to(device=hessian_raw.device, dtype=hessian_raw.dtype)


def compute_gad_vector_projected(
    forces: torch.Tensor,
    hessian_proj: torch.Tensor,
) -> torch.Tensor:
    """Compute GAD direction using PROJECTED Hessian.

    The projected Hessian has 6 near-zero eigenvalues (TR modes).
    We skip these and use the first vibrational mode (most negative eigenvalue
    among the non-TR modes, or smallest positive if at minimum).

    Args:
        forces: Forces tensor (N, 3) or (1, N, 3)
        hessian_proj: Projected mass-weighted Hessian (3N, 3N)

    Returns:
        GAD vector in Cartesian space (N, 3)
    """
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    forces = forces.reshape(-1, 3)
    num_atoms = int(forces.shape[0])

    hess = hessian_proj
    if hess.dim() != 2 or hess.shape[0] != 3 * num_atoms:
        hess = prepare_hessian(hess, num_atoms)

    # Get eigenvalues and eigenvectors
    evals, evecs = torch.linalg.eigh(hess)

    # Skip near-zero TR modes (eigenvalue magnitude < threshold)
    tr_threshold = 1e-6
    vib_mask = torch.abs(evals) > tr_threshold

    if not vib_mask.any():
        # All eigenvalues are near-zero (shouldn't happen for valid molecule)
        # Fall back to using first eigenvector
        v = evecs[:, 0].to(forces.dtype)
    else:
        # Find first vibrational mode (smallest eigenvalue among non-TR modes)
        # evals are sorted ascending, so we want the first one above threshold
        vib_indices = torch.where(vib_mask)[0]
        first_vib_idx = vib_indices[0]
        v = evecs[:, first_vib_idx].to(forces.dtype)

    v = v / (v.norm() + 1e-12)

    # GAD formula: F_gad = F + 2 * (F . v) * v
    # (inverting force along lowest mode)
    f_flat = forces.reshape(-1)
    gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v

    return gad_flat.view(num_atoms, 3)


def gad_euler_step_projected(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    dt: float,
    out: Optional[Dict[str, Any]] = None,
    scine_elements: Optional[list] = None,
) -> Dict[str, Any]:
    """GAD Euler step using PROJECTED Hessian for eigenvector computation.

    This is the Eckart-MW variant of gad_euler_step.
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)

    if out is None:
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)

    forces = out["forces"]
    hessian = out["hessian"]

    # Get projected Hessian
    hessian_proj = get_projected_hessian(hessian, coords, atomic_nums, scine_elements)

    # Compute GAD vector using projected Hessian
    gad_vec = compute_gad_vector_projected(forces, hessian_proj)
    new_coords = coords + dt * gad_vec

    return {
        "new_coords": new_coords,
        "gad_vec": gad_vec,
        "out": out,
        "hessian_proj": hessian_proj,
    }


def _prepare_hessian(hess: torch.Tensor, num_atoms: int) -> torch.Tensor:
    """Reshape hessian to (3*N, 3*N) matrix."""
    if hess.dim() == 1:
        side = int(hess.numel() ** 0.5)
        return hess.view(side, side)
    if hess.dim() == 3 and hess.shape[0] == 1:
        hess = hess[0]
    if hess.dim() > 2:
        return hess.reshape(3 * num_atoms, 3 * num_atoms)
    return hess


def _min_interatomic_distance(coords: torch.Tensor) -> float:
    """Compute minimum pairwise distance between atoms."""
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    n = coords.shape[0]
    if n < 2:
        return float("inf")

    # Compute pairwise distances
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 3)
    dist = diff.norm(dim=2)  # (N, N)

    # Set diagonal to inf to ignore self-distances
    dist = dist + torch.eye(n, device=coords.device, dtype=coords.dtype) * 1e10

    return float(dist.min().item())


def _geometry_is_valid(coords: torch.Tensor, min_dist_threshold: float) -> bool:
    """Check if geometry has all atom pairs above minimum distance threshold."""
    return _min_interatomic_distance(coords) >= min_dist_threshold


def perform_escape_perturbation(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    hessian: torch.Tensor,
    *,
    escape_delta: float,
    adaptive_delta: bool = True,
    min_interatomic_dist: float = 0.5,
    delta_shrink_factor: float = 0.5,
    max_shrink_attempts: int = 5,
    scine_elements: Optional[list] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Perturb geometry along v2 (second-lowest VIBRATIONAL eigenvector) to escape high-index saddle.

    Uses PROJECTED Hessian to find v2, skipping translation/rotation modes.
    Tries both +delta and -delta directions, picks the one with lower energy.
    Validates that new geometry doesn't have atoms too close together.
    If both directions create invalid geometries, tries progressively smaller deltas.

    Args:
        predict_fn: Energy/forces prediction function
        coords: Current coordinates (N, 3)
        atomic_nums: Atomic numbers
        hessian: Raw Hessian matrix (will be projected internally)
        escape_delta: Base displacement magnitude in Angstrom
        adaptive_delta: If True, scale delta by 1/sqrt(|lambda2|)
        min_interatomic_dist: Minimum allowed distance between atoms (A)
        delta_shrink_factor: Factor to reduce delta when geometry is invalid
        max_shrink_attempts: Max number of times to try smaller delta
        scine_elements: SCINE element types (if using SCINE calculator)

    Returns:
        new_coords: Perturbed coordinates (or original if all attempts fail)
        info: Dict with escape details (delta_used, direction, energy_change, etc.)
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    num_atoms = int(coords.shape[0])

    # Get PROJECTED Hessian (Eckart + mass-weighted)
    hess_proj = get_projected_hessian(hessian, coords, atomic_nums, scine_elements)

    # Get eigenvalues and eigenvectors from PROJECTED Hessian
    evals, evecs = torch.linalg.eigh(hess_proj)

    # Skip near-zero TR modes and get second vibrational mode (v2)
    tr_threshold = 1e-6
    vib_mask = torch.abs(evals) > tr_threshold
    vib_indices = torch.where(vib_mask)[0]

    if len(vib_indices) < 2:
        # Not enough vibrational modes, fall back to second eigenvector
        v2 = evecs[:, 1]
        lambda2 = float(evals[1].item())
    else:
        # Second vibrational mode (skip TR modes)
        second_vib_idx = vib_indices[1]
        v2 = evecs[:, second_vib_idx]
        lambda2 = float(evals[second_vib_idx].item())

    v2 = v2 / (v2.norm() + 1e-12)  # Normalize

    # Adaptive delta scaling based on curvature
    base_delta = float(escape_delta)
    if adaptive_delta and lambda2 < -0.01:
        # Scale larger for strong negative curvature
        base_delta = float(escape_delta) / np.sqrt(abs(lambda2))
        base_delta = min(base_delta, 1.0)  # Cap at 1 Angstrom

    # Reshape v2 to (N, 3)
    v2_3d = v2.reshape(num_atoms, 3)

    # Get current energy once
    E_current = _to_float(predict_fn(coords, atomic_nums, do_hessian=False, require_grad=False)["energy"])

    # Try progressively smaller deltas if geometry becomes invalid
    delta = base_delta
    for shrink_attempt in range(max_shrink_attempts + 1):
        # Try both directions
        coords_plus = coords + delta * v2_3d
        coords_minus = coords - delta * v2_3d

        plus_valid = _geometry_is_valid(coords_plus, min_interatomic_dist)
        minus_valid = _geometry_is_valid(coords_minus, min_interatomic_dist)

        if not plus_valid and not minus_valid:
            # Both invalid, try smaller delta
            delta = delta * delta_shrink_factor
            continue

        # At least one direction is valid, evaluate energies for valid ones
        candidates = []

        if plus_valid:
            try:
                out_plus = predict_fn(coords_plus, atomic_nums, do_hessian=False, require_grad=False)
                E_plus = _to_float(out_plus["energy"])
                if np.isfinite(E_plus):
                    candidates.append((coords_plus, +1, E_plus))
            except Exception:
                pass  # Model error, skip this direction

        if minus_valid:
            try:
                out_minus = predict_fn(coords_minus, atomic_nums, do_hessian=False, require_grad=False)
                E_minus = _to_float(out_minus["energy"])
                if np.isfinite(E_minus):
                    candidates.append((coords_minus, -1, E_minus))
            except Exception:
                pass  # Model error, skip this direction

        if candidates:
            # Pick lowest energy valid candidate
            candidates.sort(key=lambda x: x[2])
            new_coords, direction, energy_after = candidates[0]

            disp_per_atom = float(v2_3d.norm(dim=1).mean().item()) * delta
            min_dist_after = _min_interatomic_distance(new_coords)

            info = {
                "delta_used": delta,
                "delta_base": base_delta,
                "shrink_attempts": shrink_attempt,
                "direction": direction,
                "lambda2": lambda2,
                "energy_before": E_current,
                "energy_after": energy_after,
                "energy_change": energy_after - E_current,
                "disp_per_atom": disp_per_atom,
                "min_dist_after": min_dist_after,
                "escape_success": True,
            }

            return new_coords.detach(), info

        # Both directions valid but model errored, try smaller delta
        delta = delta * delta_shrink_factor

    # All attempts failed - return original coords with failure info
    info = {
        "delta_used": 0.0,
        "delta_base": base_delta,
        "shrink_attempts": max_shrink_attempts + 1,
        "direction": 0,
        "lambda2": lambda2,
        "energy_before": E_current,
        "energy_after": E_current,
        "energy_change": 0.0,
        "disp_per_atom": 0.0,
        "min_dist_after": _min_interatomic_distance(coords),
        "escape_success": False,
    }

    return coords.detach(), info


def _check_plateau_convergence(
    disp_history: list[float],
    neg_vib_history: list[int],
    current_neg_vib: int,
    *,
    window: int,
    disp_threshold: float,
    neg_vib_std_threshold: float,
) -> bool:
    """Check if we've converged to a plateau based on displacement history.

    Triggers when:
    1. mean(disp[-window:]) < disp_threshold (tiny steps)
    2. std(neg_vib[-window:]) <= neg_vib_std_threshold (stable saddle index)
    3. current_neg_vib > 1 (high-index saddle)
    """
    if len(disp_history) < window:
        return False

    recent_disp = disp_history[-window:]
    recent_neg_vib = neg_vib_history[-window:]

    mean_disp = float(np.mean(recent_disp))
    std_neg_vib = float(np.std(recent_neg_vib))

    return (
        mean_disp < disp_threshold
        and std_neg_vib <= neg_vib_std_threshold
        and current_neg_vib > 1
    )


def run_multi_mode_escape(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    n_steps: int,
    dt: float,
    stop_at_ts: bool,
    ts_eps: float,
    dt_control: str,
    dt_min: float,
    dt_max: float,
    max_atom_disp: Optional[float],
    plateau_patience: int,
    plateau_boost: float,
    plateau_shrink: float,
    # Multi-mode escape parameters (displacement-based detection)
    escape_disp_threshold: float,
    escape_window: int,
    hip_vib_mode: str = "projected",
    hip_rigid_tol: float = 1e-6,
    hip_eigh_device: str = "auto",
    escape_neg_vib_std: float,
    escape_delta: float,
    adaptive_delta: bool,
    min_interatomic_dist: float,
    max_escape_cycles: int,
    profile_every: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run GAD with multi-mode escape mechanism.

    Uses displacement-based plateau detection:
    - Triggers escape when mean(disp[-window:]) < threshold AND neg_vib stable AND >1
    - This is more robust than GAD norm, which stays ~0.1-1 eV/Å even at plateaus

    The algorithm:
    1. Run GAD, accumulating displacement history
    2. Check for plateau: tiny displacements + stable neg_vib + index > 1
    3. If plateau at index > 1: perturb along v2 and restart GAD
    4. Repeat until index = 1 or max_escape_cycles reached
    """
    coords = coords0.detach().clone().to(torch.float32)

    trajectory = {k: [] for k in [
        "energy",
        "force_mean",
        "eig0",
        "eig1",
        "eig_product",
        "neg_vib",
        "disp_from_last",
        "disp_from_start",
        "dt_eff",
        "gad_norm",
        "escape_cycle",  # Track which escape cycle we're in
    ]}

    start_pos = coords.clone()
    prev_pos = coords.clone()

    steps_to_ts: Optional[int] = None
    total_steps = 0
    escape_cycle = 0
    escape_events: list[Dict[str, Any]] = []

    # Rolling history for displacement-based plateau detection
    disp_history: list[float] = []
    neg_vib_history: list[int] = []

    # Stateful dt controller variables
    dt_eff_state = float(dt)
    best_neg_vib: Optional[int] = None
    no_improve = 0

    while escape_cycle < max_escape_cycles and total_steps < n_steps:
        t_predict0 = time.time() if profile_every and (total_steps % profile_every == 0) else None
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        t_predict1 = time.time() if t_predict0 is not None else None
        energy = out.get("energy")
        forces = out.get("forces")
        hessian = out.get("hessian")

        energy_value = _to_float(energy)
        force_mean = _force_mean(forces)

        scine_elements = get_scine_elements_from_predict_output(out)

        t_eigs0 = time.time() if t_predict0 is not None else None
        if scine_elements is None and hip_vib_mode == "proj_tol":
            hess_proj = get_projected_hessian(hessian, coords, atomic_nums, scine_elements=None)
            gad_vec, eig0, eig1, eig_prod, neg_vib = _step_metrics_from_projected_hessian(
                forces=forces,
                hessian_proj=hess_proj,
                tr_threshold=hip_rigid_tol,
                eigh_device=hip_eigh_device,
            )
        else:
            vib = vibrational_eigvals(hessian, coords, atomic_nums, scine_elements=scine_elements)
            eig0 = float(vib[0].item()) if vib.numel() >= 1 else float("nan")
            eig1 = float(vib[1].item()) if vib.numel() >= 2 else float("nan")
            eig_prod = float((vib[0] * vib[1]).item()) if vib.numel() >= 2 else float("inf")
            neg_vib = int((vib < 0).sum().item()) if vib.numel() > 0 else -1

            # Compute GAD vector using PROJECTED Hessian (Eckart-MW)
            step_out = gad_euler_step_projected(
                predict_fn, coords, atomic_nums, dt=0.0, out=out, scine_elements=scine_elements
            )
            gad_vec = step_out["gad_vec"]

        disp_from_last = float((coords - prev_pos).norm(dim=1).mean().item()) if total_steps > 0 else 0.0
        disp_from_start = float((coords - start_pos).norm(dim=1).mean().item()) if total_steps > 0 else 0.0

        gad_norm = _mean_atom_norm(gad_vec)

        if t_predict0 is not None and t_predict1 is not None and t_eigs0 is not None:
            t_eigs1 = time.time()
            print(
                f"[profile] step={total_steps} predict={t_predict1 - t_predict0:.4f}s eigs+gad={t_eigs1 - t_eigs0:.4f}s "
                f"mode={hip_vib_mode} eigh_device={hip_eigh_device}"
            )

        # Update rolling history
        if total_steps > 0:  # Only add after first step (disp_from_last=0 at step 0)
            disp_history.append(disp_from_last)
            neg_vib_history.append(neg_vib)

        trajectory["energy"].append(energy_value)
        trajectory["force_mean"].append(force_mean)
        trajectory["eig0"].append(eig0)
        trajectory["eig1"].append(eig1)
        trajectory["eig_product"].append(eig_prod)
        trajectory["neg_vib"].append(int(neg_vib))
        trajectory["disp_from_last"].append(disp_from_last)
        trajectory["disp_from_start"].append(disp_from_start)
        trajectory["gad_norm"].append(gad_norm)
        trajectory["escape_cycle"].append(escape_cycle)

        # Check for TS (index = 1)
        if stop_at_ts and steps_to_ts is None and np.isfinite(eig_prod) and eig_prod < -abs(ts_eps):
            steps_to_ts = total_steps
            trajectory["dt_eff"].append(float("nan"))
            break

        # Check for plateau convergence (displacement-based)
        is_plateau = _check_plateau_convergence(
            disp_history,
            neg_vib_history,
            neg_vib,
            window=escape_window,
            disp_threshold=escape_disp_threshold,
            neg_vib_std_threshold=escape_neg_vib_std,
        )

        if is_plateau:
            # Converged to high-index saddle, perform escape
            trajectory["dt_eff"].append(float("nan"))

            new_coords, escape_info = perform_escape_perturbation(
                predict_fn,
                coords,
                atomic_nums,
                hessian,
                escape_delta=escape_delta,
                adaptive_delta=adaptive_delta,
                min_interatomic_dist=min_interatomic_dist,
                scine_elements=scine_elements,
            )
            coords = new_coords
            escape_info["step"] = total_steps
            escape_info["neg_vib_before"] = neg_vib
            escape_info["gad_norm"] = gad_norm
            escape_info["mean_disp_at_trigger"] = float(np.mean(disp_history[-escape_window:]))
            escape_events.append(escape_info)

            # Reset state after escape
            disp_history.clear()
            neg_vib_history.clear()
            best_neg_vib = None
            no_improve = 0
            dt_eff_state = float(dt)
            prev_pos = coords.clone()

            escape_cycle += 1
            total_steps += 1
            continue

        # Compute dt_eff using plateau controller
        if dt_control == "neg_eig_plateau":
            if best_neg_vib is None:
                best_neg_vib = int(neg_vib)
                no_improve = 0
            else:
                if int(neg_vib) < int(best_neg_vib):
                    best_neg_vib = int(neg_vib)
                    no_improve = 0
                    dt_eff_state = min(float(dt_eff_state), float(dt))
                elif int(neg_vib) > int(best_neg_vib):
                    dt_eff_state = max(float(dt_eff_state) * float(plateau_shrink), float(dt_min))
                    no_improve = 0
                else:
                    no_improve += 1

            if no_improve >= int(max(1, plateau_patience)):
                dt_eff_state = min(float(dt_eff_state) * float(plateau_boost), float(dt_max))
                no_improve = 0

            dt_eff = float(np.clip(dt_eff_state, float(dt_min), float(dt_max)))
        else:
            dt_eff = float(dt)

        # Apply max atom displacement cap
        if max_atom_disp is not None and max_atom_disp > 0:
            step = dt_eff * gad_vec
            max_disp = _max_atom_norm(step)
            if np.isfinite(max_disp) and max_disp > float(max_atom_disp) and max_disp > 0:
                dt_eff = dt_eff * (float(max_atom_disp) / float(max_disp))

        trajectory["dt_eff"].append(float(dt_eff))

        # Take GAD step
        prev_pos = coords.clone()
        coords = (coords + dt_eff * gad_vec).detach()
        total_steps += 1

    # Pad trajectories to same length
    while len(trajectory["dt_eff"]) < len(trajectory["energy"]):
        trajectory["dt_eff"].append(float("nan"))
    while len(trajectory["gad_norm"]) < len(trajectory["energy"]):
        trajectory["gad_norm"].append(float("nan"))
    while len(trajectory["escape_cycle"]) < len(trajectory["energy"]):
        trajectory["escape_cycle"].append(escape_cycle)

    # Final vibrational analysis
    final_neg_vib = -1
    try:
        final_out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        scine_elements = get_scine_elements_from_predict_output(final_out)
        final_vib_eigvals = vibrational_eigvals(final_out["hessian"], coords, atomic_nums, scine_elements=scine_elements)
        final_neg_vib = int((final_vib_eigvals < 0).sum().item())
    except Exception:
        pass

    aux = {
        "steps_to_ts": steps_to_ts,
        "escape_cycles_used": escape_cycle,
        "escape_events": escape_events,
        "total_steps": total_steps,
    }

    final_out_dict = {
        "final_coords": coords.detach().cpu(),
        "trajectory": trajectory,
        "steps_taken": total_steps,
        "steps_to_ts": steps_to_ts,
        "final_eig0": trajectory["eig0"][-1] if trajectory["eig0"] else None,
        "final_eig1": trajectory["eig1"][-1] if trajectory["eig1"] else None,
        "final_eig_product": trajectory["eig_product"][-1] if trajectory["eig_product"] else None,
        "final_neg_vibrational": final_neg_vib,
    }

    return final_out_dict, aux


def _save_trajectory_json(logger: ExperimentLogger, result: RunResult, trajectory: Dict[str, Any], escape_events: list) -> Optional[str]:
    transition_dir = logger.run_dir / result.transition_key
    transition_dir.mkdir(parents=True, exist_ok=True)
    path = transition_dir / f"trajectory_{result.sample_index:03d}.json"
    try:
        data = {
            "trajectory": trajectory,
            "escape_events": escape_events,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return str(path)
    except Exception:
        return None


def main(
    argv: Optional[list[str]] = None,
    *,
    default_calculator: Optional[str] = None,
    enforce_calculator: bool = False,
    script_name_prefix: str = "exp-multi-mode",
) -> None:
    parser = argparse.ArgumentParser(
        description="Experiment: Multi-mode escape for GAD (escape high-index saddles via v2 perturbation)."
    )
    parser = add_common_args(parser)

    if default_calculator is not None:
        parser.set_defaults(calculator=str(default_calculator))

    parser.add_argument("--method", type=str, default="euler", choices=["euler"])

    parser.add_argument("--n-steps", type=int, default=1500, help="Total max GAD steps across all cycles")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--start-from", type=str, default="midpoint_rt")
    parser.add_argument("--stop-at-ts", action="store_true")
    parser.add_argument("--ts-eps", type=float, default=1e-5)

    # dt control (plateau-based, but NO floor)
    parser.add_argument(
        "--dt-control",
        type=str,
        default="neg_eig_plateau",
        choices=["fixed", "neg_eig_plateau"],
        help="How to choose dt each step.",
    )
    parser.add_argument("--dt-min", type=float, default=1e-6)
    parser.add_argument("--dt-max", type=float, default=0.05)
    parser.add_argument(
        "--max-atom-disp",
        type=float,
        default=0.25,
        help="Safety cap: max per-atom displacement (A) per step.",
    )

    # Plateau controller knobs
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--plateau-boost", type=float, default=1.5)
    parser.add_argument("--plateau-shrink", type=float, default=0.5)

    # Multi-mode escape parameters (displacement-based detection)
    parser.add_argument(
        "--escape-disp-threshold",
        type=float,
        default=5e-4,
        help="Mean displacement threshold (A) for plateau detection. Trigger escape when "
             "mean(disp[-window:]) < this value.",
    )
    parser.add_argument(
        "--escape-window",
        type=int,
        default=20,
        help="Number of recent steps to consider for plateau detection.",
    )

    # HIP performance knobs
    parser.add_argument(
        "--hip-vib-mode",
        type=str,
        default="projected",
        choices=["projected", "proj_tol"],
        help="How to compute eigenvalue-based metrics on HIP. "
             "'projected' uses the full Eckart-projected pipeline (more accurate, slower). "
             "'proj_tol' uses a single eigendecomp of the projected Hessian and filters TR modes by tolerance (faster).",
    )
    parser.add_argument(
        "--hip-rigid-tol",
        type=float,
        default=1e-6,
        help="Tolerance for treating eigenvalues as translation/rotation modes when --hip-vib-mode=proj_tol.",
    )
    parser.add_argument(
        "--hip-eigh-device",
        type=str,
        default="auto",
        choices=["auto", "cpu"],
        help="Where to run the projected-Hessian eigendecomposition for HIP fast mode. "
             "'cpu' can be faster for small/medium Hessians when you request multiple CPUs.",
    )
    parser.add_argument(
        "--profile-every",
        type=int,
        default=0,
        help="If >0, print timing every N steps (helps diagnose HIP slowness).",
    )
    parser.add_argument(
        "--escape-neg-vib-std",
        type=float,
        default=0.5,
        help="Max std(neg_vib) over window for plateau detection (stable saddle index).",
    )
    parser.add_argument(
        "--escape-delta",
        type=float,
        default=0.1,
        help="Base displacement magnitude (A) for v2 perturbation.",
    )
    parser.add_argument(
        "--adaptive-delta",
        action="store_true",
        default=True,
        help="Scale delta by 1/sqrt(|lambda2|) for adaptive perturbation.",
    )
    parser.add_argument(
        "--no-adaptive-delta",
        action="store_false",
        dest="adaptive_delta",
        help="Disable adaptive delta scaling.",
    )
    parser.add_argument(
        "--min-interatomic-dist",
        type=float,
        default=0.5,
        help="Minimum allowed interatomic distance (A). Perturbations creating closer atoms are rejected.",
    )
    parser.add_argument(
        "--max-escape-cycles",
        type=int,
        default=1000,
        help="Maximum number of escape attempts. Set high to let n_steps be the limiting factor.",
    )

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gad-ts-search")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Optional W&B run name. If omitted, an informative name is auto-generated.",
    )

    args = parser.parse_args(argv)

    if enforce_calculator and default_calculator is not None:
        if str(getattr(args, "calculator", "")).lower() != str(default_calculator).lower():
            raise ValueError(
                f"This entrypoint enforces --calculator={default_calculator}. "
                f"Got --calculator={getattr(args, 'calculator', None)}."
            )

    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    calculator_type = getattr(args, "calculator", "hip").lower()
    if calculator_type == "scine":
        device = "cpu"

    predict_fn = make_predict_fn_from_calculator(calculator, calculator_type)

    loss_type_flags = build_loss_type_flags(args)
    script_name = f"{script_name_prefix}-{args.method}"
    logger = ExperimentLogger(
        base_dir=out_dir,
        script_name=script_name,
        loss_type_flags=loss_type_flags,
        max_graphs_per_transition=10,
        random_seed=42,
    )

    if args.wandb:
        wandb_config = {
            "script": script_name,
            "method": args.method,
            "start_from": args.start_from,
            "stop_at_ts": bool(args.stop_at_ts),
            "calculator": getattr(args, "calculator", "hip"),
            "dt": args.dt,
            "n_steps": args.n_steps,
            "dt_control": args.dt_control,
            "dt_min": args.dt_min,
            "dt_max": args.dt_max,
            "max_atom_disp": args.max_atom_disp,
            "plateau_patience": args.plateau_patience,
            "plateau_boost": args.plateau_boost,
            "plateau_shrink": args.plateau_shrink,
            "escape_disp_threshold": args.escape_disp_threshold,
            "escape_window": args.escape_window,
            "hip_vib_mode": args.hip_vib_mode,
            "hip_rigid_tol": args.hip_rigid_tol,
            "hip_eigh_device": args.hip_eigh_device,
            "profile_every": args.profile_every,
            "escape_neg_vib_std": args.escape_neg_vib_std,
            "escape_delta": args.escape_delta,
            "adaptive_delta": args.adaptive_delta,
            "min_interatomic_dist": args.min_interatomic_dist,
            "max_escape_cycles": args.max_escape_cycles,
        }
        wandb_name = args.wandb_name
        if not wandb_name:
            wandb_name = _auto_wandb_name(script=script_name, loss_type_flags=loss_type_flags, args=args)

        init_wandb_run(
            project=args.wandb_project,
            name=str(wandb_name),
            config=wandb_config,
            entity=args.wandb_entity,
            tags=[args.start_from, args.method, "multi-mode", str(args.dt_control)],
            run_dir=out_dir,
        )

    all_metrics = defaultdict(list)

    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break

        batch = batch.to(device)
        atomic_nums = batch.z.detach().cpu().to(device)
        formula = getattr(batch, "formula", "sample")

        start_coords = parse_starting_geometry(
            args.start_from,
            batch,
            noise_seed=getattr(args, "noise_seed", None),
            sample_index=i,
        ).detach().to(device)

        # Initial vibrational order for transition bucketing
        try:
            init_out = predict_fn(start_coords, atomic_nums, do_hessian=True, require_grad=False)
            init_scine_elements = get_scine_elements_from_predict_output(init_out)
            init_vib = vibrational_eigvals(init_out["hessian"], start_coords, atomic_nums, scine_elements=init_scine_elements)
            initial_neg = int((init_vib < 0).sum().item())
        except Exception:
            initial_neg = -1

        t0 = time.time()
        out_dict, aux = run_multi_mode_escape(
            predict_fn,
            start_coords,
            atomic_nums,
            n_steps=int(args.n_steps),
            dt=float(args.dt),
            stop_at_ts=bool(args.stop_at_ts),
            ts_eps=float(args.ts_eps),
            dt_control=str(args.dt_control),
            dt_min=float(args.dt_min),
            dt_max=float(args.dt_max),
            max_atom_disp=float(args.max_atom_disp) if args.max_atom_disp is not None else None,
            plateau_patience=int(args.plateau_patience),
            plateau_boost=float(args.plateau_boost),
            plateau_shrink=float(args.plateau_shrink),
            escape_disp_threshold=float(args.escape_disp_threshold),
            escape_window=int(args.escape_window),
            hip_vib_mode=str(args.hip_vib_mode),
            hip_rigid_tol=float(args.hip_rigid_tol),
            hip_eigh_device=str(args.hip_eigh_device),
            escape_neg_vib_std=float(args.escape_neg_vib_std),
            escape_delta=float(args.escape_delta),
            adaptive_delta=bool(args.adaptive_delta),
            min_interatomic_dist=float(args.min_interatomic_dist),
            max_escape_cycles=int(args.max_escape_cycles),
            profile_every=int(args.profile_every),
        )
        wall = time.time() - t0

        final_neg = out_dict.get("final_neg_vibrational", -1)

        result = RunResult(
            sample_index=i,
            formula=str(formula),
            initial_neg_eigvals=initial_neg,
            final_neg_eigvals=int(final_neg) if final_neg is not None else -1,
            initial_neg_vibrational=None,
            final_neg_vibrational=int(final_neg) if final_neg is not None else None,
            steps_taken=int(out_dict["steps_taken"]),
            steps_to_ts=aux.get("steps_to_ts"),
            final_time=float(wall),
            final_eig0=out_dict.get("final_eig0"),
            final_eig1=out_dict.get("final_eig1"),
            final_eig_product=out_dict.get("final_eig_product"),
            final_loss=None,
            rmsd_to_known_ts=None,
            stop_reason=None,
            plot_path=None,
            extra_data={
                "method": str(args.method),
                "dt_control": str(args.dt_control),
                "escape_cycles_used": aux.get("escape_cycles_used"),
                "escape_events": aux.get("escape_events"),
                "escape_disp_threshold": float(args.escape_disp_threshold),
                "escape_window": int(args.escape_window),
                "escape_neg_vib_std": float(args.escape_neg_vib_std),
                "escape_delta": float(args.escape_delta),
                "adaptive_delta": bool(args.adaptive_delta),
                "min_interatomic_dist": float(args.min_interatomic_dist),
                "max_escape_cycles": int(args.max_escape_cycles),
            },
        )

        logger.add_result(result)

        # Generate interactive figure
        fig_interactive = plot_gad_trajectory_interactive(
            out_dict["trajectory"],
            sample_index=i,
            formula=str(formula),
            start_from=args.start_from,
            initial_neg_num=initial_neg,
            final_neg_num=int(final_neg) if final_neg is not None else -1,
            steps_to_ts=aux.get("steps_to_ts"),
        )

        # Save HTML plot
        html_path = Path(out_dir) / f"traj_{i:03d}.html"
        fig_interactive.write_html(str(html_path))
        result.plot_path = str(html_path)

        # Save trajectory JSON with escape events
        _save_trajectory_json(logger, result, out_dict["trajectory"], aux.get("escape_events", []))

        metrics = {
            "steps_taken": result.steps_taken,
            "steps_to_ts": result.steps_to_ts,
            "final_eig0": result.final_eig0,
            "final_eig1": result.final_eig1,
            "final_eig_product": result.final_eig_product,
            "final_neg_vibrational": result.final_neg_vibrational,
            "wallclock_s": result.final_time,
            "escape_cycles_used": aux.get("escape_cycles_used"),
        }

        for k, v in metrics.items():
            if v is not None:
                all_metrics[k].append(v)

        if args.wandb:
            log_sample(i, metrics, fig=fig_interactive, plot_name="trajectory_interactive")

    all_runs_path, aggregate_stats_path = logger.save_all_results()
    summary = logger.compute_aggregate_stats()
    logger.print_summary()

    if args.wandb:
        log_summary(summary)
        finish_wandb()

    print(f"Saved results: {all_runs_path}")
    print(f"Saved stats:   {aggregate_stats_path}")


if __name__ == "__main__":
    # Default to SCINE for this module (Eckart-MW variant)
    main(default_calculator="scine", enforce_calculator=True, script_name_prefix="exp-scine-multi-mode-eckartmw")
