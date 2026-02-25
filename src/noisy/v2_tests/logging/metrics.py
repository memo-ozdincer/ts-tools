"""Extended metrics computation for GAD failure diagnosis.

This module computes comprehensive metrics informed by the Levitt-Ortner and iHiSD papers:

1. **Eigenvalue Spectrum** (first 6 vibrational modes)
   - Captures the full local curvature landscape
   - Important for understanding index-k saddles

2. **Eigenvalue Gaps** (critical for singularity detection)
   - eig_gap_01: |λ₂ - λ₁| absolute gap
   - eig_gap_01_rel: (λ₂ - λ₁) / |λ₁| relative gap
   - Small gaps indicate proximity to singularity set S = {x : λ₁ = λ₂}
   - Per Levitt-Ortner, GAD is undefined on S and oscillates near it

3. **Morse Index**
   - Count of negative eigenvalues at current point
   - Index > 1 means high-index saddle where GAD can get stuck
   - iHiSD paper shows adaptive k-reflection can escape these

4. **Singularity Proximity**
   - Minimum gap between adjacent eigenvalue pairs
   - Smaller values indicate proximity to higher-order singularities

5. **GAD Direction Quality**
   - rayleigh_v1: ⟨v₁, Gv₁⟩ (should equal λ₁)
   - grad_proj_v1/v2: projection of gradient onto eigenvectors
   - gad_grad_angle: angle between GAD vector and -∇E

6. **Mode Tracking Diagnostics**
   - v1_v2_overlap: |⟨v₁(t), v₂(t-dt)⟩| for mode swap detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class ExtendedMetrics:
    """Extended metrics for a single GAD step.

    These metrics are designed to diagnose WHY GAD gets stuck, informed by
    the Levitt-Ortner (singularities, cycling) and iHiSD (adaptive k) papers.
    """

    # Step identification
    step: int

    # Eigenvalue spectrum (first 6 vibrational modes, skipping TR)
    eig_0: float
    eig_1: float
    eig_2: float
    eig_3: float
    eig_4: float
    eig_5: float

    # Eigenvalue gaps (critical for singularity detection)
    eig_gap_01: float          # |λ₂ - λ₁| (absolute)
    eig_gap_01_rel: float      # (λ₂ - λ₁) / |λ₁| (relative)
    eig_gap_12: float          # |λ₃ - λ₂| (absolute)

    # Morse index (count of negative vibrational eigenvalues)
    morse_index: int
    neg_eig_sum: float         # Σ λᵢ for λᵢ < 0

    # Singularity proximity (minimum gap between adjacent pairs)
    singularity_metric: float

    # GAD direction quality
    rayleigh_v1: float         # ⟨v₁, Gv₁⟩ (should equal λ₁)
    grad_proj_v1: float        # |⟨∇E, v₁⟩| / ‖∇E‖
    grad_proj_v2: float        # |⟨∇E, v₂⟩| / ‖∇E‖
    gad_grad_angle: float      # angle (degrees) between GAD vector and -∇E

    # Mode tracking diagnostics
    mode_overlap: float        # existing: ⟨v₁(t), v₁(t-dt)⟩
    mode_index: int            # existing: which eigenvector was tracked
    v1_v2_overlap: float       # |⟨v₁(t), v₂(t-dt)⟩| (mode swap detection)

    # Convergence diagnostics
    grad_norm: float           # ‖∇E‖
    step_size_eff: float       # effective dt after capping
    x_disp_step: float         # ‖x(t) - x(t-dt)‖

    # Position/Energy
    energy: float
    energy_delta: float        # E(x) - E(x_prev)

    # TR mode verification (near-zero eigenvalue diagnostics)
    # These track whether translation/rotation modes are properly projected out
    n_tr_modes: int            # Count of eigenvalues with |λ| < tr_threshold (should be 5-6 for nonlinear, 6 for linear)
    tr_eig_max: float          # Maximum absolute value among TR eigenvalues (should be ~0)
    tr_eig_mean: float         # Mean absolute value of TR eigenvalues (should be ~0)
    tr_eig_std: float          # Std dev of TR eigenvalues (monitors drift)

    # Optional: distance to known TS (for validation)
    dist_to_ts: Optional[float] = None
    # Optional: displacement window (mean over last N steps)
    x_disp_window: float = float("nan")

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "step": self.step,
            "eig_0": self.eig_0,
            "eig_1": self.eig_1,
            "eig_2": self.eig_2,
            "eig_3": self.eig_3,
            "eig_4": self.eig_4,
            "eig_5": self.eig_5,
            "eig_gap_01": self.eig_gap_01,
            "eig_gap_01_rel": self.eig_gap_01_rel,
            "eig_gap_12": self.eig_gap_12,
            "morse_index": self.morse_index,
            "neg_eig_sum": self.neg_eig_sum,
            "singularity_metric": self.singularity_metric,
            "rayleigh_v1": self.rayleigh_v1,
            "grad_proj_v1": self.grad_proj_v1,
            "grad_proj_v2": self.grad_proj_v2,
            "gad_grad_angle": self.gad_grad_angle,
            "mode_overlap": self.mode_overlap,
            "mode_index": self.mode_index,
            "v1_v2_overlap": self.v1_v2_overlap,
            "grad_norm": self.grad_norm,
            "step_size_eff": self.step_size_eff,
            "x_disp_step": self.x_disp_step,
            "x_disp_window": self.x_disp_window,
            "energy": self.energy,
            "energy_delta": self.energy_delta,
            "n_tr_modes": self.n_tr_modes,
            "tr_eig_max": self.tr_eig_max,
            "tr_eig_mean": self.tr_eig_mean,
            "tr_eig_std": self.tr_eig_std,
            "dist_to_ts": self.dist_to_ts,
        }


def _to_float(x) -> float:
    """Convert tensor/array to float."""
    if isinstance(x, torch.Tensor):
        return float(x.detach().reshape(-1)[0].item())
    if isinstance(x, np.ndarray):
        return float(x.reshape(-1)[0])
    return float(x)


def _safe_angle_deg(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute angle between two vectors in degrees, handling edge cases."""
    v1_flat = v1.reshape(-1).to(torch.float64)
    v2_flat = v2.reshape(-1).to(torch.float64)

    norm1 = v1_flat.norm()
    norm2 = v2_flat.norm()

    if norm1 < 1e-12 or norm2 < 1e-12:
        return float("nan")

    cos_angle = torch.dot(v1_flat, v2_flat) / (norm1 * norm2)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    return float(torch.acos(cos_angle).item() * 180.0 / np.pi)


def compute_extended_metrics(
    step: int,
    coords: torch.Tensor,
    coords_prev: Optional[torch.Tensor],
    energy: float,
    energy_prev: Optional[float],
    forces: torch.Tensor,
    hessian_proj: torch.Tensor,
    gad_vec: torch.Tensor,
    v_prev: Optional[torch.Tensor],
    dt_eff: float,
    mode_index: Optional[int] = None,
    x_disp_window: Optional[float] = None,
    *,
    tr_threshold: float = 1e-6,
    n_eigs_to_compute: int = 6,
    known_ts_coords: Optional[torch.Tensor] = None,
    vib_evals: Optional[torch.Tensor] = None,
    vib_evecs_full: Optional[torch.Tensor] = None,
) -> Tuple[ExtendedMetrics, torch.Tensor, torch.Tensor]:
    """Compute extended metrics for a single GAD step.

    Args:
        step: Current step number
        coords: Current coordinates (N, 3)
        coords_prev: Previous coordinates (N, 3) or None if first step
        energy: Current energy
        energy_prev: Previous energy or None if first step
        forces: Current forces (N, 3) - note: forces = -∇E
        hessian_proj: Projected Hessian (3N, 3N) with TR modes zeroed
        gad_vec: GAD direction vector (N, 3)
        v_prev: Previous tracked eigenvector (3N,) or None
        dt_eff: Effective timestep used
        tr_threshold: Threshold for TR mode filtering
        n_eigs_to_compute: Number of eigenvalues to include in spectrum
        known_ts_coords: Known TS coordinates for distance computation (optional)

    Returns:
        metrics: ExtendedMetrics dataclass
        v1_new: New tracked eigenvector (3N,) for mode tracking
        v2_new: Second eigenvector (3N,) for mode swap detection
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    num_atoms = int(coords.shape[0])

    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    forces_flat = forces.reshape(-1)  # (3N,)
    grad_flat = -forces_flat  # ∇E = -forces

    # Eigendecomposition of projected Hessian — used only for TR diagnostics.
    evals_full, evecs_full = torch.linalg.eigh(hessian_proj)

    # TR mode diagnostics: how well did the Eckart projection zero the TR modes?
    tr_mask = torch.abs(evals_full) <= tr_threshold
    tr_evals = evals_full[tr_mask]
    n_tr_modes = int(tr_mask.sum().item())

    if n_tr_modes > 0:
        tr_evals_abs = torch.abs(tr_evals)
        tr_eig_max = float(tr_evals_abs.max().item())
        tr_eig_mean = float(tr_evals_abs.mean().item())
        tr_eig_std = float(tr_evals_abs.std().item()) if n_tr_modes > 1 else 0.0
    else:
        tr_eig_max = float("nan")
        tr_eig_mean = float("nan")
        tr_eig_std = float("nan")

    # Vibrational modes for spectrum / Morse index / eigenvector extraction.
    # Prefer the pre-computed reduced-basis eigenvalues (exact 3N-k, no threshold)
    # when provided by the caller.  Fall back to threshold filtering otherwise.
    if vib_evals is not None and vib_evecs_full is not None:
        vib_evals_use = vib_evals
        vib_evecs_use = vib_evecs_full
        # Morse index: every eigenvalue < 0 is genuinely negative (no threshold dead-zone)
        neg_mask = vib_evals_use < 0.0
    else:
        vib_mask = ~tr_mask
        vib_indices = torch.where(vib_mask)[0]
        if len(vib_indices) < n_eigs_to_compute:
            vib_evals_use = evals_full
            vib_evecs_use = evecs_full
        else:
            vib_evals_use = evals_full[vib_indices]
            vib_evecs_use = evecs_full[:, vib_indices]
        neg_mask = vib_evals_use < -tr_threshold

    # Extract first 6 eigenvalues (or fewer if not available)
    n_avail = min(n_eigs_to_compute, int(vib_evals_use.numel()))
    eig_spectrum = [float("nan")] * n_eigs_to_compute
    for i in range(n_avail):
        eig_spectrum[i] = float(vib_evals_use[i].item())

    eig_0, eig_1, eig_2, eig_3, eig_4, eig_5 = eig_spectrum

    # Eigenvalue gaps
    if n_avail >= 2:
        eig_gap_01 = abs(eig_1 - eig_0)
        eig_gap_01_rel = (eig_1 - eig_0) / abs(eig_0) if abs(eig_0) > 1e-12 else float("nan")
    else:
        eig_gap_01 = float("nan")
        eig_gap_01_rel = float("nan")

    if n_avail >= 3:
        eig_gap_12 = abs(eig_2 - eig_1)
    else:
        eig_gap_12 = float("nan")

    # Morse index (count negative vibrational eigenvalues)
    morse_index = int(neg_mask.sum().item())
    neg_eig_sum = float(vib_evals_use[neg_mask].sum().item()) if neg_mask.any() else 0.0

    # Singularity metric: minimum gap between adjacent eigenvalue pairs
    if n_avail >= 2:
        gaps = torch.abs(vib_evals_use[1:n_avail] - vib_evals_use[:n_avail-1])
        singularity_metric = float(gaps.min().item()) if len(gaps) > 0 else float("nan")
    else:
        singularity_metric = float("nan")

    # Extract v1 and v2 from vibrational modes
    v1 = vib_evecs_use[:, 0] if n_avail >= 1 else torch.zeros(3 * num_atoms, device=coords.device)
    v2 = vib_evecs_use[:, 1] if n_avail >= 2 else torch.zeros(3 * num_atoms, device=coords.device)

    # Normalize
    v1 = v1 / (v1.norm() + 1e-12)
    v2 = v2 / (v2.norm() + 1e-12)

    # Rayleigh quotient: ⟨v₁, Gv₁⟩ (should equal λ₁)
    rayleigh_v1 = float(torch.dot(v1, hessian_proj @ v1).item())

    # Gradient projections
    grad_norm = float(grad_flat.norm().item())
    if grad_norm > 1e-12:
        grad_proj_v1 = float(torch.abs(torch.dot(grad_flat, v1)).item() / grad_norm)
        grad_proj_v2 = float(torch.abs(torch.dot(grad_flat, v2)).item() / grad_norm)
    else:
        grad_proj_v1 = float("nan")
        grad_proj_v2 = float("nan")

    # GAD-gradient angle
    gad_flat = gad_vec.reshape(-1)
    gad_grad_angle = _safe_angle_deg(gad_flat, -grad_flat)  # angle between GAD and -∇E

    # Mode tracking diagnostics
    if v_prev is not None:
        v_prev_flat = v_prev.reshape(-1).to(v1.device, v1.dtype)
        mode_overlap = float(torch.abs(torch.dot(v1, v_prev_flat)).item())
        v1_v2_overlap = float(torch.abs(torch.dot(v1, v_prev_flat)).item())  # overlap of current v1 with prev

        # Also check if current v1 aligns with previous v2 (mode swap)
        # This requires storing v2_prev, which we'll handle in trajectory logger
        v1_v2_overlap = float("nan")  # Will be computed by trajectory logger
    else:
        mode_overlap = 1.0
        v1_v2_overlap = float("nan")

    # Find which mode index v1 corresponds to (for mode_index tracking)
    # This is the index within the vibrational subspace
    mode_index = 0  # Always tracking lowest vibrational mode
    mode_index = int(mode_index) if mode_index is not None else 0

    # Displacement
    if coords_prev is not None:
        x_disp_step = float((coords - coords_prev.reshape(-1, 3)).norm().item())
    else:
        x_disp_step = 0.0

    # Energy delta
    energy_delta = energy - energy_prev if energy_prev is not None else 0.0

    # Distance to known TS
    dist_to_ts = None
    if known_ts_coords is not None:
        if known_ts_coords.dim() == 3 and known_ts_coords.shape[0] == 1:
            known_ts_coords = known_ts_coords[0]
        dist_to_ts = float((coords - known_ts_coords.reshape(-1, 3)).norm().item())

    metrics = ExtendedMetrics(
        step=step,
        eig_0=eig_0,
        eig_1=eig_1,
        eig_2=eig_2,
        eig_3=eig_3,
        eig_4=eig_4,
        eig_5=eig_5,
        eig_gap_01=eig_gap_01,
        eig_gap_01_rel=eig_gap_01_rel,
        eig_gap_12=eig_gap_12,
        morse_index=morse_index,
        neg_eig_sum=neg_eig_sum,
        singularity_metric=singularity_metric,
        rayleigh_v1=rayleigh_v1,
        grad_proj_v1=grad_proj_v1,
        grad_proj_v2=grad_proj_v2,
        gad_grad_angle=gad_grad_angle,
        mode_overlap=mode_overlap,
        mode_index=mode_index,
        v1_v2_overlap=v1_v2_overlap,
        grad_norm=grad_norm,
        step_size_eff=dt_eff,
        x_disp_step=x_disp_step,
        energy=energy,
        energy_delta=energy_delta,
        n_tr_modes=n_tr_modes,
        tr_eig_max=tr_eig_max,
        tr_eig_mean=tr_eig_mean,
        tr_eig_std=tr_eig_std,
        dist_to_ts=dist_to_ts,
    )

    return metrics, v1.detach().clone(), v2.detach().clone()


def compute_eigenvalue_spectrum(
    hessian_proj: torch.Tensor,
    tr_threshold: float = 1e-6,
    n_eigs: int = 6,
) -> List[float]:
    """Compute first n vibrational eigenvalues from projected Hessian.

    Args:
        hessian_proj: Projected Hessian (3N, 3N)
        tr_threshold: Threshold for TR mode filtering
        n_eigs: Number of eigenvalues to return

    Returns:
        List of first n vibrational eigenvalues (padded with nan if fewer available)
    """
    evals, _ = torch.linalg.eigh(hessian_proj)

    vib_mask = torch.abs(evals) > tr_threshold
    vib_evals = evals[vib_mask]

    result = [float("nan")] * n_eigs
    for i in range(min(n_eigs, len(vib_evals))):
        result[i] = float(vib_evals[i].item())

    return result


def compute_morse_index(
    hessian_proj: torch.Tensor,
    tr_threshold: float = 1e-6,
) -> int:
    """Compute Morse index (count of negative vibrational eigenvalues).

    Args:
        hessian_proj: Projected Hessian (3N, 3N)
        tr_threshold: Threshold for TR mode filtering

    Returns:
        Number of negative vibrational eigenvalues
    """
    evals, _ = torch.linalg.eigh(hessian_proj)

    vib_mask = torch.abs(evals) > tr_threshold
    vib_evals = evals[vib_mask]

    return int((vib_evals < -tr_threshold).sum().item())


def compute_singularity_proximity(
    hessian_proj: torch.Tensor,
    tr_threshold: float = 1e-6,
) -> Tuple[float, float]:
    """Compute singularity proximity metrics.

    Per Levitt-Ortner, GAD is undefined on the singularity set S = {x : λ₁ = λ₂}
    and exhibits quasi-periodic orbits near S.

    Args:
        hessian_proj: Projected Hessian (3N, 3N)
        tr_threshold: Threshold for TR mode filtering

    Returns:
        eig_gap_01: |λ₂ - λ₁| (absolute gap)
        min_adjacent_gap: minimum gap between any adjacent eigenvalue pair
    """
    evals, _ = torch.linalg.eigh(hessian_proj)

    vib_mask = torch.abs(evals) > tr_threshold
    vib_evals = evals[vib_mask]

    if len(vib_evals) < 2:
        return float("nan"), float("nan")

    eig_gap_01 = float(torch.abs(vib_evals[1] - vib_evals[0]).item())

    gaps = torch.abs(vib_evals[1:] - vib_evals[:-1])
    min_adjacent_gap = float(gaps.min().item())

    return eig_gap_01, min_adjacent_gap