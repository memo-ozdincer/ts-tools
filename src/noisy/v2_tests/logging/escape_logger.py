"""Escape event logging for GAD diagnosis.

When GAD gets stuck at a high-index saddle (Morse index > 1), the v₂ kicking
mechanism attempts to escape. This module provides comprehensive logging of
these escape events to understand:

1. **When escapes are triggered** - What conditions led to the escape?
2. **Pre-kick state** - Full eigenvalue spectrum, Morse index, gradients
3. **Kick parameters** - Direction, magnitude, scaling
4. **Post-kick state** - How did the eigenvalue landscape change?
5. **Outcome** - Did the kick succeed? Why/why not?

Per the iHiSD paper (Theorem 3.2), kicking in v₂ when stuck at index-2 saddle
is equivalent to a single step of index-2 HiSD, which should move toward
an index-1 saddle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class EscapeEvent:
    """Comprehensive logging for a single escape/kick event.

    Captures the full eigenvalue spectrum and state before and after
    the kick to understand escape dynamics.
    """

    # Event identification
    step: int
    escape_cycle: int
    trigger_reason: str  # "displacement_plateau", "force_plateau", etc.

    # Pre-kick eigenvalue spectrum (first 6 vibrational modes)
    pre_eig_0: float
    pre_eig_1: float
    pre_eig_2: float
    pre_eig_3: float
    pre_eig_4: float
    pre_eig_5: float

    # Pre-kick derived metrics
    pre_morse_index: int
    pre_eig_gap_01: float      # |λ₂ - λ₁| critical for singularity detection
    pre_eig_gap_01_rel: float  # (λ₂ - λ₁) / |λ₁|
    pre_grad_norm: float
    pre_energy: float
    pre_singularity_metric: float  # min gap between adjacent eigenvalues

    # Kick parameters
    kick_mode: int             # 1=v₁, 2=v₂, 3=v₃, etc.
    kick_direction: str        # "+v2", "-v2", "random", etc.
    kick_delta_base: float     # Base delta before adaptive scaling
    kick_delta_effective: float  # Actual delta used after scaling
    kick_lambda: float         # Eigenvalue of kick mode (λ₂ for v₂ kick)

    # Post-kick eigenvalue spectrum (first 6 vibrational modes)
    post_eig_0: float
    post_eig_1: float
    post_eig_2: float
    post_eig_3: float
    post_eig_4: float
    post_eig_5: float

    # Post-kick derived metrics
    post_morse_index: int
    post_eig_gap_01: float
    post_eig_gap_01_rel: float
    post_grad_norm: float
    post_energy: float
    post_singularity_metric: float

    # Outcome metrics
    energy_change: float       # post_energy - pre_energy
    morse_index_change: int    # post_morse_index - pre_morse_index
    eig_gap_change: float      # post_eig_gap_01 - pre_eig_gap_01
    accepted: bool             # Was the kick accepted?
    rejection_reason: Optional[str] = None  # If rejected, why?

    # Displacement metrics
    displacement_magnitude: float = 0.0  # Actual displacement in Angstrom
    min_dist_after: float = float("inf")  # Minimum interatomic distance after kick

    # Context from preceding trajectory
    mean_disp_at_trigger: float = 0.0  # Mean displacement in escape window
    neg_vib_std_at_trigger: float = 0.0  # Std of neg_vib in escape window

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "step": self.step,
            "escape_cycle": self.escape_cycle,
            "trigger_reason": self.trigger_reason,
            # Pre-kick spectrum
            "pre_eig_0": self.pre_eig_0,
            "pre_eig_1": self.pre_eig_1,
            "pre_eig_2": self.pre_eig_2,
            "pre_eig_3": self.pre_eig_3,
            "pre_eig_4": self.pre_eig_4,
            "pre_eig_5": self.pre_eig_5,
            # Pre-kick metrics
            "pre_morse_index": self.pre_morse_index,
            "pre_eig_gap_01": self.pre_eig_gap_01,
            "pre_eig_gap_01_rel": self.pre_eig_gap_01_rel,
            "pre_grad_norm": self.pre_grad_norm,
            "pre_energy": self.pre_energy,
            "pre_singularity_metric": self.pre_singularity_metric,
            # Kick parameters
            "kick_mode": self.kick_mode,
            "kick_direction": self.kick_direction,
            "kick_delta_base": self.kick_delta_base,
            "kick_delta_effective": self.kick_delta_effective,
            "kick_lambda": self.kick_lambda,
            # Post-kick spectrum
            "post_eig_0": self.post_eig_0,
            "post_eig_1": self.post_eig_1,
            "post_eig_2": self.post_eig_2,
            "post_eig_3": self.post_eig_3,
            "post_eig_4": self.post_eig_4,
            "post_eig_5": self.post_eig_5,
            # Post-kick metrics
            "post_morse_index": self.post_morse_index,
            "post_eig_gap_01": self.post_eig_gap_01,
            "post_eig_gap_01_rel": self.post_eig_gap_01_rel,
            "post_grad_norm": self.post_grad_norm,
            "post_energy": self.post_energy,
            "post_singularity_metric": self.post_singularity_metric,
            # Outcome
            "energy_change": self.energy_change,
            "morse_index_change": self.morse_index_change,
            "eig_gap_change": self.eig_gap_change,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            # Displacement
            "displacement_magnitude": self.displacement_magnitude,
            "min_dist_after": self.min_dist_after,
            # Context
            "mean_disp_at_trigger": self.mean_disp_at_trigger,
            "neg_vib_std_at_trigger": self.neg_vib_std_at_trigger,
        }

    def summary(self) -> str:
        """Return a one-line summary for logging."""
        status = "ACCEPTED" if self.accepted else f"REJECTED ({self.rejection_reason})"
        return (
            f"[Escape {self.escape_cycle}] step={self.step} "
            f"index: {self.pre_morse_index}→{self.post_morse_index} "
            f"gap: {self.pre_eig_gap_01:.2e}→{self.post_eig_gap_01:.2e} "
            f"ΔE: {self.energy_change:+.4f} "
            f"delta: {self.kick_delta_effective:.4f} {status}"
        )

    @property
    def is_singularity_escape(self) -> bool:
        """Was this escape triggered near a singularity (small eig gap)?"""
        return self.pre_eig_gap_01 < 0.01  # Arbitrary threshold

    @property
    def index_improved(self) -> bool:
        """Did the Morse index decrease (move toward index-1)?"""
        return self.post_morse_index < self.pre_morse_index


def _compute_spectrum_metrics(
    hessian_proj: torch.Tensor,
    forces: torch.Tensor,
    tr_threshold: float = 1e-6,
    n_eigs: int = 6,
) -> dict:
    """Compute eigenvalue spectrum and derived metrics.

    Args:
        hessian_proj: Projected Hessian (3N, 3N)
        forces: Forces tensor (N, 3)
        tr_threshold: Threshold for TR mode filtering
        n_eigs: Number of eigenvalues to extract

    Returns:
        Dictionary with spectrum, morse_index, gaps, grad_norm, singularity_metric
    """
    evals, _ = torch.linalg.eigh(hessian_proj)

    # Filter TR modes
    vib_mask = torch.abs(evals) > tr_threshold
    vib_evals = evals[vib_mask]

    # Extract spectrum
    spectrum = [float("nan")] * n_eigs
    n_avail = min(n_eigs, len(vib_evals))
    for i in range(n_avail):
        spectrum[i] = float(vib_evals[i].item())

    # Morse index
    morse_index = int((vib_evals < -tr_threshold).sum().item())

    # Eigenvalue gaps
    if n_avail >= 2:
        eig_gap_01 = abs(spectrum[1] - spectrum[0])
        eig_gap_01_rel = (spectrum[1] - spectrum[0]) / abs(spectrum[0]) if abs(spectrum[0]) > 1e-12 else float("nan")
    else:
        eig_gap_01 = float("nan")
        eig_gap_01_rel = float("nan")

    # Singularity metric
    if n_avail >= 2:
        gaps = torch.abs(vib_evals[1:n_avail] - vib_evals[:n_avail-1])
        singularity_metric = float(gaps.min().item()) if len(gaps) > 0 else float("nan")
    else:
        singularity_metric = float("nan")

    # Gradient norm
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    grad_norm = float(forces.reshape(-1).norm().item())

    return {
        "spectrum": spectrum,
        "morse_index": morse_index,
        "eig_gap_01": eig_gap_01,
        "eig_gap_01_rel": eig_gap_01_rel,
        "singularity_metric": singularity_metric,
        "grad_norm": grad_norm,
    }


def create_escape_event(
    step: int,
    escape_cycle: int,
    trigger_reason: str,
    *,
    # Pre-kick state
    pre_hessian_proj: torch.Tensor,
    pre_forces: torch.Tensor,
    pre_energy: float,
    # Kick parameters
    kick_mode: int,
    kick_direction: str,
    kick_delta_base: float,
    kick_delta_effective: float,
    kick_lambda: float,
    # Post-kick state
    post_hessian_proj: torch.Tensor,
    post_forces: torch.Tensor,
    post_energy: float,
    # Outcome
    accepted: bool,
    rejection_reason: Optional[str] = None,
    # Displacement
    displacement_magnitude: float = 0.0,
    min_dist_after: float = float("inf"),
    # Context
    mean_disp_at_trigger: float = 0.0,
    neg_vib_std_at_trigger: float = 0.0,
    tr_threshold: float = 1e-6,
) -> EscapeEvent:
    """Create an EscapeEvent with full pre/post diagnostics.

    This is the main factory function for creating escape events with
    automatic computation of derived metrics.

    Args:
        step: Current step number
        escape_cycle: Which escape cycle this is
        trigger_reason: What triggered the escape
        pre_hessian_proj: Projected Hessian before kick
        pre_forces: Forces before kick
        pre_energy: Energy before kick
        kick_mode: Which mode was kicked (1=v₁, 2=v₂, etc.)
        kick_direction: Direction descriptor (+v2, -v2, random, etc.)
        kick_delta_base: Base delta before adaptive scaling
        kick_delta_effective: Actual delta used
        kick_lambda: Eigenvalue of kicked mode
        post_hessian_proj: Projected Hessian after kick
        post_forces: Forces after kick
        post_energy: Energy after kick
        accepted: Whether the kick was accepted
        rejection_reason: Why rejected (if applicable)
        displacement_magnitude: Actual displacement in Angstrom
        min_dist_after: Minimum interatomic distance after kick
        mean_disp_at_trigger: Mean displacement in escape window
        neg_vib_std_at_trigger: Std of neg_vib in escape window
        tr_threshold: Threshold for TR mode filtering

    Returns:
        EscapeEvent with all metrics computed
    """
    # Compute pre-kick metrics
    pre_metrics = _compute_spectrum_metrics(pre_hessian_proj, pre_forces, tr_threshold)
    pre_spec = pre_metrics["spectrum"]

    # Compute post-kick metrics
    post_metrics = _compute_spectrum_metrics(post_hessian_proj, post_forces, tr_threshold)
    post_spec = post_metrics["spectrum"]

    return EscapeEvent(
        step=step,
        escape_cycle=escape_cycle,
        trigger_reason=trigger_reason,
        # Pre-kick spectrum
        pre_eig_0=pre_spec[0],
        pre_eig_1=pre_spec[1],
        pre_eig_2=pre_spec[2],
        pre_eig_3=pre_spec[3],
        pre_eig_4=pre_spec[4],
        pre_eig_5=pre_spec[5],
        # Pre-kick metrics
        pre_morse_index=pre_metrics["morse_index"],
        pre_eig_gap_01=pre_metrics["eig_gap_01"],
        pre_eig_gap_01_rel=pre_metrics["eig_gap_01_rel"],
        pre_grad_norm=pre_metrics["grad_norm"],
        pre_energy=pre_energy,
        pre_singularity_metric=pre_metrics["singularity_metric"],
        # Kick parameters
        kick_mode=kick_mode,
        kick_direction=kick_direction,
        kick_delta_base=kick_delta_base,
        kick_delta_effective=kick_delta_effective,
        kick_lambda=kick_lambda,
        # Post-kick spectrum
        post_eig_0=post_spec[0],
        post_eig_1=post_spec[1],
        post_eig_2=post_spec[2],
        post_eig_3=post_spec[3],
        post_eig_4=post_spec[4],
        post_eig_5=post_spec[5],
        # Post-kick metrics
        post_morse_index=post_metrics["morse_index"],
        post_eig_gap_01=post_metrics["eig_gap_01"],
        post_eig_gap_01_rel=post_metrics["eig_gap_01_rel"],
        post_grad_norm=post_metrics["grad_norm"],
        post_energy=post_energy,
        post_singularity_metric=post_metrics["singularity_metric"],
        # Outcome
        energy_change=post_energy - pre_energy,
        morse_index_change=post_metrics["morse_index"] - pre_metrics["morse_index"],
        eig_gap_change=post_metrics["eig_gap_01"] - pre_metrics["eig_gap_01"],
        accepted=accepted,
        rejection_reason=rejection_reason,
        # Displacement
        displacement_magnitude=displacement_magnitude,
        min_dist_after=min_dist_after,
        # Context
        mean_disp_at_trigger=mean_disp_at_trigger,
        neg_vib_std_at_trigger=neg_vib_std_at_trigger,
    )


def summarize_escape_events(events: List[EscapeEvent]) -> dict:
    """Compute summary statistics over a list of escape events.

    Args:
        events: List of EscapeEvent objects

    Returns:
        Dictionary with summary statistics
    """
    if not events:
        return {
            "total_escapes": 0,
            "accepted_escapes": 0,
            "rejected_escapes": 0,
            "success_rate": float("nan"),
            "mean_index_before": float("nan"),
            "mean_index_after": float("nan"),
            "mean_index_improvement": float("nan"),
            "escapes_near_singularity": 0,
            "mean_energy_change": float("nan"),
            "mean_gap_before": float("nan"),
            "mean_gap_after": float("nan"),
        }

    total = len(events)
    accepted = sum(1 for e in events if e.accepted)
    rejected = total - accepted

    pre_indices = [e.pre_morse_index for e in events]
    post_indices = [e.post_morse_index for e in events if e.accepted]
    index_improvements = [e.pre_morse_index - e.post_morse_index for e in events if e.accepted]

    energy_changes = [e.energy_change for e in events if e.accepted]
    gaps_before = [e.pre_eig_gap_01 for e in events if np.isfinite(e.pre_eig_gap_01)]
    gaps_after = [e.post_eig_gap_01 for e in events if e.accepted and np.isfinite(e.post_eig_gap_01)]

    return {
        "total_escapes": total,
        "accepted_escapes": accepted,
        "rejected_escapes": rejected,
        "success_rate": accepted / total if total > 0 else float("nan"),
        "mean_index_before": float(np.mean(pre_indices)) if pre_indices else float("nan"),
        "mean_index_after": float(np.mean(post_indices)) if post_indices else float("nan"),
        "mean_index_improvement": float(np.mean(index_improvements)) if index_improvements else float("nan"),
        "escapes_near_singularity": sum(1 for e in events if e.is_singularity_escape),
        "mean_energy_change": float(np.mean(energy_changes)) if energy_changes else float("nan"),
        "mean_gap_before": float(np.mean(gaps_before)) if gaps_before else float("nan"),
        "mean_gap_after": float(np.mean(gaps_after)) if gaps_after else float("nan"),
    }
