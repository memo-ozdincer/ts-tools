from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class SignEnforcerConfig:
    """Configuration for the sign-enforcer objective."""

    sign_neg_target: float = -5e-3
    sign_pos_floor: float = 1e-3


def sign_enforcer_loss(
    vibrational_eigvals: torch.Tensor,
    *,
    sign_neg_target: float = -5e-3,
    sign_pos_floor: float = 1e-3,
) -> Tuple[torch.Tensor, int]:
    """Loss used by the 'sign enforcer' algorithm.

    Copied logically from `src/gad_eigenvalue_descent.py`.

    Returns:
        (loss, neg_vibrational)
    """

    if vibrational_eigvals.numel() == 0:
        loss = vibrational_eigvals.new_tensor(float("inf"))
        return loss, -1

    neg_vibrational = int((vibrational_eigvals < 0).sum().item())
    eig0 = vibrational_eigvals[0]

    neg_target = eig0.new_tensor(sign_neg_target)
    pos_floor = eig0.new_tensor(sign_pos_floor)

    if neg_vibrational == 0:
        loss = (eig0 - neg_target).pow(2)
    elif neg_vibrational == 1:
        loss = eig0.new_tensor(0.0)
    else:
        trailing_eigs = vibrational_eigvals[1:]
        if trailing_eigs.numel() == 0:
            loss = eig0.new_tensor(0.0)
        else:
            penalties = (pos_floor - trailing_eigs).pow(2)
            loss = penalties.sum()

    return loss, neg_vibrational


def sign_enforcer_should_stop(neg_vibrational: int, loss_value: float, sign_neg_target: float) -> Optional[str]:
    """Replicates the early-stop heuristics used in the original script."""

    if neg_vibrational == 1:
        return "Exactly one negative eigenvalue achieved"
    if neg_vibrational == 0 and loss_value < 1e-8:
        return f"λ₀ pushed below target ({sign_neg_target})"
    return None
