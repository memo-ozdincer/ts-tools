"""Alternative kick strategies for escaping high-index saddles.

This module implements various kick strategies to compare against v₂ kicking.
The goal is to understand:
1. WHY v₂ kicking works (what makes it special?)
2. Are there better/more elegant alternatives?

Strategies implemented:
- v2_kick: Current implementation (kick along second vibrational mode)
- v1_kick: Kick along first vibrational mode
- random_kick: Random direction (control experiment)
- random_ortho_v1: Random direction orthogonal to v₁
- gradient_descent: Take gradient descent steps to escape
- ortho_v1_grad_descent: Gradient descent constrained to subspace orthogonal to v₁
- higher_modes: Try v₂, v₃, v₄ sequentially
- adaptive_k_reflect: Single step reflecting along full unstable subspace
- energy_steepest: Steepest descent direction (most energy decrease)

Each strategy returns:
- new_coords: The perturbed coordinates
- info: Dictionary with diagnostic information
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def _geometry_is_valid(coords: torch.Tensor, min_dist_threshold: float) -> bool:
    """Check if geometry has all atom pairs above minimum distance threshold."""
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    n = coords.shape[0]
    if n < 2:
        return True
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    dist = diff.norm(dim=2)
    dist = dist + torch.eye(n, device=coords.device, dtype=coords.dtype) * 1e10
    return float(dist.min().item()) >= min_dist_threshold


def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().reshape(-1)[0].item())
    return float(x)


def _get_vibrational_modes(
    hessian_proj: torch.Tensor,
    tr_threshold: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get vibrational eigenvalues and eigenvectors from projected Hessian.

    Returns:
        vib_evals: Vibrational eigenvalues (sorted ascending)
        vib_evecs: Vibrational eigenvectors
        vib_indices: Indices into full eigenvalue array
    """
    evals, evecs = torch.linalg.eigh(hessian_proj)
    vib_mask = torch.abs(evals) > tr_threshold
    vib_indices = torch.where(vib_mask)[0]

    if len(vib_indices) == 0:
        return evals, evecs, torch.arange(len(evals))

    return evals[vib_indices], evecs[:, vib_indices], vib_indices


def kick_v2(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    hessian_proj: torch.Tensor,
    *,
    delta: float = 0.3,
    adaptive_delta: bool = True,
    min_interatomic_dist: float = 0.5,
    max_shrink_attempts: int = 5,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Kick along v₂ (second vibrational mode). This is the current strategy.

    Why it might work:
    - v₂ is the direction of second-smallest curvature
    - At high-index saddles, this is often another "unstable" direction
    - Moving in v₂ changes the local Hessian structure
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    num_atoms = coords.shape[0]

    vib_evals, vib_evecs, _ = _get_vibrational_modes(hessian_proj)

    if len(vib_evals) < 2:
        return coords, {"success": False, "reason": "insufficient_modes", "strategy": "v2"}

    v2 = vib_evecs[:, 1]
    v2 = v2 / (v2.norm() + 1e-12)
    lambda2 = float(vib_evals[1].item())

    # Adaptive scaling
    base_delta = delta
    if adaptive_delta and lambda2 < -0.01:
        base_delta = delta / np.sqrt(abs(lambda2))
        base_delta = min(base_delta, 1.0)

    v2_3d = v2.reshape(num_atoms, 3)
    E_current = _to_float(predict_fn(coords, atomic_nums, do_hessian=False, require_grad=False)["energy"])

    # Try both directions
    for attempt in range(max_shrink_attempts + 1):
        current_delta = base_delta * (0.5 ** attempt)

        coords_plus = coords + current_delta * v2_3d
        coords_minus = coords - current_delta * v2_3d

        candidates = []

        if _geometry_is_valid(coords_plus, min_interatomic_dist):
            try:
                E_plus = _to_float(predict_fn(coords_plus, atomic_nums, do_hessian=False, require_grad=False)["energy"])
                if np.isfinite(E_plus):
                    candidates.append((coords_plus, +1, E_plus))
            except Exception:
                pass

        if _geometry_is_valid(coords_minus, min_interatomic_dist):
            try:
                E_minus = _to_float(predict_fn(coords_minus, atomic_nums, do_hessian=False, require_grad=False)["energy"])
                if np.isfinite(E_minus):
                    candidates.append((coords_minus, -1, E_minus))
            except Exception:
                pass

        if candidates:
            candidates.sort(key=lambda x: x[2])
            new_coords, direction, E_new = candidates[0]
            return new_coords.detach(), {
                "success": True,
                "strategy": "v2",
                "delta_used": current_delta,
                "direction": direction,
                "lambda2": lambda2,
                "energy_before": E_current,
                "energy_after": E_new,
                "energy_change": E_new - E_current,
                "shrink_attempts": attempt,
            }

    return coords, {"success": False, "reason": "all_attempts_failed", "strategy": "v2"}


def kick_v1(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    hessian_proj: torch.Tensor,
    *,
    delta: float = 0.3,
    adaptive_delta: bool = True,
    min_interatomic_dist: float = 0.5,
    max_shrink_attempts: int = 5,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Kick along v₁ (first vibrational mode).

    This tests whether perturbing along the tracked unstable direction
    can revive motion without needing v₂ specifically.
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    num_atoms = coords.shape[0]

    vib_evals, vib_evecs, _ = _get_vibrational_modes(hessian_proj)

    if len(vib_evals) < 1:
        return coords, {"success": False, "reason": "insufficient_modes", "strategy": "v1"}

    v1 = vib_evecs[:, 0]
    v1 = v1 / (v1.norm() + 1e-12)
    lambda1 = float(vib_evals[0].item())

    base_delta = delta
    if adaptive_delta and lambda1 < -0.01:
        base_delta = delta / np.sqrt(abs(lambda1))
        base_delta = min(base_delta, 1.0)

    v1_3d = v1.reshape(num_atoms, 3)
    E_current = _to_float(predict_fn(coords, atomic_nums, do_hessian=False, require_grad=False)["energy"])

    for attempt in range(max_shrink_attempts + 1):
        current_delta = base_delta * (0.5 ** attempt)

        coords_plus = coords + current_delta * v1_3d
        coords_minus = coords - current_delta * v1_3d

        candidates = []

        if _geometry_is_valid(coords_plus, min_interatomic_dist):
            try:
                E_plus = _to_float(predict_fn(coords_plus, atomic_nums, do_hessian=False, require_grad=False)["energy"])
                if np.isfinite(E_plus):
                    candidates.append((coords_plus, +1, E_plus))
            except Exception:
                pass

        if _geometry_is_valid(coords_minus, min_interatomic_dist):
            try:
                E_minus = _to_float(predict_fn(coords_minus, atomic_nums, do_hessian=False, require_grad=False)["energy"])
                if np.isfinite(E_minus):
                    candidates.append((coords_minus, -1, E_minus))
            except Exception:
                pass

        if candidates:
            candidates.sort(key=lambda x: x[2])
            new_coords, direction, E_new = candidates[0]
            return new_coords.detach(), {
                "success": True,
                "strategy": "v1",
                "delta_used": current_delta,
                "direction": direction,
                "lambda1": lambda1,
                "energy_before": E_current,
                "energy_after": E_new,
                "energy_change": E_new - E_current,
                "shrink_attempts": attempt,
            }

    return coords, {"success": False, "reason": "all_attempts_failed", "strategy": "v1"}


def kick_random(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    hessian_proj: torch.Tensor,  # Not used, but kept for API consistency
    *,
    delta: float = 0.3,
    min_interatomic_dist: float = 0.5,
    max_shrink_attempts: int = 5,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Kick in a random direction (CONTROL EXPERIMENT).

    If v₂ is not special, random kicks should work equally well.
    If v₂ is special, random kicks should perform worse.
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    num_atoms = coords.shape[0]

    if seed is not None:
        torch.manual_seed(seed)

    # Generate random direction
    r = torch.randn(num_atoms, 3, device=coords.device, dtype=coords.dtype)
    r = r / (r.norm() + 1e-12)

    E_current = _to_float(predict_fn(coords, atomic_nums, do_hessian=False, require_grad=False)["energy"])

    for attempt in range(max_shrink_attempts + 1):
        current_delta = delta * (0.5 ** attempt)

        coords_plus = coords + current_delta * r
        coords_minus = coords - current_delta * r

        candidates = []

        if _geometry_is_valid(coords_plus, min_interatomic_dist):
            try:
                E_plus = _to_float(predict_fn(coords_plus, atomic_nums, do_hessian=False, require_grad=False)["energy"])
                if np.isfinite(E_plus):
                    candidates.append((coords_plus, +1, E_plus))
            except Exception:
                pass

        if _geometry_is_valid(coords_minus, min_interatomic_dist):
            try:
                E_minus = _to_float(predict_fn(coords_minus, atomic_nums, do_hessian=False, require_grad=False)["energy"])
                if np.isfinite(E_minus):
                    candidates.append((coords_minus, -1, E_minus))
            except Exception:
                pass

        if candidates:
            candidates.sort(key=lambda x: x[2])
            new_coords, direction, E_new = candidates[0]
            return new_coords.detach(), {
                "success": True,
                "strategy": "random",
                "delta_used": current_delta,
                "direction": direction,
                "energy_before": E_current,
                "energy_after": E_new,
                "energy_change": E_new - E_current,
                "shrink_attempts": attempt,
            }

    return coords, {"success": False, "reason": "all_attempts_failed", "strategy": "random"}


def kick_random_ortho_v1(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    hessian_proj: torch.Tensor,
    *,
    delta: float = 0.3,
    min_interatomic_dist: float = 0.5,
    max_shrink_attempts: int = 5,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Kick in a random direction ORTHOGONAL to v₁.

    Hypothesis: Maybe any perturbation orthogonal to the tracked mode works,
    not specifically v₂. This tests if v₂'s orthogonality to v₁ is the key.
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    num_atoms = coords.shape[0]

    vib_evals, vib_evecs, _ = _get_vibrational_modes(hessian_proj)

    if len(vib_evals) < 1:
        return coords, {"success": False, "reason": "no_modes", "strategy": "random_ortho_v1"}

    v1 = vib_evecs[:, 0]
    v1 = v1 / (v1.norm() + 1e-12)

    if seed is not None:
        torch.manual_seed(seed)

    # Generate random direction and project out v1
    r = torch.randn(3 * num_atoms, device=coords.device, dtype=coords.dtype)
    r = r - torch.dot(r, v1) * v1  # Project out v1
    r = r / (r.norm() + 1e-12)
    r = r.reshape(num_atoms, 3)

    E_current = _to_float(predict_fn(coords, atomic_nums, do_hessian=False, require_grad=False)["energy"])

    for attempt in range(max_shrink_attempts + 1):
        current_delta = delta * (0.5 ** attempt)

        coords_plus = coords + current_delta * r
        coords_minus = coords - current_delta * r

        candidates = []

        if _geometry_is_valid(coords_plus, min_interatomic_dist):
            try:
                E_plus = _to_float(predict_fn(coords_plus, atomic_nums, do_hessian=False, require_grad=False)["energy"])
                if np.isfinite(E_plus):
                    candidates.append((coords_plus, +1, E_plus))
            except Exception:
                pass

        if _geometry_is_valid(coords_minus, min_interatomic_dist):
            try:
                E_minus = _to_float(predict_fn(coords_minus, atomic_nums, do_hessian=False, require_grad=False)["energy"])
                if np.isfinite(E_minus):
                    candidates.append((coords_minus, -1, E_minus))
            except Exception:
                pass

        if candidates:
            candidates.sort(key=lambda x: x[2])
            new_coords, direction, E_new = candidates[0]
            return new_coords.detach(), {
                "success": True,
                "strategy": "random_ortho_v1",
                "delta_used": current_delta,
                "direction": direction,
                "energy_before": E_current,
                "energy_after": E_new,
                "energy_change": E_new - E_current,
                "shrink_attempts": attempt,
            }

    return coords, {"success": False, "reason": "all_attempts_failed", "strategy": "random_ortho_v1"}


def kick_gradient_descent(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    hessian_proj: torch.Tensor,  # Not used
    *,
    n_steps: int = 5,
    step_size: float = 0.05,
    min_interatomic_dist: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Take gradient descent steps to escape.

    Hypothesis: Since we're at a high-index saddle, gradient descent
    will naturally move us toward a lower-energy region (minimum or
    lower-index saddle), from which we can restart GAD.

    This is motivated by iHiSD's gradient flow component.
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)

    E_initial = _to_float(predict_fn(coords, atomic_nums, do_hessian=False, require_grad=False)["energy"])
    current_coords = coords.clone()

    energies = [E_initial]
    steps_taken = 0

    for _ in range(n_steps):
        out = predict_fn(current_coords, atomic_nums, do_hessian=False, require_grad=False)
        forces = out["forces"]
        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)

        # Gradient descent: move in direction of forces (= -gradient)
        new_coords = current_coords + step_size * forces

        if not _geometry_is_valid(new_coords, min_interatomic_dist):
            break

        E_new = _to_float(predict_fn(new_coords, atomic_nums, do_hessian=False, require_grad=False)["energy"])
        if not np.isfinite(E_new):
            break

        current_coords = new_coords
        energies.append(E_new)
        steps_taken += 1

    E_final = energies[-1] if energies else E_initial

    return current_coords.detach(), {
        "success": steps_taken > 0,
        "strategy": "gradient_descent",
        "steps_taken": steps_taken,
        "step_size": step_size,
        "energy_before": E_initial,
        "energy_after": E_final,
        "energy_change": E_final - E_initial,
        "energy_trajectory": energies,
    }


def kick_ortho_v1_grad_descent(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    hessian_proj: torch.Tensor,
    *,
    n_steps: int = 5,
    step_size: float = 0.05,
    min_interatomic_dist: float = 0.5,
    tr_threshold: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Gradient descent constrained to the subspace orthogonal to v₁.

    This tests whether energy descent in the orthogonal manifold (to the tracked
    mode) is sufficient to escape without explicitly using v₂.
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    num_atoms = coords.shape[0]

    vib_evals, vib_evecs, _ = _get_vibrational_modes(hessian_proj, tr_threshold)
    if len(vib_evals) < 1:
        return coords, {"success": False, "reason": "no_modes", "strategy": "ortho_v1_grad_descent"}

    v1 = vib_evecs[:, 0]
    v1 = v1 / (v1.norm() + 1e-12)

    E_initial = _to_float(predict_fn(coords, atomic_nums, do_hessian=False, require_grad=False)["energy"])
    current_coords = coords.clone()

    energies = [E_initial]
    steps_taken = 0

    for _ in range(n_steps):
        out = predict_fn(current_coords, atomic_nums, do_hessian=False, require_grad=False)
        forces = out["forces"]
        if forces.dim() == 3 and forces.shape[0] == 1:
            forces = forces[0]
        forces = forces.reshape(-1, 3)

        forces_flat = forces.reshape(-1)
        proj = forces_flat - torch.dot(forces_flat, v1) * v1
        proj_norm = proj.norm()
        if proj_norm < 1e-12:
            break
        proj = proj.reshape(num_atoms, 3)

        new_coords = current_coords + step_size * proj

        if not _geometry_is_valid(new_coords, min_interatomic_dist):
            break

        E_new = _to_float(predict_fn(new_coords, atomic_nums, do_hessian=False, require_grad=False)["energy"])
        if not np.isfinite(E_new):
            break

        current_coords = new_coords
        energies.append(E_new)
        steps_taken += 1

    E_final = energies[-1] if energies else E_initial

    return current_coords.detach(), {
        "success": steps_taken > 0,
        "strategy": "ortho_v1_grad_descent",
        "steps_taken": steps_taken,
        "step_size": step_size,
        "energy_before": E_initial,
        "energy_after": E_final,
        "energy_change": E_final - E_initial,
        "energy_trajectory": energies,
    }


def kick_higher_modes(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    hessian_proj: torch.Tensor,
    *,
    max_mode: int = 4,  # Try v2, v3, v4
    delta: float = 0.3,
    adaptive_delta: bool = True,
    min_interatomic_dist: float = 0.5,
    max_shrink_attempts: int = 3,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Try kicking along v₂, v₃, v₄, etc. sequentially.

    For high-index saddles (index 5+), maybe we need to kick along
    a higher mode that corresponds to the "stuck" direction.
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    num_atoms = coords.shape[0]

    vib_evals, vib_evecs, _ = _get_vibrational_modes(hessian_proj)

    E_current = _to_float(predict_fn(coords, atomic_nums, do_hessian=False, require_grad=False)["energy"])

    # Try modes 1 (v2), 2 (v3), ... up to max_mode-1
    for mode_idx in range(1, min(max_mode, len(vib_evals))):
        v = vib_evecs[:, mode_idx]
        v = v / (v.norm() + 1e-12)
        lambda_v = float(vib_evals[mode_idx].item())

        # Adaptive delta
        base_delta = delta
        if adaptive_delta and lambda_v < -0.01:
            base_delta = delta / np.sqrt(abs(lambda_v))
            base_delta = min(base_delta, 1.0)

        v_3d = v.reshape(num_atoms, 3)

        for attempt in range(max_shrink_attempts + 1):
            current_delta = base_delta * (0.5 ** attempt)

            coords_plus = coords + current_delta * v_3d
            coords_minus = coords - current_delta * v_3d

            candidates = []

            if _geometry_is_valid(coords_plus, min_interatomic_dist):
                try:
                    E_plus = _to_float(predict_fn(coords_plus, atomic_nums, do_hessian=False, require_grad=False)["energy"])
                    if np.isfinite(E_plus):
                        candidates.append((coords_plus, +1, E_plus))
                except Exception:
                    pass

            if _geometry_is_valid(coords_minus, min_interatomic_dist):
                try:
                    E_minus = _to_float(predict_fn(coords_minus, atomic_nums, do_hessian=False, require_grad=False)["energy"])
                    if np.isfinite(E_minus):
                        candidates.append((coords_minus, -1, E_minus))
                except Exception:
                    pass

            if candidates:
                candidates.sort(key=lambda x: x[2])
                new_coords, direction, E_new = candidates[0]

                # Accept if energy decreased meaningfully
                if E_new < E_current - 1e-6:
                    return new_coords.detach(), {
                        "success": True,
                        "strategy": "higher_modes",
                        "mode_used": mode_idx + 1,  # v2 is mode 1, etc.
                        "lambda": lambda_v,
                        "delta_used": current_delta,
                        "direction": direction,
                        "energy_before": E_current,
                        "energy_after": E_new,
                        "energy_change": E_new - E_current,
                        "shrink_attempts": attempt,
                    }

    return coords, {"success": False, "reason": "no_mode_improved_energy", "strategy": "higher_modes"}


def kick_adaptive_k_reflect(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    hessian_proj: torch.Tensor,
    *,
    delta: float = 0.3,
    min_interatomic_dist: float = 0.5,
    max_shrink_attempts: int = 5,
    neg_threshold: float = -1e-4,
    tr_threshold: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Take one step reflecting along the FULL unstable subspace.

    This is the "academically defensible" version from iHiSD:
    At a Morse-index-m saddle, reflect along all m negative modes.

    R_m = I - 2∑ᵢ₌₁ᵐ vᵢvᵢᵀ
    direction = -R_m · ∇E

    This should destabilize the current high-index saddle.
    """
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)
    num_atoms = coords.shape[0]

    vib_evals, vib_evecs, _ = _get_vibrational_modes(hessian_proj, tr_threshold)

    # Count negative eigenvalues (Morse index)
    neg_mask = vib_evals < neg_threshold
    morse_index = int(neg_mask.sum().item())

    if morse_index <= 1:
        return coords, {"success": False, "reason": "already_at_low_index", "strategy": "adaptive_k_reflect", "morse_index": morse_index}

    # Get forces (= -gradient)
    out = predict_fn(coords, atomic_nums, do_hessian=False, require_grad=False)
    forces = out["forces"]
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    grad = -forces.reshape(-1)  # ∇E

    E_current = _to_float(out["energy"])

    # Build reflection matrix R_m = I - 2∑ᵢ₌₁ᵐ vᵢvᵢᵀ
    # Direction = -R_m · ∇E
    V_neg = vib_evecs[:, :morse_index]  # First m eigenvectors (negative eigenvalues)
    reflected_grad = grad - 2.0 * (V_neg @ (V_neg.T @ grad))
    direction = -reflected_grad  # -R_m ∇E

    direction_norm = direction.norm()
    if direction_norm < 1e-12:
        return coords, {"success": False, "reason": "zero_direction", "strategy": "adaptive_k_reflect"}

    direction = direction / direction_norm
    direction_3d = direction.reshape(num_atoms, 3)

    for attempt in range(max_shrink_attempts + 1):
        current_delta = delta * (0.5 ** attempt)

        new_coords = coords + current_delta * direction_3d

        if _geometry_is_valid(new_coords, min_interatomic_dist):
            try:
                E_new = _to_float(predict_fn(new_coords, atomic_nums, do_hessian=False, require_grad=False)["energy"])
                if np.isfinite(E_new):
                    return new_coords.detach(), {
                        "success": True,
                        "strategy": "adaptive_k_reflect",
                        "morse_index": morse_index,
                        "delta_used": current_delta,
                        "energy_before": E_current,
                        "energy_after": E_new,
                        "energy_change": E_new - E_current,
                        "shrink_attempts": attempt,
                        "direction_norm_raw": float(direction_norm.item()),
                    }
            except Exception:
                pass

    return coords, {"success": False, "reason": "all_attempts_failed", "strategy": "adaptive_k_reflect"}


# Registry of all kick strategies for easy access
KICK_STRATEGIES = {
    "v1": kick_v1,
    "v2": kick_v2,
    "random": kick_random,
    "random_ortho_v1": kick_random_ortho_v1,
    "gradient_descent": kick_gradient_descent,
    "ortho_v1_grad_descent": kick_ortho_v1_grad_descent,
    "higher_modes": kick_higher_modes,
    "adaptive_k_reflect": kick_adaptive_k_reflect,
}


def get_kick_strategy(name: str):
    """Get kick strategy by name."""
    if name not in KICK_STRATEGIES:
        raise ValueError(f"Unknown kick strategy: {name}. Available: {list(KICK_STRATEGIES.keys())}")
    return KICK_STRATEGIES[name]
