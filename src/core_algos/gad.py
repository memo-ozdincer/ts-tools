from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import torch

from .types import PredictFn, ensure_2d_coords


@dataclass(frozen=True)
class GADConfig:
    method: Literal["euler", "rk45"] = "euler"
    n_steps: int = 50
    dt: float = 0.005  # Euler dt
    rk45_t1: float = 1.0
    rk45_rtol: float = 1e-6
    rk45_atol: float = 1e-9
    rk45_max_steps: int = 10_000


def _prepare_hessian(hess: torch.Tensor, num_atoms: int) -> torch.Tensor:
    if hess.dim() == 1:
        side = int(hess.numel() ** 0.5)
        return hess.view(side, side)
    if hess.dim() == 3 and hess.shape[0] == 1:
        hess = hess[0]
    if hess.dim() > 2:
        return hess.reshape(3 * num_atoms, 3 * num_atoms)
    return hess


def compute_gad_vector(forces: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    """Compute the GAD direction from forces and (unprojected) Hessian."""

    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]

    forces = forces.reshape(-1, 3)
    num_atoms = int(forces.shape[0])

    hess = _prepare_hessian(hessian, num_atoms)

    # Lowest eigenvector of full Hessian
    evals, evecs = torch.linalg.eigh(hess)
    v = evecs[:, 0].to(forces.dtype)
    v = v / (v.norm() + 1e-12)

    f_flat = forces.reshape(-1)
    gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
    return gad_flat.view(num_atoms, 3)


def gad_euler_step(
    predict_fn: PredictFn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    dt: float,
    out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    coords0 = ensure_2d_coords(coords)
    if out is None:
        out = predict_fn(coords0, atomic_nums, do_hessian=True, require_grad=False)

    forces = out["forces"]
    hessian = out["hessian"]

    gad_vec = compute_gad_vector(forces, hessian)
    new_coords = coords0 + dt * gad_vec

    return {
        "new_coords": new_coords,
        "gad_vec": gad_vec,
        "out": out,
    }


class RK45:
    """Minimal adaptive RK45 integrator (ported from `src/gad_rk45_search.py`)."""

    def __init__(
        self,
        f,
        t0: float,
        y0: np.ndarray,
        t1: float,
        *,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        h0: Optional[float] = None,
        h_max: float = np.inf,
        safety: float = 0.9,
        max_steps: int = 10_000,
    ):
        self.f = f
        self.t = float(t0)
        self.t_end = float(t1)
        self.y = np.array(y0, dtype=float)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.h_max = float(h_max)
        self.safety = float(safety)
        self.max_steps = int(max_steps)
        self.step_count = 0
        self.direction = np.sign(self.t_end - self.t) if self.t_end != self.t else 1.0

        if h0 is None:
            f0 = np.asarray(self.f(self.t, self.y))
            scale = self.atol + np.abs(self.y) * self.rtol
            d0 = np.linalg.norm(self.y / scale)
            d1 = np.linalg.norm(f0 / scale)
            h0 = 0.01 * d0 / d1 if d0 > 1e-5 and d1 > 1e-5 else 1e-6

        self.h = self.direction * min(abs(float(h0)), self.h_max)

        self.c = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1], dtype=float)
        self.a = [
            [],
            [1 / 5],
            [3 / 40, 9 / 40],
            [44 / 45, -56 / 15, 32 / 9],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
        ]
        self.b5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=float)
        self.b4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40], dtype=float)

    def step(self):
        t, y, h = self.t, self.y, self.h
        h = min(h, self.t_end - t) if self.direction > 0 else max(h, self.t_end - t)
        if abs(h) < 1e-15:
            return None

        k = []
        for i in range(7):
            yi = y.copy() + h * sum(a_ij * k[j] for j, a_ij in enumerate(self.a[i]))
            k.append(np.asarray(self.f(t + self.c[i] * h, yi)))
        k = np.array(k)

        y5 = y + h * np.tensordot(self.b5, k, axes=(0, 0))
        y4 = y + h * np.tensordot(self.b4, k, axes=(0, 0))
        err = y5 - y4

        scale = self.atol + np.maximum(np.abs(y), np.abs(y5)) * self.rtol
        err_norm = np.linalg.norm(err / scale) / np.sqrt(err.size)

        if err_norm <= 1.0:
            self.t, self.y = t + h, y5
            self.step_count += 1
            factor = self.safety * (1.0 / err_norm) ** (1 / 5) if err_norm > 0 else 2.0
            self.h *= np.clip(factor, 0.2, 5.0)
            return "accepted"

        self.h *= np.clip(self.safety * (1.0 / err_norm) ** (1 / 5), 0.2, 5.0)
        return "rejected"

    def solve(self):
        while self.direction * (self.t - self.t_end) < 0:
            if self.step_count >= self.max_steps:
                break
            status = self.step()
            if status is None:
                break


def gad_rk45_integrate(
    predict_fn: PredictFn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    t1: float = 1.0,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 10_000,
) -> Dict[str, Any]:
    """Integrate GAD dynamics with RK45.

    Note: this is not differentiable; it is intended for robust dynamics.
    """

    coords0 = ensure_2d_coords(coords0)
    device = coords0.device
    n_atoms = int(coords0.shape[0])

    trajectory: List[torch.Tensor] = []

    def f(_t: float, y_flat: np.ndarray) -> np.ndarray:
        coords = torch.from_numpy(y_flat).to(device=device, dtype=torch.float32).reshape(n_atoms, 3)
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        gad_vec = compute_gad_vector(out["forces"], out["hessian"])
        trajectory.append(coords.detach().cpu())
        return gad_vec.detach().cpu().numpy().reshape(-1)

    solver = RK45(f, 0.0, coords0.detach().cpu().numpy().reshape(-1), float(t1), rtol=rtol, atol=atol, max_steps=max_steps)
    solver.solve()

    final_coords = torch.from_numpy(solver.y).to(device=device, dtype=torch.float32).reshape(n_atoms, 3)

    return {
        "final_coords": final_coords,
        "trajectory": trajectory,
        "steps": solver.step_count,
        "t_final": solver.t,
    }
