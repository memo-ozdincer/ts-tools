from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .types import PredictFn, ensure_2d_coords
from ..dependencies.hessian import vibrational_eigvals


@dataclass(frozen=True)
class EigenProductConfig:
    lr: float = 0.01
    max_step: float = 0.1
    max_grad_norm: float = 100.0


def eig_product_from_vib(vib_eigvals: torch.Tensor) -> torch.Tensor:
    if vib_eigvals.numel() < 2:
        return vib_eigvals.new_tensor(float("inf"))
    return vib_eigvals[0] * vib_eigvals[1]


def eig_product_descent_step(
    predict_fn: PredictFn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    lr: float = 0.01,
    max_step: float = 0.1,
    max_grad_norm: float = 100.0,
) -> Dict[str, Any]:
    """One differentiable step minimizing $\lambda_0 \lambda_1$.

    This is the core of the 'eigenvalue-product' approach.

    Requires `predict_fn(..., require_grad=True)` to be autograd-differentiable.
    """

    coords0 = ensure_2d_coords(coords)
    coords_opt = coords0.clone().detach().to(torch.float32)
    coords_opt.requires_grad = True

    try:
        with torch.enable_grad():
            out = predict_fn(coords_opt, atomic_nums, do_hessian=True, require_grad=True)
            hess_raw = out.get("hessian")
            if not isinstance(hess_raw, torch.Tensor):
                raise ValueError("predict_fn must return a tensor 'hessian' when require_grad=True")

            vib = vibrational_eigvals(hess_raw, coords_opt, atomic_nums)
            if vib.numel() < 2:
                return {
                    "new_coords": coords0,
                    "eig0": float("nan"),
                    "eig1": float("nan"),
                    "eig_product": float("inf"),
                    "loss": float("inf"),
                    "grad_norm": 0.0,
                    "max_atom_disp": 0.0,
                    "success": False,
                }

            eig0 = vib[0]
            eig1 = vib[1]
            loss = eig0 * eig1
            grad = torch.autograd.grad(loss, coords_opt, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            grad_norm = float(grad.norm().item())
            if grad_norm > max_grad_norm and grad_norm > 0:
                grad = grad * (max_grad_norm / grad.norm())
                grad_norm = float(max_grad_norm)

            update = lr * grad
            update_per_atom = update.reshape(-1, 3)
            atom_displacements = torch.norm(update_per_atom, dim=1)
            max_atom_disp = float(atom_displacements.max().item()) if atom_displacements.numel() else 0.0

            if max_atom_disp > max_step and max_atom_disp > 0:
                scale = max_step / max_atom_disp
                update = update * scale
                max_atom_disp = float(max_step)

            new_coords = coords_opt.detach() - update

        return {
            "new_coords": new_coords.detach(),
            "eig0": float(eig0.detach().cpu().item()),
            "eig1": float(eig1.detach().cpu().item()),
            "eig_product": float((eig0 * eig1).detach().cpu().item()),
            "loss": float(loss.detach().cpu().item()),
            "grad_norm": float(grad_norm),
            "max_atom_disp": float(max_atom_disp),
            "success": True,
        }

    except Exception as exc:
        return {
            "new_coords": coords0,
            "eig0": float("nan"),
            "eig1": float("nan"),
            "eig_product": float("inf"),
            "loss": float("inf"),
            "grad_norm": 0.0,
            "max_atom_disp": 0.0,
            "success": False,
            "error": str(exc),
        }
