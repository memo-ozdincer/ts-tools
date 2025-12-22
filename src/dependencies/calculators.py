from __future__ import annotations

from typing import Any, Dict

import torch

from ..core_algos.types import PredictFn
from .pyg_batch import coords_to_pyg_batch


def make_hip_predict_fn(calculator) -> PredictFn:
    """Adapter for HIP EquiformerTorchCalculator.

    - `require_grad=False`: uses `calculator.predict(...)` (fast, no autograd).
    - `require_grad=True`: calls `calculator.potential.forward(...)` so downstream
      code can differentiate w.r.t. `coords`.

    This matches patterns already used in your repo.
    """

    model = calculator.potential

    def _predict(
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        *,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> Dict[str, Any]:
        device = coords.device
        batch = coords_to_pyg_batch(coords, atomic_nums, device=device)

        if require_grad:
            if not do_hessian:
                raise ValueError("HIP differentiable path expects do_hessian=True")
            with torch.enable_grad():
                # Equiformer returns (energy, forces, out) in some checkpoints;
                # we standardize to a dict.
                _, _, out = model.forward(batch, otf_graph=True)
                # Common keys used across your scripts
                energy = out.get("energy")
                forces = out.get("forces")
                hessian = out.get("hessian")
                if energy is None and "energy" in out:
                    energy = out["energy"]
                return {"energy": energy, "forces": forces, "hessian": hessian}

        # Non-differentiable fast path
        with torch.no_grad():
            return calculator.predict(batch, do_hessian=do_hessian)

    return _predict


def make_scine_predict_fn(scine_calculator) -> PredictFn:
    """Adapter for `ScineSparrowCalculator`.

    SCINE is CPU-only and not differentiable w.r.t coords via autograd.
    """

    def _predict(
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        *,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> Dict[str, Any]:
        if require_grad:
            raise NotImplementedError(
                "SCINE backend is not autograd-differentiable; use require_grad=False"
            )

        batch = coords_to_pyg_batch(coords.detach().cpu(), atomic_nums.detach().cpu(), device=torch.device("cpu"))
        with torch.no_grad():
            return scine_calculator.predict(batch, do_hessian=do_hessian)

    return _predict
