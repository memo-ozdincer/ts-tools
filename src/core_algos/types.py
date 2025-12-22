from __future__ import annotations

from typing import Any, Callable, Dict, Protocol

import torch


class PredictFn(Protocol):
    """Callable interface for model/FF predictions.

    Core algorithms call this to obtain `energy`, `forces`, and optionally a
    `hessian` for a single geometry.

    Notes:
    - `require_grad=True` should return tensors connected to `coords` so autograd
      can compute gradients through the prediction pipeline.
    - For non-differentiable backends (e.g., SCINE), `require_grad=True` may not
      be supported.
    """

    def __call__(
        self,
        coords: torch.Tensor,
        atomic_nums: torch.Tensor,
        *,
        do_hessian: bool = True,
        require_grad: bool = False,
    ) -> Dict[str, Any]: ...


TensorDict = Dict[str, torch.Tensor]


def ensure_2d_coords(coords: torch.Tensor) -> torch.Tensor:
    if coords.dim() == 1:
        return coords.reshape(-1, 3)
    return coords
