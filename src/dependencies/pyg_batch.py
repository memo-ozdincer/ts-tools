from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data as TGData


def coords_to_pyg_batch(
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
) -> Batch:
    """Create a single-structure PyG `Batch` in the format HIP expects."""

    if coords.dim() == 1:
        coords = coords.reshape(-1, 3)

    if device is None:
        device = coords.device

    data = TGData(
        pos=torch.as_tensor(coords, dtype=torch.float32),
        z=torch.as_tensor(atomic_nums, dtype=torch.int64),
        charges=torch.as_tensor(atomic_nums, dtype=torch.int64),
        natoms=torch.tensor([int(atomic_nums.numel())], dtype=torch.int64),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool),
    )

    return Batch.from_data_list([data]).to(device)
