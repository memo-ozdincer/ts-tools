import os
import json
from typing import Any, Dict, List

import torch
import numpy as np
from torch_geometric.data import Data as TGData, Batch
from torch_geometric.loader import DataLoader

from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.horm.ff_lmdb import LmdbDataset
from ocpmodels.hessian_graph_transform import HessianGraphTransform


# -----------------------------
# Alignment + RMSD utilities
# -----------------------------
def find_rigid_alignment(A: np.ndarray, B: np.ndarray):
    """Kabsch (no masses), handles reflection."""
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    H = A_c.T @ B_c
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    t = b_mean - R @ a_mean
    return R, t


def get_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(((A - B) ** 2).sum(axis=1).mean()))


def align_ordered_and_get_rmsd(A, B) -> float:
    """Rigid-align A to B and compute RMSD. A,B: (N,3), same ordering."""
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor):
        B = B.detach().cpu().numpy()
    if A.shape != B.shape:
        return float("inf")
    R, t = find_rigid_alignment(A, B)
    A_aligned = (R @ A.T).T + t
    return get_rmsd(A_aligned, B)


# -----------------------------
# Pre-transform: use TS coords
# -----------------------------
class UsePos:
    def __init__(self, attr: str = "pos_transition"):
        self.attr = attr

    def __call__(self, data: TGData) -> TGData:
        pos = getattr(data, self.attr, None)
        if pos is None:
            raise ValueError(f"Data missing '{self.attr}'. Keys: {list(data.keys())}")
        data.pos = pos
        return data


# -----------------------------
# GAD (Euler) loop
# -----------------------------
def run_gad_euler_on_batch(
    calculator: EquiformerTorchCalculator,
    batch: Batch,
    n_steps: int = 200,
    dt: float = 0.01,
) -> Dict[str, Any]:
    """
    Runs GAD Euler updates on positions in `batch` using predicted Hessians.
    Returns dict with RMSD, forces, natoms.
    """
    assert int(batch.batch.max().item()) + 1 == 1, "Use batch_size=1."
    start_pos = batch.pos.detach().clone()

    for _ in range(n_steps):
        results = calculator.predict(batch, do_hessian=True)
        forces = results["forces"]  # (N,3)
        N = forces.shape[0]
        hess = results["hessian"]

        # reshape Hessian if necessary
        if hess.dim() == 1:
            side = int(hess.numel() ** 0.5)
            hess = hess.view(side, side)
        elif hess.dim() == 3 and hess.shape[0] == 1:
            hess = hess[0]
        elif hess.dim() > 2:
            hess = hess.reshape(3 * N, 3 * N)

        # smallest eigenvector
        evals, evecs = torch.linalg.eigh(hess)
        v = evecs[:, 0]
        v = v / (v.norm() + 1e-12)

        # GAD velocity: F + 2(-F·v)v   (since F = -∇V)
        f_flat = forces.reshape(-1)
        gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
        gad = gad_flat.view(N, 3)

        # Euler update
        batch.pos = batch.pos + dt * gad

    end_pos = batch.pos.detach().clone()
    rmsd_val = align_ordered_and_get_rmsd(start_pos, end_pos)

    # diagnostics at end
    res_end = calculator.predict(batch, do_hessian=False)
    F_end = res_end["forces"]
    rms_force = F_end.pow(2).mean().sqrt().item()
    max_atom_force = F_end.norm(dim=1).max().item()

    return {
        "rmsd": float(rmsd_val),
        "rms_force_end": float(rms_force),
        "max_atom_force_end": float(max_atom_force),
        "natoms": int(start_pos.shape[0]),
    }


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    HOME = os.path.expanduser("~")
    PROJECT = "/project/memo"  # big files here

    # I/O layout
    checkpoint_path = os.path.join(PROJECT, "ts-tools/ckpt/hesspred_v1.ckpt")
    dataset_path = os.path.join(PROJECT, "ts-tools/data/rgd1_minimal_val.lmdb")
    out_dir = os.path.join(HOME, "ts-tools", "out")
    os.makedirs(out_dir, exist_ok=True)

    calculator = EquiformerTorchCalculator(
        checkpoint_path=checkpoint_path,
        hessian_method="predict",
    )

    use_pos_tf = UsePos("pos_transition")
    hess_tf = HessianGraphTransform(
        cutoff=calculator.potential.cutoff,
        max_neighbors=calculator.potential.max_neighbors,
        use_pbc=getattr(calculator.potential, "use_pbc", False),
    )
    composed_tf = lambda d: hess_tf(use_pos_tf(d))

    dataset = LmdbDataset(dataset_path, transform=composed_tf)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # config
    MAX_SAMPLES = 30
    N_STEPS = 50
    DT = 0.005

    results_summary: List[Dict[str, Any]] = []

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset:    {dataset_path}")
    print(f"Device:     {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Processing up to {MAX_SAMPLES} samples with GAD-Euler (steps={N_STEPS}, dt={DT})")

    for i, batch in enumerate(dataloader):
        if i >= MAX_SAMPLES:
            break
        try:
            out = run_gad_euler_on_batch(calculator, batch, n_steps=N_STEPS, dt=DT)
            results_summary.append({"index": i, **out})
            print(f"[{i}] RMSD={out['rmsd']:.4f} Å, "
                  f"RMS|F|_end={out['rms_force_end']:.3f} eV/Å, "
                  f"max|F_i|_end={out['max_atom_force_end']:.3f} eV/Å, "
                  f"N={out['natoms']}")
        except Exception as e:
            print(f"[{i}] ERROR: {e}")

    out_json = os.path.join(out_dir, f"rgd1_gad_rmsd_{len(results_summary)}.json")
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved GAD RMSD summary for {len(results_summary)} samples → {out_json}")