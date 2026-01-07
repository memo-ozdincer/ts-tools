"""Analyze discrepancy between HIP's predicted Hessian and autograd Hessian.

This script compares two ways of computing the Hessian with HIP:
1. Predicted Hessian: Direct neural network output (hessian_method="predict")
2. Autograd Hessian: Computed as H = -∂F/∂x via automatic differentiation

For Sella's trust radius mechanism to work correctly, the Hessian must be
consistent with the forces (H = -∂F/∂x). If the predicted Hessian differs
significantly from the autograd Hessian, Sella's quadratic energy model
will be inaccurate, leading to erratic trust radius updates.

Usage:
    python -m src.experiments.Sella.analyze_hessian_discrepancy \
        --checkpoint-path /path/to/hip_v2.ckpt \
        --h5-path /path/to/transition1x.h5 \
        --max-samples 10
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data as TGData
from torch_geometric.data import Batch as TGBatch

from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.hessian_utils import compute_hessian
from hip.inference_utils import get_model_from_checkpoint


def coords_to_batch(coords: torch.Tensor, z: torch.Tensor, device: str) -> TGBatch:
    """Convert coords and atomic numbers to PyG batch."""
    data = TGData(
        pos=coords.reshape(-1, 3),
        z=z,
        charges=z,
        natoms=torch.tensor([len(z)], dtype=torch.int64, device=device),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool, device=device),
    )
    return TGBatch.from_data_list([data])


def compute_both_hessians(
    model: torch.nn.Module,
    coords: torch.Tensor,
    z: torch.Tensor,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Compute both predicted and autograd Hessians.

    Returns:
        H_pred: Predicted Hessian (3N, 3N)
        H_autograd: Autograd Hessian (3N, 3N)
        stats: Dictionary of comparison statistics
    """
    n_atoms = len(z)

    # 1. Compute predicted Hessian (direct model output)
    coords_pred = coords.clone().to(device)
    z_dev = z.to(device)
    batch_pred = coords_to_batch(coords_pred, z_dev, device)

    with torch.no_grad():
        energy_pred, forces_pred, out = model.forward(batch_pred, otf_graph=True)
        H_pred = out["hessian"].reshape(3 * n_atoms, 3 * n_atoms).cpu().numpy()

    # 2. Compute autograd Hessian (H = -∂F/∂x)
    coords_ag = coords.clone().to(device).requires_grad_(True)
    batch_ag = coords_to_batch(coords_ag, z_dev, device)
    batch_ag.pos = coords_ag  # Ensure we use the coords with requires_grad

    with torch.enable_grad():
        energy_ag, forces_ag, _ = model.forward(batch_ag, otf_graph=True)
        H_autograd_t = compute_hessian(
            coords=batch_ag.pos,
            energy=energy_ag,
            forces=forces_ag,
        )

    H_autograd = H_autograd_t.detach().reshape(3 * n_atoms, 3 * n_atoms).cpu().numpy()

    # 3. Compute comparison statistics
    diff = H_pred - H_autograd

    # Frobenius norm of difference
    frob_diff = np.linalg.norm(diff)
    frob_pred = np.linalg.norm(H_pred)
    frob_autograd = np.linalg.norm(H_autograd)
    rel_diff = frob_diff / (0.5 * (frob_pred + frob_autograd) + 1e-10)

    # Eigenvalue comparison
    eigvals_pred = np.linalg.eigvalsh(H_pred)
    eigvals_autograd = np.linalg.eigvalsh(H_autograd)
    eigval_diff = np.abs(eigvals_pred - eigvals_autograd)

    # Element-wise correlation
    corr = np.corrcoef(H_pred.ravel(), H_autograd.ravel())[0, 1]

    # Max absolute difference
    max_diff = np.max(np.abs(diff))

    # Number of negative eigenvalues
    n_neg_pred = np.sum(eigvals_pred < -1e-6)
    n_neg_autograd = np.sum(eigvals_autograd < -1e-6)

    stats = {
        "frob_diff": float(frob_diff),
        "frob_pred": float(frob_pred),
        "frob_autograd": float(frob_autograd),
        "rel_diff": float(rel_diff),
        "correlation": float(corr),
        "max_element_diff": float(max_diff),
        "mean_eigval_diff": float(np.mean(eigval_diff)),
        "max_eigval_diff": float(np.max(eigval_diff)),
        "eigval0_pred": float(eigvals_pred[0]),
        "eigval0_autograd": float(eigvals_autograd[0]),
        "eigval1_pred": float(eigvals_pred[1]),
        "eigval1_autograd": float(eigvals_autograd[1]),
        "n_neg_pred": int(n_neg_pred),
        "n_neg_autograd": int(n_neg_autograd),
    }

    return H_pred, H_autograd, stats


def main():
    parser = argparse.ArgumentParser(description="Analyze HIP Hessian discrepancy")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    print(f"Loading HIP model from: {args.checkpoint_path}")
    model = get_model_from_checkpoint(args.checkpoint_path, args.device)
    model.eval()

    # Load dataset
    from ...dependencies.data import Transition1xDataset, UsePos
    from torch.utils.data import DataLoader

    print(f"Loading dataset: {args.h5_path}")
    dataset = Transition1xDataset(
        h5_path=args.h5_path,
        split=args.split,
        max_samples=args.max_samples,
        transform=UsePos("pos_transition"),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"\nAnalyzing {len(dataset)} samples...")
    print("=" * 80)

    all_stats: List[Dict[str, float]] = []

    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break

        coords = batch.pos.to(args.device)
        z = batch.z.to(args.device)
        formula = getattr(batch, "formula", ["unknown"])[0]

        try:
            H_pred, H_autograd, stats = compute_both_hessians(model, coords, z, args.device)
            all_stats.append(stats)

            print(f"\n[{i}] {formula} ({len(z)} atoms)")
            print(f"  Frobenius norm: pred={stats['frob_pred']:.2f}, autograd={stats['frob_autograd']:.2f}")
            print(f"  Frobenius diff: {stats['frob_diff']:.2f} (relative: {stats['rel_diff']:.2%})")
            print(f"  Element correlation: {stats['correlation']:.4f}")
            print(f"  Max element diff: {stats['max_element_diff']:.4f}")
            print(f"  Eigenvalue diff: mean={stats['mean_eigval_diff']:.4f}, max={stats['max_eigval_diff']:.4f}")
            print(f"  λ₀: pred={stats['eigval0_pred']:.4f}, autograd={stats['eigval0_autograd']:.4f}")
            print(f"  λ₁: pred={stats['eigval1_pred']:.4f}, autograd={stats['eigval1_autograd']:.4f}")
            print(f"  Neg eigenvalues: pred={stats['n_neg_pred']}, autograd={stats['n_neg_autograd']}")

        except Exception as e:
            print(f"\n[{i}] FAILED: {e}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if all_stats:
        keys = ["rel_diff", "correlation", "max_element_diff", "mean_eigval_diff", "max_eigval_diff"]
        for key in keys:
            values = [s[key] for s in all_stats]
            print(f"{key}:")
            print(f"  mean={np.mean(values):.4f}, std={np.std(values):.4f}")
            print(f"  min={np.min(values):.4f}, max={np.max(values):.4f}")

        # Eigenvalue sign agreement
        n_neg_pred = [s["n_neg_pred"] for s in all_stats]
        n_neg_autograd = [s["n_neg_autograd"] for s in all_stats]
        sign_match = sum(1 for p, a in zip(n_neg_pred, n_neg_autograd) if p == a)
        print(f"\nNeg eigenvalue count agreement: {sign_match}/{len(all_stats)} ({sign_match/len(all_stats):.1%})")

        # Correlation summary
        corrs = [s["correlation"] for s in all_stats]
        print(f"\nOverall Hessian correlation: {np.mean(corrs):.4f} ± {np.std(corrs):.4f}")

        # Relative difference summary
        rel_diffs = [s["rel_diff"] for s in all_stats]
        print(f"Mean relative Frobenius diff: {np.mean(rel_diffs):.2%} ± {np.std(rel_diffs):.2%}")


if __name__ == "__main__":
    main()
