# src/gad_eigenvalue_descent.py
import os
import json
import argparse
from typing import Any, Dict, List
from collections import defaultdict

import torch
import numpy as np
from torch_geometric.data import Data as TGData, Batch as TGBatch

from .common_utils import setup_experiment, add_common_args
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.frequency_analysis import analyze_frequencies_torch
from .differentiable_projection import differentiable_massweigh_and_eckartprojection_torch as massweigh_and_eckartprojection_torch
from nets.prediction_utils import Z_TO_ATOM_SYMBOL

def find_rigid_alignment(A: np.ndarray, B: np.ndarray):
    """Kabsch algorithm. A and B are expected to be NumPy arrays."""
    a_mean = A.mean(axis=0); b_mean = B.mean(axis=0)
    A_c = A - a_mean; B_c = B - b_mean
    H = A_c.T @ B_c
    U, _, Vt = np.linalg.svd(H); V = Vt.T
    R = V @ U.T
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    t = b_mean - (R @ a_mean)
    return R, t

def get_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(((A - B) ** 2).sum(axis=1).mean()))

def align_ordered_and_get_rmsd(A, B) -> float:
    """Rigid-align A to B and compute RMSD. Handles both torch/numpy inputs."""
    if isinstance(A, torch.Tensor): A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor): B = B.detach().cpu().numpy()
    
    if A.shape != B.shape: return float("inf")
    
    # Now that A and B are guaranteed to be NumPy arrays, the rest of the function works.
    R, t = find_rigid_alignment(A, B)
    A_aligned = (R @ A.T).T + t
    return get_rmsd(A_aligned, B)
# --- END OF CORRECTION ---

def coord_atoms_to_torch_geometric(coords, atomic_nums, device):
    data = TGData(
        pos=coords.reshape(-1, 3),
        z=torch.as_tensor(atomic_nums, dtype=torch.int64, device=device),
        natoms=torch.tensor([len(atomic_nums)], dtype=torch.int64, device=device),
        cell=torch.zeros(3, 3, dtype=torch.float32, device=device),
        pbc=torch.tensor([False, False, False], dtype=torch.bool, device=device),
    )
    return TGBatch.from_data_list([data])

# --- Core Optimization Function with Improved Loss Functions ---
def run_eigenvalue_descent(
    calculator: EquiformerTorchCalculator,
    initial_coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    n_steps: int = 50,
    lr: float = 0.01,
    loss_type: str = "targeted_magnitude",
    target_eig0: float = -0.05,
    target_eig1: float = 0.10,
) -> Dict[str, Any]:
    """
    Run gradient descent on eigenvalues to find transition states.

    Args:
        loss_type: Type of loss function to use:
            - "relu": Original ReLU loss (allows infinitesimal eigenvalues)
            - "targeted_magnitude": Target specific eigenvalue magnitudes (RECOMMENDED)
            - "midpoint_squared": Minimize squared midpoint between eigenvalues
        target_eig0: Target value for most negative eigenvalue (eV/Å²)
        target_eig1: Target value for second smallest eigenvalue (eV/Å²)
    """
    model = calculator.potential
    device = model.device
    coords = initial_coords.clone().to(torch.float32).to(device)
    coords.requires_grad = True
    atomsymbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in atomic_nums]
    history = defaultdict(list)
    final_eigvals = None

    for step in range(n_steps):
        with torch.enable_grad():
            batch = coord_atoms_to_torch_geometric(coords, atomic_nums, device)
            _, _, out = model.forward(batch, otf_graph=True)
            hess_raw = out["hessian"].reshape(coords.numel(), coords.numel())
            hess_proj = massweigh_and_eckartprojection_torch(hess_raw, coords, atomsymbols)
            eigvals, _ = torch.linalg.eigh(hess_proj)
            final_eigvals = eigvals

            # Compute loss based on selected type
            if loss_type == "relu":
                # Original: allows infinitesimal eigenvalues
                loss = torch.relu(eigvals[0]) + torch.relu(-eigvals[1])

            elif loss_type == "targeted_magnitude":
                # Target specific magnitudes (RECOMMENDED)
                # Push λ₀ to be meaningfully negative, λ₁ to be meaningfully positive
                loss = (eigvals[0] - target_eig0)**2 + (eigvals[1] - target_eig1)**2

            elif loss_type == "midpoint_squared":
                # Minimize squared midpoint (from your description)
                midpoint = (eigvals[0] + eigvals[1]) / 2.0
                loss = midpoint**2

            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            # Early stopping based on loss type
            if loss_type == "relu" and loss.item() < 1e-8:
                print(f"  Stopping early at step {step}: Converged (Loss ~ 0).")
                break
            elif loss_type in ["targeted_magnitude", "midpoint_squared"] and loss.item() < 1e-6:
                print(f"  Stopping early at step {step}: Converged (Loss < 1e-6).")
                break

            grad = torch.autograd.grad(loss, coords)[0]

        with torch.no_grad():
            coords -= lr * grad

        coords.requires_grad = True

        history["loss"].append(loss.item())
        history["eig_product"].append((eigvals[0] * eigvals[1]).item())
        history["eig0"].append(eigvals[0].item())
        history["eig1"].append(eigvals[1].item())

        if step % 10 == 0:
            eig_prod = (eigvals[0] * eigvals[1]).item()
            print(f"  Step {step:03d}: Loss={loss.item():.6e}, λ₀*λ₁={eig_prod:.6e} (λ₀={eigvals[0].item():.6f}, λ₁={eigvals[1].item():.6f})")

    final_coords = coords.detach()

    return {
        "final_coords": final_coords.cpu(),
        "history": history,
        "final_loss": loss.item(),
        "final_eig_product": (final_eigvals[0] * final_eigvals[1]).item(),
        "final_eig0": final_eigvals[0].item(),
        "final_eig1": final_eigvals[1].item(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gradient descent on eigenvalues to find transition states.")
    parser = add_common_args(parser)
    parser.add_argument("--n-steps-opt", type=int, default=200,
                        help="Number of optimization steps (default: 200)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--start-from", type=str, default="reactant",
                        choices=["reactant", "ts", "midpoint_rt", "three_quarter_rt"],
                        help="Starting point for optimization")

    # Loss function arguments
    parser.add_argument("--loss-type", type=str, default="targeted_magnitude",
                        choices=["relu", "targeted_magnitude", "midpoint_squared"],
                        help="Loss function type (default: targeted_magnitude)")
    parser.add_argument("--target-eig0", type=float, default=-0.05,
                        help="Target for most negative eigenvalue in eV/Å² (default: -0.05)")
    parser.add_argument("--target-eig1", type=float, default=0.10,
                        help="Target for second smallest eigenvalue in eV/Å² (default: 0.10)")
    args = parser.parse_args()

    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)

    print(f"Running Eigenvalue Descent to Find Transition States")
    print(f"Starting From: {args.start_from.upper()}, Steps: {args.n_steps_opt}, LR: {args.lr}")
    print(f"Loss Type: {args.loss_type}")
    if args.loss_type == "targeted_magnitude":
        print(f"  Target λ₀: {args.target_eig0:.4f} eV/Å²")
        print(f"  Target λ₁: {args.target_eig1:.4f} eV/Å²")

    results_summary = []
    for i, batch in enumerate(dataloader):
        if i >= args.max_samples: break
        print(f"\n--- Processing Sample {i} (Formula: {batch.formula[0]}) ---")
        try:
            if args.start_from == "reactant": initial_coords = batch.pos_reactant
            elif args.start_from == "midpoint_rt": initial_coords = 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition
            elif args.start_from == "three_quarter_rt": initial_coords = 0.25 * batch.pos_reactant + 0.75 * batch.pos_transition
            else: initial_coords = batch.pos_transition

            opt_results = run_eigenvalue_descent(
                calculator=calculator,
                initial_coords=initial_coords,
                atomic_nums=batch.z,
                n_steps=args.n_steps_opt,
                lr=args.lr,
                loss_type=args.loss_type,
                target_eig0=args.target_eig0,
                target_eig1=args.target_eig1,
            )

            with torch.no_grad():
                final_batch = coord_atoms_to_torch_geometric(opt_results['final_coords'].to(device), batch.z, device)
                _, _, final_out = calculator.potential.forward(final_batch, otf_graph=True)
                final_freq_info = analyze_frequencies_torch(final_out['hessian'], opt_results['final_coords'].to(device), batch.z)

            summary = {
                "sample_index": i, "formula": batch.formula[0],
                "final_loss": opt_results["final_loss"],
                "final_eig_product": opt_results["final_eig_product"],
                "final_eig0": opt_results["final_eig0"],
                "final_eig1": opt_results["final_eig1"],
                "final_neg_eigvals": int(final_freq_info.get("neg_num", -1)),
                "rmsd_to_known_ts": align_ordered_and_get_rmsd(opt_results["final_coords"], batch.pos_transition),
            }
            results_summary.append(summary)

            print(f"Result for Sample {i}:")
            print(f"  Final Loss: {summary['final_loss']:.6e}")
            print(f"  Final λ₀: {summary['final_eig0']:.6f} eV/Å², λ₁: {summary['final_eig1']:.6f} eV/Å²")
            print(f"  Final λ₀*λ₁: {summary['final_eig_product']:.6e}")
            print(f"  Final Neg Eigs: {summary['final_neg_eigvals']}")
            print(f"  RMSD to T1x TS: {summary['rmsd_to_known_ts']:.4f} Å")
            
        except Exception as e:
            print(f"[ERROR] Sample {i} failed: {e}")
            import traceback
            traceback.print_exc()

    out_json = os.path.join(out_dir, f"eigdesc_{args.loss_type}_{args.start_from}_{len(results_summary)}.json")
    with open(out_json, "w") as f: json.dump(results_summary, f, indent=2)
    print(f"\nSaved summary to {out_json}")

    # Print final statistics
    if results_summary:
        print("\n" + "="*60)
        print(" " * 18 + "EIGENVALUE DESCENT SUMMARY")
        print("="*60)

        avg_eig0 = np.mean([r["final_eig0"] for r in results_summary])
        avg_eig1 = np.mean([r["final_eig1"] for r in results_summary])
        avg_eig_prod = np.mean([r["final_eig_product"] for r in results_summary])
        avg_rmsd = np.mean([r["rmsd_to_known_ts"] for r in results_summary])

        print(f"\nAverage Final Values (across {len(results_summary)} samples):")
        print(f"  λ₀: {avg_eig0:.6f} eV/Å²")
        print(f"  λ₁: {avg_eig1:.6f} eV/Å²")
        print(f"  λ₀*λ₁: {avg_eig_prod:.6e}")
        print(f"  RMSD to known TS: {avg_rmsd:.4f} Å")

        # Count how many have exactly 1 negative eigenvalue
        one_neg = sum(1 for r in results_summary if r["final_neg_eigvals"] == 1)
        print(f"\nSamples with exactly 1 negative eigenvalue (TS signature): {one_neg}/{len(results_summary)}")

        # Show distribution
        from collections import Counter
        neg_dist = Counter(r["final_neg_eigvals"] for r in results_summary)
        print(f"Distribution of negative eigenvalues:")
        for key in sorted(neg_dist.keys()):
            print(f"  {key} neg eig: {neg_dist[key]} samples")
        print("="*60)