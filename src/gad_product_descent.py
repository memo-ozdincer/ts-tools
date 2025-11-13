# src/gad_direct_gradient.py
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
from hip.frequency_analysis import analyze_frequencies_torch # For final analysis

# --- (Helper functions for RMSD are unchanged) ---
def find_rigid_alignment(A, B):
    if isinstance(A, torch.Tensor): A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor): B = B.detach().cpu().numpy()
    a_mean = A.mean(axis=0); b_mean = B.mean(axis=0)
    A_c = A - a_mean; B_c = B - b_mean
    H = A_c.T @ B_c
    U, _, Vt = np.linalg.svd(H); V = Vt.T
    R = V @ U.T
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    t = b_mean - R @ a_mean
    return R, t

def get_rmsd(A, B): return float(np.sqrt(((A - B) ** 2).sum(axis=1).mean()))

def align_ordered_and_get_rmsd(A, B):
    if A.shape != B.shape: return float("inf")
    R, t = find_rigid_alignment(A, B)
    A_aligned = (R @ A.T).T + t
    return get_rmsd(A_aligned, B)

# --- (Helper function for reshaping the Hessian) ---
def _prepare_hessian(hess: torch.Tensor, num_atoms: int) -> torch.Tensor:
    if hess.dim() == 1:
        side_dim = 3 * num_atoms
        hess = hess.reshape(side_dim, side_dim)
    elif hess.dim() == 3 and hess.shape[0] == 1:
        hess = hess[0]
    return hess

# --- YOUR PROVIDED FUNCTIONS, INTEGRATED ---
def coord_atoms_to_torch_geometric(coords, atomic_nums):
    data = TGData(
        pos=torch.as_tensor(coords, dtype=torch.float32).reshape(-1, 3),
        z=torch.as_tensor(atomic_nums, dtype=torch.int64),
        charges=torch.as_tensor(atomic_nums, dtype=torch.int64),
        natoms=torch.tensor([len(atomic_nums)], dtype=torch.int64),
        cell=torch.zeros(3, 3, dtype=torch.float32),
        pbc=torch.tensor([False, False, False], dtype=torch.bool),
    )
    return TGBatch.from_data_list([data])

def get_gradient_of_eigprod(
    potential: torch.nn.Module,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    """
    Computes the gradient of the eigenvalue-product w.r.t. input positions,
    exactly as you specified.
    """
    batch = coord_atoms_to_torch_geometric(coords, atomic_nums)
    batch = batch.to(potential.device)
    
    # Enable gradients for this specific calculation
    with torch.enable_grad():
        # Mark the input tensor as requiring a gradient
        batch.pos.requires_grad_(True)
        
        # Run the forward pass to get the hessian
        _, _, out = potential.forward(batch, otf_graph=True)
        
        hess_flat = out["hessian"]
        hess = _prepare_hessian(hess_flat, len(atomic_nums))
        
        eigvals, _ = torch.linalg.eigh(hess)
        
        prod = eigvals[0] * eigvals[1]
        
        # Explicitly ask for the gradient of the product w.r.t. positions
        grad_prod = torch.autograd.grad(prod, batch.pos, retain_graph=False)[0]
        
    return prod.detach(), grad_prod

# --- Main Optimization Loop built around YOUR function ---
def run_direct_gradient_descent(
    calculator: EquiformerTorchCalculator,
    initial_coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    n_steps: int = 50,
    lr: float = 0.01,
) -> Dict[str, Any]:
    
    potential = calculator.potential
    device = potential.device
    
    coords = initial_coords.clone().to(device)
    history = defaultdict(list)

    for step in range(n_steps):
        # 1. Call your function to get the current value and the gradient
        eig_prod, grad = get_gradient_of_eigprod(potential, coords, atomic_nums)
        
        # Log history
        history["eig_product"].append(eig_prod.item())
        if step % 10 == 0:
            print(f"  Step {step:03d}: Eig Product = {eig_prod.item():.5f}")

        # 2. Manually update the coordinates using the calculated gradient
        #    We do this in a no_grad block because the update itself should not be tracked.
        with torch.no_grad():
            coords -= lr * grad
            
    final_coords = coords.clone()

    # Final analysis
    with torch.no_grad():
        final_batch = coord_atoms_to_torch_geometric(final_coords, atomic_nums).to(device)
        _, _, final_out = potential.forward(final_batch, otf_graph=True)
        final_freq_info = analyze_frequencies_torch(final_out["hessian"], final_coords, atomic_nums)

    return {
        "final_coords": final_coords.cpu(),
        "history": history,
        "final_eig_product": (final_freq_info["eigvals"][0] * final_freq_info["eigvals"][1]).item(),
        "final_neg_eigvals": int(final_freq_info.get("neg_num", -1)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DIRECT gradient descent on Hessian eigenvalues.")
    parser = add_common_args(parser)
    parser.add_argument("--n-steps-opt", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--start-from", type=str, default="reactant", 
                        choices=["reactant", "ts", "midpoint_rt", "three_quarter_rt"])
    args = parser.parse_args()
    
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    
    print(f"Running DIRECT Eigenvalue Product Minimization.")
    print(f"Starting From: {args.start_from.upper()}, Steps: {args.n_steps_opt}, LR: {args.lr}")

    results_summary = []
    for i, batch in enumerate(dataloader):
        if i >= args.max_samples: break
        print(f"\n--- Processing Sample {i} (Formula: {batch.formula[0]}) ---")
        try:
            if args.start_from == "reactant": initial_coords = batch.pos_reactant
            elif args.start_from == "midpoint_rt": initial_coords = 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition
            elif args.start_from == "three_quarter_rt": initial_coords = 0.25 * batch.pos_reactant + 0.75 * batch.pos_transition
            else: initial_coords = batch.pos_transition

            opt_results = run_direct_gradient_descent(
                calculator=calculator,
                initial_coords=initial_coords,
                atomic_nums=batch.z,
                n_steps=args.n_steps_opt,
                lr=args.lr,
            )
            
            # ... (rest of the analysis and summary saving is similar)
            with torch.no_grad():
                initial_freq_info = analyze_frequencies_torch(
                    calculator.predict(TGBatch.from_data_list([TGData(pos=initial_coords, z=batch.z, natoms=torch.tensor([len(batch.z)]))]).to(device), do_hessian=True)["hessian"], 
                    initial_coords, 
                    batch.z
                )
            initial_eig_product = (initial_freq_info["eigvals"][0] * initial_freq_info["eigvals"][1]).item()

            summary = {
                "sample_index": i, "formula": batch.formula[0],
                "initial_eig_product": initial_eig_product,
                "final_eig_product": opt_results["final_eig_product"],
                "final_neg_eigvals": opt_results["final_neg_eigvals"],
            }
            results_summary.append(summary)
            
            print("Result:")
            print(f"  Eig Product: {summary['initial_eig_product']:.4f} -> {summary['final_eig_product']:.4f}")
            print(f"  Neg Eigs: {initial_freq_info.get('neg_num', -1)} -> {summary['final_neg_eigvals']}")
            
        except Exception as e:
            print(f"[ERROR] Sample {i} failed: {e}")
            import traceback
            traceback.print_exc()

    out_json = os.path.join(out_dir, f"direct_grad_descent_{args.start_from}_{len(results_summary)}.json")
    with open(out_json, "w") as f: json.dump(results_summary, f, indent=2)
    print(f"\nSaved summary to {out_json}")