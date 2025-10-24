# src/gad_gad_euler_rmsd.py
import os
import json
import re
import argparse
from typing import Any, Dict, List, Optional
from collections import defaultdict

import torch
import numpy as np
from torch_geometric.data import Batch

from .common_utils import setup_experiment, add_common_args
from hip.frequency_analysis import analyze_frequencies_torch, eckart_projection_notmw_torch
from torch_geometric.data import Data as TGData
from torch_geometric.data import Batch as TGBatch
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.masses import MASS_DICT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Helper functions for RMSD (copied from previous script) ---
def find_rigid_alignment(A: np.ndarray, B: np.ndarray):
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

def get_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(((A - B) ** 2).sum(axis=1).mean()))

def align_ordered_and_get_rmsd(A, B) -> float:
    if isinstance(A, torch.Tensor): A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor): B = B.detach().cpu().numpy()
    if A.shape != B.shape: return float("inf")
    R, t = find_rigid_alignment(A, B)
    A_aligned = (R @ A.T).T + t
    return get_rmsd(A_aligned, B)


def coord_atoms_to_torch_geometric(
    coords,  # (N, 3)
    atomic_nums,  # (N,)
):
    """
    Convert raw coords and atomic numbers to torch_geometric Data format expected by Equiformer.
    """
    data = TGData(
        pos=torch.as_tensor(coords, dtype=torch.float32).reshape(-1, 3),
        z=torch.as_tensor(atomic_nums, dtype=torch.int64),
        charges=torch.as_tensor(atomic_nums, dtype=torch.int64),
        natoms=torch.tensor([len(atomic_nums)], dtype=torch.int64),
        cell=torch.zeros(3, 3, dtype=torch.float32), # dummy cell
        pbc=torch.tensor([False, False, False], dtype=torch.bool),
    )
    return TGBatch.from_data_list([data])


# --- Core Optimization Function on PROJECTED Eigenvalues ---
def run_projected_eigenvalue_descent(
    calculator: EquiformerTorchCalculator,
    initial_coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    n_steps: int = 50,
    lr: float = 0.01,
) -> Dict[str, Any]:
    """
    Performs gradient descent to minimize the product of the two smallest
    PROJECTED (vibrational) Hessian eigenvalues.
    """
    potential = calculator.potential
    device = potential.device
    
    coords = torch.nn.Parameter(initial_coords.clone().to(device))
    optimizer = torch.optim.Adam([coords], lr=lr)
    history = defaultdict(list)

    for step in range(n_steps):
        optimizer.zero_grad()
        batch = coord_atoms_to_torch_geometric(coords, atomic_nums).to(device)

        with torch.enable_grad():
            _, _, out = potential.forward(batch, otf_graph=True)
            
        hessian = out["hessian"]
        
        # --- KEY CHANGE: Use the projected eigenvalues for the loss ---
        # We backpropagate through the Eckart projection performed by this function.
        freq_info = analyze_frequencies_torch(hessian, coords, atomic_nums)
        projected_eigvals = freq_info["eigvals"]
        
        # The loss is the product of the two smallest PROJECTED eigenvalues
        loss = projected_eigvals[0] * projected_eigvals[1]
        
        loss.backward()
        optimizer.step()

        # Logging
        history["loss"].append(loss.item())
        history["eigval_0"].append(projected_eigvals[0].item())
        history["eigval_1"].append(projected_eigvals[1].item())
        
        if step % 10 == 0:
            print(f"  Step {step:03d}: Loss (λ0*λ1) = {loss.item():.5f}, λ0 = {projected_eigvals[0].item():.5f}, λ1 = {projected_eigvals[1].item():.5f}")

    final_coords = coords.detach().clone()
    
    # Final analysis is just the last step's result
    final_freq_info = analyze_frequencies_torch(hessian.detach(), final_coords, atomic_nums)

    return {
        "final_coords": final_coords.cpu(),
        "history": history,
        "final_eig_product": history["loss"][-1],
        "final_neg_eigvals": int(final_freq_info.get("neg_num", -1)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gradient descent on PROJECTED Hessian eigenvalues.")
    parser = add_common_args(parser)
    parser.add_argument("--n-steps-opt", type=int, default=50, help="Number of optimization steps.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the Adam optimizer.")
    parser.add_argument("--start-from", type=str, default="reactant", 
                        choices=["reactant", "ts", "midpoint_rt", "three_quarter_rt"],
                        help="Which geometry to start the optimization from.")
    args = parser.parse_args()
    
    # Use shuffle=False for reproducible comparison runs
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    
    print(f"Running Projected Eigenvalue Product Minimization for {args.max_samples} samples.")
    print(f"Starting From: {args.start_from.upper()}, Optimizer: Adam, Steps: {args.n_steps_opt}, LR: {args.lr}")

    results_summary: List[Dict[str, Any]] = []

    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break
        
        print(f"\n--- Processing Sample {i} (Formula: {batch.formula[0]}) ---")
        
        try:
            # Select the starting coordinates
            if args.start_from == "reactant":
                initial_coords = batch.pos_reactant
            elif args.start_from == "midpoint_rt":
                initial_coords = 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition
            elif args.start_from == "three_quarter_rt":
                initial_coords = 0.25 * batch.pos_reactant + 0.75 * batch.pos_transition
            else: # "ts"
                initial_coords = batch.pos_transition

            atomic_nums = batch.z
            target_ts_coords = batch.pos_transition
            
            # Run the optimization
            opt_results = run_projected_eigenvalue_descent(
                calculator=calculator,
                initial_coords=initial_coords,
                atomic_nums=atomic_nums,
                n_steps=args.n_steps_opt,
                lr=args.lr,
            )
            
            # Calculate initial state for comparison
            initial_batch = coord_atoms_to_torch_geometric(initial_coords, atomic_nums).to(device)
            _, _, initial_out = calculator.potential.forward(initial_batch, otf_graph=True)
            initial_freq_info = analyze_frequencies_torch(initial_out["hessian"], initial_coords, atomic_nums)
            initial_eigvals = initial_freq_info["eigvals"].cpu().numpy()
            initial_eig_product = initial_eigvals[0] * initial_eigvals[1]

            rmsd_to_target = align_ordered_and_get_rmsd(opt_results["final_coords"], target_ts_coords)

            summary = {
                "sample_index": i,
                "formula": batch.formula[0],
                "initial_eig_product": initial_eig_product,
                "final_eig_product": opt_results["final_eig_product"],
                "initial_neg_eigvals": int(initial_freq_info.get("neg_num", -1)),
                "final_neg_eigvals": opt_results["final_neg_eigvals"],
                "rmsd_to_known_ts": rmsd_to_target,
                "history": opt_results["history"],
            }
            results_summary.append(summary)
            
            print(f"Result for Sample {i}:")
            print(f"  Eigenvalue Product: {summary['initial_eig_product']:.5f} -> {summary['final_eig_product']:.5f}")
            print(f"  Negative Eigenvalues: {summary['initial_neg_eigvals']} -> {summary['final_neg_eigvals']}")
            print(f"  RMSD to T1x TS: {summary['rmsd_to_known_ts']:.4f} Å")
            
        except Exception as e:
            print(f"[ERROR] Failed to process sample {i}: {e}")
            import traceback
            traceback.print_exc()

    out_json = os.path.join(out_dir, f"projected_eig_descent_{args.start_from}_{len(results_summary)}.json")
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved summary for {len(results_summary)} samples to {out_json}")