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
from hip.frequency_analysis import analyze_frequencies_torch # For ANALYSIS ONLY

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

# --- (coord_atoms_to_torch_geometric is unchanged) ---
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

# --- Core Optimization Function with HYBRID approach AND EARLY STOPPING ---
def run_eigenvalue_descent(
    calculator: EquiformerTorchCalculator,
    initial_coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    n_steps: int = 50,
    lr: float = 0.01,
    stop_at_ts: bool = False,
) -> Dict[str, Any]:
    """
    Performs gradient descent by optimizing eigenvalues of the FULL Hessian,
    analyzing with the PROJECTED Hessian, and optionally stopping when a TS is found.
    """
    potential = calculator.potential
    device = potential.device
    
    coords = torch.nn.Parameter(initial_coords.clone().to(device))
    optimizer = torch.optim.Adam([coords], lr=lr)
    history = defaultdict(list)
    steps_taken = 0

    for step in range(n_steps):
        steps_taken = step + 1
        optimizer.zero_grad()
        batch = coord_atoms_to_torch_geometric(coords, atomic_nums).to(device)

        with torch.enable_grad():
            _, _, out = potential.forward(batch, otf_graph=True)
            
        hessian = out["hessian"]
        full_eigvals, _ = torch.linalg.eigh(hessian)
        loss = full_eigvals[0] * full_eigvals[1]
        
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            projected_freq_info = analyze_frequencies_torch(hessian.detach(), coords.detach(), atomic_nums)
            projected_eigvals = projected_freq_info["eigvals"].cpu()
            projected_product = (projected_eigvals[0] * projected_eigvals[1]).item()
        
        history["loss"].append(loss.item())
        history["projected_eigval_0"].append(projected_eigvals[0].item())
        history["projected_eigval_1"].append(projected_eigvals[1].item())
        
        if step % 10 == 0:
            print(f"  Step {step:03d}: Loss(unproj)={loss.item():.4f}, Proj λ0*λ1={projected_product:.4f}")
            
        # --- EARLY STOPPING CHECK ---
        if stop_at_ts and projected_product < -1e-5: # Use a small threshold
            print(f"  Stopping early at step {step}: Found TS signature (product = {projected_product:.4f})")
            break

    final_coords = coords.detach().clone()
    
    with torch.no_grad():
        final_batch = coord_atoms_to_torch_geometric(final_coords, atomic_nums).to(device)
        _, _, final_out = potential.forward(final_batch, otf_graph=True)
        final_freq_info = analyze_frequencies_torch(final_out["hessian"], final_coords, atomic_nums)

    return {
        "final_coords": final_coords.cpu(),
        "history": history,
        "steps_taken": steps_taken,
        "final_eig_product": (final_freq_info["eigvals"][0] * final_freq_info["eigvals"][1]).item(),
        "final_neg_eigvals": int(final_freq_info.get("neg_num", -1)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gradient descent on Hessian eigenvalues to find transition states.")
    parser = add_common_args(parser)
    parser.add_argument("--n-steps-opt", type=int, default=50, help="Number of optimization steps.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the Adam optimizer.")
    parser.add_argument("--start-from", type=str, default="reactant", 
                        choices=["reactant", "ts", "midpoint_rt", "three_quarter_rt"],
                        help="Which geometry to start the optimization from.")
    # --- NEW ARGUMENT ---
    parser.add_argument("--stop-at-ts", action="store_true", 
                        help="Stop optimization as soon as projected eigenvalue product becomes negative.")
    args = parser.parse_args()
    
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    
    print(f"Running Eigenvalue Product Minimization for {args.max_samples} samples.")
    mode_str = "Search until TS found" if args.stop_at_ts else f"Fixed {args.n_steps_opt} steps"
    print(f"Mode: {mode_str}, Starting From: {args.start_from.upper()}, LR: {args.lr}")

    results_summary = []
    for i, batch in enumerate(dataloader):
        if i >= args.max_samples: break
        print(f"\n--- Processing Sample {i} (Formula: {batch.formula[0]}) ---")
        try:
            if args.start_from == "reactant":
                initial_coords = batch.pos_reactant
            elif args.start_from == "midpoint_rt":
                initial_coords = 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition
            elif args.start_from == "three_quarter_rt":
                initial_coords = 0.25 * batch.pos_reactant + 0.75 * batch.pos_transition
            else:
                initial_coords = batch.pos_transition

            atomic_nums = batch.z
            target_ts_coords = batch.pos_transition
            
            opt_results = run_eigenvalue_descent(
                calculator=calculator,
                initial_coords=initial_coords,
                atomic_nums=atomic_nums,
                n_steps=args.n_steps_opt,
                lr=args.lr,
                stop_at_ts=args.stop_at_ts, # Pass the flag
            )
            
            with torch.no_grad():
                initial_batch = coord_atoms_to_torch_geometric(initial_coords, atomic_nums).to(device)
                _, _, initial_out = calculator.potential.forward(initial_batch, otf_graph=True)
                initial_freq_info = analyze_frequencies_torch(initial_out["hessian"], initial_coords, atomic_nums)
            initial_eigvals = initial_freq_info["eigvals"].cpu()
            initial_eig_product = (initial_eigvals[0] * initial_eigvals[1]).item()

            summary = {
                "sample_index": i, "formula": batch.formula[0],
                "steps_taken": opt_results["steps_taken"],
                "initial_eig_product": initial_eig_product,
                "final_eig_product": opt_results["final_eig_product"],
                "initial_neg_eigvals": int(initial_freq_info.get("neg_num", -1)),
                "final_neg_eigvals": opt_results["final_neg_eigvals"],
                "rmsd_to_known_ts": align_ordered_and_get_rmsd(opt_results["final_coords"], target_ts_coords),
                "history": opt_results["history"],
            }
            results_summary.append(summary)
            
            stop_reason = "Found TS" if (args.stop_at_ts and opt_results['steps_taken'] < args.n_steps_opt) else "Max Steps"
            print(f"Result for Sample {i}:")
            print(f"  Steps Taken: {summary['steps_taken']} ({stop_reason})")
            print(f"  Projected Eig Product: {summary['initial_eig_product']:.4f} -> {summary['final_eig_product']:.4f}")
            print(f"  Projected Neg Eigs: {summary['initial_neg_eigvals']} -> {summary['final_neg_eigvals']}")
            print(f"  RMSD to T1x TS: {summary['rmsd_to_known_ts']:.4f} Å")
            
        except Exception as e:
            print(f"[ERROR] Failed to process sample {i}: {e}")
            import traceback
            traceback.print_exc()

    filename_suffix = f"{args.start_from}"
    if args.stop_at_ts: filename_suffix += "_stopts"
    out_json = os.path.join(out_dir, f"full_eig_descent_{filename_suffix}_{len(results_summary)}.json")
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved summary for {len(results_summary)} samples to {out_json}")