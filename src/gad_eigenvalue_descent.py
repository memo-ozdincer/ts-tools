# src/gad_manual_eig_descent.py
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

# --- CORRECTED HELPER FUNCTIONS ---
def find_rigid_alignment(A: np.ndarray, B: np.ndarray):
    """Kabsch algorithm for rigid alignment. Expects NumPy arrays."""
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
    """Rigid-align A to B and compute RMSD, ensuring inputs are NumPy arrays."""
    # THE FIX: Convert any PyTorch Tensors to NumPy arrays right at the start.
    if isinstance(A, torch.Tensor): A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor): B = B.detach().cpu().numpy()
    
    if A.shape != B.shape: return float("inf")
    
    # Now that A and B are guaranteed to be NumPy arrays, the rest works.
    R, t = find_rigid_alignment(A, B)
    A_aligned = (R @ A.T).T + t
    return get_rmsd(A_aligned, B)
# --- END CORRECTION ---


def coord_atoms_to_torch_geometric(coords, atomic_nums, device):
    data = TGData(
        pos=coords.reshape(-1, 3),
        z=torch.as_tensor(atomic_nums, dtype=torch.int64, device=device),
        natoms=torch.tensor([len(atomic_nums)], dtype=torch.int64, device=device),
        cell=torch.zeros(3, 3, dtype=torch.float32, device=device),
        pbc=torch.tensor([False, False, False], dtype=torch.bool, device=device),
    )
    return TGBatch.from_data_list([data])

# --- Core Optimization Function with YOUR LOGIC ---
def run_manual_eigenvalue_descent(
    calculator: EquiformerTorchCalculator,
    initial_coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    n_steps: int = 50,
    lr: float = 0.01,
) -> Dict[str, Any]:
    
    model = calculator.potential
    device = model.device
    
    coords = initial_coords.clone().to(torch.float32).to(device)
    coords.requires_grad = True 
    
    atomsymbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in atomic_nums]
    history = defaultdict(list)
    current_prod = float('nan')

    for step in range(n_steps):
        with torch.enable_grad():
            batch = coord_atoms_to_torch_geometric(coords, atomic_nums, device)
            _, _, out = model.forward(batch, otf_graph=True)
            hess_raw = out["hessian"].reshape(coords.numel(), coords.numel())
            hess_proj = massweigh_and_eckartprojection_torch(hess_raw, coords, atomsymbols)
            eigvals, _ = torch.linalg.eigh(hess_proj)
            value_to_minimize = eigvals[0] * eigvals[1]
            grad = torch.autograd.grad(value_to_minimize, coords)[0]
        
        with torch.no_grad():
            coords -= lr * grad
            
        coords.requires_grad = True
            
        current_prod = value_to_minimize.item()
        history["eig_product"].append(current_prod)
        if step % 10 == 0:
            print(f"  Step {step:03d}: Proj λ0*λ1 = {current_prod:.6f} (λ0={eigvals[0].item():.4f}, λ1={eigvals[1].item():.4f})")
            
    final_coords = coords.detach()
    
    return {
        "final_coords": final_coords.cpu(),
        "history": history,
        "final_eig_product": current_prod,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MANUAL gradient descent on the eigenvalue product.")
    parser = add_common_args(parser)
    parser.add_argument("--n-steps-opt", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--start-from", type=str, default="reactant", 
                        choices=["reactant", "ts", "midpoint_rt", "three_quarter_rt"])
    args = parser.parse_args()
    
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    
    print(f"Running MANUAL Eigenvalue Product Minimization.")
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

            opt_results = run_manual_eigenvalue_descent(
                calculator=calculator,
                initial_coords=initial_coords,
                atomic_nums=batch.z,
                n_steps=args.n_steps_opt,
                lr=args.lr,
            )
            
            with torch.no_grad():
                final_batch = coord_atoms_to_torch_geometric(opt_results['final_coords'].to(device), batch.z, device)
                _, _, final_out = calculator.potential.forward(final_batch, otf_graph=True)
                final_freq_info = analyze_frequencies_torch(final_out['hessian'], opt_results['final_coords'].to(device), batch.z)

            summary = {
                "sample_index": i, "formula": batch.formula[0],
                "final_eig_product": opt_results["final_eig_product"],
                "final_neg_eigvals": int(final_freq_info.get("neg_num", -1)),
                "rmsd_to_known_ts": align_ordered_and_get_rmsd(opt_results["final_coords"], batch.pos_transition),
            }
            results_summary.append(summary)
            
            print(f"Result for Sample {i}:")
            print(f"  Final Proj λ0*λ1: {summary['final_eig_product']:.5f}")
            print(f"  Final Neg Eigs: {summary['final_neg_eigvals']}")
            print(f"  RMSD to T1x TS: {summary['rmsd_to_known_ts']:.4f} Å")
            
        except Exception as e:
            print(f"[ERROR] Sample {i} failed: {e}")
            import traceback
            traceback.print_exc()

    out_json = os.path.join(out_dir, f"manual_eig_descent_{args.start_from}_{len(results_summary)}.json")
    with open(out_json, "w") as f: json.dump(results_summary, f, indent=2)
    print(f"\nSaved summary to {out_json}")