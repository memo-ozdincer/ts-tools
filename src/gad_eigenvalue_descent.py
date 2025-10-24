import os
import json
import argparse
from typing import Any, Dict
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data as TGData, Batch as TGBatch

# Use the trusted, installed library for all frequency analysis
from .common_utils import setup_experiment, add_common_args
from hip.frequency_analysis import analyze_frequencies_torch
from hip.equiformer_torch_calculator import EquiformerTorchCalculator

# --- Helper functions ---
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

# --- Main Optimization Function with Selectable Loss ---
def run_direct_gradient_descent(
    calculator: EquiformerTorchCalculator,
    initial_coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    n_steps: int = 50,
    lr: float = 0.01,
    loss_fn_type: str = "product",
) -> Dict[str, Any]:
    
    potential = calculator.potential
    device = potential.device
    
    coords = initial_coords.clone().to(device)
    history = defaultdict(list)

    with torch.enable_grad():
        coords = torch.nn.Parameter(coords)
        optimizer = torch.optim.Adam([coords], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()
            
            batch = coord_atoms_to_torch_geometric(coords, atomic_nums).to(device)
            _, _, out = potential.forward(batch, otf_graph=True)
            
            # This is the core: get projected eigenvalues, which are differentiable
            freq_info = analyze_frequencies_torch(out["hessian"], coords, atomic_nums)
            eigvals = freq_info["eigvals"]
            
            # --- Select the loss function ---
            if loss_fn_type == "product":
                loss = eigvals[0] * eigvals[1]
            elif loss_fn_type == "relu_sum":
                loss = eigvals[0] + F.relu(eigvals[1])
            else:
                raise ValueError(f"Unknown loss function: {loss_fn_type}")
            
            loss.backward()
            optimizer.step()
            
            # Logging
            history["loss"].append(loss.item())
            history["eig_product"].append((eigvals[0] * eigvals[1]).item())
            if step % 10 == 0:
                print(f"  Step {step:03d}: Loss={loss.item():.5f}, Eig Product={(eigvals[0] * eigvals[1]).item():.5f}")

    final_coords = coords.detach().clone()
    
    with torch.no_grad():
        final_batch = coord_atoms_to_torch_geometric(final_coords, atomic_nums).to(device)
        _, _, final_out = potential.forward(final_batch, otf_graph=True)
        final_freq_info = analyze_frequencies_torch(final_out["hessian"], final_coords, atomic_nums)

    return {
        "final_coords": final_coords.cpu(),
        "history": history,
        "final_freq_info": {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v for k, v in final_freq_info.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DIRECT gradient descent on projected Hessian eigenvalues.")
    parser = add_common_args(parser)
    parser.add_argument("--n-steps-opt", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--start-from", type=str, default="reactant", 
                        choices=["reactant", "ts", "midpoint_rt", "three_quarter_rt"])
    # --- NEW: Argument to select loss function ---
    parser.add_argument("--loss-function", type=str, default="product",
                        choices=["product", "relu_sum"],
                        help="Loss function to minimize: 'product' (lambda_0*lambda_1) or 'relu_sum' (lambda_0 + relu(lambda_1)).")
    
    args = parser.parse_args()
    
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    
    print(f"Running DIRECT Projected Eigenvalue Gradient Descent.")
    print(f"Loss Function: {args.loss_function.upper()}")
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
            
            atomic_nums = batch.z
            
            with torch.no_grad():
                initial_batch = coord_atoms_to_torch_geometric(initial_coords, atomic_nums).to(device)
                _, _, initial_out = calculator.potential.forward(initial_batch, otf_graph=True)
                initial_freq_info = analyze_frequencies_torch(initial_out["hessian"], initial_coords, atomic_nums)

            opt_results = run_direct_gradient_descent(
                calculator=calculator,
                initial_coords=initial_coords,
                atomic_nums=atomic_nums,
                n_steps=args.n_steps_opt,
                lr=args.lr,
                loss_fn_type=args.loss_function,
            )
            
            initial_eigvals = initial_freq_info["eigvals"]
            initial_prod = (initial_eigvals[0] * initial_eigvals[1]).item()
            initial_neg_num = initial_freq_info["neg_num"]
            
            final_freq_info = opt_results["final_freq_info"]
            final_eigvals = np.array(final_freq_info["eigvals"])
            final_prod = final_eigvals[0] * final_eigvals[1]
            final_neg_num = final_freq_info["neg_num"]
            
            summary = {
                "sample_index": i, "formula": batch.formula[0],
                "initial_eig_product": initial_prod,
                "final_eig_product": final_prod,
                "initial_neg_eigvals": initial_neg_num,
                "final_neg_eigvals": final_neg_num,
            }
            results_summary.append(summary)
            
            print("Result:")
            print(f"  Eig Product: {initial_prod:.5f} -> {final_prod:.5f}")
            print(f"  Neg Eigs: {initial_neg_num} -> {final_neg_num}")
            
        except RuntimeError as e:
            if "does not require grad" in str(e):
                print(f"[FATAL ERROR] Sample {i} failed because autograd is not supported by the model's forward pass.")
                print("  This confirms the model is in inference-only mode and cannot be used for this gradient descent method.")
                break # Stop the entire job, as no other samples will work either.
            else:
                print(f"[ERROR] Sample {i} failed: {e}")
        except Exception as e:
            print(f"[ERROR] Sample {i} failed: {e}")

    # Generate a filename that includes the loss function used
    filename = f"direct_grad_{args.loss_function}_{args.start_from}_{len(results_summary)}.json"
    out_json = os.path.join(out_dir, filename)
    with open(out_json, "w") as f: json.dump(results_summary, f, indent=2)
    print(f"\nSaved summary to {out_json}")