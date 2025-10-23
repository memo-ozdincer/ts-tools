# src/gad_eigen_optimization.py
import os
import json
import re
import argparse
from typing import Any, Dict, List, Optional
from collections import defaultdict

import torch
import numpy as np
from torch_geometric.data import Batch, Data as TGDData

from .common_utils import setup_experiment, add_common_args
from hip.frequency_analysis import analyze_frequencies_torch
from hip.equiformer_torch_calculator import EquiformerTorchCalculator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- (Helper functions are unchanged) ---
def find_rigid_alignment(A, B):
    a_mean = A.mean(axis=0); b_mean = B.mean(axis=0)
    A_c = A - a_mean; B_c = B - b_mean
    H = A_c.T @ B_c
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    t = b_mean - R @ a_mean
    return R, t

def get_rmsd(A, B) -> float:
    return float(np.sqrt(((A - B) ** 2).sum(axis=1).mean()))

def align_ordered_and_get_rmsd(A, B) -> float:
    if isinstance(A, torch.Tensor): A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor): B = B.detach().cpu().numpy()
    if A.shape != B.shape: return float("inf")
    R, t = find_rigid_alignment(A, B)
    A_aligned = (R @ A.T).T + t
    return get_rmsd(A_aligned, B)

def _prepare_hessian(hess: torch.Tensor, num_atoms: int) -> torch.Tensor:
    if hess.dim() == 1: hess = hess.view(int(hess.numel()**0.5), -1)
    elif hess.dim() == 3 and hess.shape[0] == 1: hess = hess[0]
    elif hess.dim() > 2: hess = hess.reshape(3 * num_atoms, 3 * num_atoms)
    return hess

def _sanitize_formula(formula: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", formula)
    return (safe.strip("_") or "sample")

def plot_trajectory(trajectory, sample_index, formula, out_dir):
    num_steps = len(trajectory["loss"])
    timesteps = np.arange(num_steps)
    def _nanify(values): return np.array([v if v is not None else np.nan for v in values], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    fig.suptitle(f"Eigenvalue Product Optimization for Sample {sample_index}: {formula}", fontsize=14)
    
    axes[0].plot(timesteps, _nanify(trajectory["loss"]), marker=".", lw=1.2, color="tab:purple")
    axes[0].axhline(0, color='grey', linestyle='--', linewidth=1)
    axes[0].set_ylabel("Loss (λ_0 * λ_1)")
    axes[0].set_title("Loss (Eigenvalue Product) vs. Step")
    
    axes[1].plot(timesteps, _nanify(trajectory["energy"]), marker=".", color="tab:blue")
    axes[1].set_ylabel("Energy (eV)")
    axes[1].set_title("Energy vs. Step")

    axes[2].plot(timesteps, _nanify(trajectory["rms_force"]), marker=".", color="tab:orange")
    axes[2].set_ylabel("RMS Force (eV/Å)")
    axes[2].set_xlabel("Step")
    axes[2].set_title("RMS Force vs. Step")
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f"rgd1_eigen_opt_traj_{sample_index:03d}_{_sanitize_formula(formula)}.png"
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=200); plt.close(fig)
    return out_path


# --- THE NEW OPTIMIZATION FUNCTION ---
def run_eigen_optimization_on_batch(
    calculator: EquiformerTorchCalculator,
    batch: Batch,
    n_steps: int,
    lr: float,
    stop_at_ts: bool,
) -> Dict[str, Any]:
    """Optimizes atomic positions to minimize the product of the two smallest Hessian eigenvalues."""
    
    with torch.set_grad_enabled(True):
        start_pos = batch.pos.detach().clone()
        pos = start_pos.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([pos], lr=lr)
        trajectory = defaultdict(list)
        steps_taken = 0
        
        # Get initial state
        with torch.no_grad():
            # --- FIX #1: Add `natoms` to the temporary data object ---
            data_start = TGDData(
                z=batch.z,
                pos=start_pos,
                batch=torch.zeros_like(batch.z),
                natoms=torch.tensor([start_pos.shape[0]], dtype=torch.long)
            )
            res_start = calculator.predict(Batch.from_data_list([data_start]), do_hessian=True)
            fi_start = analyze_frequencies_torch(res_start["hessian"], start_pos, batch.z)
            neg_eigvals_start = int(fi_start.get("neg_num", -1))

        for i in range(n_steps):
            steps_taken = i + 1
            optimizer.zero_grad()
            
            # --- FIX #2: Add `natoms` to the data object inside the loop ---
            current_data = TGDData(
                z=batch.z,
                pos=pos,
                batch=torch.zeros_like(batch.z),
                natoms=torch.tensor([pos.shape[0]], dtype=torch.long)
            )
            current_batch = Batch.from_data_list([current_data])
            results = calculator.predict(current_batch, do_hessian=True)
            
            hess = _prepare_hessian(results["hessian"], pos.shape[0])
            evals = torch.linalg.eigvalsh(hess) 
            evals_sorted, _ = torch.sort(evals)
            loss = evals_sorted[0] * evals_sorted[1]
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                trajectory["loss"].append(loss.item())
                trajectory["energy"].append(results.get("energy").item())
                trajectory["rms_force"].append(results.get("forces").pow(2).mean().sqrt().item())
                if stop_at_ts and loss.item() < -1e-5:
                    break

    with torch.no_grad():
        end_pos = pos.detach().clone()
        # --- FIX #3: Add `natoms` to the final data object ---
        data_end = TGDData(
            z=batch.z,
            pos=end_pos,
            batch=torch.zeros_like(batch.z),
            natoms=torch.tensor([end_pos.shape[0]], dtype=torch.long)
        )
        final_results = calculator.predict(Batch.from_data_list([data_end]), do_hessian=True)
        final_freq_info = analyze_frequencies_torch(final_results["hessian"], end_pos, batch.z)
        neg_eigvals_end = int(final_freq_info.get("neg_num", -1))
        
        return {
            "steps_taken": steps_taken,
            "rmsd": align_ordered_and_get_rmsd(start_pos, end_pos),
            "rms_force_end": final_results["forces"].pow(2).mean().sqrt().item(),
            "max_atom_force_end": final_results["forces"].norm(dim=1).max().item(),
            "natoms": int(start_pos.shape[0]),
            "energy_start": trajectory["energy"][0],
            "energy_end": trajectory["energy"][-1],
            "eig_product_start": trajectory["loss"][0],
            "eig_product_end": trajectory["loss"][-1],
            "neg_eigvals_start": neg_eigvals_start,
            "neg_eigvals_end": neg_eigvals_end,
            "trajectory": trajectory,
        }

# --- (The __main__ block is unchanged and correct) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize geometry to find transition states by minimizing the eigenvalue product.")
    parser = add_common_args(parser)
    parser.add_argument("--n-steps", type=int, default=100, help="Maximum number of optimization steps.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the Adam optimizer.")
    parser.add_argument("--start-from", type=str, default="reactant", 
                        choices=["ts", "reactant", "midpoint_rt", "three_quarter_rt"],
                        help="Which geometry to start from.")
    parser.add_argument("--stop-at-ts", action="store_true", 
                        help="Stop optimization as soon as eigenvalue product becomes negative.")
    
    args = parser.parse_args()
    
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=True)
    results_summary: List[Dict[str, Any]] = []
    plot_dir = os.path.join(out_dir, "eigen_opt_trajectories"); os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Starting Eigenvalue Product Optimization from: {args.start_from.upper()}")
    print(f"Max steps: {args.n_steps}, Learning Rate: {args.lr}, Stop at TS: {args.stop_at_ts}")

    processed_count = 0
    neg_eig_transitions = defaultdict(int)
    
    for dataset_idx, batch in enumerate(dataloader):
        if processed_count >= args.max_samples: break
        try:
            if args.start_from == "reactant":
                if not hasattr(batch, 'pos_reactant'): continue
                batch.pos = batch.pos_reactant
            elif args.start_from == "midpoint_rt":
                if not (hasattr(batch, 'pos_reactant') and hasattr(batch, 'pos_transition')): continue
                batch.pos = 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition
            elif args.start_from == "three_quarter_rt":
                if not (hasattr(batch, 'pos_reactant') and hasattr(batch, 'pos_transition')): continue
                batch.pos = 0.25 * batch.pos_reactant + 0.75 * batch.pos_transition

            formula = getattr(batch, "formula", [""])[0]
            if isinstance(formula, bytes): formula = formula.decode("utf-8", "ignore")
            
            # This is important: the original batch from the dataloader does not have `natoms`
            # We must add it here before passing it to the optimization function.
            batch.natoms = torch.tensor([batch.pos.shape[0]], dtype=torch.long)
            batch = batch.to(device)
            
            out = run_eigen_optimization_on_batch(calculator, batch, n_steps=args.n_steps, lr=args.lr, stop_at_ts=args.stop_at_ts)
            
            start_n, end_n = out['neg_eigvals_start'], out['neg_eigvals_end']
            neg_eig_transitions[f"{start_n} -> {end_n}"] += 1
            
            plot_path = plot_trajectory(out["trajectory"], processed_count, formula, plot_dir)
            out["plot_path"] = os.path.relpath(plot_path, out_dir)
            results_summary.append(out)
            
            stop_reason = "Found TS" if (args.stop_at_ts and out['steps_taken'] < args.n_steps) else "Max Steps"
            print(f"[sample {processed_count}] N={out['natoms']}, Steps={out['steps_taken']}({stop_reason}), neg_eigs: {start_n}->{end_n}, Final Loss={out['eig_product_end']:.4f}, RMS|F|={out['rms_force_end']:.3f}, formula={formula}")
            processed_count += 1
        except Exception as e:
            print(f"[{dataset_idx}] ERROR: {e}")
            import traceback
            traceback.print_exc()

    if processed_count > 0:
        print("\n" + "="*50)
        print(" " * 15 + "OPTIMIZATION SUMMARY")
        print("="*50)
        print(f"Total samples processed: {processed_count}")
        print("\nNegative Eigenvalue Transitions (Start -> End):")
        found_ts_count = 0
        for key, count in sorted(neg_eig_transitions.items()):
            print(f"  {key}: {count} samples")
            if key.endswith(" -> 1"): found_ts_count += count
        print(f"  Total ending in TS (1 neg eig): {found_ts_count} ({found_ts_count/processed_count*100:.1f}%)")
    
    filename_suffix = f"from_{args.start_from}"
    if args.stop_at_ts: filename_suffix += "_stopts"
    out_json = os.path.join(out_dir, f"rgd1_eigen_opt_{filename_suffix}_{len(results_summary)}.json")
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved summary → {out_json}")