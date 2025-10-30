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
from hip.frequency_analysis import analyze_frequencies_torch
from hip.equiformer_torch_calculator import EquiformerTorchCalculator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- (Helper functions remain unchanged) ---
def find_rigid_alignment(A: np.ndarray, B: np.ndarray):
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

def get_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(((A - B) ** 2).sum(axis=1).mean()))

def align_ordered_and_get_rmsd(A, B) -> float:
    if isinstance(A, torch.Tensor): A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor): B = B.detach().cpu().numpy()
    if A.shape != B.shape: return float("inf")
    R, t = find_rigid_alignment(A, B)
    A_aligned = (R @ A.T).T + t
    return get_rmsd(A_aligned, B)

def _scalar_from(results: Dict[str, Any], key: str) -> Optional[float]:
    value = results.get(key)
    if value is None: return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0: return None
        return float(value.detach().cpu().view(-1)[0].item())
    try: return float(value)
    except (TypeError, ValueError): return None

def _prepare_hessian(hess: torch.Tensor, num_atoms: int) -> torch.Tensor:
    if hess.dim() == 1: hess = hess.view(int(hess.numel()**0.5), -1)
    elif hess.dim() == 3 and hess.shape[0] == 1: hess = hess[0]
    elif hess.dim() > 2: hess = hess.reshape(3 * num_atoms, 3 * num_atoms)
    return hess

def _extract_eig_product(freq_info: Dict[str, Any]) -> Optional[float]:
    """Extracts product of two smallest eigenvalues (projected)."""
    eigvals = freq_info.get("eigvals")
    if eigvals is None or not isinstance(eigvals, torch.Tensor) or eigvals.numel() < 2:
        return None
    
    eigvals = eigvals.detach().cpu().flatten()
    eigvals_sorted, _ = torch.sort(eigvals)
    # Threshold to treat extremely small values as zero to avoid numerical noise flipping signs
    e0 = float(eigvals_sorted[0].item())
    e1 = float(eigvals_sorted[1].item())
    return e0 * e1

def _mean_vector_magnitude(vec: torch.Tensor) -> float:
    return float(vec.detach().cpu().norm(dim=1).mean().item())


# --- MODIFIED FUNCTION IMPLEMENTING EARLY STOPPING ---
def run_gad_euler_on_batch(
    calculator: EquiformerTorchCalculator, batch: Batch, n_steps: int, dt: float, stop_at_ts: bool = False
) -> Dict[str, Any]:
    """Runs GAD Euler updates, optionally stopping when a TS signature is found."""
    assert int(batch.batch.max().item()) + 1 == 1, "Use batch_size=1."
    start_pos = batch.pos.detach().clone()
    
    trajectory = {k: [] for k in ["energy", "force_mean", "gad_mean", "eig_product"]}

    # Modified to return the product for checking
    def _record_step(predictions: Dict[str, Any], gad_vec: Optional[torch.Tensor]) -> Optional[float]:
        trajectory["energy"].append(_scalar_from(predictions, "energy"))
        trajectory["force_mean"].append(_mean_vector_magnitude(predictions["forces"]))
        trajectory["gad_mean"].append(_mean_vector_magnitude(gad_vec) if gad_vec is not None else None)
        try:
            # Use projected analysis for the product check
            freq_info = analyze_frequencies_torch(predictions["hessian"], batch.pos, batch.z)
            eig_prod = _extract_eig_product(freq_info)
        except Exception:
            eig_prod = None
        trajectory["eig_product"].append(eig_prod)
        return eig_prod

    # Initial state
    results = calculator.predict(batch, do_hessian=True)
    current_eig_prod = _record_step(results, gad_vec=None)
    
    steps_taken = 0
    # Check if we start at a TS and early stopping is enabled
    if stop_at_ts and current_eig_prod is not None and current_eig_prod < -1e-5:
        # Already in TS region, don't take steps
        pass
    else:
        # GAD loop
        for step_i in range(1, n_steps + 1):
            steps_taken = step_i
            
            # 1. Get Gad direction from FULL Hessian
            hess_full = _prepare_hessian(results["hessian"], batch.pos.shape[0])
            evals, evecs = torch.linalg.eigh(hess_full)
            v = evecs[:, 0].to(results["forces"].dtype) # Lowest eigenvector
            v = v / (v.norm() + 1e-12)

            # 2. Calculate GAD vector
            f_flat = results["forces"].reshape(-1)
            gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
            gad = gad_flat.view(batch.pos.shape[0], 3)

            # 3. Euler step
            batch.pos = batch.pos + dt * gad
            
            # 4. Calculate new state
            results = calculator.predict(batch, do_hessian=True)
            current_eig_prod = _record_step(results, gad)

            # --- EARLY STOPPING CHECK ---
            # If product is negative, we have exactly one negative eigenvalue (assuming ordered e0 < e1)
            # Use a small epsilon to avoid stopping on numerical noise around 0
            if stop_at_ts and current_eig_prod is not None and current_eig_prod < -1e-5:
                break

    # Final analysis of the point where we stopped
    end_pos = batch.pos.detach().clone()
    forces_end = results["forces"]
    
    # Get final negative eigenvalue count using the proper analysis function
    try:
        final_freq_info = analyze_frequencies_torch(results["hessian"], batch.pos, batch.z)
        neg_eigvals_end = int(final_freq_info.get("neg_num", -1))
    except Exception:
        neg_eigvals_end = -1

    # Get initial negative eigenvalue count (re-analyzing trajectory[0] implicitly)
    try:
        # create temp calculator just to re-run hessian on start_pos is inefficient, 
        # better to grab neg_num from the very first analyze_frequencies_torch call inside _record_step
        # but _record_step is streamlined. Let's re-calculate freq info on start_pos just for neg_num
        # Actually, simpler: look at first eig_product. If < 0, neg_num is likely 1, if >0, could be 0 or 2.
        # To be perfectly accurate, let's analyze start_pos again.
        batch_start = batch.clone()
        batch_start.pos = start_pos
        res_start = calculator.predict(batch_start, do_hessian=True)
        fi_start = analyze_frequencies_torch(res_start["hessian"], start_pos, batch.z)
        neg_eigvals_start = int(fi_start.get("neg_num", -1))
    except Exception:
        neg_eigvals_start = -1

    displacement = (end_pos - start_pos).norm(dim=1)

    return {
        "steps_taken": steps_taken, # NEW
        "rmsd": align_ordered_and_get_rmsd(start_pos, end_pos),
        "rms_force_end": forces_end.pow(2).mean().sqrt().item(),
        "max_atom_force_end": forces_end.norm(dim=1).max().item(),
        "natoms": int(start_pos.shape[0]),
        "energy_start": _scalar_from({"energy": trajectory["energy"][0]}, "energy"),
        "energy_end": _scalar_from({"energy": trajectory["energy"][-1]}, "energy"),
        "eig_product_start": trajectory["eig_product"][0],
        "eig_product_end": trajectory["eig_product"][-1],
        "mean_displacement": float(displacement.mean().item()),
        "max_displacement": float(displacement.max().item()),
        "trajectory": trajectory,
        "neg_eigvals_start": neg_eigvals_start,
        "neg_eigvals_end": neg_eigvals_end,
    }

# --- (Plotting and helper functions remain unchanged) ---
def _sanitize_formula(formula: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", formula)
    return (safe.strip("_") or "sample")

def plot_trajectory(trajectory: Dict[str, List[Optional[float]]], sample_index: int, formula: str, out_dir: str) -> str:
    num_steps = len(trajectory.get("energy", []))
    timesteps = np.arange(num_steps)

    def _nanify(values: List[Optional[float]]) -> np.ndarray:
        return np.array([v if v is not None else np.nan for v in values], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(f"GAD Trajectory for Sample {sample_index}: {formula}", fontsize=14)
    
    axes[0].plot(timesteps, _nanify(trajectory["energy"]), marker=".", lw=1.2)
    axes[0].set_ylabel("Energy (eV)"); axes[0].set_title("Energy")
    
    axes[1].plot(timesteps, _nanify(trajectory["force_mean"]), marker=".", color="tab:orange", lw=1.2)
    axes[1].set_ylabel("Mean |F| (eV/Å)"); axes[1].set_title("Force Magnitude")

    eig_ax = axes[2]
    eig_product = _nanify(trajectory["eig_product"])
    
    eig_ax.plot(timesteps, eig_product, marker=".", color="tab:purple", lw=1.2, label="λ_0 * λ_1")
    eig_ax.axhline(0, color='grey', linestyle='--', linewidth=1, zorder=1)
    
    if len(eig_product) > 0 and not np.isnan(eig_product[0]):
        eig_ax.text(0.02, 0.95, f"Start: {eig_product[0]:.4f}", 
                    transform=eig_ax.transAxes, ha='left', va='top', 
                    color='tab:purple', fontsize=9)
    if len(eig_product) > 0 and not np.isnan(eig_product[-1]):
        eig_ax.text(0.98, 0.95, f"End: {eig_product[-1]:.4f}", 
                    transform=eig_ax.transAxes, ha='right', va='top', 
                    color='tab:purple', fontsize=9)

    eig_ax.set_ylabel("Eigenvalue Product"); eig_ax.set_title("Product of Two Smallest Hessian Eigenvalues")
    eig_ax.legend(loc='best')
    
    axes[3].plot(timesteps, _nanify(trajectory["gad_mean"]), marker=".", color="tab:green", lw=1.2)
    axes[3].set_ylabel("Mean |GAD| (Å)"); axes[3].set_xlabel("Step"); axes[3].set_title("GAD Vector Magnitude")
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f"rgd1_gad_traj_{sample_index:03d}_{_sanitize_formula(formula)}.png"
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=200); plt.close(fig)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GAD-Euler dynamics and analyze RMSD.")
    parser = add_common_args(parser)
    parser.add_argument("--n-steps", type=int, default=50, help="Maximum number of Euler steps.")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step for Euler integration.")
    parser.add_argument("--unique-formulas", action="store_true", help="Only process one sample for each unique chemical formula.")
    parser.add_argument("--dataset-load-multiplier", type=int, default=5, help="Factor to multiply max_samples by for initial data loading.")
    parser.add_argument("--convergence-rms-force", type=float, default=0.01, help="Convergence threshold for RMS force (eV/Å) (used for stats, not stopping).")
    parser.add_argument("--convergence-max-force", type=float, default=0.03, help="Convergence threshold for max atomic force (eV/Å).")
    parser.add_argument("--start-from", type=str, default="ts", 
                        choices=["ts", "reactant", "midpoint_rt", "three_quarter_rt"],
                        help="Which geometry to start from.")
    
    # --- NEW ARGUMENT ---
    parser.add_argument("--stop-at-ts", action="store_true", 
                        help="Stop simulation as soon as eigenvalue product becomes negative (TS region found).")

    args = parser.parse_args()
    
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=True, dataset_load_multiplier=args.dataset_load_multiplier if args.unique_formulas else 1)
    results_summary: List[Dict[str, Any]] = []
    plot_dir = os.path.join(out_dir, "gad_trajectories"); os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Starting GAD from: {args.start_from.upper()}")
    mode_str = "Search until TS found (eig_prod < 0)" if args.stop_at_ts else f"Fixed trajectory ({args.n_steps} steps)"
    print(f"Mode: {mode_str}")
    print(f"Processing up to {args.max_samples} samples (dt={args.dt})")

    seen_formulas, processed_count, converged_count = set(), 0, 0
    final_rms_forces = []
    neg_eig_transitions = defaultdict(int)
    steps_taken_list = []

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
            if args.unique_formulas and formula in seen_formulas: continue
            seen_formulas.add(formula)

            batch.natoms = torch.tensor([batch.pos.shape[0]], dtype=torch.long)
            batch = batch.to(device)
            
            # Pass the new flag
            out = run_gad_euler_on_batch(calculator, batch, n_steps=args.n_steps, dt=args.dt, stop_at_ts=args.stop_at_ts)
            
            final_rms_forces.append(out['rms_force_end'])
            steps_taken_list.append(out['steps_taken'])

            # Convergence check (for statistics, separate from stopping condition)
            is_converged = (out['rms_force_end'] < args.convergence_rms_force and out['max_atom_force_end'] < args.convergence_max_force)
            if is_converged: converged_count += 1
            
            start_n, end_n = out['neg_eigvals_start'], out['neg_eigvals_end']
            if start_n >= 0 and end_n >= 0:
                neg_eig_transitions[f"{start_n} -> {end_n}"] += 1
            
            plot_path = plot_trajectory(out["trajectory"], processed_count, formula, plot_dir)
            result = {"dataset_index": dataset_idx, "sample_order": processed_count, "formula": formula, "plot_path": os.path.relpath(plot_path, out_dir), "converged": is_converged, **out}
            results_summary.append(result)
            
            # Updated print to show steps taken
            stop_reason = "Found TS" if (args.stop_at_ts and out['steps_taken'] < args.n_steps) else "Max Steps"
            print(f"[sample {processed_count}] N={out['natoms']}, Steps={out['steps_taken']}({stop_reason}), neg_eigs: {start_n}->{end_n}, RMS|F|_end={out['rms_force_end']:.3f}, formula={formula}")
            processed_count += 1
        except Exception as e:
            print(f"[{dataset_idx}] ERROR: {e}")
            import traceback
            traceback.print_exc()

    if processed_count > 0:
        print("\n" + "="*50)
        print(" " * 15 + "RUN SUMMARY")
        print("="*50)
        print(f"Total samples processed: {processed_count}")
        
        print("\nNegative Eigenvalue Transitions (Start -> End):")
        if not neg_eig_transitions:
            print("  No successful frequency analyses.")
        else:
            found_ts_count = 0
            for key, count in sorted(neg_eig_transitions.items()):
                print(f"  {key}: {count} samples")
                if key.endswith(" -> 1"): found_ts_count += count
            print(f"  Total ending in TS (1 neg eig): {found_ts_count} ({found_ts_count/processed_count*100:.1f}%)")

        steps_np = np.array(steps_taken_list)
        print(f"\nSteps taken (Mean): {np.mean(steps_np):.1f}")
        print(f"Steps taken (Median): {np.median(steps_np):.1f}")

        # Histogram of final forces
        forces_np = np.array(final_rms_forces)
        plt.figure(figsize=(10, 6))
        plt.hist(forces_np, bins=50, alpha=0.7)
        plt.xlabel("Final RMS Force (eV/Å)"); plt.ylabel("Count"); plt.title("Final RMS Force Distribution")
        hist_path = os.path.join(out_dir, f"force_hist_{processed_count}_{args.start_from}{'_stopts' if args.stop_at_ts else ''}.png")
        plt.savefig(hist_path, dpi=100); plt.close()

    print("="*50)
    
    # Append _stopts to filename if flag is active
    filename_suffix = f"from_{args.start_from}"
    if args.stop_at_ts: filename_suffix += "_stopts"
    
    out_json = os.path.join(out_dir, f"rgd1_gad_rmsd_{filename_suffix}_{len(results_summary)}.json")
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved summary → {out_json}")