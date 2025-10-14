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

# --- (Helper functions like find_rigid_alignment, etc. are unchanged) ---
def find_rigid_alignment(A: np.ndarray, B: np.ndarray):
    """Kabsch (no masses), handles reflection."""
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
    """Rigid-align A to B and compute RMSD. A,B: (N,3), same ordering."""
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

def _extract_eig_product(freq_info: Dict[str, Any]):
    """Extracts product of two smallest eigenvalues from frequency analysis info."""
    eigvals = freq_info.get("eigvals")
    if eigvals is None or not isinstance(eigvals, torch.Tensor) or eigvals.numel() < 2:
        return None
    
    eigvals = eigvals.detach().cpu().flatten()
    eigvals_sorted, _ = torch.sort(eigvals)
    return float(eigvals_sorted[0].item()) * float(eigvals_sorted[1].item())

def _mean_vector_magnitude(vec: torch.Tensor) -> float:
    return float(vec.detach().cpu().norm(dim=1).mean().item())


# --- MODIFIED: Streamlined run_gad_euler_on_batch function ---
def run_gad_euler_on_batch(
    calculator: EquiformerTorchCalculator, batch: Batch, n_steps: int, dt: float
) -> Dict[str, Any]:
    """Runs GAD Euler updates and analyzes start/end frequency modes."""
    assert int(batch.batch.max().item()) + 1 == 1, "Use batch_size=1."
    start_pos = batch.pos.detach().clone()
    
    # MODIFIED: Trajectory no longer stores individual eigenvalues
    trajectory = {k: [] for k in ["energy", "force_mean", "gad_mean", "eig_product"]}

    def _record_step(predictions: Dict[str, Any], gad_vec: Optional[torch.Tensor]):
        trajectory["energy"].append(_scalar_from(predictions, "energy"))
        trajectory["force_mean"].append(_mean_vector_magnitude(predictions["forces"]))
        trajectory["gad_mean"].append(_mean_vector_magnitude(gad_vec) if gad_vec is not None else None)
        try:
            freq_info = analyze_frequencies_torch(predictions["hessian"], batch.pos, batch.z)
            eig_prod = _extract_eig_product(freq_info)
        except Exception:
            eig_prod = None
        trajectory["eig_product"].append(eig_prod)

    results = calculator.predict(batch, do_hessian=True)
    try:
        initial_freq_info = analyze_frequencies_torch(results["hessian"], batch.pos, batch.z)
        neg_eigvals_start = int(initial_freq_info.get("neg_num", -1))
    except Exception:
        neg_eigvals_start = -1
    _record_step(results, gad_vec=None)

    for _ in range(n_steps):
        forces = results["forces"]
        N = forces.shape[0]
        hess = _prepare_hessian(results["hessian"], N)

        evals, evecs = torch.linalg.eigh(hess)
        v = evecs[:, 0]
        v = v.to(forces.dtype)
        v = v / (v.norm() + 1e-12)

        f_flat = forces.reshape(-1)
        gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
        gad = gad_flat.view(N, 3)

        batch.pos = batch.pos + dt * gad
        results = calculator.predict(batch, do_hessian=True)
        _record_step(results, gad)

    end_pos = batch.pos.detach().clone()
    forces_end = results["forces"]
    try:
        final_freq_info = analyze_frequencies_torch(results["hessian"], batch.pos, batch.z)
        neg_eigvals_end = int(final_freq_info.get("neg_num", -1))
    except Exception:
        neg_eigvals_end = -1

    displacement = (end_pos - start_pos).norm(dim=1)

    def _to_float(value: Optional[float]) -> float:
        return float(value) if value is not None else float("nan")

    # MODIFIED: Return dictionary is cleaner, no individual eigenvalues
    return {
        "rmsd": align_ordered_and_get_rmsd(start_pos, end_pos),
        "rms_force_end": forces_end.pow(2).mean().sqrt().item(),
        "max_atom_force_end": forces_end.norm(dim=1).max().item(),
        "natoms": int(start_pos.shape[0]),
        "energy_start": _scalar_from({"energy": trajectory["energy"][0]}, "energy"),
        "energy_end": _scalar_from({"energy": trajectory["energy"][-1]}, "energy"),
        "eig_product_start": _to_float(trajectory["eig_product"][0]),
        "eig_product_end": _to_float(trajectory["eig_product"][-1]),
        "mean_displacement": float(displacement.mean().item()),
        "max_displacement": float(displacement.max().item()),
        "trajectory": trajectory,
        "neg_eigvals_start": neg_eigvals_start,
        "neg_eigvals_end": neg_eigvals_end,
    }

# --- (plot_trajectory is unchanged from the last version, already plots only the product) ---
def plot_trajectory(trajectory: Dict[str, List[Optional[float]]], sample_index: int, formula: str, out_dir: str) -> str:
    # ... (This function is identical to the previous version) ...
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
    
    if not np.isnan(eig_product[0]):
        eig_ax.text(0.02, 0.95, f"Start: {eig_product[0]:.4f}", 
                    transform=eig_ax.transAxes, ha='left', va='top', 
                    color='tab:purple', fontsize=9)
    if not np.isnan(eig_product[-1]):
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
    parser.add_argument("--n-steps", type=int, default=50, help="Number of Euler steps.")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step for Euler integration.")
    parser.add_argument("--unique-formulas", action="store_true", help="Only process one sample for each unique chemical formula.")
    parser.add_argument("--dataset-load-multiplier", type=int, default=5, help="Factor to multiply max_samples by for initial data loading.")
    parser.add_argument("--convergence-rms-force", type=float, default=0.01, help="Convergence threshold for RMS force (eV/Å).")
    parser.add_argument("--convergence-max-force", type=float, default=0.03, help="Convergence threshold for max atomic force (eV/Å).")

    # --- MODIFIED: Add new choices for the starting geometry ---
    parser.add_argument("--start-from", type=str, default="ts", 
                        choices=["ts", "reactant", "midpoint_rt", "three_quarter_rt"],
                        help="Which geometry to start from. 'rt' refers to Reactant-to-TS interpolation.")

    args = parser.parse_args()
    # ... (setup and loop initialization is the same) ...
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=True, dataset_load_multiplier=args.dataset_load_multiplier if args.unique_formulas else 1)
    results_summary: List[Dict[str, Any]] = []
    plot_dir = os.path.join(out_dir, "gad_trajectories"); os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Starting GAD from: {args.start_from.upper()}")
    print(f"Processing up to {args.max_samples} samples with GAD-Euler (steps={args.n_steps}, dt={args.dt})")
    print(f"Convergence criteria: RMS|F| < {args.convergence_rms_force:.4f}, Max|F| < {args.convergence_max_force:.4f} eV/Å")

    seen_formulas, processed_count, converged_count = set(), 0, 0
    final_rms_forces = []
    neg_eig_transitions = defaultdict(int)

    for dataset_idx, batch in enumerate(dataloader):
        if processed_count >= args.max_samples: break
        try:
            # --- MODIFIED: Logic to set the starting position based on the argument ---
            start_pos_source = "ts_guess" # Default
            if args.start_from == "reactant":
                if not hasattr(batch, 'pos_reactant'): continue
                batch.pos = batch.pos_reactant
                start_pos_source = "reactant"
            elif args.start_from == "midpoint_rt":
                if not (hasattr(batch, 'pos_reactant') and hasattr(batch, 'pos_transition')): continue
                batch.pos = 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition
                start_pos_source = "midpoint_rt"
            elif args.start_from == "three_quarter_rt":
                if not (hasattr(batch, 'pos_reactant') and hasattr(batch, 'pos_transition')): continue
                batch.pos = 0.25 * batch.pos_reactant + 0.75 * batch.pos_transition
                start_pos_source = "three_quarter_rt"
            # --- END MODIFICATION ---

            formula = getattr(batch, "formula", [""])[0]
            if isinstance(formula, bytes): formula = formula.decode("utf-8", "ignore")
            if args.unique_formulas and formula in seen_formulas: continue
            seen_formulas.add(formula)

            batch.natoms = torch.tensor([batch.pos.shape[0]], dtype=torch.long)
            batch = batch.to(device)
            out = run_gad_euler_on_batch(calculator, batch, n_steps=args.n_steps, dt=args.dt)
            
            # ... (rest of the loop and summary is unchanged) ...
            final_rms_forces.append(out['rms_force_end'])
            is_converged = (out['rms_force_end'] < args.convergence_rms_force and out['max_atom_force_end'] < args.convergence_max_force)
            if is_converged: converged_count += 1
            convergence_status = "✅ Converged" if is_converged else "❌ Not Converged"
            
            start_n, end_n = out['neg_eigvals_start'], out['neg_eigvals_end']
            if start_n >= 0 and end_n >= 0:
                neg_eig_transitions[f"{start_n} -> {end_n}"] += 1
            
            plot_path = plot_trajectory(out["trajectory"], processed_count, formula, plot_dir)
            result = {"dataset_index": dataset_idx, "sample_order": processed_count, "formula": formula, "plot_path": os.path.relpath(plot_path, out_dir), "converged": is_converged, **out}
            results_summary.append(result)
            
            print(f"[sample {processed_count}] N={out['natoms']}, RMS|F|_end={out['rms_force_end']:.4f}, neg_eigs: {start_n} -> {end_n}, {convergence_status}, formula={formula}")
            processed_count += 1
        except Exception as e:
            print(f"[{dataset_idx}] ERROR: {e}")

    # --- (Final summary block is unchanged) ---
    # ...
    out_json = os.path.join(out_dir, f"rgd1_gad_rmsd_from_{args.start_from}_{len(results_summary)}.json")
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved GAD RMSD summary for {len(results_summary)} samples → {out_json}")