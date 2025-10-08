# gad_gad_euler_rmsd.py
import os
import json
import re
import argparse
from typing import Any, Dict, List, Optional

import torch
import numpy as np
from torch_geometric.data import Batch

# Import shared utilities
from .common_utils import setup_experiment, add_common_args
# Import the required analysis function, which is now used exclusively
from hip.frequency_analysis import analyze_frequencies_torch
from hip.equiformer_torch_calculator import EquiformerTorchCalculator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Alignment + RMSD utilities (Specific to this script) ---
def find_rigid_alignment(A: np.ndarray, B: np.ndarray):
    """Kabsch (no masses), handles reflection."""
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
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

# --- GAD (Euler) loop and helpers (Specific to this script) ---
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

def _extract_smallest_eigs(freq_info: Dict[str, Any]):
    """Extracts smallest eigenvalues from pre-computed frequency analysis info."""
    eigvals = freq_info.get("eigvals")
    if eigvals is None or not isinstance(eigvals, torch.Tensor) or eigvals.numel() < 2:
        return None, None, None
    
    eigvals = eigvals.detach().cpu().flatten()
    eigvals_sorted, _ = torch.sort(eigvals)
    smallest = float(eigvals_sorted[0].item())
    second_smallest = float(eigvals_sorted[1].item())
    return smallest, second_smallest, smallest * second_smallest

def _mean_vector_magnitude(vec: torch.Tensor) -> float:
    return float(vec.detach().cpu().norm(dim=1).mean().item())

def run_gad_euler_on_batch(
    calculator: EquiformerTorchCalculator, batch: Batch, n_steps: int, dt: float
) -> Dict[str, Any]:
    """Runs GAD Euler updates on positions in `batch` using predicted Hessians."""
    assert int(batch.batch.max().item()) + 1 == 1, "Use batch_size=1."
    start_pos = batch.pos.detach().clone()
    
    trajectory = {k: [] for k in ["energy", "force_mean", "gad_mean", "eig_min1", "eig_min2", "eig_product"]}

    def _record_step(predictions: Dict[str, Any], gad_vec: Optional[torch.Tensor]):
        trajectory["energy"].append(_scalar_from(predictions, "energy"))
        trajectory["force_mean"].append(_mean_vector_magnitude(predictions["forces"]))
        trajectory["gad_mean"].append(_mean_vector_magnitude(gad_vec) if gad_vec is not None else None)

        try:
            freq_info = analyze_frequencies_torch(predictions["hessian"], batch.pos, batch.z)
            eig0, eig1, eig_prod = _extract_smallest_eigs(freq_info)
        except Exception:
            eig0, eig1, eig_prod = None, None, None

        trajectory["eig_min1"].append(eig0)
        trajectory["eig_min2"].append(eig1)
        trajectory["eig_product"].append(eig_prod)
        return freq_info

    results = calculator.predict(batch, do_hessian=True)
    _record_step(results, gad_vec=None)

    for _ in range(n_steps):
        forces = results["forces"]
        N = forces.shape[0]
        hess = _prepare_hessian(results["hessian"], N)

        freq_info = analyze_frequencies_torch(hess, batch.pos, batch.z)
        v = freq_info["eigvecs"][:, 0]  # Smallest eigenvector

        # --- FIX: DATA TYPE MISMATCH ---
        # Cast the eigenvector 'v' (likely float64) to match the forces' dtype (float32).
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
    displacement = (end_pos - start_pos).norm(dim=1)

    def _to_float(value: Optional[float]) -> float:
        return float(value) if value is not None else float("nan")

    return {
        "rmsd": align_ordered_and_get_rmsd(start_pos, end_pos),
        "rms_force_end": forces_end.pow(2).mean().sqrt().item(),
        "max_atom_force_end": forces_end.norm(dim=1).max().item(),
        "natoms": int(start_pos.shape[0]),
        "energy_start": _scalar_from({"energy": trajectory["energy"][0]}, "energy"),
        "energy_end": _scalar_from({"energy": trajectory["energy"][-1]}, "energy"),
        "min_hess_eig_start": _to_float(trajectory["eig_min1"][0]),
        "min_hess_eig_end": _to_float(trajectory["eig_min1"][-1]),
        "eig_product_start": _to_float(trajectory["eig_product"][0]),
        "eig_product_end": _to_float(trajectory["eig_product"][-1]),
        "mean_displacement": float(displacement.mean().item()),
        "max_displacement": float(displacement.max().item()),
        "trajectory": trajectory,
    }

# --- Plotting utilities (Specific to this script) ---
def _sanitize_formula(formula: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", formula)
    return (safe.strip("_") or "sample")

def plot_trajectory(trajectory: Dict[str, List[Optional[float]]], sample_index: int, formula: str, out_dir: str) -> str:
    num_steps = len(trajectory.get("energy", []))
    timesteps = np.arange(num_steps)

    def _nanify(values: List[Optional[float]]) -> np.ndarray:
        return np.array([v if v is not None else np.nan for v in values], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)
    axes[0].plot(timesteps, _nanify(trajectory["energy"]), marker=".", lw=1.2)
    axes[0].set_ylabel("Energy (eV)"); axes[0].set_title("Energy")
    axes[1].plot(timesteps, _nanify(trajectory["force_mean"]), marker=".", color="tab:orange", lw=1.2)
    axes[1].set_ylabel("Mean |F| (eV/Å)"); axes[1].set_title("Force Magnitude")

    eig_ax = axes[2]
    eig_ax.plot(timesteps, _nanify(trajectory["eig_min1"]), marker=".", color="tab:blue", lw=1.2, label="λ_0")
    eig_ax.plot(timesteps, _nanify(trajectory["eig_min2"]), marker=".", color="tab:red", lw=1.2, label="λ_1")
    eig_ax.set_ylabel("Eigenvalues"); eig_ax.set_title("Smallest Hessian Eigenvalues")
    prod_ax = eig_ax.twinx()
    prod_ax.plot(timesteps, _nanify(trajectory["eig_product"]), marker=".", color="tab:purple", lw=1.2, ls="--", label="λ_0*λ_1")
    prod_ax.set_ylabel("Product")
    lines, labels = eig_ax.get_legend_handles_labels()
    plines, plabels = prod_ax.get_legend_handles_labels()
    eig_ax.legend(lines + plines, labels + plabels, loc="best", fontsize="small")

    axes[3].plot(timesteps, _nanify(trajectory["gad_mean"]), marker=".", color="tab:green", lw=1.2)
    axes[3].set_ylabel("Mean |GAD| (Å)"); axes[3].set_xlabel("Step"); axes[3].set_title("GAD Vector Magnitude")
    fig.tight_layout()

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
    parser.add_argument("--dataset-load-multiplier", type=int, default=5, help="Factor to multiply max_samples by for initial data loading, used for finding unique formulas.")
    args = parser.parse_args()

    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=True, dataset_load_multiplier=args.dataset_load_multiplier if args.unique_formulas else 1)

    results_summary: List[Dict[str, Any]] = []
    plot_dir = os.path.join(out_dir, "gad_trajectories"); os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Processing up to {args.max_samples} samples with GAD-Euler (steps={args.n_steps}, dt={args.dt})")

    seen_formulas = set()
    processed_count = 0
    for dataset_idx, batch in enumerate(dataloader):
        if processed_count >= args.max_samples: break

        try:
            formula = getattr(batch, "formula", [""])[0]
            if isinstance(formula, bytes): formula = formula.decode("utf-8", "ignore")
            
            if args.unique_formulas and formula in seen_formulas: continue
            seen_formulas.add(formula)

            batch.natoms = torch.tensor([batch.pos.shape[0]], dtype=torch.long)
            batch = batch.to(device)
            out = run_gad_euler_on_batch(calculator, batch, n_steps=args.n_steps, dt=args.dt)
            
            plot_path = plot_trajectory(out["trajectory"], processed_count, formula, plot_dir)
            
            result = {"dataset_index": dataset_idx, "sample_order": processed_count, "formula": formula, "plot_path": os.path.relpath(plot_path, out_dir), **out}
            results_summary.append(result)
            
            e_start, e_end = out["energy_start"], out["energy_end"]
            delta_e_str = f"ΔE={e_end - e_start:.5f}" if e_start is not None and e_end is not None else "ΔE=NA"
            print(f"[sample {processed_count}] N={out['natoms']}, RMSD={out['rmsd']:.4f} Å, {delta_e_str}, λ_min_end={out['min_hess_eig_end']:.5f}, formula={formula}")
            processed_count += 1
            
        except Exception as e:
            print(f"[{dataset_idx}] ERROR: {e}")

    out_json = os.path.join(out_dir, f"rgd1_gad_rmsd_{len(results_summary)}.json")
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved GAD RMSD summary for {len(results_summary)} samples → {out_json}")