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

from .common_utils import setup_experiment, add_common_args, parse_starting_geometry
from hip.frequency_analysis import analyze_frequencies_torch
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from .experiment_logger import ExperimentLogger, RunResult, build_loss_type_flags

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


def detect_stall(
    eig_product_history: List[Optional[float]],
    disp_history: List[float],
    window: int = 50,
    disp_threshold: float = 0.02,
    eig_change_threshold: float = 5.0,
) -> bool:
    """
    Detect if GAD optimization has stalled.
    
    Stall is detected when:
    1. Eigenvalue product remains positive (not at TS) for the window
    2. Average displacement is below threshold (not making progress)
    3. Eigenvalue product change is below threshold (not improving)
    
    Args:
        eig_product_history: History of eigenvalue products
        disp_history: History of per-step displacements
        window: Number of steps to check
        disp_threshold: Minimum average displacement to not be stalled (Å)
        eig_change_threshold: Minimum change in eigenvalue product
        
    Returns:
        True if stalled, False otherwise
    """
    if len(eig_product_history) < window or len(disp_history) < window:
        return False
    
    # Get the last 'window' steps
    recent_eig_prods = [e for e in eig_product_history[-window:] if e is not None]
    recent_disps = disp_history[-window:]
    
    if len(recent_eig_prods) < window // 2:
        return False  # Not enough valid data
    
    # Check 1: All eigenvalue products are positive (not at TS)
    all_positive = all(e > 0 for e in recent_eig_prods)
    if not all_positive:
        return False  # Already found TS region
    
    # Check 2: Average displacement is very small
    avg_disp = np.mean(recent_disps)
    small_displacement = avg_disp < disp_threshold
    
    # Check 3: Eigenvalue product hasn't changed much
    eig_change = abs(recent_eig_prods[-1] - recent_eig_prods[0])
    small_eig_change = eig_change < eig_change_threshold
    
    # Stalled if both displacement and eigenvalue change are small
    return small_displacement and small_eig_change


# --- MODIFIED FUNCTION IMPLEMENTING EARLY STOPPING, KICK, AND MINIMIZATION FALLBACK ---
def run_gad_euler_on_batch(
    calculator: EquiformerTorchCalculator, batch: Batch, n_steps: int, dt: float, stop_at_ts: bool = False,
    kick_enabled: bool = False, kick_force_threshold: float = 0.015, kick_magnitude: float = 0.1,
    minimization_fallback: bool = False, stall_window: int = 50, 
    stall_disp_threshold: float = 0.02, stall_eig_change_threshold: float = 5.0,
) -> Dict[str, Any]:
    """
    Runs GAD Euler updates, optionally stopping when a TS signature is found.
    
    Can fallback to minimization mode when GAD stalls, which follows forces
    directly to find a local minimum (all eigenvalues positive).

    Args:
        kick_enabled: Enable kick mechanism to escape local minima
        kick_force_threshold: Force threshold in eV/Å below which kick is considered
        kick_magnitude: Magnitude of kick displacement in Å
        minimization_fallback: Enable fallback to minimization when GAD stalls
        stall_window: Number of steps to check for stall detection
        stall_disp_threshold: Minimum displacement to not be considered stalled (Å)
        stall_eig_change_threshold: Minimum eigenvalue product change to not be stalled
    """
    assert int(batch.batch.max().item()) + 1 == 1, "Use batch_size=1."
    start_pos = batch.pos.detach().clone()
    prev_pos = start_pos.clone()  # Track previous position for step-wise displacement
    num_kicks = 0
    
    # Track mode: 0 = GAD, 1 = minimization
    current_mode = 0  # Start in GAD mode
    mode_switch_step = None  # Step at which we switched to minimization

    trajectory = {k: [] for k in ["energy", "force_mean", "gad_mean", "eig_product", "kick_applied",
                                   "disp_from_last", "disp_from_start", "mode"]}

    # Modified to return the product and freq_info for checking
    def _record_step(predictions: Dict[str, Any], gad_vec: Optional[torch.Tensor], 
                     disp_from_last: float, disp_from_start: float, kick_applied: bool = False,
                     mode: int = 0) -> tuple:
        trajectory["energy"].append(_scalar_from(predictions, "energy"))
        trajectory["force_mean"].append(_mean_vector_magnitude(predictions["forces"]))
        trajectory["gad_mean"].append(_mean_vector_magnitude(gad_vec) if gad_vec is not None else None)
        trajectory["kick_applied"].append(1 if kick_applied else 0)
        trajectory["disp_from_last"].append(disp_from_last)
        trajectory["disp_from_start"].append(disp_from_start)
        trajectory["mode"].append(mode)  # 0 = GAD, 1 = minimization
        try:
            # Use projected analysis for the product check
            freq_info = analyze_frequencies_torch(predictions["hessian"], batch.pos, batch.z)
            eig_prod = _extract_eig_product(freq_info)
        except Exception:
            eig_prod = None
            freq_info = {}
        trajectory["eig_product"].append(eig_prod)
        return eig_prod, freq_info

    # Initial state (no displacement at step 0)
    results = calculator.predict(batch, do_hessian=True)
    current_eig_prod, freq_info = _record_step(results, gad_vec=None, disp_from_last=0.0, disp_from_start=0.0, mode=current_mode)

    steps_taken = 0
    # Check if we start at a TS and early stopping is enabled
    if stop_at_ts and current_eig_prod is not None and current_eig_prod < -1e-5:
        # Already in TS region, don't take steps
        pass
    else:
        # GAD loop (with optional minimization fallback)
        for step_i in range(1, n_steps + 1):
            steps_taken = step_i

            # --- STALL DETECTION AND MODE SWITCH ---
            if minimization_fallback and current_mode == 0:  # Only check if still in GAD mode
                is_stalled = detect_stall(
                    trajectory["eig_product"],
                    trajectory["disp_from_last"],
                    window=stall_window,
                    disp_threshold=stall_disp_threshold,
                    eig_change_threshold=stall_eig_change_threshold,
                )
                # Debug output every 50 steps to see stall metrics
                if step_i % 50 == 0 and len(trajectory["disp_from_last"]) >= stall_window:
                    recent_disps = trajectory["disp_from_last"][-stall_window:]
                    recent_eigs = [e for e in trajectory["eig_product"][-stall_window:] if e is not None]
                    avg_disp = np.mean(recent_disps) if recent_disps else 0
                    eig_change = abs(recent_eigs[-1] - recent_eigs[0]) if len(recent_eigs) >= 2 else 0
                    print(f"  [STALL CHECK] step {step_i}: avg_disp={avg_disp:.4f} (thresh={stall_disp_threshold}), "
                          f"eig_change={eig_change:.4f} (thresh={stall_eig_change_threshold}), stalled={is_stalled}")
                
                if is_stalled:
                    current_mode = 1  # Switch to minimization mode
                    mode_switch_step = step_i
                    eig_prod_at_switch = trajectory["eig_product"][-1] if trajectory["eig_product"] else None
                    print(f"  [MODE SWITCH] GAD stalled at step {step_i}, switching to MINIMIZATION mode")
                    print(f"    Eigenvalue product at switch: {eig_prod_at_switch:.4f}" if eig_prod_at_switch else "    Eigenvalue product: N/A")

            # Check if we should apply a kick (escape local minimum)
            apply_kick = False
            if kick_enabled and current_mode == 0:  # Only kick in GAD mode
                force_magnitude = results["forces"].norm(dim=1).mean().item()
                # Check if force is below threshold AND both smallest projected eigenvalues are positive (local min)
                eigvals = freq_info.get("eigvals")
                if eigvals is not None and force_magnitude < kick_force_threshold and eigvals.numel() >= 2:
                    eig0, eig1 = eigvals[0].item(), eigvals[1].item()
                    if eig0 > 0 and eig1 > 0:  # Local minimum
                        apply_kick = True
                        num_kicks += 1
                        print(f"  [KICK {num_kicks}] at step {step_i}: |F|={force_magnitude:.4f} eV/Å, λ₀={eig0:.6f}, λ₁={eig1:.6f}")

            if apply_kick:
                # Apply kick: random perturbation scaled by kick_magnitude
                kick_direction = torch.randn_like(batch.pos)
                kick_direction = kick_direction / (kick_direction.norm() + 1e-12)
                batch.pos = batch.pos + kick_direction * kick_magnitude
                gad = kick_direction * kick_magnitude
            elif current_mode == 1:
                # MINIMIZATION MODE: Follow forces directly (gradient descent on energy)
                # Forces = -gradient of energy, so following forces decreases energy
                forces = results["forces"]
                gad = forces  # Use forces directly as the update direction
                
                # Euler step with forces
                batch.pos = batch.pos + dt * gad
            else:
                # GAD MODE: Use GAD direction
                # 1. Get GAD direction from Hessian
                hess_full = _prepare_hessian(results["hessian"], batch.pos.shape[0])
                evals, evecs = torch.linalg.eigh(hess_full)
                v = evecs[:, 0].to(results["forces"].dtype)  # Lowest eigenvector
                v = v / (v.norm() + 1e-12)

                # 2. Calculate GAD vector
                f_flat = results["forces"].reshape(-1)
                gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
                gad = gad_flat.view(batch.pos.shape[0], 3)

                # 3. Euler step
                batch.pos = batch.pos + dt * gad

            # 4. Calculate new state
            results = calculator.predict(batch, do_hessian=True)
            
            # Compute displacements
            disp_from_last = (batch.pos - prev_pos).norm(dim=1).mean().item()
            disp_from_start = (batch.pos - start_pos).norm(dim=1).mean().item()
            prev_pos = batch.pos.detach().clone()  # Update for next iteration
            
            current_eig_prod, freq_info = _record_step(results, gad, disp_from_last, disp_from_start, 
                                                        kick_applied=apply_kick, mode=current_mode)

            # --- EARLY STOPPING CHECK ---
            # If product is negative, we have exactly one negative eigenvalue (assuming ordered e0 < e1)
            # Use a small epsilon to avoid stopping on numerical noise around 0
            if stop_at_ts and current_eig_prod is not None and current_eig_prod < -1e-5:
                break
            
            # --- MINIMIZATION CONVERGENCE CHECK ---
            # In minimization mode, stop if forces become very small (found a minimum)
            if current_mode == 1:
                force_magnitude = results["forces"].norm(dim=1).mean().item()
                if force_magnitude < 0.01:  # Converged to minimum
                    print(f"  [MINIMIZATION] Converged to minimum at step {step_i}: |F|={force_magnitude:.6f} eV/Å")
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
    
    # Determine final mode
    final_mode = "minimization" if current_mode == 1 else "gad"

    return {
        "steps_taken": steps_taken,
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
        "num_kicks": num_kicks,
        "mode_switch_step": mode_switch_step,
        "final_mode": final_mode,
    }

# --- (Plotting and helper functions remain unchanged) ---
def _sanitize_formula(formula: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", formula)
    return (safe.strip("_") or "sample")

def plot_trajectory_new(
    trajectory: Dict[str, List[Optional[float]]],
    sample_index: int,
    formula: str,
    start_from: str,
    initial_neg_num: int,
    final_neg_num: int,
    steps_to_ts: Optional[int] = None,
    mode_switch_step: Optional[int] = None,
) -> tuple:
    """
    Plot GAD trajectory with 6 panels in a 3x2 grid.

    Args:
        trajectory: Dictionary containing trajectory data
        sample_index: Index of the sample
        formula: Chemical formula
        start_from: Starting geometry type
        initial_neg_num: Initial number of negative eigenvalues
        final_neg_num: Final number of negative eigenvalues
        steps_to_ts: Step at which TS was found (optional)
        mode_switch_step: Step at which mode switched to minimization (optional)

    Returns:
        Tuple of (matplotlib Figure, suggested filename)
    """
    num_steps = len(trajectory.get("energy", []))
    timesteps = np.arange(num_steps)

    def _nanify(values: List[Optional[float]]) -> np.ndarray:
        return np.array([v if v is not None else np.nan for v in values], dtype=float)

    # Extract noise level from start_from for title (e.g., "reactant_noise2A" -> "noise 2Å")
    noise_info = ""
    if "_noise" in start_from:
        noise_str = start_from.split("_noise")[1]
        noise_info = f" (noise {noise_str})"

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f"Sample {sample_index}: {formula}{noise_info}", fontsize=14)

    # Panel 1 (0,0): Energy
    ax = axes[0, 0]
    ax.plot(timesteps, _nanify(trajectory["energy"]), marker=".", lw=1.2, markersize=3)
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Energy")
    ax.set_xlabel("Step")
    if mode_switch_step is not None:
        ax.axvline(mode_switch_step, color='red', linestyle=':', linewidth=2, alpha=0.7, label=f'→MIN @ {mode_switch_step}')

    # Panel 2 (0,1): Force Magnitude
    ax = axes[0, 1]
    ax.plot(timesteps, _nanify(trajectory["force_mean"]), marker=".", color="tab:orange", lw=1.2, markersize=3)
    ax.set_ylabel("Mean |F| (eV/Å)")
    ax.set_title("Force Magnitude")
    ax.set_xlabel("Step")

    # Panel 3 (1,0): Eigenvalue Product
    ax = axes[1, 0]
    eig_product = _nanify(trajectory["eig_product"])
    ax.plot(timesteps, eig_product, marker=".", color="tab:purple", lw=1.2, markersize=3, label="λ₀ * λ₁")
    ax.axhline(0, color='grey', linestyle='--', linewidth=1, zorder=1)
    if len(eig_product) > 0 and not np.isnan(eig_product[0]):
        ax.text(0.02, 0.95, f"Start: {eig_product[0]:.4f}",
                transform=ax.transAxes, ha='left', va='top', color='tab:purple', fontsize=9)
    if len(eig_product) > 0 and not np.isnan(eig_product[-1]):
        ax.text(0.98, 0.95, f"End: {eig_product[-1]:.4f}",
                transform=ax.transAxes, ha='right', va='top', color='tab:purple', fontsize=9)
    if steps_to_ts is not None:
        ax.axvline(steps_to_ts, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'TS @ {steps_to_ts}')
    if mode_switch_step is not None:
        ax.axvline(mode_switch_step, color='red', linestyle=':', linewidth=2, alpha=0.7, label=f'→MIN @ {mode_switch_step}')
    ax.set_ylabel("Eigenvalue Product")
    ax.set_title("Eigenvalue Product (λ₀ * λ₁)")
    ax.set_xlabel("Step")
    ax.legend(loc='best', fontsize=8)

    # Panel 4 (1,1): GAD Vector Magnitude
    ax = axes[1, 1]
    ax.plot(timesteps, _nanify(trajectory["gad_mean"]), marker=".", color="tab:green", lw=1.2, markersize=3)
    ax.set_ylabel("Mean |GAD| (Å)")
    ax.set_title("GAD Vector Magnitude")
    ax.set_xlabel("Step")

    # Panel 5 (2,0): Displacement from Last Step
    ax = axes[2, 0]
    disp_last = _nanify(trajectory.get("disp_from_last", []))
    ax.plot(timesteps, disp_last, marker=".", color="tab:red", lw=1.2, markersize=3)
    ax.set_ylabel("Mean Disp (Å)")
    ax.set_title("Displacement from Last Step")
    ax.set_xlabel("Step")
    if len(disp_last) > 1:
        valid_disp = disp_last[1:]  # Skip step 0
        valid_disp = valid_disp[~np.isnan(valid_disp)]
        if len(valid_disp) > 0:
            ax.text(0.98, 0.95, f"Avg: {np.mean(valid_disp):.4f} Å",
                    transform=ax.transAxes, ha='right', va='top', fontsize=9)

    # Panel 6 (2,1): Displacement from Start
    ax = axes[2, 1]
    disp_start = _nanify(trajectory.get("disp_from_start", []))
    ax.plot(timesteps, disp_start, marker=".", color="tab:blue", lw=1.2, markersize=3)
    ax.set_ylabel("Mean Disp (Å)")
    ax.set_title("Displacement from Start")
    ax.set_xlabel("Step")
    if len(disp_start) > 0 and not np.isnan(disp_start[-1]):
        ax.text(0.98, 0.95, f"Final: {disp_start[-1]:.4f} Å",
                transform=ax.transAxes, ha='right', va='top', fontsize=9)
    # Add summary text
    summary_parts = [f"neg eig: {initial_neg_num} → {final_neg_num}"]
    if steps_to_ts is not None:
        summary_parts.append(f"TS @ {steps_to_ts}")
    if mode_switch_step is not None:
        summary_parts.append(f"→MIN @ {mode_switch_step}")
    ax.text(0.02, 0.05,
            ", ".join(summary_parts),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f"traj_{sample_index:03d}_{_sanitize_formula(formula)}_{start_from}_{initial_neg_num}to{final_neg_num}.png"
    return fig, filename


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
                        help="Starting geometry: 'reactant', 'ts', 'midpoint_rt', 'three_quarter_rt', "
                             "or add noise: 'reactant_noise0.5A', 'reactant_noise1A', 'reactant_noise2A', etc.")
    
    # --- NEW ARGUMENT ---
    parser.add_argument("--stop-at-ts", action="store_true",
                        help="Stop simulation as soon as eigenvalue product becomes negative (TS region found).")

    # Kick mechanism arguments
    parser.add_argument("--enable-kick", action="store_true", help="Enable kick mechanism to escape local minima.")
    parser.add_argument("--kick-force-threshold", type=float, default=0.015,
                        help="Force threshold in eV/Å below which kick is considered (default: 0.015).")
    parser.add_argument("--kick-magnitude", type=float, default=0.1,
                        help="Magnitude of kick displacement in Å (default: 0.1).")

    # Minimization fallback arguments
    parser.add_argument("--enable-minimization-fallback", action="store_true",
                        help="Enable fallback to minimization mode when GAD stalls. "
                             "Minimization follows forces to find a local minimum.")
    parser.add_argument("--stall-window", type=int, default=50,
                        help="Number of steps to check for stall detection (default: 50).")
    parser.add_argument("--stall-disp-threshold", type=float, default=0.02,
                        help="Minimum average displacement (Å) to not be considered stalled (default: 0.02).")
    parser.add_argument("--stall-eig-change-threshold", type=float, default=5.0,
                        help="Minimum eigenvalue product change to not be stalled (default: 5.0).")

    # W&B arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="gad-ts-search",
                        help="W&B project name (default: gad-ts-search)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity/username (optional)")

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=True, dataset_load_multiplier=args.dataset_load_multiplier if args.unique_formulas else 1)

    # Set up experiment logger
    loss_type_flags = build_loss_type_flags(args)

    # Prepare W&B config
    wandb_config = {
        "script": "gad_gad_euler_rmsd",
        "start_from": args.start_from,
        "n_steps": args.n_steps,
        "dt": args.dt,
        "stop_at_ts": args.stop_at_ts,
        "enable_kick": args.enable_kick,
        "kick_force_threshold": args.kick_force_threshold if args.enable_kick else None,
        "kick_magnitude": args.kick_magnitude if args.enable_kick else None,
        "enable_minimization_fallback": args.enable_minimization_fallback,
        "stall_window": args.stall_window if args.enable_minimization_fallback else None,
        "stall_disp_threshold": args.stall_disp_threshold if args.enable_minimization_fallback else None,
        "stall_eig_change_threshold": args.stall_eig_change_threshold if args.enable_minimization_fallback else None,
        "max_samples": args.max_samples,
    }

    logger = ExperimentLogger(
        base_dir=out_dir,
        script_name="gad-euler",
        loss_type_flags=loss_type_flags,
        max_graphs_per_transition=10,
        random_seed=42,  # For reproducible sampling
        use_wandb=args.wandb,
        wandb_project=args.wandb_project if args.wandb else None,
        wandb_entity=args.wandb_entity,
        wandb_tags=[args.start_from, "euler"],
        wandb_config=wandb_config,
    )

    print(f"Running GAD-Euler Search. Start: {args.start_from.upper()}, Stop at TS: {args.stop_at_ts}")
    print(f"Output directory: {logger.run_dir}")
    print(f"Mode: {'Search until TS found (eig_prod < 0)' if args.stop_at_ts else f'Fixed trajectory ({args.n_steps} steps)'}")
    print(f"Processing up to {args.max_samples} samples (dt={args.dt})")
    print(f"Kick enabled: {args.enable_kick}")
    if args.enable_kick:
        print(f"  Force threshold: {args.kick_force_threshold} eV/Å, Kick magnitude: {args.kick_magnitude} Å")
    print(f"Minimization fallback enabled: {args.enable_minimization_fallback}")
    if args.enable_minimization_fallback:
        print(f"  Stall window: {args.stall_window} steps")
        print(f"  Stall displacement threshold: {args.stall_disp_threshold} Å")
        print(f"  Stall eig product change threshold: {args.stall_eig_change_threshold}")

    for i, batch in enumerate(dataloader):
        if i >= args.max_samples: break
        print(f"\n--- Processing Sample {i} (Formula: {batch.formula[0]}) ---")
        try:
            # Use parse_starting_geometry to handle both standard and noisy starting points
            initial_coords = parse_starting_geometry(args.start_from, batch, noise_seed=42, sample_index=i)
            batch.pos = initial_coords
            batch.natoms = torch.tensor([batch.pos.shape[0]], dtype=torch.long)
            batch = batch.to(device)

            # Run GAD Euler
            out = run_gad_euler_on_batch(
                calculator, batch,
                n_steps=args.n_steps,
                dt=args.dt,
                stop_at_ts=args.stop_at_ts,
                kick_enabled=args.enable_kick,
                kick_force_threshold=args.kick_force_threshold,
                kick_magnitude=args.kick_magnitude,
                minimization_fallback=args.enable_minimization_fallback,
                stall_window=args.stall_window,
                stall_disp_threshold=args.stall_disp_threshold,
                stall_eig_change_threshold=args.stall_eig_change_threshold,
            )

            # Extract final results from trajectory
            traj = out["trajectory"]
            initial_neg_num = out['neg_eigvals_start']
            final_neg_num = out['neg_eigvals_end']

            # Compute steps_to_ts: find first step where eig_product < 0
            steps_to_ts = None
            eig_prod_series = traj.get("eig_product", [])
            if eig_prod_series:
                for step_idx, eig_prod in enumerate(eig_prod_series):
                    if eig_prod is not None and eig_prod < -1e-5:
                        steps_to_ts = step_idx
                        break

            # Create RunResult
            result = RunResult(
                sample_index=i,
                formula=batch.formula[0],
                initial_neg_eigvals=initial_neg_num,
                final_neg_eigvals=final_neg_num,
                initial_neg_vibrational=None,  # Not tracked in euler
                final_neg_vibrational=None,
                steps_taken=out['steps_taken'],
                steps_to_ts=steps_to_ts,
                final_time=None,  # Euler uses discrete steps
                final_eig0=None,  # Not separately tracked
                final_eig1=None,
                final_eig_product=out['eig_product_end'],
                final_loss=None,  # Not applicable for Euler
                rmsd_to_known_ts=out['rmsd'],
                stop_reason="ts_found" if (args.stop_at_ts and out['steps_taken'] < args.n_steps) else None,
                plot_path=None,  # Will be set below
                extra_data={
                    "initial_eig_product": out['eig_product_start'],
                    "rms_force_end": out['rms_force_end'],
                    "max_atom_force_end": out['max_atom_force_end'],
                    "num_kicks": out['num_kicks'],
                    "mean_displacement": out['mean_displacement'],
                    "max_displacement": out['max_displacement'],
                    "mode_switch_step": out['mode_switch_step'],
                    "final_mode": out['final_mode'],
                }
            )

            # Add result to logger
            logger.add_result(result)

            # Create and save plot
            fig, filename = plot_trajectory_new(
                trajectory=traj,
                sample_index=i,
                formula=batch.formula[0],
                start_from=args.start_from,
                initial_neg_num=initial_neg_num,
                final_neg_num=final_neg_num,
                steps_to_ts=steps_to_ts,
                mode_switch_step=out['mode_switch_step'],
            )

            # Save plot using logger (handles sampling)
            plot_path = logger.save_graph(result, fig, filename)
            if plot_path:
                result.plot_path = plot_path
                print(f"  Saved plot to: {plot_path}")
            else:
                print(f"  Skipped plot (max samples for {result.transition_key} reached)")
            plt.close(fig)

            print("Result:")
            print(f"  Transition: {result.transition_key}")
            print(f"  Steps: {result.steps_taken}" + (f", Steps to TS: {steps_to_ts}" if steps_to_ts is not None else ""))
            print(f"  Neg Eigs: {result.initial_neg_eigvals} -> {result.final_neg_eigvals}")
            print(f"  RMS Force: {out['rms_force_end']:.3e} eV/Å")
            print(f"  Final Disp from Start: {traj['disp_from_start'][-1]:.4f} Å")
            if args.enable_kick and out['num_kicks'] > 0:
                print(f"  Kicks applied: {out['num_kicks']}")
            if out['mode_switch_step'] is not None:
                print(f"  Mode switched to minimization at step: {out['mode_switch_step']}")
                print(f"  Final mode: {out['final_mode']}")

        except Exception as e:
            print(f"[ERROR] Sample {i} failed: {e}")
            import traceback
            traceback.print_exc()

    # Save all results and aggregate statistics
    all_runs_path, aggregate_path = logger.save_all_results()
    print(f"\nSaved all runs to: {all_runs_path}")
    print(f"Saved aggregate stats to: {aggregate_path}")

    # Print summary
    logger.print_summary()

    # Finish W&B run
    logger.finish()