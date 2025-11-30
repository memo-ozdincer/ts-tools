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


# --- MODIFIED FUNCTION IMPLEMENTING EARLY STOPPING AND KICK MECHANISM ---
def run_gad_euler_on_batch(
    calculator: EquiformerTorchCalculator, batch: Batch, n_steps: int, dt: float, stop_at_ts: bool = False,
    kick_enabled: bool = False, kick_force_threshold: float = 0.015, kick_magnitude: float = 0.1
) -> Dict[str, Any]:
    """
    Runs GAD Euler updates, optionally stopping when a TS signature is found.

    Args:
        kick_enabled: Enable kick mechanism to escape local minima
        kick_force_threshold: Force threshold in eV/Å below which kick is considered
        kick_magnitude: Magnitude of kick displacement in Å
    """
    assert int(batch.batch.max().item()) + 1 == 1, "Use batch_size=1."
    start_pos = batch.pos.detach().clone()
    num_kicks = 0

    trajectory = {k: [] for k in ["energy", "force_mean", "gad_mean", "eig_product", "kick_applied"]}

    # Modified to return the product and freq_info for checking
    def _record_step(predictions: Dict[str, Any], gad_vec: Optional[torch.Tensor], kick_applied: bool = False) -> tuple:
        trajectory["energy"].append(_scalar_from(predictions, "energy"))
        trajectory["force_mean"].append(_mean_vector_magnitude(predictions["forces"]))
        trajectory["gad_mean"].append(_mean_vector_magnitude(gad_vec) if gad_vec is not None else None)
        trajectory["kick_applied"].append(1 if kick_applied else 0)
        try:
            # Use projected analysis for the product check
            freq_info = analyze_frequencies_torch(predictions["hessian"], batch.pos, batch.z)
            eig_prod = _extract_eig_product(freq_info)
        except Exception:
            eig_prod = None
            freq_info = {}
        trajectory["eig_product"].append(eig_prod)
        return eig_prod, freq_info

    # Initial state
    results = calculator.predict(batch, do_hessian=True)
    current_eig_prod, freq_info = _record_step(results, gad_vec=None)

    steps_taken = 0
    # Check if we start at a TS and early stopping is enabled
    if stop_at_ts and current_eig_prod is not None and current_eig_prod < -1e-5:
        # Already in TS region, don't take steps
        pass
    else:
        # GAD loop
        for step_i in range(1, n_steps + 1):
            steps_taken = step_i

            # Check if we should apply a kick (escape local minimum)
            apply_kick = False
            if kick_enabled:
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
            else:
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
            current_eig_prod, freq_info = _record_step(results, gad, kick_applied=apply_kick)

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
        "num_kicks": num_kicks,
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
) -> tuple:
    """
    Plot GAD trajectory.

    Returns:
        Tuple of (matplotlib Figure, suggested filename)
    """
    num_steps = len(trajectory.get("energy", []))
    timesteps = np.arange(num_steps)

    def _nanify(values: List[Optional[float]]) -> np.ndarray:
        return np.array([v if v is not None else np.nan for v in values], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(f"GAD Euler Trajectory for Sample {sample_index}: {formula}", fontsize=14)

    axes[0].plot(timesteps, _nanify(trajectory["energy"]), marker=".", lw=1.2)
    axes[0].set_ylabel("Energy (eV)"); axes[0].set_title("Energy")

    axes[1].plot(timesteps, _nanify(trajectory["force_mean"]), marker=".", color="tab:orange", lw=1.2)
    axes[1].set_ylabel("Mean |F| (eV/Å)"); axes[1].set_title("Force Magnitude")

    eig_ax = axes[2]
    eig_product = _nanify(trajectory["eig_product"])

    eig_ax.plot(timesteps, eig_product, marker=".", color="tab:purple", lw=1.2, label="λ₀ * λ₁")
    eig_ax.axhline(0, color='grey', linestyle='--', linewidth=1, zorder=1)

    if len(eig_product) > 0 and not np.isnan(eig_product[0]):
        eig_ax.text(0.02, 0.95, f"Start: {eig_product[0]:.4f}",
                    transform=eig_ax.transAxes, ha='left', va='top',
                    color='tab:purple', fontsize=9)
    if len(eig_product) > 0 and not np.isnan(eig_product[-1]):
        eig_ax.text(0.98, 0.95, f"End: {eig_product[-1]:.4f}",
                    transform=eig_ax.transAxes, ha='right', va='top',
                    color='tab:purple', fontsize=9)

    eig_ax.set_ylabel("Eigenvalue Product")
    eig_ax.set_title("Product of Two Smallest Hessian Eigenvalues")
    eig_ax.legend(loc='best')

    axes[3].plot(timesteps, _nanify(trajectory["gad_mean"]), marker=".", color="tab:green", lw=1.2)
    axes[3].set_ylabel("Mean |GAD| (Å)")
    axes[3].set_xlabel("Step")
    axes[3].set_title("GAD Vector Magnitude")

    axes[3].text(
        0.98, 0.05,
        f"neg eig: {initial_neg_num} → {final_neg_num}",
        transform=axes[3].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f"traj_{sample_index:03d}_{_sanitize_formula(formula)}_from_{start_from}_{initial_neg_num}to{final_neg_num}.png"
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
                        choices=["ts", "reactant", "midpoint_rt", "three_quarter_rt"],
                        help="Which geometry to start from.")
    
    # --- NEW ARGUMENT ---
    parser.add_argument("--stop-at-ts", action="store_true",
                        help="Stop simulation as soon as eigenvalue product becomes negative (TS region found).")

    # Kick mechanism arguments
    parser.add_argument("--enable-kick", action="store_true", help="Enable kick mechanism to escape local minima.")
    parser.add_argument("--kick-force-threshold", type=float, default=0.015,
                        help="Force threshold in eV/Å below which kick is considered (default: 0.015).")
    parser.add_argument("--kick-magnitude", type=float, default=0.1,
                        help="Magnitude of kick displacement in Å (default: 0.1).")

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

    for i, batch in enumerate(dataloader):
        if i >= args.max_samples: break
        print(f"\n--- Processing Sample {i} (Formula: {batch.formula[0]}) ---")
        try:
            # Use parse_starting_geometry to handle both standard and noisy starting points
            initial_coords = parse_starting_geometry(args.start_from, batch, noise_seed=42)
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
                kick_magnitude=args.kick_magnitude
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
            print(f"  Steps: {result.steps_taken}")
            print(f"  Neg Eigs: {result.initial_neg_eigvals} -> {result.final_neg_eigvals}")
            print(f"  RMS Force: {out['rms_force_end']:.3e} eV/Å")
            if args.enable_kick:
                print(f"  Kicks applied: {out['num_kicks']}")

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