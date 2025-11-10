# src/gad_eigenvalue_descent.py
import os
import json
import argparse
from typing import Any, Dict, List, Optional
from collections import defaultdict

import torch
import numpy as np
from torch_geometric.data import Data as TGData, Batch as TGBatch

from .common_utils import setup_experiment, add_common_args
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.frequency_analysis import analyze_frequencies_torch
from .differentiable_projection import differentiable_massweigh_and_eckartprojection_torch as massweigh_and_eckartprojection_torch
from nets.prediction_utils import Z_TO_ATOM_SYMBOL

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def find_rigid_alignment(A: np.ndarray, B: np.ndarray):
    """Kabsch algorithm. A and B are expected to be NumPy arrays."""
    a_mean = A.mean(axis=0); b_mean = B.mean(axis=0)
    A_c = A - a_mean; B_c = B - b_mean
    H = A_c.T @ B_c
    U, _, Vt = np.linalg.svd(H); V = Vt.T
    R = V @ U.T
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    t = b_mean - (R @ a_mean)
    return R, t

def get_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(((A - B) ** 2).sum(axis=1).mean()))

def align_ordered_and_get_rmsd(A, B) -> float:
    """Rigid-align A to B and compute RMSD. Handles both torch/numpy inputs."""
    if isinstance(A, torch.Tensor): A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor): B = B.detach().cpu().numpy()
    
    if A.shape != B.shape: return float("inf")
    
    # Now that A and B are guaranteed to be NumPy arrays, the rest of the function works.
    R, t = find_rigid_alignment(A, B)
    A_aligned = (R @ A.T).T + t
    return get_rmsd(A_aligned, B)
# --- END OF CORRECTION ---

def _sanitize_formula(formula: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(formula))
    keep = keep.strip("_")
    return keep or "sample"

def coord_atoms_to_torch_geometric(coords, atomic_nums, device):
    """Convert coordinates and atomic numbers to a PyG batch for model inference.

    Important: Must move batch to device AFTER creating it from data list.
    Matches the format expected by Equiformer models.
    """
    # Ensure coords has the right shape (num_atoms, 3)
    if isinstance(coords, torch.Tensor) and coords.dim() == 1:
        coords = coords.reshape(-1, 3)

    # Safety check: ensure coordinates are valid
    if isinstance(coords, torch.Tensor):
        if torch.isnan(coords).any() or torch.isinf(coords).any():
            raise ValueError("Invalid coordinates detected (NaN or Inf)")

    # Create TGData with all required fields (CPU tensors first)
    data = TGData(
        pos=torch.as_tensor(coords, dtype=torch.float32).reshape(-1, 3),
        z=torch.as_tensor(atomic_nums, dtype=torch.int64),
        charges=torch.as_tensor(atomic_nums, dtype=torch.int64),
        natoms=torch.tensor([len(atomic_nums)], dtype=torch.int64),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool),
    )
    # Create batch and THEN move to device
    return TGBatch.from_data_list([data]).to(device)

def plot_eig_descent_history(
    history: Dict[str, List[float]],
    sample_index: int,
    formula: str,
    out_dir: str,
    start_from: str,
    target_eig0: float,
    target_eig1: float,
    final_neg_vibrational: int,
    final_neg_eigvals: int,
) -> Optional[str]:
    """Plot optimization traces for eigenvalue descent."""
    def _series(key: str) -> np.ndarray:
        values = history.get(key, [])
        if not values:
            return np.array([], dtype=float)
        return np.asarray(values, dtype=float)

    loss = _series("loss")
    if loss.size == 0:
        return None  # Nothing to plot
    steps = np.arange(loss.size)
    eig0 = _series("eig0")
    eig1 = _series("eig1")
    eig_prod = _series("eig_product")
    grad_norm = _series("grad_norm")
    max_disp = _series("max_atom_disp")
    neg_vib = _series("neg_vibrational")

    fig, axes = plt.subplots(5, 1, figsize=(8, 15), sharex=True)
    fig.suptitle(f"Eigenvalue Descent (Sample {sample_index}): {formula}", fontsize=14)

    loss_for_plot = np.maximum(loss, 1e-12)
    axes[0].plot(steps, loss_for_plot, marker=".", lw=1.2, color="tab:blue")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].set_title("Optimization Loss (log scale)")

    axes[1].plot(steps, eig0, marker=".", lw=1.2, color="tab:red", label="λ₀")
    axes[1].plot(steps, eig1, marker=".", lw=1.2, color="tab:green", label="λ₁")
    axes[1].axhline(target_eig0, color="tab:red", ls="--", lw=1, alpha=0.6, label="Target λ₀")
    axes[1].axhline(target_eig1, color="tab:green", ls="--", lw=1, alpha=0.6, label="Target λ₁")
    axes[1].axhline(0.0, color="grey", ls=":", lw=1)
    axes[1].set_ylabel("Eigenvalue (eV/Å²)")
    axes[1].set_title("Tracked Vibrational Eigenvalues")
    axes[1].legend(loc="best", fontsize=9)

    axes[2].plot(steps, eig_prod, marker=".", lw=1.2, color="tab:purple")
    axes[2].axhline(0.0, color="grey", ls=":", lw=1)
    axes[2].set_ylabel("λ₀ · λ₁")
    axes[2].set_title("Eigenvalue Product")
    if eig_prod.size:
        axes[2].text(0.02, 0.9, f"Start {eig_prod[0]:.3e}", transform=axes[2].transAxes, ha="left", va="top", fontsize=9, color="tab:purple")
        axes[2].text(0.98, 0.9, f"End {eig_prod[-1]:.3e}", transform=axes[2].transAxes, ha="right", va="top", fontsize=9, color="tab:purple")

    axes[3].plot(steps, np.maximum(grad_norm, 1e-12), marker=".", lw=1.2, color="tab:orange")
    axes[3].set_ylabel("‖∇‖")
    axes[3].set_yscale("log")
    axes[3].set_title("Gradient Norm (log scale)")

    axes[4].plot(steps, max_disp, marker=".", lw=1.2, color="tab:cyan", label="Max atom Δ (Å)")
    axes[4].axhline(0.2, color="tab:cyan", ls="--", lw=1, alpha=0.6, label="Clamp limit")
    axes[4].set_ylabel("Displacement (Å)")
    axes[4].set_xlabel("Optimization Step")
    axes[4].set_title("Per-step Displacement & Neg Eigenvalue Count")

    if neg_vib.size:
        ax4b = axes[4].twinx()
        ax4b.step(steps, neg_vib, where="post", color="tab:red", lw=1.2, label="# neg vib eig")
        ax4b.set_ylabel("# Neg Vibrational")
        handles, labels = axes[4].get_legend_handles_labels()
        handles2, labels2 = ax4b.get_legend_handles_labels()
        axes[4].legend(handles + handles2, labels + labels2, loc="best", fontsize=9)
        ax4b.set_ylim(-0.5, max(np.max(neg_vib) + 0.5, 1.5))
        ax4b.set_yticks(sorted(set(int(v) for v in neg_vib)))
    else:
        axes[4].legend(loc="best", fontsize=9)

    axes[4].text(
        0.98, 0.05,
        f"Final neg vib: {final_neg_vibrational}\nFreq analysis neg: {final_neg_eigvals}",
        transform=axes[4].transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    filename = f"eigdesc_{sample_index:03d}_{_sanitize_formula(formula)}_{start_from}_steps{loss.size}.png"
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

# --- Core Optimization Function with Improved Loss Functions ---
def run_eigenvalue_descent(
    calculator: EquiformerTorchCalculator,
    initial_coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    n_steps: int = 50,
    lr: float = 0.01,
    loss_type: str = "targeted_magnitude",
    target_eig0: float = -0.05,
    target_eig1: float = 0.10,
    adaptive_targets: bool = False,
    adaptive_relax_steps: int = 50,
    adaptive_final_eig0: float = -0.02,
    adaptive_final_eig1: float = 0.05,
    early_stop_eig_product_threshold: Optional[float] = 5e-4,
) -> Dict[str, Any]:
    """
    Run gradient descent on eigenvalues to find transition states.

    Args:
        loss_type: Type of loss function to use:
            - "relu": Original ReLU loss (allows infinitesimal eigenvalues)
            - "targeted_magnitude": Target specific eigenvalue magnitudes (RECOMMENDED)
            - "midpoint_squared": Minimize squared midpoint between eigenvalues
        target_eig0: Initial target value for most negative eigenvalue (eV/Å²)
        target_eig1: Initial target value for second smallest eigenvalue (eV/Å²)
        adaptive_targets: Enable adaptive target relaxation over final steps
        adaptive_relax_steps: Number of final steps over which to relax targets
        adaptive_final_eig0: Final relaxed target for λ₀ (eV/Å²)
        adaptive_final_eig1: Final relaxed target for λ₁ (eV/Å²)
        early_stop_eig_product_threshold: Stop once λ₀·λ₁ ≤ -THRESH (set ≤0 to disable)
    """
    model = calculator.potential
    device = model.device
    coords = initial_coords.clone().to(torch.float32).to(device)
    coords.requires_grad = True
    atomsymbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in atomic_nums]
    history = defaultdict(list)
    final_eigvals = None
    loss = torch.tensor(float('inf'), device=device)  # Initialize in case of early termination
    stop_reason: Optional[str] = None
    eig_product_threshold = (
        abs(float(early_stop_eig_product_threshold))
        if (early_stop_eig_product_threshold is not None and early_stop_eig_product_threshold > 0)
        else None
    )

    # Store initial targets for adaptive relaxation
    initial_target_eig0 = target_eig0
    initial_target_eig1 = target_eig1

    for step in range(n_steps):
        # Adaptive target relaxation: linearly interpolate targets over final steps
        if adaptive_targets and loss_type == "targeted_magnitude":
            relax_start_step = n_steps - adaptive_relax_steps
            if step >= relax_start_step:
                # Linear interpolation: alpha = 0 at relax_start_step, alpha = 1 at n_steps
                alpha = (step - relax_start_step) / adaptive_relax_steps
                target_eig0 = initial_target_eig0 + alpha * (adaptive_final_eig0 - initial_target_eig0)
                target_eig1 = initial_target_eig1 + alpha * (adaptive_final_eig1 - initial_target_eig1)

                if step == relax_start_step:
                    print(f"  [ADAPTIVE] Starting target relaxation over {adaptive_relax_steps} steps")
                    print(f"    λ₀: {initial_target_eig0:.4f} → {adaptive_final_eig0:.4f}")
                    print(f"    λ₁: {initial_target_eig1:.4f} → {adaptive_final_eig1:.4f}")

        try:
            with torch.enable_grad():
                batch = coord_atoms_to_torch_geometric(coords, atomic_nums, device)
                _, _, out = model.forward(batch, otf_graph=True)
                hess_raw = out["hessian"].reshape(coords.numel(), coords.numel())
                hess_proj = massweigh_and_eckartprojection_torch(hess_raw, coords, atomsymbols)
                eigvals, _ = torch.linalg.eigh(hess_proj)

                # Remove rigid-body modes (translations / rotations) before building losses.
                with torch.no_grad():
                    coords_cent = coords.detach().reshape(-1, 3).to(torch.float64)
                    coords_cent = coords_cent - coords_cent.mean(dim=0, keepdim=True)
                    # Linear molecules have 5 zero modes, non-linear have 6.
                    geom_rank = torch.linalg.matrix_rank(coords_cent.cpu(), tol=1e-8).item()
                    expected_rigid = 5 if geom_rank <= 2 else 6
                total_modes = eigvals.shape[0]
                expected_rigid = min(expected_rigid, max(0, total_modes - 2))
                abs_sorted_idx = torch.argsort(torch.abs(eigvals))
                keep_idx = abs_sorted_idx[expected_rigid:]
                keep_idx, _ = torch.sort(keep_idx)
                vibrational_eigvals = eigvals[keep_idx]
                if vibrational_eigvals.numel() < 2:
                    raise RuntimeError(
                        f"Insufficient vibrational eigenvalues after removing {expected_rigid} rigid modes."
                    )

                eig0 = vibrational_eigvals[0]
                eig1 = vibrational_eigvals[1]
                neg_vibrational = (vibrational_eigvals < 0).sum().item()
                final_eigvals = vibrational_eigvals

                # Compute loss based on selected type
                if loss_type == "relu":
                    # Original: allows infinitesimal eigenvalues
                    loss = torch.relu(eig0) + torch.relu(-eig1)

                elif loss_type == "targeted_magnitude":
                    # Target specific magnitudes (RECOMMENDED)
                    # Push λ₀ to be meaningfully negative, λ₁ to be meaningfully positive
                    loss = (eig0 - target_eig0)**2 + (eig1 - target_eig1)**2

                elif loss_type == "midpoint_squared":
                    # Minimize squared midpoint (from your description)
                    midpoint = (eig0 + eig1) / 2.0
                    loss = midpoint**2

                else:
                    raise ValueError(f"Unknown loss_type: {loss_type}")

                # Early stopping based on loss type
                if loss_type == "relu" and loss.item() < 1e-8:
                    print(f"  Stopping early at step {step}: Converged (Loss ~ 0).")
                    break
                elif loss_type in ["targeted_magnitude", "midpoint_squared"] and loss.item() < 1e-6:
                    print(f"  Stopping early at step {step}: Converged (Loss < 1e-6).")
                    break

                grad = torch.autograd.grad(loss, coords)[0]

        except (AssertionError, RuntimeError) as e:
            # Handle case where atoms drift too far apart (empty edge graph)
            if "edge_distance_vec is empty" in str(e) or "edge_index" in str(e):
                print(f"  [WARNING] Step {step}: Atoms too far apart, stopping optimization early.")
                print(f"    This usually means the optimization diverged. Returning last valid state.")
                break
            else:
                raise  # Re-raise if it's a different error

        with torch.no_grad():
            # Gradient clipping to prevent exploding gradients
            grad_norm = torch.norm(grad)
            grad_norm_value = grad_norm.item()
            max_grad_norm = 100.0  # Maximum allowed gradient norm
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / grad_norm)
                grad_norm_value = float(max_grad_norm)

            # Compute proposed update
            update = lr * grad

            # Limit maximum displacement per atom to prevent atoms from flying apart
            # Max displacement per atom should be ~0.1-0.3 Å per step
            max_displacement_per_atom = 0.2  # Angstroms
            update_per_atom = update.reshape(-1, 3)
            atom_displacements = torch.norm(update_per_atom, dim=1)
            max_atom_disp = atom_displacements.max()

            if max_atom_disp > max_displacement_per_atom:
                scale_factor = max_displacement_per_atom / max_atom_disp
                update = update * scale_factor
                atom_displacements = atom_displacements * scale_factor
                max_atom_disp = max_atom_disp * scale_factor

            max_atom_disp_value = max_atom_disp.item()

            coords -= update

        coords.requires_grad = True

        history["loss"].append(loss.item())
        eig_prod_value = (final_eigvals[0] * final_eigvals[1]).item()
        history["eig_product"].append(eig_prod_value)
        history["eig0"].append(final_eigvals[0].item())
        history["eig1"].append(final_eigvals[1].item())
        history["neg_vibrational"].append(neg_vibrational)
        history["grad_norm"].append(grad_norm_value)
        history["max_atom_disp"].append(max_atom_disp_value)

        if eig_product_threshold is not None:
            if eig_prod_value < 0 and abs(eig_prod_value) >= eig_product_threshold:
                stop_reason = (
                    f"eig_product_threshold |λ₀·λ₁|={abs(eig_prod_value):.3e} ≥ {eig_product_threshold:.1e}"
                )
                print(
                    f"  Stopping early at step {step}: λ₀*λ₁={eig_prod_value:.6e} crossed threshold "
                    f"-{eig_product_threshold:.1e}."
                )
                break

        if step % 10 == 0:
            eig_prod = eig_prod_value
            print(
                f"  Step {step:03d}: Loss={loss.item():.6e}, λ₀*λ₁={eig_prod:.6e} "
                f"(λ₀={final_eigvals[0].item():.6f}, λ₁={final_eigvals[1].item():.6f}, neg_vib={neg_vibrational})"
            )

    final_coords = coords.detach()

    # Handle case where optimization failed/diverged early
    if final_eigvals is None:
        print(f"  [WARNING] Optimization failed - no valid eigenvalues computed")
        return {
            "final_coords": final_coords.cpu(),
            "history": history,
            "final_loss": float('inf'),
            "final_eig_product": float('inf'),
            "final_eig0": float('nan'),
            "final_eig1": float('nan'),
            "final_neg_vibrational": -1,
        }

    return {
        "final_coords": final_coords.cpu(),
        "history": history,
        "final_loss": loss.item(),
        "final_eig_product": (final_eigvals[0] * final_eigvals[1]).item(),
        "final_eig0": final_eigvals[0].item(),
        "final_eig1": final_eigvals[1].item(),
        "final_neg_vibrational": int((final_eigvals < 0).sum().item()),
        "stop_reason": stop_reason,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gradient descent on eigenvalues to find transition states.")
    parser = add_common_args(parser)
    parser.add_argument("--n-steps-opt", type=int, default=200,
                        help="Number of optimization steps (default: 200)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--start-from", type=str, default="reactant",
                        choices=["reactant", "ts", "midpoint_rt", "three_quarter_rt"],
                        help="Starting point for optimization")

    # Loss function arguments
    parser.add_argument("--loss-type", type=str, default="targeted_magnitude",
                        choices=["relu", "targeted_magnitude", "midpoint_squared"],
                        help="Loss function type (default: targeted_magnitude)")
    parser.add_argument("--target-eig0", type=float, default=-0.05,
                        help="Initial target for most negative eigenvalue in eV/Å² (default: -0.05)")
    parser.add_argument("--target-eig1", type=float, default=0.10,
                        help="Initial target for second smallest eigenvalue in eV/Å² (default: 0.10)")

    # Adaptive target relaxation
    parser.add_argument("--adaptive-targets", action="store_true",
                        help="Enable adaptive target relaxation over final steps")
    parser.add_argument("--adaptive-relax-steps", type=int, default=50,
                        help="Number of final steps over which to relax targets (default: 50)")
    parser.add_argument("--adaptive-final-eig0", type=float, default=-0.02,
                        help="Final relaxed target for λ₀ in eV/Å² (default: -0.02)")
    parser.add_argument("--adaptive-final-eig1", type=float, default=0.05,
                        help="Final relaxed target for λ₁ in eV/Å² (default: 0.05)")
    parser.add_argument("--early-stop-eig-product", type=float, default=5e-4,
                        help="Stop optimization once λ₀·λ₁ ≤ -THRESH (default: 5e-4). "
                             "Set to ≤0 to disable.")
    args = parser.parse_args()

    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)

    print(f"Running Eigenvalue Descent to Find Transition States")
    print(f"Starting From: {args.start_from.upper()}, Steps: {args.n_steps_opt}, LR: {args.lr}")
    print(f"Loss Type: {args.loss_type}")
    if args.early_stop_eig_product and args.early_stop_eig_product > 0:
        print(f"  Early stop when λ₀·λ₁ ≤ -{args.early_stop_eig_product:.1e}")
    else:
        print(f"  Early stop based on λ₀·λ₁: DISABLED")
    if args.loss_type == "targeted_magnitude":
        print(f"  Initial Target λ₀: {args.target_eig0:.4f} eV/Å²")
        print(f"  Initial Target λ₁: {args.target_eig1:.4f} eV/Å²")
        if args.adaptive_targets:
            print(f"  Adaptive Relaxation: ENABLED")
            print(f"    Relax over final {args.adaptive_relax_steps} steps")
            print(f"    Final Target λ₀: {args.adaptive_final_eig0:.4f} eV/Å²")
            print(f"    Final Target λ₁: {args.adaptive_final_eig1:.4f} eV/Å²")

    results_summary = []
    for i, batch in enumerate(dataloader):
        if i >= args.max_samples: break
        print(f"\n--- Processing Sample {i} (Formula: {batch.formula[0]}) ---")
        try:
            if args.start_from == "reactant": initial_coords = batch.pos_reactant
            elif args.start_from == "midpoint_rt": initial_coords = 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition
            elif args.start_from == "three_quarter_rt": initial_coords = 0.25 * batch.pos_reactant + 0.75 * batch.pos_transition
            else: initial_coords = batch.pos_transition

            opt_results = run_eigenvalue_descent(
                calculator=calculator,
                initial_coords=initial_coords,
                atomic_nums=batch.z,
                n_steps=args.n_steps_opt,
                lr=args.lr,
                loss_type=args.loss_type,
                target_eig0=args.target_eig0,
                target_eig1=args.target_eig1,
                adaptive_targets=args.adaptive_targets,
                adaptive_relax_steps=args.adaptive_relax_steps,
                adaptive_final_eig0=args.adaptive_final_eig0,
                adaptive_final_eig1=args.adaptive_final_eig1,
                early_stop_eig_product_threshold=args.early_stop_eig_product,
            )

            with torch.no_grad():
                final_batch = coord_atoms_to_torch_geometric(opt_results['final_coords'].to(device), batch.z, device)
                _, _, final_out = calculator.potential.forward(final_batch, otf_graph=True)
                final_freq_info = analyze_frequencies_torch(final_out['hessian'], opt_results['final_coords'].to(device), batch.z)

            summary = {
                "sample_index": i, "formula": batch.formula[0],
                "final_loss": opt_results["final_loss"],
                "final_eig_product": opt_results["final_eig_product"],
                "final_eig0": opt_results["final_eig0"],
                "final_eig1": opt_results["final_eig1"],
                "final_neg_vibrational": opt_results["final_neg_vibrational"],
                "final_neg_eigvals": int(final_freq_info.get("neg_num", -1)),
                "rmsd_to_known_ts": align_ordered_and_get_rmsd(opt_results["final_coords"], batch.pos_transition),
                "stop_reason": opt_results.get("stop_reason"),
            }
            results_summary.append(summary)

            plot_path = plot_eig_descent_history(
                history=opt_results["history"],
                sample_index=i,
                formula=batch.formula[0],
                out_dir=out_dir,
                start_from=args.start_from,
                target_eig0=args.target_eig0,
                target_eig1=args.target_eig1,
                final_neg_vibrational=summary["final_neg_vibrational"],
                final_neg_eigvals=summary["final_neg_eigvals"],
            )
            if plot_path:
                summary["plot_path"] = plot_path
                print(f"  Saved optimization plot to: {os.path.basename(plot_path)}")

            print(f"Result for Sample {i}:")
            print(f"  Final Loss: {summary['final_loss']:.6e}")
            print(f"  Final λ₀: {summary['final_eig0']:.6f} eV/Å², λ₁: {summary['final_eig1']:.6f} eV/Å²")
            print(f"  Final λ₀*λ₁: {summary['final_eig_product']:.6e}")
            print(f"  Final Neg Vibrational: {summary['final_neg_vibrational']} (Freq Analysis Negs: {summary['final_neg_eigvals']})")
            print(f"  RMSD to T1x TS: {summary['rmsd_to_known_ts']:.4f} Å")
            
        except Exception as e:
            print(f"[ERROR] Sample {i} failed: {e}")
            import traceback
            traceback.print_exc()

    out_json = os.path.join(out_dir, f"eigdesc_{args.loss_type}_{args.start_from}_{len(results_summary)}.json")
    with open(out_json, "w") as f: json.dump(results_summary, f, indent=2)
    print(f"\nSaved summary to {out_json}")

    # Print final statistics
    if results_summary:
        print("\n" + "="*60)
        print(" " * 18 + "EIGENVALUE DESCENT SUMMARY")
        print("="*60)

        avg_eig0 = np.mean([r["final_eig0"] for r in results_summary])
        avg_eig1 = np.mean([r["final_eig1"] for r in results_summary])
        avg_eig_prod = np.mean([r["final_eig_product"] for r in results_summary])
        avg_rmsd = np.mean([r["rmsd_to_known_ts"] for r in results_summary])

        print(f"\nAverage Final Values (across {len(results_summary)} samples):")
        print(f"  λ₀: {avg_eig0:.6f} eV/Å²")
        print(f"  λ₁: {avg_eig1:.6f} eV/Å²")
        print(f"  λ₀*λ₁: {avg_eig_prod:.6e}")
        print(f"  RMSD to known TS: {avg_rmsd:.4f} Å")

        # Count how many have exactly 1 negative eigenvalue
        one_neg = sum(1 for r in results_summary if r["final_neg_eigvals"] == 1)
        print(f"\nSamples with exactly 1 negative eigenvalue (TS signature): {one_neg}/{len(results_summary)}")

        # Show distribution
        from collections import Counter
        neg_dist = Counter(r["final_neg_eigvals"] for r in results_summary)
        print(f"Distribution of negative eigenvalues:")
        for key in sorted(neg_dist.keys()):
            print(f"  {key} neg eig: {neg_dist[key]} samples")
        print("="*60)
