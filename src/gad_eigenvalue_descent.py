# src/gad_eigenvalue_descent.py
import os
import json
import argparse
from typing import Any, Dict, List, Optional
from collections import defaultdict

import torch
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from torch_geometric.data import Data as TGData, Batch as TGBatch

from .common_utils import setup_experiment, add_common_args, parse_starting_geometry, extract_vibrational_eigenvalues
from .saddle_detection import classify_saddle_point, compute_adaptive_step_scale
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.frequency_analysis import analyze_frequencies_torch
from .differentiable_projection import differentiable_massweigh_and_eckartprojection_torch as massweigh_and_eckartprojection_torch
from nets.prediction_utils import Z_TO_ATOM_SYMBOL
from .experiment_logger import (
    ExperimentLogger, RunResult, build_loss_type_flags,
    init_wandb_run, log_sample, log_summary, finish_wandb,
)
import time

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
    start_from: str,
    target_eig0: float,
    target_eig1: float,
    final_neg_vibrational: int,
    final_neg_eigvals: int,
):
    """
    Plot optimization traces for eigenvalue descent.

    Returns:
        Tuple of (matplotlib Figure, suggested filename) or (None, None) if nothing to plot
    """
    def _series(key: str) -> np.ndarray:
        values = history.get(key, [])
        if not values:
            return np.array([], dtype=float)
        cleaned: List[float] = []
        for v in values:
            if v is None:
                cleaned.append(np.nan)
            else:
                try:
                    cleaned.append(float(v))
                except (TypeError, ValueError):
                    cleaned.append(np.nan)
        return np.asarray(cleaned, dtype=float)

    def _isfinite_scalar(val) -> bool:
        try:
            return bool(np.isfinite(float(val)))
        except (TypeError, ValueError):
            return False

    loss = _series("loss")
    if loss.size == 0:
        return None  # Nothing to plot
    steps = np.arange(loss.size)
    energy = _series("energy")
    force_mean = _series("force_mean")
    force_max = _series("force_max")
    eig0 = _series("eig0")
    eig1 = _series("eig1")
    eig_prod = _series("eig_product")
    grad_norm = _series("grad_norm")
    max_disp = _series("max_atom_disp")
    neg_vib = _series("neg_vibrational")

    fig, axes = plt.subplots(5, 1, figsize=(8, 15), sharex=True)
    fig.suptitle(f"Eigenvalue Descent (Sample {sample_index}): {formula}", fontsize=14)

    loss_for_plot = loss.copy()
    finite_loss = np.isfinite(loss_for_plot)
    loss_for_plot[finite_loss] = np.maximum(np.abs(loss_for_plot[finite_loss]), 1e-12)
    has_energy = energy.size > 0 and np.isfinite(energy).any()
    if has_energy:
        axes[0].plot(steps, energy, marker=".", lw=1.2, color="tab:blue", label="Energy")
        axes[0].set_ylabel("Energy (eV)")
        axes[0].set_title("Energy & Loss")
        ax0b = axes[0].twinx()
        ax0b.plot(steps, loss_for_plot, marker=".", lw=1.2, color="tab:orange", label="|Loss|")
        ax0b.set_ylabel("|Loss|")
        ax0b.set_yscale("log")
        handles0, labels0 = axes[0].get_legend_handles_labels()
        handles0b, labels0b = ax0b.get_legend_handles_labels()
        if handles0 or handles0b:
            axes[0].legend(handles0 + handles0b, labels0 + labels0b, loc="best", fontsize=9)
    else:
        axes[0].plot(steps, loss_for_plot, marker=".", lw=1.2, color="tab:orange")
        axes[0].set_ylabel("|Loss|")
        axes[0].set_yscale("log")
        axes[0].set_title("Optimization Loss (log scale)")

    def _plot_log_series(ax, series: np.ndarray, color: str, label: str) -> bool:
        if series.size == 0:
            return False
        finite = np.isfinite(series)
        if not finite.any():
            return False
        data = series.copy()
        data[finite] = np.maximum(np.abs(data[finite]), 1e-12)
        ax.plot(steps, data, marker=".", lw=1.2, color=color, label=label)
        return True

    axes[1].set_yscale("log")
    plotted_force = _plot_log_series(axes[1], force_mean, "tab:orange", "Mean |F|")
    plotted_force |= _plot_log_series(axes[1], force_max, "tab:brown", "Max |F|")
    plotted_grad = _plot_log_series(axes[1], grad_norm, "tab:purple", "‖∇loss‖")
    axes[1].set_ylabel("Magnitude (log)")
    axes[1].set_title("Force & Gradient Norms")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(loc="best", fontsize=9)
    if not (plotted_force or plotted_grad):
        axes[1].set_yscale("linear")
        axes[1].text(0.5, 0.5, "No force/grad data", transform=axes[1].transAxes,
                     ha="center", va="center", fontsize=9, color="grey")

    axes[2].plot(steps, eig0, marker=".", lw=1.2, color="tab:red", label="λ₀")
    axes[2].plot(steps, eig1, marker=".", lw=1.2, color="tab:green", label="λ₁")
    if _isfinite_scalar(target_eig0):
        axes[2].axhline(target_eig0, color="tab:red", ls="--", lw=1, alpha=0.6, label="Target λ₀")
    if _isfinite_scalar(target_eig1):
        axes[2].axhline(target_eig1, color="tab:green", ls="--", lw=1, alpha=0.6, label="Target λ₁")
    axes[2].axhline(0.0, color="grey", ls=":", lw=1)
    axes[2].set_ylabel("Eigenvalue (eV/Å²)")
    axes[2].set_title("Tracked Vibrational Eigenvalues")
    axes[2].legend(loc="best", fontsize=9)
    if eig0.size and np.isfinite(eig0[-1]):
        axes[2].text(0.98, 0.05, f"Final λ₀={eig0[-1]:.6f}", transform=axes[2].transAxes,
                     ha="right", va="bottom", fontsize=9, color="tab:red")

    axes[3].plot(steps, eig_prod, marker=".", lw=1.2, color="tab:purple")
    axes[3].axhline(0.0, color="grey", ls=":", lw=1)
    axes[3].set_ylabel("λ₀ · λ₁")
    axes[3].set_title("Eigenvalue Product")
    if eig_prod.size:
        if np.isfinite(eig_prod[0]):
            axes[3].text(0.02, 0.9, f"Start {eig_prod[0]:.3e}", transform=axes[3].transAxes,
                         ha="left", va="top", fontsize=9, color="tab:purple")
        if np.isfinite(eig_prod[-1]):
            axes[3].text(0.98, 0.9, f"End {eig_prod[-1]:.3e}", transform=axes[3].transAxes,
                         ha="right", va="top", fontsize=9, color="tab:purple")

    max_disp_plot = max_disp.copy()
    finite_disp = np.isfinite(max_disp_plot)
    max_disp_plot[finite_disp] = np.maximum(max_disp_plot[finite_disp], 0.0)
    axes[4].plot(steps, max_disp_plot, marker=".", lw=1.2, color="tab:cyan", label="Max atom Δ (Å)")
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
        finite_neg = neg_vib[np.isfinite(neg_vib)]
        if finite_neg.size:
            ax4b.set_ylim(-0.5, max(finite_neg.max() + 0.5, 1.5))
            ax4b.set_yticks(sorted(set(int(v) for v in finite_neg)))
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
    return fig, filename

# --- Helper function for line search ---
def evaluate_step(
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    calculator: EquiformerTorchCalculator,
    loss_type: str,
    target_eig0: float,
    target_eig1: float,
    sign_neg_target: float,
    sign_pos_floor: float,
) -> tuple[float, float, float, float, int]:
    """
    Evaluate loss and eigenvalues at given coordinates.

    Returns:
        (loss_value, eig0, eig1, eig_product, neg_vibrational)
    """
    model = calculator.potential
    device = model.device
    atomsymbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in atomic_nums]

    try:
        with torch.no_grad():
            batch = coord_atoms_to_torch_geometric(coords, atomic_nums, device)
            _, _, out = model.forward(batch, otf_graph=True)
            hess_raw = out["hessian"].reshape(coords.numel(), coords.numel())
            hess_proj = massweigh_and_eckartprojection_torch(hess_raw, coords, atomsymbols)

            # Extract vibrational eigenvalues (centralized function)
            vibrational_eigvals = extract_vibrational_eigenvalues(hess_proj, coords)

            if vibrational_eigvals.numel() < 2:
                return float('inf'), float('nan'), float('nan'), float('inf'), -1

            eig0 = vibrational_eigvals[0]
            eig1 = vibrational_eigvals[1]
            eig_product = eig0 * eig1
            neg_vibrational = (vibrational_eigvals < 0).sum().item()

            # Compute loss based on loss type
            if loss_type == "relu":
                loss = torch.relu(eig0) + torch.relu(-eig1)
            elif loss_type == "targeted_magnitude":
                loss = (eig0 - target_eig0)**2 + (eig1 - target_eig1)**2
            elif loss_type == "midpoint_squared":
                midpoint = (eig0 + eig1) / 2.0
                loss = midpoint**2
            elif loss_type == "eig_product":
                loss = eig_product
            elif loss_type == "sign_enforcer":
                neg_target = eig0.new_tensor(sign_neg_target)
                pos_floor = eig0.new_tensor(sign_pos_floor)
                if neg_vibrational == 0:
                    loss = (eig0 - neg_target).pow(2)
                elif neg_vibrational == 1:
                    loss = eig0.new_tensor(0.0)
                else:
                    trailing_eigs = vibrational_eigvals[1:]
                    if trailing_eigs.numel() == 0:
                        loss = eig0.new_tensor(0.0)
                    else:
                        penalties = (pos_floor - trailing_eigs).pow(2)
                        loss = penalties.sum()
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            return (
                loss.item(),
                eig0.item(),
                eig1.item(),
                eig_product.item(),
                neg_vibrational
            )
    except Exception:
        return float('inf'), float('nan'), float('nan'), float('inf'), -1


def run_eigprod_bfgs(
    calculator: EquiformerTorchCalculator,
    initial_coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    maxiter: int = 100,
    gtol: float = 1e-5,
    max_step: float = 1.0,
    loss_type: str = "eig_product",
    target_eig0: float = -0.05,
    target_eig1: float = 0.10,
) -> Dict[str, Any]:
    """
    Run L-BFGS-B optimization to find transition states via eigenvalue-based loss.
    
    Args:
        calculator: The Equiformer calculator
        initial_coords: Starting coordinates
        atomic_nums: Atomic numbers
        maxiter: Maximum BFGS iterations
        gtol: Gradient tolerance for convergence
        max_step: Maximum displacement per atom from start (Å)
        loss_type: "eig_product" (minimize λ₀*λ₁) or "targeted_magnitude" (target specific values)
        target_eig0: Target value for λ₀ (only used if loss_type="targeted_magnitude")
        target_eig1: Target value for λ₁ (only used if loss_type="targeted_magnitude")
        
    Returns:
        Dictionary with optimization results
    """
    model = calculator.potential
    device = model.device
    atomsymbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in atomic_nums]
    start_coords = initial_coords.clone().detach().cpu().numpy().flatten()
    
    # Track optimization history
    history = {
        "loss": [],
        "eig0": [],
        "eig1": [],
        "eig_product": [],
        "neg_vibrational": [],
    }
    
    # Track best result
    best_result = {
        "loss": float('inf'),
        "eig0": float('nan'),
        "eig1": float('nan'),
        "eig_product": float('inf'),
        "neg_vibrational": -1,
    }
    
    def objective_and_grad(x_flat):
        """Compute loss and gradient for eigenvalue-based optimization."""
        nonlocal best_result
        
        coords = torch.tensor(
            x_flat.reshape(-1, 3),
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )
        
        try:
            with torch.enable_grad():
                batch = coord_atoms_to_torch_geometric(coords, atomic_nums, device)
                _, _, out = model.forward(batch, otf_graph=True)
                hess_raw = out["hessian"].reshape(coords.numel(), coords.numel())
                hess_proj = massweigh_and_eckartprojection_torch(hess_raw, coords, atomsymbols)
                
                vibrational_eigvals = extract_vibrational_eigenvalues(hess_proj, coords)
                if vibrational_eigvals.numel() < 2:
                    return 1e10, np.zeros_like(x_flat)
                
                eig0 = vibrational_eigvals[0]
                eig1 = vibrational_eigvals[1]
                eig_product = eig0 * eig1
                neg_vibrational = (vibrational_eigvals < 0).sum().item()
                
                # Compute loss
                if loss_type == "eig_product":
                    loss = eig_product
                elif loss_type == "targeted_magnitude":
                    loss = (eig0 - target_eig0)**2 + (eig1 - target_eig1)**2
                else:
                    loss = eig_product  # Default to eig_product
                
                # Track history
                history["loss"].append(loss.item())
                history["eig0"].append(eig0.item())
                history["eig1"].append(eig1.item())
                history["eig_product"].append(eig_product.item())
                history["neg_vibrational"].append(neg_vibrational)
                
                # Update best result
                if loss.item() < best_result["loss"]:
                    best_result["loss"] = loss.item()
                    best_result["eig0"] = eig0.item()
                    best_result["eig1"] = eig1.item()
                    best_result["eig_product"] = eig_product.item()
                    best_result["neg_vibrational"] = neg_vibrational
                
                # Compute gradient
                grad = torch.autograd.grad(loss, coords)[0]
                
            return loss.item(), grad.detach().cpu().numpy().flatten().astype(np.float64)
            
        except Exception as e:
            print(f"    [EIGPROD-BFGS] Evaluation failed: {e}")
            return 1e10, np.zeros_like(x_flat)
    
    # Set up bounds
    if max_step is not None and max_step > 0:
        lower_bounds = start_coords - max_step
        upper_bounds = start_coords + max_step
        bounds = list(zip(lower_bounds, upper_bounds))
    else:
        bounds = None
    
    try:
        result = scipy_minimize(
            objective_and_grad,
            x0=start_coords.astype(np.float64),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': maxiter,
                'gtol': gtol,
                'disp': False,
            }
        )
        
        # Convert result back to torch
        final_coords = torch.tensor(
            result.x.reshape(-1, 3),
            dtype=torch.float32,
            device=device
        )
        
        # Compute final displacement
        displacement = (final_coords - initial_coords.to(device)).norm(dim=1)
        max_atom_disp = displacement.max().item()
        mean_disp = displacement.mean().item()
        
        # Get final state info
        with torch.no_grad():
            batch = coord_atoms_to_torch_geometric(final_coords, atomic_nums, device)
            _, _, out = model.forward(batch, otf_graph=True)
            hess_raw = out["hessian"].reshape(final_coords.numel(), final_coords.numel())
            
            # Get negative eigenvalue count from raw Hessian
            hess_np = hess_raw.detach().cpu().numpy()
            eigvals_raw = np.linalg.eigvalsh(hess_np)
            neg_count_raw = int((eigvals_raw < 0).sum())
        
        return {
            "final_coords": final_coords,
            "history": history,
            "nit": result.nit,
            "success": result.success,
            "message": result.message,
            "final_loss": best_result["loss"],
            "final_eig0": best_result["eig0"],
            "final_eig1": best_result["eig1"],
            "final_eig_product": best_result["eig_product"],
            "final_neg_vibrational": best_result["neg_vibrational"],
            "final_neg_eigvals": neg_count_raw,
            "max_atom_disp": max_atom_disp,
            "mean_disp": mean_disp,
            "stop_reason": "bfgs_converged" if result.success else "bfgs_maxiter",
        }
        
    except Exception as e:
        print(f"    [EIGPROD-BFGS] Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "final_coords": initial_coords,
            "history": history,
            "nit": 0,
            "success": False,
            "message": str(e),
            "final_loss": float('inf'),
            "final_eig0": float('nan'),
            "final_eig1": float('nan'),
            "final_eig_product": float('inf'),
            "final_neg_vibrational": -1,
            "final_neg_eigvals": -1,
            "max_atom_disp": 0.0,
            "mean_disp": 0.0,
            "stop_reason": "error",
        }


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
    sign_neg_target: float = -5e-3,
    sign_pos_floor: float = 1e-3,
    use_line_search: bool = True,
    adaptive_max_displacement: bool = True,
    initial_max_displacement: float = 2.0,
    final_max_displacement: float = 0.1,
) -> Dict[str, Any]:
    """
    Run gradient descent on eigenvalues to find transition states.

    Args:
        loss_type: Type of loss function to use:
            - "relu": Original ReLU loss (allows infinitesimal eigenvalues)
            - "targeted_magnitude": Target specific eigenvalue magnitudes (RECOMMENDED)
            - "midpoint_squared": Minimize squared midpoint between eigenvalues
            - "eig_product": Direct gradient descent on λ₀·λ₁
            - "sign_enforcer": Dynamically push eigenvalues into the TS pattern
        target_eig0: Initial target value for most negative eigenvalue (eV/Å²)
        target_eig1: Initial target value for second smallest eigenvalue (eV/Å²)
        adaptive_targets: Enable adaptive target relaxation over final steps
        adaptive_relax_steps: Number of final steps over which to relax targets
        adaptive_final_eig0: Final relaxed target for λ₀ (eV/Å²)
        adaptive_final_eig1: Final relaxed target for λ₁ (eV/Å²)
        early_stop_eig_product_threshold: Stop once λ₀·λ₁ ≤ -THRESH (set ≤0 to disable)
        sign_neg_target: Target (slightly negative) value for λ₀ when no negatives exist
        sign_pos_floor: Minimum positive floor for secondary negative eigenvalues
        use_line_search: Enable line search to find optimal step size
        adaptive_max_displacement: Adaptively adjust max displacement based on progress
        initial_max_displacement: Starting max displacement per atom (Å)
        final_max_displacement: Final max displacement per atom (Å)
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
        # Initialize default values for saddle detection and adaptive scaling
        saddle_info = {'saddle_order': 0, 'classification': 'unconverged'}
        step_scale = 1.0

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

                # Extract vibrational eigenvalues (centralized function)
                vibrational_eigvals = extract_vibrational_eigenvalues(hess_proj, coords)
                if vibrational_eigvals.numel() < 2:
                    raise RuntimeError(
                        f"Insufficient vibrational eigenvalues after removing rigid modes."
                    )

                eig0 = vibrational_eigvals[0]
                eig1 = vibrational_eigvals[1]
                eig_product = eig0 * eig1
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

                elif loss_type == "eig_product":
                    # Direct gradient descent on the product of the first two vibrational eigenvalues
                    loss = eig_product

                elif loss_type == "sign_enforcer":
                    # Dynamically enforce "exactly one negative eigenvalue" pattern.
                    neg_target = eig0.new_tensor(sign_neg_target)
                    pos_floor = eig0.new_tensor(sign_pos_floor)

                    if neg_vibrational == 0:
                        # No negative eigenvalues: push λ₀ below a small negative target.
                        loss = (eig0 - neg_target).pow(2)
                    elif neg_vibrational == 1:
                        # Desired configuration: nothing to optimize.
                        loss = eig0.new_tensor(0.0)
                    else:
                        # More than one negative: leave λ₀ alone and push the rest positive.
                        trailing_eigs = vibrational_eigvals[1:]
                        if trailing_eigs.numel() == 0:
                            loss = eig0.new_tensor(0.0)
                        else:
                            penalties = (pos_floor - trailing_eigs).pow(2)
                            loss = penalties.sum()

                else:
                    raise ValueError(f"Unknown loss_type: {loss_type}")

                # Early stopping based on loss type
                if loss_type == "relu" and loss.item() < 1e-8:
                    print(f"  Stopping early at step {step}: Converged (Loss ~ 0).")
                    break
                elif loss_type in ["targeted_magnitude", "midpoint_squared"] and loss.item() < 1e-6:
                    print(f"  Stopping early at step {step}: Converged (Loss < 1e-6).")
                    break
                elif loss_type == "sign_enforcer":
                    if neg_vibrational == 1:
                        print(f"  Stopping early at step {step}: Exactly one negative eigenvalue achieved.")
                        break
                    if neg_vibrational == 0 and loss.item() < 1e-8:
                        print(f"  Stopping early at step {step}: λ₀ pushed below target ({sign_neg_target}).")
                        break

                grad = torch.autograd.grad(loss, coords)[0]

                # Classify geometry and compute adaptive step scale
                force_norm = grad.norm().item()
                saddle_info = classify_saddle_point(vibrational_eigvals, force_norm)

                if args.adaptive_step_sizing:
                    step_scale = compute_adaptive_step_scale(
                        saddle_info,
                        base_scale=1.0,
                        higher_order_mult=args.higher_order_multiplier,
                        ts_mult=args.ts_multiplier
                    )
                    if step_scale != 1.0:
                        print(f"  [ADAPTIVE] Order={saddle_info['saddle_order']}, "
                              f"classification={saddle_info['classification']}, "
                              f"scale={step_scale:.1f}×")
                else:
                    step_scale = 1.0

        except (AssertionError, RuntimeError) as e:
            # Handle case where atoms drift too far apart (empty edge graph)
            if "edge_distance_vec is empty" in str(e) or "edge_index" in str(e):
                print(f"  [WARNING] Step {step}: Atoms too far apart, stopping optimization early.")
                print(f"    This usually means the optimization diverged. Returning last valid state.")
                break
            else:
                raise  # Re-raise if it's a different error

        # Track auxiliary statistics for plotting/logging
        energy_value = float("nan")
        force_mean_value = float("nan")
        force_max_value = float("nan")

        energy_tensor = out.get("energy") if isinstance(out, dict) else None
        if isinstance(energy_tensor, torch.Tensor):
            energy_value = float(energy_tensor.detach().reshape(-1)[0].item())
        elif isinstance(energy_tensor, (float, int)):
            energy_value = float(energy_tensor)

        forces_tensor = out.get("forces") if isinstance(out, dict) else None
        if isinstance(forces_tensor, torch.Tensor):
            forces_detached = forces_tensor.detach()
            if forces_detached.dim() == 3 and forces_detached.shape[0] == 1:
                forces_detached = forces_detached[0]
            if forces_detached.numel() > 0:
                force_vectors = forces_detached.reshape(-1, 3)
                force_norms = force_vectors.norm(dim=1)
                force_mean_value = float(force_norms.mean().item())
                force_max_value = float(force_norms.max().item())

        with torch.no_grad():
            # Gradient clipping to prevent exploding gradients
            grad_norm = torch.norm(grad)
            grad_norm_value = grad_norm.item()
            max_grad_norm = 100.0  # Maximum allowed gradient norm
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / grad_norm)
                grad_norm_value = float(max_grad_norm)

            # Adaptive maximum displacement based on progress
            if adaptive_max_displacement:
                # Linearly interpolate max displacement from initial to final
                progress = step / max(n_steps - 1, 1)
                max_displacement_per_atom = (
                    initial_max_displacement * (1 - progress) +
                    final_max_displacement * progress
                )
            else:
                max_displacement_per_atom = 0.2  # Original fixed value

            # Compute base update direction (normalized)
            base_update = lr * grad

            # Line search: try multiple step sizes if enabled
            if use_line_search:
                # Test multiple step sizes: [0.25×, 0.5×, 1×, 2×, 4×] of base update
                # Apply adaptive scaling based on saddle order if enabled
                base_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
                step_multipliers = [m * step_scale for m in base_multipliers]
                best_loss = loss.item()
                best_multiplier = 1.0
                best_coords = coords.clone()
                best_eig0 = eig0.item()
                best_eig1 = eig1.item()
                best_eig_prod = eig_product.item()
                best_neg_vib = neg_vibrational

                for multiplier in step_multipliers:
                    # Compute candidate update
                    candidate_update = base_update * multiplier

                    # Apply displacement limit to candidate
                    update_per_atom = candidate_update.reshape(-1, 3)
                    atom_displacements = torch.norm(update_per_atom, dim=1)
                    max_atom_disp = atom_displacements.max()

                    if max_atom_disp > max_displacement_per_atom:
                        scale_factor = max_displacement_per_atom / max_atom_disp
                        candidate_update = candidate_update * scale_factor

                    # Evaluate candidate position
                    candidate_coords = coords - candidate_update
                    candidate_loss, cand_eig0, cand_eig1, cand_eig_prod, cand_neg_vib = evaluate_step(
                        candidate_coords,
                        atomic_nums,
                        calculator,
                        loss_type,
                        target_eig0,
                        target_eig1,
                        sign_neg_target,
                        sign_pos_floor,
                    )

                    # Accept step if it improves loss (or if it's the first valid step)
                    if candidate_loss < best_loss or (best_loss == float('inf') and candidate_loss != float('inf')):
                        best_loss = candidate_loss
                        best_multiplier = multiplier
                        best_coords = candidate_coords
                        best_eig0 = cand_eig0
                        best_eig1 = cand_eig1
                        best_eig_prod = cand_eig_prod
                        best_neg_vib = cand_neg_vib

                # Compute actual displacement for logging (before updating coords)
                displacement = (coords - best_coords).reshape(-1, 3)
                max_atom_disp_value = torch.norm(displacement, dim=1).max().item()

                # Apply best step found
                coords = best_coords
                # Update tracked values with best step's values
                final_eigvals = torch.tensor([best_eig0, best_eig1], device=device)
                neg_vibrational = best_neg_vib
                loss = torch.tensor(best_loss, device=device)
                eig_product = torch.tensor(best_eig_prod, device=device)

                if step % 10 == 0 and best_multiplier != 1.0:
                    print(f"    [Line search] Selected step multiplier: {best_multiplier:.2f}×")

            else:
                # Original behavior: fixed step size with displacement clamping
                update = base_update
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
        history["energy"].append(energy_value)
        history["force_mean"].append(force_mean_value)
        history["saddle_order"].append(saddle_info['saddle_order'])
        history["step_scale"].append(step_scale)
        history["classification"].append(saddle_info['classification'])
        history["force_max"].append(force_max_value)
        history["step"].append(step)

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
                        help="Starting geometry: 'reactant', 'ts', 'midpoint_rt', 'three_quarter_rt', "
                             "or add noise: 'reactant_noise0.5A', 'reactant_noise1A', 'reactant_noise2A', 'reactant_noise10A', etc.")

    # Loss function arguments
    parser.add_argument("--loss-type", type=str, default="targeted_magnitude",
                        choices=["relu", "targeted_magnitude", "midpoint_squared", "eig_product", "sign_enforcer"],
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
    parser.add_argument("--sign-neg-target", type=float, default=-5e-3,
                        help="For 'sign_enforcer': target value (in eV/Å²) to push λ₀ below when no negatives.")
    parser.add_argument("--sign-pos-floor", type=float, default=1e-3,
                        help="For 'sign_enforcer': positive floor (in eV/Å²) to push additional negatives above.")

    # Adaptive step size arguments
    parser.add_argument("--use-line-search", action="store_true", default=True,
                        help="Enable line search to find optimal step size (default: True)")
    parser.add_argument("--no-line-search", action="store_false", dest="use_line_search",
                        help="Disable line search (use fixed step size)")
    parser.add_argument("--adaptive-max-displacement", action="store_true", default=True,
                        help="Adaptively adjust max displacement based on progress (default: True)")
    parser.add_argument("--no-adaptive-max-displacement", action="store_false", dest="adaptive_max_displacement",
                        help="Disable adaptive max displacement (use fixed 0.2 Å)")
    parser.add_argument("--initial-max-displacement", type=float, default=2.0,
                        help="Starting max displacement per atom in Å (default: 2.0)")
    parser.add_argument("--final-max-displacement", type=float, default=0.1,
                        help="Final max displacement per atom in Å (default: 0.1)")

    # Adaptive step sizing based on saddle order
    parser.add_argument("--adaptive-step-sizing", action="store_true",
                        help="Enable adaptive step sizing based on saddle order (number of negative eigenvalues)")
    parser.add_argument("--higher-order-multiplier", type=float, default=5.0,
                        help="Step size multiplier for higher-order saddles (default: 5.0). "
                             "Order-2 saddles use 5×, order-3 use 10×, etc.")
    parser.add_argument("--ts-multiplier", type=float, default=0.5,
                        help="Step size multiplier near TS (order-1) for refinement (default: 0.5)")

    # BFGS optimization arguments
    parser.add_argument("--use-bfgs", action="store_true",
                        help="Use L-BFGS-B optimizer instead of gradient descent")
    parser.add_argument("--bfgs-maxiter", type=int, default=100,
                        help="Maximum iterations for BFGS optimization (default: 100)")
    parser.add_argument("--bfgs-gtol", type=float, default=1e-5,
                        help="Gradient tolerance for BFGS convergence (default: 1e-5)")
    parser.add_argument("--bfgs-max-step", type=float, default=1.0,
                        help="Maximum displacement per atom during BFGS (default: 1.0 Å)")

    # W&B arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="gad-ts-search",
                        help="W&B project name (default: gad-ts-search)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity/username (optional)")

    args = parser.parse_args()

    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)

    # Set up experiment logger
    loss_type_flags = build_loss_type_flags(args)

    # Set up experiment logger (file management only)
    logger = ExperimentLogger(
        base_dir=out_dir,
        script_name="gad-eigdescent",
        loss_type_flags=loss_type_flags,
        max_graphs_per_transition=10,
        random_seed=42,
    )

    # Initialize W&B if requested
    if args.wandb:
        wandb_config = {
            "script": "gad_eigenvalue_descent",
            "loss_type": args.loss_type,
            "start_from": args.start_from,
            "n_steps_opt": args.n_steps_opt,
            "lr": args.lr,
            "target_eig0": args.target_eig0,
            "target_eig1": args.target_eig1,
            "adaptive_targets": args.adaptive_targets,
            "max_samples": args.max_samples,
        }
        init_wandb_run(
            project=args.wandb_project,
            name=f"gad-eigdescent_{loss_type_flags}",
            config=wandb_config,
            entity=args.wandb_entity,
            tags=[args.loss_type, args.start_from],
            run_dir=str(logger.run_dir),
        )

    # Track metrics for summary statistics
    all_metrics = {
        "wallclock_time": [],
        "steps_taken": [],
        "final_neg_vibrational": [],  # For saddle order distribution
        "final_eig_product": [],
        "final_eig0": [],
        "final_eig1": [],
        "final_loss": [],
        "rmsd_to_known_ts": [],
    }

    print(f"Running Eigenvalue Descent to Find Transition States")
    print(f"Output directory: {logger.run_dir}")
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
    elif args.loss_type == "sign_enforcer":
        print(f"  λ₀ target when all positive: {args.sign_neg_target:.4f} eV/Å²")
        print(f"  Positive floor for extra negatives: {args.sign_pos_floor:.4f} eV/Å²")
    elif args.adaptive_targets:
        print("  [WARNING] Adaptive targets requested but only supported with 'targeted_magnitude'; disabling.")
        args.adaptive_targets = False

    # Print adaptive step size settings
    print(f"\nAdaptive Step Size Settings:")
    print(f"  Line Search: {'ENABLED' if args.use_line_search else 'DISABLED'}")
    if args.use_line_search:
        print(f"    Testing step multipliers: [0.25×, 0.5×, 1×, 2×, 4×]")
    print(f"  Adaptive Max Displacement: {'ENABLED' if args.adaptive_max_displacement else 'DISABLED'}")
    if args.adaptive_max_displacement:
        print(f"    Initial: {args.initial_max_displacement:.2f} Å → Final: {args.final_max_displacement:.2f} Å")
    else:
        print(f"    Fixed: 0.2 Å")

    for i, batch in enumerate(dataloader):
        if i >= args.max_samples: break
        print(f"\n--- Processing Sample {i} (Formula: {batch.formula[0]}) ---")
        sample_start_time = time.time()
        try:
            # Use parse_starting_geometry to handle both standard and noisy starting points
            initial_coords = parse_starting_geometry(args.start_from, batch, noise_seed=42, sample_index=i)

            if args.use_bfgs:
                # --- BFGS MODE ---
                print(f"  Running BFGS optimization...")
                opt_results = run_eigprod_bfgs(
                    calculator=calculator,
                    initial_coords=initial_coords,
                    atomic_nums=batch.z,
                    maxiter=args.bfgs_maxiter,
                    gtol=args.bfgs_gtol,
                    max_step=args.bfgs_max_step,
                    loss_type=args.loss_type,
                    target_eig0=args.target_eig0,
                    target_eig1=args.target_eig1,
                )
                print(f"  [BFGS] Done: nit={opt_results['nit']}, success={opt_results['success']}, "
                      f"final_eig_product={opt_results['final_eig_product']:.6f}")
            else:
                # --- GRADIENT DESCENT MODE ---
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
                    sign_neg_target=args.sign_neg_target,
                    sign_pos_floor=args.sign_pos_floor,
                    use_line_search=args.use_line_search,
                    adaptive_max_displacement=args.adaptive_max_displacement,
                    initial_max_displacement=args.initial_max_displacement,
                    final_max_displacement=args.final_max_displacement,
                )

            with torch.no_grad():
                final_batch = coord_atoms_to_torch_geometric(opt_results['final_coords'].to(device), batch.z, device)
                _, _, final_out = calculator.potential.forward(final_batch, otf_graph=True)
                final_freq_info = analyze_frequencies_torch(final_out['hessian'], opt_results['final_coords'].to(device), batch.z)

            # Compute initial eigenvalues for transition tracking
            initial_batch = coord_atoms_to_torch_geometric(initial_coords.to(device), batch.z, device)
            _, _, initial_out = calculator.potential.forward(initial_batch, otf_graph=True)
            initial_freq_info = analyze_frequencies_torch(initial_out['hessian'], initial_coords.to(device), batch.z)

            # Compute steps_to_ts: find first step where neg_vibrational == 1
            steps_to_ts = None
            neg_vib_history = opt_results["history"].get("neg_vibrational", [])
            if neg_vib_history:
                for step_idx, neg_vib in enumerate(neg_vib_history):
                    if neg_vib == 1:
                        steps_to_ts = step_idx
                        break

            # Create RunResult
            # Extract adaptive step sizing metrics from history
            extra_data = {}
            if "saddle_order" in opt_results["history"] and opt_results["history"]["saddle_order"]:
                # Average saddle order across trajectory
                saddle_orders = opt_results["history"]["saddle_order"]
                extra_data["avg_saddle_order"] = float(np.mean(saddle_orders))
                extra_data["final_saddle_order"] = int(saddle_orders[-1]) if saddle_orders else None

                # Count how many steps were at each saddle order
                unique_orders, counts = np.unique(saddle_orders, return_counts=True)
                for order, count in zip(unique_orders, counts):
                    extra_data[f"steps_at_order_{int(order)}"] = int(count)

                # Step scale statistics
                if "step_scale" in opt_results["history"] and opt_results["history"]["step_scale"]:
                    step_scales = opt_results["history"]["step_scale"]
                    extra_data["avg_step_scale"] = float(np.mean(step_scales))
                    extra_data["max_step_scale"] = float(np.max(step_scales))
                    extra_data["min_step_scale"] = float(np.min(step_scales))

            result = RunResult(
                sample_index=i,
                formula=batch.formula[0],
                initial_neg_eigvals=int(initial_freq_info.get("neg_num", -1)),
                final_neg_eigvals=int(final_freq_info.get("neg_num", -1)),
                initial_neg_vibrational=None,  # Could compute if needed
                final_neg_vibrational=opt_results["final_neg_vibrational"],
                steps_taken=len(opt_results["history"]["loss"]),
                steps_to_ts=steps_to_ts,
                final_time=None,  # Not applicable for optimization
                final_eig0=opt_results["final_eig0"],
                final_eig1=opt_results["final_eig1"],
                final_eig_product=opt_results["final_eig_product"],
                final_loss=opt_results["final_loss"],
                rmsd_to_known_ts=align_ordered_and_get_rmsd(opt_results["final_coords"], batch.pos_transition),
                stop_reason=opt_results.get("stop_reason"),
                plot_path=None,  # Will be set below
                extra_data=extra_data,
            )

            # Track wallclock time and metrics for summary
            sample_wallclock = time.time() - sample_start_time
            all_metrics["wallclock_time"].append(sample_wallclock)
            all_metrics["steps_taken"].append(result.steps_taken)
            all_metrics["final_neg_vibrational"].append(result.final_neg_vibrational)
            if result.final_eig_product is not None:
                all_metrics["final_eig_product"].append(result.final_eig_product)
            if result.final_eig0 is not None:
                all_metrics["final_eig0"].append(result.final_eig0)
            if result.final_eig1 is not None:
                all_metrics["final_eig1"].append(result.final_eig1)
            if result.final_loss is not None:
                all_metrics["final_loss"].append(result.final_loss)
            if result.rmsd_to_known_ts is not None:
                all_metrics["rmsd_to_known_ts"].append(result.rmsd_to_known_ts)

            # Add result to logger (file management)
            logger.add_result(result)

            # Create plot
            fig_and_filename = plot_eig_descent_history(
                history=opt_results["history"],
                sample_index=i,
                formula=batch.formula[0],
                start_from=args.start_from,
                target_eig0=args.target_eig0,
                target_eig1=args.target_eig1,
                final_neg_vibrational=result.final_neg_vibrational,
                final_neg_eigvals=result.final_neg_eigvals,
            )

            # Save plot to disk (handles sampling)
            fig = None
            if fig_and_filename[0] is not None:
                fig, filename = fig_and_filename
                plot_path = logger.save_graph(result, fig, filename)
                if plot_path:
                    result.plot_path = plot_path
                    print(f"  Saved plot to: {plot_path}")
                else:
                    print(f"  Skipped plot (max samples for {result.transition_key} reached)")
                    fig = None  # Don't log to W&B if not saved

            # Log metrics + plot together to W&B (once per sample)
            metrics = {
                "formula": batch.formula[0],
                "transition_type": result.transition_key,
                "initial_neg_eigvals": result.initial_neg_eigvals,
                "final_neg_eigvals": result.final_neg_eigvals,
                "final_neg_vibrational": result.final_neg_vibrational,
                "steps_taken": result.steps_taken,
                "steps_to_ts": result.steps_to_ts,
                "final_loss": result.final_loss,
                "final_eig0": result.final_eig0,
                "final_eig1": result.final_eig1,
                "final_eig_product": result.final_eig_product,
                "rmsd_to_known_ts": result.rmsd_to_known_ts,
                "reached_ts": int(result.final_neg_vibrational == 1) if result.final_neg_vibrational is not None else 0,
                "wallclock_time": sample_wallclock,
            }
            log_sample(i, metrics, fig=fig, plot_name=f"eigdescent_{result.transition_key}")
            if fig is not None:
                plt.close(fig)

            print(f"Result for Sample {i}:")
            print(f"  Transition: {result.transition_key}")
            print(f"  Steps taken: {result.steps_taken}, Wallclock: {sample_wallclock:.2f}s")
            print(f"  Final Loss: {result.final_loss:.6e}")
            print(f"  Final λ₀: {result.final_eig0:.6f} eV/Å², λ₁: {result.final_eig1:.6f} eV/Å²")
            print(f"  Final λ₀*λ₁: {result.final_eig_product:.6e}")
            print(f"  Final Neg Vibrational: {result.final_neg_vibrational} (Freq Analysis Negs: {result.final_neg_eigvals})")
            print(f"  RMSD to T1x TS: {result.rmsd_to_known_ts:.4f} Å")
            if result.stop_reason:
                print(f"  Stop Reason: {result.stop_reason}")

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

    # Compute and log summary statistics to W&B
    if all_metrics["wallclock_time"]:
        total_samples = len(all_metrics["wallclock_time"])
        
        # Count final saddle orders (how many converged to order 0, 1, 2, 3, etc.)
        from collections import Counter
        # Filter out None values for counting
        valid_orders = [x for x in all_metrics["final_neg_vibrational"] if x is not None]
        order_counts = Counter(valid_orders)
        
        # Build summary dict
        summary = {
            "total_samples": total_samples,
            "avg_steps": np.mean(all_metrics["steps_taken"]),
            "std_steps": np.std(all_metrics["steps_taken"]),
            "avg_wallclock_time": np.mean(all_metrics["wallclock_time"]),
            "std_wallclock_time": np.std(all_metrics["wallclock_time"]),
            "total_wallclock_time": sum(all_metrics["wallclock_time"]),
            # Saddle order distribution
            "ts_success_rate": order_counts.get(1, 0) / total_samples if total_samples > 0 else 0,
            "count_order_0": order_counts.get(0, 0),
            "count_order_1_ts": order_counts.get(1, 0),
            "count_order_2": order_counts.get(2, 0),
            "count_order_3": order_counts.get(3, 0),
            "count_order_4_plus": sum(v for k, v in order_counts.items() if k is not None and k >= 4),
        }
        
        # Add averages for other metrics
        if all_metrics["final_eig_product"]:
            summary["avg_final_eig_product"] = np.mean(all_metrics["final_eig_product"])
            summary["std_final_eig_product"] = np.std(all_metrics["final_eig_product"])
        if all_metrics["final_eig0"]:
            summary["avg_final_eig0"] = np.mean(all_metrics["final_eig0"])
        if all_metrics["final_eig1"]:
            summary["avg_final_eig1"] = np.mean(all_metrics["final_eig1"])
        if all_metrics["final_loss"]:
            summary["avg_final_loss"] = np.mean(all_metrics["final_loss"])
        if all_metrics["rmsd_to_known_ts"]:
            summary["avg_rmsd_to_known_ts"] = np.mean(all_metrics["rmsd_to_known_ts"])
            summary["std_rmsd_to_known_ts"] = np.std(all_metrics["rmsd_to_known_ts"])
        
        log_summary(summary)
    finish_wandb()
