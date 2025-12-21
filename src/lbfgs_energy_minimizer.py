# src/lbfgs_energy_minimizer.py
"""
L-BFGS Energy Minimizer for pre-conditioning noisy starting geometries.

This module provides a modular L-BFGS energy minimization algorithm that uses
HIP's direct force/Hessian outputs to descend from noisy starting geometries
(4+ negative eigenvalues) to stable points (0-1 negative eigenvalues),
preparing them for subsequent TS-finding algorithms like GAD.

Key features:
- Uses HIP's direct energy, forces, and Hessian outputs (no autograd)
- Monitors vibrational eigenvalue count to detect convergence to minimum
- Stopping criterion: 0 or 1 negative vibrational eigenvalues
- Integrates with existing logging infrastructure (ExperimentLogger, W&B)
"""

import os
import json
import argparse
import time
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from torch_geometric.data import Data as TGData, Batch as TGBatch

from .common_utils import (
    setup_experiment, add_common_args, parse_starting_geometry
)
from .differentiable_projection import (
    differentiable_massweigh_and_eckartprojection_torch as massweigh_and_eckartprojection_torch
)
from hip.ff_lmdb import Z_TO_ATOM_SYMBOL
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from .experiment_logger import (
    ExperimentLogger, RunResult, build_loss_type_flags,
    init_wandb_run, log_sample, log_summary, finish_wandb,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Helper Functions
# =============================================================================

def coord_atoms_to_torch_geometric(coords, atomic_nums, device):
    """
    Convert coordinates and atomic numbers to a PyG batch for model inference.
    
    Important: Must move batch to device AFTER creating it from data list.
    Matches the format expected by Equiformer models.
    """
    if isinstance(coords, torch.Tensor) and coords.dim() == 1:
        coords = coords.reshape(-1, 3)

    if isinstance(coords, torch.Tensor):
        if torch.isnan(coords).any() or torch.isinf(coords).any():
            raise ValueError("Invalid coordinates detected (NaN or Inf)")

    data = TGData(
        pos=torch.as_tensor(coords, dtype=torch.float32).reshape(-1, 3),
        z=torch.as_tensor(atomic_nums, dtype=torch.int64),
        charges=torch.as_tensor(atomic_nums, dtype=torch.int64),
        natoms=torch.tensor([len(atomic_nums)], dtype=torch.int64),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool),
    )
    return TGBatch.from_data_list([data]).to(device)


def compute_vibrational_eigenvalues(
    hessian: torch.Tensor,
    coords: torch.Tensor,
    atomsymbols: List[str],
) -> Tuple[torch.Tensor, int, int]:
    """
    Compute vibrational eigenvalues from Hessian using mass-weighting and Eckart projection.
    
    Args:
        hessian: Raw Hessian from HIP (3N x 3N)
        coords: Atomic coordinates (N, 3) or (3N,)
        atomsymbols: List of atom symbols (e.g., ['C', 'H', 'H', 'H', 'H'])
    
    Returns:
        Tuple of:
            - vibrational_eigvals: Eigenvalues with rigid modes removed, sorted ascending
            - neg_count: Number of negative vibrational eigenvalues
            - n_rigid_removed: Number of rigid-body modes removed (5 or 6)
    """
    coords_3d = coords.reshape(-1, 3)
    
    # Mass-weigh and Eckart-project the Hessian
    hess_proj = massweigh_and_eckartprojection_torch(hessian, coords_3d, atomsymbols)
    
    # Compute full eigenvalue spectrum
    eigvals, _ = torch.linalg.eigh(hess_proj)
    
    # Determine number of rigid modes based on molecular geometry
    coords_cent = coords_3d.detach().to(torch.float64)
    coords_cent = coords_cent - coords_cent.mean(dim=0, keepdim=True)
    geom_rank = torch.linalg.matrix_rank(coords_cent.cpu(), tol=1e-8).item()
    expected_rigid = 5 if geom_rank <= 2 else 6  # Linear vs non-linear
    
    total_modes = eigvals.shape[0]
    n_rigid_removed = min(expected_rigid, max(0, total_modes - 2))
    
    # Remove rigid modes (smallest by absolute value)
    abs_sorted_idx = torch.argsort(torch.abs(eigvals))
    keep_idx = abs_sorted_idx[n_rigid_removed:]
    keep_idx, _ = torch.sort(keep_idx)  # Re-sort to maintain ascending order
    vibrational_eigvals = eigvals[keep_idx]
    
    # Count negative eigenvalues
    neg_count = (vibrational_eigvals < 0).sum().item()
    
    return vibrational_eigvals, neg_count, n_rigid_removed


def evaluate_geometry(
    calculator: EquiformerTorchCalculator,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    atomsymbols: List[str],
    device: str,
) -> Dict[str, Any]:
    """
    Evaluate energy, forces, and eigenvalue statistics at a geometry.
    
    Uses HIP's direct outputs (no autograd needed).
    
    Args:
        calculator: HIP calculator instance
        coords: Atomic coordinates (N, 3) or (3N,)
        atomic_nums: Atomic numbers tensor
        atomsymbols: List of atom symbols
        device: Device string
    
    Returns:
        Dictionary with:
            - energy: Potential energy (eV)
            - forces: Forces tensor (N, 3)
            - force_rms: RMS force magnitude
            - force_max: Maximum force magnitude
            - neg_eig_count: Number of negative vibrational eigenvalues
            - vibrational_eigvals: All vibrational eigenvalues
            - eig0, eig1: Two smallest eigenvalues
            - n_rigid_removed: Number of rigid modes removed
    """
    coords_3d = coords.reshape(-1, 3).to(device)
    
    with torch.no_grad():
        batch = coord_atoms_to_torch_geometric(coords_3d, atomic_nums, device)
        results = calculator.predict(batch, do_hessian=True)
        
        energy = results["energy"].item()
        forces = results["forces"]  # (N, 3)
        hessian = results["hessian"].reshape(coords_3d.numel(), coords_3d.numel())
        
        # Compute vibrational eigenvalues
        vib_eigvals, neg_count, n_rigid = compute_vibrational_eigenvalues(
            hessian, coords_3d, atomsymbols
        )
        
        # Force statistics
        force_norms = forces.norm(dim=1)
        force_rms = force_norms.mean().item()
        force_max = force_norms.max().item()
        
        return {
            "energy": energy,
            "forces": forces,
            "force_rms": force_rms,
            "force_max": force_max,
            "neg_eig_count": neg_count,
            "vibrational_eigvals": vib_eigvals.cpu().numpy().tolist(),
            "eig0": vib_eigvals[0].item() if len(vib_eigvals) > 0 else None,
            "eig1": vib_eigvals[1].item() if len(vib_eigvals) > 1 else None,
            "n_rigid_removed": n_rigid,
        }


# =============================================================================
# L-BFGS Energy Minimizer Class
# =============================================================================

class LBFGSEnergyMinimizer:
    """
    L-BFGS energy minimizer for pre-conditioning noisy geometries.
    
    Uses scipy's L-BFGS-B optimizer with HIP's direct force outputs as gradients.
    Monitors vibrational eigenvalue count and stops when reaching a minimum
    (0 or 1 negative eigenvalues).
    
    Example usage:
        minimizer = LBFGSEnergyMinimizer(calculator, atomic_nums, device)
        result = minimizer.minimize(noisy_coords)
        clean_coords = result["final_coords"]
    """
    
    def __init__(
        self,
        calculator: EquiformerTorchCalculator,
        atomic_nums: torch.Tensor,
        device: str,
        max_iterations: int = 200,
        force_tol: float = 0.01,  # eV/Å - convergence criterion
        max_step: float = 0.5,     # Å - maximum displacement per atom from start
        target_neg_eig_count: int = 1,  # Stop when neg_eig_count <= this
        eigenvalue_check_freq: int = 5,  # Check eigenvalues every N iterations
        verbose: bool = True,
    ):
        """
        Initialize the L-BFGS energy minimizer.
        
        Args:
            calculator: HIP EquiformerTorchCalculator instance
            atomic_nums: Atomic numbers as torch tensor
            device: Device string ('cuda' or 'cpu')
            max_iterations: Maximum L-BFGS iterations
            force_tol: Force convergence tolerance (eV/Å)
            max_step: Maximum displacement per atom from starting position (Å)
            target_neg_eig_count: Stop when negative eigenvalue count <= this
            eigenvalue_check_freq: Check eigenvalues every N iterations
            verbose: Print progress information
        """
        self.calculator = calculator
        self.atomic_nums = torch.as_tensor(atomic_nums, dtype=torch.int64)
        self.atomsymbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in self.atomic_nums]
        self.num_atoms = len(atomic_nums)
        self.device = device
        self.max_iterations = max_iterations
        self.force_tol = force_tol
        self.max_step = max_step
        self.target_neg_eig_count = target_neg_eig_count
        self.eigenvalue_check_freq = eigenvalue_check_freq
        self.verbose = verbose
        
        # Trajectory storage (populated during minimize)
        self.trajectory: Dict[str, List] = {}
        self._iteration_count = 0
        self._should_stop = False
        self._current_neg_eig_count = None
    
    def _objective_and_grad(self, x_flat: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute energy and gradient (negative forces) for L-BFGS.
        
        This is called by scipy.optimize.minimize. HIP gives us forces directly,
        which are -∇E, so we return -forces as the gradient.
        """
        coords = torch.tensor(
            x_flat.reshape(-1, 3),
            dtype=torch.float32,
            device=self.device,
        )
        
        with torch.no_grad():
            batch = coord_atoms_to_torch_geometric(coords, self.atomic_nums, self.device)
            results = self.calculator.predict(batch, do_hessian=True)
            
            energy = results["energy"].item()
            forces = results["forces"]  # (N, 3)
            
            # Gradient = -forces (since forces = -∇E)
            grad = -forces.cpu().numpy().flatten().astype(np.float64)
            
            # Track trajectory
            force_rms = forces.norm(dim=1).mean().item()
            force_max = forces.norm(dim=1).max().item()
            
            # Check eigenvalues periodically
            if self._iteration_count % self.eigenvalue_check_freq == 0:
                hessian = results["hessian"].reshape(coords.numel(), coords.numel())
                vib_eigvals, neg_count, _ = compute_vibrational_eigenvalues(
                    hessian, coords, self.atomsymbols
                )
                self._current_neg_eig_count = neg_count
                eig0 = vib_eigvals[0].item() if len(vib_eigvals) > 0 else None
                eig1 = vib_eigvals[1].item() if len(vib_eigvals) > 1 else None
                
                # Check stopping criterion
                if neg_count <= self.target_neg_eig_count:
                    if self.verbose:
                        print(f"    [L-BFGS] Target reached: {neg_count} negative eigenvalues")
                    self._should_stop = True
            else:
                neg_count = self._current_neg_eig_count
                eig0 = eig1 = None
            
            # Record trajectory
            self.trajectory["iteration"].append(self._iteration_count)
            self.trajectory["energy"].append(energy)
            self.trajectory["force_rms"].append(force_rms)
            self.trajectory["force_max"].append(force_max)
            self.trajectory["neg_eig_count"].append(neg_count)
            self.trajectory["eig0"].append(eig0)
            self.trajectory["eig1"].append(eig1)
            
            if self.verbose and self._iteration_count % 10 == 0:
                neg_str = f"{neg_count}" if neg_count is not None else "?"
                print(f"    [L-BFGS] Iter {self._iteration_count:4d}: "
                      f"E={energy:12.6f} eV, |F|_rms={force_rms:.4f}, neg_eig={neg_str}")
            
            self._iteration_count += 1
        
        return energy, grad
    
    def _callback(self, x_flat: np.ndarray) -> bool:
        """Callback for L-BFGS to enable early stopping."""
        # Return True to stop optimization
        return self._should_stop
    
    def minimize(
        self,
        initial_coords: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Run L-BFGS energy minimization.
        
        Args:
            initial_coords: Starting coordinates (N, 3) or (3N,)
        
        Returns:
            Dictionary with:
                - final_coords: Optimized coordinates (torch.Tensor, N x 3)
                - initial_coords: Starting coordinates (torch.Tensor, N x 3)
                - initial_neg_eig_count: Negative eigenvalue count at start
                - final_neg_eig_count: Negative eigenvalue count at end
                - initial_energy: Energy at start (eV)
                - final_energy: Energy at end (eV)
                - n_iterations: Number of iterations taken
                - converged: Whether convergence criterion was met
                - stop_reason: Reason for stopping
                - trajectory: Dictionary of iteration-wise data
        """
        # Reset state
        self._iteration_count = 0
        self._should_stop = False
        self._current_neg_eig_count = None
        self.trajectory = defaultdict(list)
        
        # Prepare coordinates
        coords_start = initial_coords.clone().detach().reshape(-1, 3).to(self.device)
        x0 = coords_start.cpu().numpy().flatten().astype(np.float64)
        
        # Evaluate initial state
        initial_eval = evaluate_geometry(
            self.calculator, coords_start, self.atomic_nums,
            self.atomsymbols, self.device
        )
        initial_neg_count = initial_eval["neg_eig_count"]
        initial_energy = initial_eval["energy"]
        
        if self.verbose:
            print(f"  [L-BFGS] Starting minimization:")
            print(f"    Initial energy: {initial_energy:.6f} eV")
            print(f"    Initial neg eigenvalues: {initial_neg_count}")
            print(f"    Target: <= {self.target_neg_eig_count} negative eigenvalues")
        
        # Check if already at target
        if initial_neg_count <= self.target_neg_eig_count:
            if self.verbose:
                print(f"  [L-BFGS] Already at target ({initial_neg_count} neg eig). No minimization needed.")
            return {
                "final_coords": coords_start,
                "initial_coords": coords_start.clone(),
                "initial_neg_eig_count": initial_neg_count,
                "final_neg_eig_count": initial_neg_count,
                "initial_energy": initial_energy,
                "final_energy": initial_energy,
                "n_iterations": 0,
                "converged": True,
                "stop_reason": "already_at_target",
                "trajectory": dict(self.trajectory),
            }
        
        # Set up bounds to limit displacement
        if self.max_step is not None and self.max_step > 0:
            lower_bounds = x0 - self.max_step
            upper_bounds = x0 + self.max_step
            bounds = list(zip(lower_bounds, upper_bounds))
        else:
            bounds = None
        
        # Run L-BFGS-B
        try:
            result = scipy_minimize(
                self._objective_and_grad,
                x0=x0,
                method='L-BFGS-B',
                jac=True,  # We provide the gradient
                bounds=bounds,
                callback=self._callback,
                options={
                    'maxiter': self.max_iterations,
                    'gtol': self.force_tol,
                    'disp': False,
                }
            )
            
            final_coords = torch.tensor(
                result.x.reshape(-1, 3),
                dtype=torch.float32,
                device=self.device,
            )
            
            # Determine stop reason
            if self._should_stop:
                stop_reason = "target_neg_eig_reached"
                converged = True
            elif result.success:
                stop_reason = "force_converged"
                converged = True
            else:
                stop_reason = result.message
                converged = False
            
        except Exception as e:
            if self.verbose:
                print(f"  [L-BFGS] Optimization failed: {e}")
            final_coords = coords_start
            stop_reason = f"error: {str(e)}"
            converged = False
        
        # Evaluate final state
        final_eval = evaluate_geometry(
            self.calculator, final_coords, self.atomic_nums,
            self.atomsymbols, self.device
        )
        final_neg_count = final_eval["neg_eig_count"]
        final_energy = final_eval["energy"]
        
        if self.verbose:
            print(f"  [L-BFGS] Finished after {self._iteration_count} iterations")
            print(f"    Final energy: {final_energy:.6f} eV (Δ = {final_energy - initial_energy:+.6f})")
            print(f"    Final neg eigenvalues: {final_neg_count}")
            print(f"    Stop reason: {stop_reason}")
        
        return {
            "final_coords": final_coords,
            "initial_coords": coords_start,
            "initial_neg_eig_count": initial_neg_count,
            "final_neg_eig_count": final_neg_count,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "n_iterations": self._iteration_count,
            "converged": converged,
            "stop_reason": stop_reason,
            "trajectory": dict(self.trajectory),
            "final_force_rms": final_eval["force_rms"],
            "final_eig0": final_eval["eig0"],
            "final_eig1": final_eval["eig1"],
        }


# =============================================================================
# Functional Interface
# =============================================================================

def lbfgs_energy_minimize(
    calculator: EquiformerTorchCalculator,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    device: str,
    max_iterations: int = 200,
    force_tol: float = 0.01,
    max_step: float = 0.5,
    target_neg_eig_count: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function for L-BFGS energy minimization.
    
    A simple functional interface that wraps LBFGSEnergyMinimizer.
    
    Args:
        calculator: HIP EquiformerTorchCalculator instance
        coords: Starting coordinates (N, 3) or (3N,)
        atomic_nums: Atomic numbers
        device: Device string
        max_iterations: Maximum L-BFGS iterations
        force_tol: Force convergence tolerance (eV/Å)
        max_step: Maximum displacement per atom (Å)
        target_neg_eig_count: Stop when neg_eig_count <= this (default: 1)
        verbose: Print progress
    
    Returns:
        Result dictionary (see LBFGSEnergyMinimizer.minimize)
    """
    minimizer = LBFGSEnergyMinimizer(
        calculator=calculator,
        atomic_nums=atomic_nums,
        device=device,
        max_iterations=max_iterations,
        force_tol=force_tol,
        max_step=max_step,
        target_neg_eig_count=target_neg_eig_count,
        verbose=verbose,
    )
    return minimizer.minimize(coords)


# =============================================================================
# Plotting
# =============================================================================

def plot_minimization_trajectory(
    trajectory: Dict[str, List],
    sample_index: int,
    formula: str,
    initial_neg_eig: int,
    final_neg_eig: int,
) -> Tuple[plt.Figure, str]:
    """
    Plot L-BFGS minimization trajectory.
    
    Returns:
        Tuple of (matplotlib Figure, suggested filename)
    """
    iterations = np.array(trajectory.get("iteration", []))
    if len(iterations) == 0:
        # Create empty figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, "No trajectory data", ha='center', va='center')
        return fig, f"lbfgs_{sample_index:03d}_empty.png"
    
    def _nanify(key):
        return np.array([v if v is not None else np.nan 
                        for v in trajectory.get(key, [])], dtype=float)
    
    energy = _nanify("energy")
    force_rms = _nanify("force_rms")
    neg_eig = _nanify("neg_eig_count")
    eig0 = _nanify("eig0")
    eig1 = _nanify("eig1")
    
    # Create 4-panel plot
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"L-BFGS Energy Minimization (Sample {sample_index}): {formula}", fontsize=14)
    
    # Panel 1: Energy
    axes[0].plot(iterations, energy, marker='.', lw=1.2, color='tab:blue')
    axes[0].set_ylabel("Energy (eV)")
    axes[0].set_title("Energy")
    if len(energy) > 0:
        e_start = energy[0] if not np.isnan(energy[0]) else 0
        e_end = energy[-1] if not np.isnan(energy[-1]) else 0
        axes[0].text(0.02, 0.95, f"Start: {e_start:.4f} eV", transform=axes[0].transAxes,
                    ha='left', va='top', fontsize=9)
        axes[0].text(0.98, 0.95, f"End: {e_end:.4f} eV (Δ={e_end-e_start:+.4f})", 
                    transform=axes[0].transAxes, ha='right', va='top', fontsize=9)
    
    # Panel 2: Force RMS (log scale)
    axes[1].semilogy(iterations, force_rms, marker='.', lw=1.2, color='tab:orange')
    axes[1].set_ylabel("Force RMS (eV/Å)")
    axes[1].set_title("Force Magnitude (log scale)")
    axes[1].axhline(0.01, color='green', ls='--', lw=1, alpha=0.7, label='Convergence threshold')
    axes[1].legend(loc='best', fontsize=8)
    
    # Panel 3: Eigenvalues
    valid_eig0 = ~np.isnan(eig0)
    valid_eig1 = ~np.isnan(eig1)
    if valid_eig0.any():
        axes[2].plot(iterations[valid_eig0], eig0[valid_eig0], marker='.', lw=1.2, 
                    color='tab:red', label='λ₀ (smallest)')
    if valid_eig1.any():
        axes[2].plot(iterations[valid_eig1], eig1[valid_eig1], marker='.', lw=1.2,
                    color='tab:green', label='λ₁ (2nd smallest)')
    axes[2].axhline(0, color='grey', ls='--', lw=1)
    axes[2].set_ylabel("Eigenvalue (eV/Å²)")
    axes[2].set_title("Vibrational Eigenvalues (sampled)")
    axes[2].legend(loc='best', fontsize=8)
    
    # Panel 4: Negative eigenvalue count
    valid_neg = ~np.isnan(neg_eig)
    if valid_neg.any():
        axes[3].step(iterations[valid_neg], neg_eig[valid_neg], where='post',
                    lw=1.5, color='tab:purple')
    axes[3].axhline(1, color='green', ls='--', lw=1, alpha=0.7, label='Target (≤1)')
    axes[3].set_ylabel("# Negative Eigenvalues")
    axes[3].set_xlabel("Iteration")
    axes[3].set_title("Saddle Order (Negative Eigenvalue Count)")
    axes[3].legend(loc='best', fontsize=8)
    
    # Set integer y-axis for eigenvalue count
    if valid_neg.any():
        max_neg = int(np.nanmax(neg_eig)) + 1
        axes[3].set_ylim(-0.5, max(max_neg, 2.5))
        axes[3].set_yticks(range(max(int(np.nanmax(neg_eig)) + 2, 3)))
    
    # Add summary text
    axes[3].text(
        0.98, 0.95,
        f"Neg eig: {initial_neg_eig} → {final_neg_eig}",
        transform=axes[3].transAxes,
        ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
    )
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Sanitize formula for filename
    safe_formula = "".join(c if c.isalnum() or c in '-_.' else '_' for c in formula).strip('_')
    filename = f"lbfgs_{sample_index:03d}_{safe_formula}_{initial_neg_eig}to{final_neg_eig}.png"
    
    return fig, filename


# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="L-BFGS Energy Minimizer for pre-conditioning noisy geometries."
    )
    parser = add_common_args(parser)
    
    # L-BFGS specific arguments
    parser.add_argument("--start-from", type=str, default="reactant_noise2A",
                        help="Starting geometry: 'reactant', 'ts', 'midpoint_rt', "
                             "or with noise: 'reactant_noise0.5A', 'reactant_noise2A', etc.")
    parser.add_argument("--max-iterations", type=int, default=200,
                        help="Maximum L-BFGS iterations (default: 200)")
    parser.add_argument("--force-tol", type=float, default=0.01,
                        help="Force convergence tolerance in eV/Å (default: 0.01)")
    parser.add_argument("--max-step", type=float, default=0.5,
                        help="Maximum displacement per atom in Å (default: 0.5)")
    parser.add_argument("--target-neg-eig", type=int, default=1,
                        help="Target negative eigenvalue count to stop (default: 1, i.e., stop at TS or minimum)")
    parser.add_argument("--eigenvalue-check-freq", type=int, default=5,
                        help="Check eigenvalues every N iterations (default: 5)")
    parser.add_argument("--noise-seed", type=int, default=42,
                        help="Random seed for noise generation (default: 42)")
    
    # W&B arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="lbfgs-precondition",
                        help="W&B project name (default: lbfgs-precondition)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity/username (optional)")
    
    args = parser.parse_args()
    
    # Setup experiment
    torch.set_grad_enabled(False)
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    
    # Build loss type flags for directory naming
    loss_type_flags = f"lbfgs-maxiter{args.max_iterations}-tol{args.force_tol}-from-{args.start_from}"
    
    # Set up experiment logger
    logger = ExperimentLogger(
        base_dir=out_dir,
        script_name="lbfgs-minimize",
        loss_type_flags=loss_type_flags,
        max_graphs_per_transition=10,
        random_seed=42,
    )
    
    # Initialize W&B if requested
    if args.wandb:
        wandb_config = {
            "script": "lbfgs_energy_minimizer",
            "start_from": args.start_from,
            "max_iterations": args.max_iterations,
            "force_tol": args.force_tol,
            "max_step": args.max_step,
            "target_neg_eig": args.target_neg_eig,
            "eigenvalue_check_freq": args.eigenvalue_check_freq,
            "max_samples": args.max_samples,
            "noise_seed": args.noise_seed,
        }
        init_wandb_run(
            project=args.wandb_project,
            name=f"lbfgs_{loss_type_flags}",
            config=wandb_config,
            entity=args.wandb_entity,
            tags=[args.start_from, "lbfgs", "precondition"],
            run_dir=str(logger.run_dir),
        )
    
    # Track metrics for summary
    all_metrics = {
        "wallclock_time": [],
        "n_iterations": [],
        "initial_neg_eig": [],
        "final_neg_eig": [],
        "initial_energy": [],
        "final_energy": [],
        "converged": [],
    }
    
    print(f"Running L-BFGS Energy Minimization")
    print(f"  Start from: {args.start_from}")
    print(f"  Target: <= {args.target_neg_eig} negative eigenvalues")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Force tolerance: {args.force_tol} eV/Å")
    print(f"  Output directory: {logger.run_dir}")
    print("-" * 60)
    
    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break
        
        print(f"\n--- Sample {i} (Formula: {batch.formula[0]}) ---")
        sample_start_time = time.time()
        
        try:
            # Get starting coordinates (with noise if specified)
            initial_coords = parse_starting_geometry(
                args.start_from, batch, 
                noise_seed=args.noise_seed, 
                sample_index=i
            )
            
            # Create minimizer
            minimizer = LBFGSEnergyMinimizer(
                calculator=calculator,
                atomic_nums=batch.z,
                device=device,
                max_iterations=args.max_iterations,
                force_tol=args.force_tol,
                max_step=args.max_step,
                target_neg_eig_count=args.target_neg_eig,
                eigenvalue_check_freq=args.eigenvalue_check_freq,
                verbose=True,
            )
            
            # Run minimization
            result = minimizer.minimize(initial_coords)
            
            sample_wallclock = time.time() - sample_start_time
            
            # Track metrics
            all_metrics["wallclock_time"].append(sample_wallclock)
            all_metrics["n_iterations"].append(result["n_iterations"])
            all_metrics["initial_neg_eig"].append(result["initial_neg_eig_count"])
            all_metrics["final_neg_eig"].append(result["final_neg_eig_count"])
            all_metrics["initial_energy"].append(result["initial_energy"])
            all_metrics["final_energy"].append(result["final_energy"])
            all_metrics["converged"].append(1 if result["converged"] else 0)
            
            # Create RunResult for logger
            run_result = RunResult(
                sample_index=i,
                formula=batch.formula[0],
                initial_neg_eigvals=result["initial_neg_eig_count"],
                final_neg_eigvals=result["final_neg_eig_count"],
                initial_neg_vibrational=result["initial_neg_eig_count"],
                final_neg_vibrational=result["final_neg_eig_count"],
                steps_taken=result["n_iterations"],
                steps_to_ts=result["n_iterations"] if result["final_neg_eig_count"] <= 1 else None,
                final_time=None,
                final_eig0=result.get("final_eig0"),
                final_eig1=result.get("final_eig1"),
                final_eig_product=(result.get("final_eig0", 0) or 0) * (result.get("final_eig1", 0) or 0),
                final_loss=result["final_energy"],
                rmsd_to_known_ts=None,
                stop_reason=result["stop_reason"],
                plot_path=None,
                extra_data={
                    "initial_energy": result["initial_energy"],
                    "final_energy": result["final_energy"],
                    "energy_change": result["final_energy"] - result["initial_energy"],
                    "converged": result["converged"],
                    "final_force_rms": result.get("final_force_rms"),
                },
            )
            
            logger.add_result(run_result)
            
            # Create and save plot
            fig, filename = plot_minimization_trajectory(
                trajectory=result["trajectory"],
                sample_index=i,
                formula=batch.formula[0],
                initial_neg_eig=result["initial_neg_eig_count"],
                final_neg_eig=result["final_neg_eig_count"],
            )
            
            plot_path = logger.save_graph(run_result, fig, filename)
            if plot_path:
                run_result.plot_path = plot_path
                print(f"  Saved plot to: {plot_path}")
            
            # Log to W&B
            metrics = {
                "formula": batch.formula[0],
                "initial_neg_eig": result["initial_neg_eig_count"],
                "final_neg_eig": result["final_neg_eig_count"],
                "n_iterations": result["n_iterations"],
                "initial_energy": result["initial_energy"],
                "final_energy": result["final_energy"],
                "energy_change": result["final_energy"] - result["initial_energy"],
                "converged": int(result["converged"]),
                "wallclock_time": sample_wallclock,
            }
            log_sample(i, metrics, fig=fig if plot_path else None, plot_name="lbfgs_trajectory")
            plt.close(fig)
            
            print(f"  Result: {result['initial_neg_eig_count']} → {result['final_neg_eig_count']} neg eig, "
                  f"{result['n_iterations']} iters, {sample_wallclock:.2f}s")
            
        except Exception as e:
            print(f"[ERROR] Sample {i} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    all_runs_path, aggregate_path = logger.save_all_results()
    print(f"\nSaved all runs to: {all_runs_path}")
    print(f"Saved aggregate stats to: {aggregate_path}")
    
    # Print and log summary
    logger.print_summary()
    
    if all_metrics["wallclock_time"]:
        summary = {
            "total_samples": len(all_metrics["wallclock_time"]),
            "avg_iterations": np.mean(all_metrics["n_iterations"]),
            "std_iterations": np.std(all_metrics["n_iterations"]),
            "avg_wallclock_time": np.mean(all_metrics["wallclock_time"]),
            "total_wallclock_time": sum(all_metrics["wallclock_time"]),
            "avg_initial_neg_eig": np.mean(all_metrics["initial_neg_eig"]),
            "avg_final_neg_eig": np.mean(all_metrics["final_neg_eig"]),
            "convergence_rate": np.mean(all_metrics["converged"]),
            "avg_energy_change": np.mean([f - i for f, i in zip(all_metrics["final_energy"], 
                                                                all_metrics["initial_energy"])]),
        }
        log_summary(summary)
    
    finish_wandb()
    print("\nDone!")

