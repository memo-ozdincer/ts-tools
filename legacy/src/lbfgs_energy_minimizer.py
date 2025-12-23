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

from .dependencies.common_utils import (
    setup_experiment, add_common_args, parse_starting_geometry
)
from .dependencies.differentiable_projection import (
    differentiable_massweigh_and_eckartprojection_torch as massweigh_and_eckartprojection_torch
)
from hip.ff_lmdb import Z_TO_ATOM_SYMBOL
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from .dependencies.experiment_logger import (
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
        max_step: float = 0.5,     # Å - maximum displacement per atom from start
        target_neg_eig_count: int = 1,  # Stop when neg_eig_count <= this
        eigenvalue_check_freq: int = 1,  # Check eigenvalues every N iterations
        verbose: bool = True,
    ):
        """
        Initialize the L-BFGS energy minimizer.
        
        Args:
            calculator: HIP EquiformerTorchCalculator instance
            atomic_nums: Atomic numbers as torch tensor
            device: Device string ('cuda' or 'cpu')
            max_iterations: Maximum L-BFGS iterations
            max_step: Maximum displacement per atom from starting position (Å)
            target_neg_eig_count: Stop when negative eigenvalue count <= this
            eigenvalue_check_freq: Check eigenvalues every N iterations (default: 1 = every iteration)
            verbose: Print progress information
            
        Note:
            The ONLY convergence criterion is negative eigenvalue count reaching the target.
            Force convergence is NOT used as a stopping criterion.
        """
        self.calculator = calculator
        self.atomic_nums = torch.as_tensor(atomic_nums, dtype=torch.int64)
        self.atomsymbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in self.atomic_nums]
        self.num_atoms = len(atomic_nums)
        self.device = device
        self.max_iterations = max_iterations
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
        # Note: gtol is set to effectively zero so scipy never stops due to gradient convergence.
        # The ONLY convergence criterion is negative eigenvalue count via our callback.
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
                    'gtol': 1e-30,  # Effectively disabled - we stop via eigenvalue count only
                    'disp': False,
                }
            )
            
            final_coords = torch.tensor(
                result.x.reshape(-1, 3),
                dtype=torch.float32,
                device=self.device,
            )
            
            # Determine stop reason - ONLY eigenvalue count is considered "converged"
            if self._should_stop:
                stop_reason = "target_neg_eig_reached"
                converged = True
            else:
                # Either max iterations reached or scipy stopped for other reasons
                # This is NOT convergence - we didn't reach target eigenvalue count
                stop_reason = f"max_iterations_reached ({result.message})"
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
# Steepest Descent Minimizer
# =============================================================================

class SteepestDescentMinimizer:
    """
    Steepest descent energy minimizer with adaptive timestep.

    Simple gradient descent: x += dt * forces
    More robust than L-BFGS for very noisy geometries with many negative eigenvalues.
    Uses line search or adaptive timestep to ensure energy decreases.
    """

    def __init__(
        self,
        calculator: EquiformerTorchCalculator,
        atomic_nums: torch.Tensor,
        device: str,
        max_iterations: int = 500,
        dt_init: float = 0.01,  # Initial timestep in Å
        dt_max: float = 0.1,
        dt_min: float = 1e-5,
        alpha: float = 0.9,  # Backtracking factor
        target_neg_eig_count: int = 1,
        eigenvalue_check_freq: int = 5,
        verbose: bool = True,
    ):
        """
        Args:
            calculator: HIP calculator
            atomic_nums: Atomic numbers
            device: Device string
            max_iterations: Maximum iterations
            dt_init: Initial timestep (Å)
            dt_max: Maximum allowed timestep
            dt_min: Minimum allowed timestep
            alpha: Backtracking factor for line search (0 < alpha < 1)
            target_neg_eig_count: Stop when neg_eig_count <= this
            eigenvalue_check_freq: Check eigenvalues every N iterations
            verbose: Print progress
        """
        self.calculator = calculator
        self.atomic_nums = torch.as_tensor(atomic_nums, dtype=torch.int64)
        self.atomsymbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in self.atomic_nums]
        self.device = device
        self.max_iterations = max_iterations
        self.dt = dt_init
        self.dt_max = dt_max
        self.dt_min = dt_min
        self.alpha = alpha
        self.target_neg_eig_count = target_neg_eig_count
        self.eigenvalue_check_freq = eigenvalue_check_freq
        self.verbose = verbose

        self.trajectory: Dict[str, List] = {}
        self._iteration_count = 0

    def minimize(self, initial_coords: torch.Tensor) -> Dict[str, Any]:
        """Run steepest descent minimization."""
        coords = initial_coords.clone().detach().reshape(-1, 3).to(self.device)
        self.trajectory = defaultdict(list)
        self._iteration_count = 0

        # Evaluate initial state
        initial_eval = evaluate_geometry(
            self.calculator, coords, self.atomic_nums,
            self.atomsymbols, self.device
        )
        initial_neg_count = initial_eval["neg_eig_count"]
        initial_energy = initial_eval["energy"]

        if self.verbose:
            print(f"  [SteepestDescent] Starting minimization:")
            print(f"    Initial energy: {initial_energy:.6f} eV")
            print(f"    Initial neg eigenvalues: {initial_neg_count}")
            print(f"    Target: <= {self.target_neg_eig_count} negative eigenvalues")

        # Check if already at target
        if initial_neg_count <= self.target_neg_eig_count:
            if self.verbose:
                print(f"  [SteepestDescent] Already at target. No minimization needed.")
            return {
                "final_coords": coords,
                "initial_coords": coords.clone(),
                "initial_neg_eig_count": initial_neg_count,
                "final_neg_eig_count": initial_neg_count,
                "initial_energy": initial_energy,
                "final_energy": initial_energy,
                "n_iterations": 0,
                "converged": True,
                "stop_reason": "already_at_target",
                "trajectory": dict(self.trajectory),
            }

        current_energy = initial_energy
        converged = False
        stop_reason = "max_iterations_reached"

        for iteration in range(self.max_iterations):
            # Compute forces
            with torch.no_grad():
                batch = coord_atoms_to_torch_geometric(coords, self.atomic_nums, self.device)
                results = self.calculator.predict(batch, do_hessian=True)

                energy = results["energy"].item()
                forces = results["forces"]  # (N, 3)
                hessian = results["hessian"].reshape(coords.numel(), coords.numel())

                # Check eigenvalues periodically
                if iteration % self.eigenvalue_check_freq == 0:
                    vib_eigvals, neg_count, _ = compute_vibrational_eigenvalues(
                        hessian, coords, self.atomsymbols
                    )
                    eig0 = vib_eigvals[0].item() if len(vib_eigvals) > 0 else None
                    eig1 = vib_eigvals[1].item() if len(vib_eigvals) > 1 else None

                    # Check convergence
                    if neg_count <= self.target_neg_eig_count:
                        converged = True
                        stop_reason = "target_neg_eig_reached"
                        if self.verbose:
                            print(f"    [SteepestDescent] Target reached: {neg_count} negative eigenvalues")
                else:
                    neg_count = None
                    eig0 = eig1 = None

                # Record trajectory
                force_rms = forces.norm(dim=1).mean().item()
                force_max = forces.norm(dim=1).max().item()
                self.trajectory["iteration"].append(iteration)
                self.trajectory["energy"].append(energy)
                self.trajectory["force_rms"].append(force_rms)
                self.trajectory["force_max"].append(force_max)
                self.trajectory["neg_eig_count"].append(neg_count)
                self.trajectory["eig0"].append(eig0)
                self.trajectory["eig1"].append(eig1)
                self.trajectory["dt"].append(self.dt)

                if self.verbose and iteration % 10 == 0:
                    neg_str = f"{neg_count}" if neg_count is not None else "?"
                    print(f"    [SteepestDescent] Iter {iteration:4d}: "
                          f"E={energy:12.6f} eV, |F|_rms={force_rms:.4f}, "
                          f"neg_eig={neg_str}, dt={self.dt:.5f}")

                if converged:
                    break

                # Steepest descent step with backtracking line search
                step_accepted = False
                trial_dt = self.dt

                for _ in range(10):  # Max 10 backtracking steps
                    trial_coords = coords + trial_dt * forces

                    # Evaluate trial step
                    batch_trial = coord_atoms_to_torch_geometric(
                        trial_coords, self.atomic_nums, self.device
                    )
                    results_trial = self.calculator.predict(batch_trial, do_hessian=False)
                    trial_energy = results_trial["energy"].item()

                    # Accept if energy decreased
                    if trial_energy < energy:
                        coords = trial_coords
                        current_energy = trial_energy
                        step_accepted = True
                        # Increase timestep for next iteration (bold driver)
                        self.dt = min(trial_dt * 1.2, self.dt_max)
                        break
                    else:
                        # Backtrack
                        trial_dt *= self.alpha
                        if trial_dt < self.dt_min:
                            break

                if not step_accepted:
                    # Force a tiny step if backtracking failed
                    coords = coords + self.dt_min * forces
                    self.dt = self.dt_min
                    if self.verbose:
                        print(f"    [SteepestDescent] Warning: Backtracking failed at iter {iteration}")

                self._iteration_count = iteration + 1

        # Final evaluation
        final_eval = evaluate_geometry(
            self.calculator, coords, self.atomic_nums,
            self.atomsymbols, self.device
        )
        final_neg_count = final_eval["neg_eig_count"]
        final_energy = final_eval["energy"]

        if self.verbose:
            print(f"  [SteepestDescent] Finished after {self._iteration_count} iterations")
            print(f"    Final energy: {final_energy:.6f} eV (Δ = {final_energy - initial_energy:+.6f})")
            print(f"    Final neg eigenvalues: {final_neg_count}")
            print(f"    Stop reason: {stop_reason}")

        return {
            "final_coords": coords,
            "initial_coords": initial_coords.reshape(-1, 3),
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
# Mode-Selective Descent Minimizer
# =============================================================================

class ModeSelectiveDescentMinimizer:
    """
    Mode-selective descent: only move along positive-eigenvalue directions.

    This is the most theoretically direct approach for reducing saddle order.
    At each step:
    1. Compute Hessian and get eigendecomposition
    2. Project forces onto positive eigenvalue subspace
    3. Take step only in "stable" directions

    Guarantees we only move in directions that reduce saddle order.
    """

    def __init__(
        self,
        calculator: EquiformerTorchCalculator,
        atomic_nums: torch.Tensor,
        device: str,
        max_iterations: int = 500,
        dt_init: float = 0.01,
        dt_max: float = 0.1,
        dt_min: float = 1e-5,
        alpha: float = 0.9,
        target_neg_eig_count: int = 1,
        eigenvalue_check_freq: int = 1,  # Must check every iteration for projection
        verbose: bool = True,
    ):
        """
        Args:
            calculator: HIP calculator
            atomic_nums: Atomic numbers
            device: Device string
            max_iterations: Maximum iterations
            dt_init: Initial timestep
            dt_max: Maximum timestep
            dt_min: Minimum timestep
            alpha: Backtracking factor
            target_neg_eig_count: Stop when neg_eig_count <= this
            eigenvalue_check_freq: Should be 1 for mode-selective (need Hessian every step)
            verbose: Print progress
        """
        self.calculator = calculator
        self.atomic_nums = torch.as_tensor(atomic_nums, dtype=torch.int64)
        self.atomsymbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in self.atomic_nums]
        self.device = device
        self.max_iterations = max_iterations
        self.dt = dt_init
        self.dt_max = dt_max
        self.dt_min = dt_min
        self.alpha = alpha
        self.target_neg_eig_count = target_neg_eig_count
        self.eigenvalue_check_freq = max(1, eigenvalue_check_freq)  # Force minimum of 1
        self.verbose = verbose

        self.trajectory: Dict[str, List] = {}
        self._iteration_count = 0

    def minimize(self, initial_coords: torch.Tensor) -> Dict[str, Any]:
        """Run mode-selective descent minimization."""
        coords = initial_coords.clone().detach().reshape(-1, 3).to(self.device)
        self.trajectory = defaultdict(list)
        self._iteration_count = 0

        # Evaluate initial state
        initial_eval = evaluate_geometry(
            self.calculator, coords, self.atomic_nums,
            self.atomsymbols, self.device
        )
        initial_neg_count = initial_eval["neg_eig_count"]
        initial_energy = initial_eval["energy"]

        if self.verbose:
            print(f"  [ModeSelective] Starting minimization:")
            print(f"    Initial energy: {initial_energy:.6f} eV")
            print(f"    Initial neg eigenvalues: {initial_neg_count}")
            print(f"    Target: <= {self.target_neg_eig_count} negative eigenvalues")

        # Check if already at target
        if initial_neg_count <= self.target_neg_eig_count:
            if self.verbose:
                print(f"  [ModeSelective] Already at target. No minimization needed.")
            return {
                "final_coords": coords,
                "initial_coords": coords.clone(),
                "initial_neg_eig_count": initial_neg_count,
                "final_neg_eig_count": initial_neg_count,
                "initial_energy": initial_energy,
                "final_energy": initial_energy,
                "n_iterations": 0,
                "converged": True,
                "stop_reason": "already_at_target",
                "trajectory": dict(self.trajectory),
            }

        converged = False
        stop_reason = "max_iterations_reached"

        for iteration in range(self.max_iterations):
            with torch.no_grad():
                batch = coord_atoms_to_torch_geometric(coords, self.atomic_nums, self.device)
                results = self.calculator.predict(batch, do_hessian=True)

                energy = results["energy"].item()
                forces = results["forces"]  # (N, 3)
                hessian = results["hessian"].reshape(coords.numel(), coords.numel())

                # Get vibrational eigenvalues and eigenvectors
                # We need to mass-weight and Eckart-project the Hessian
                coords_3d = coords.reshape(-1, 3)
                hess_proj = massweigh_and_eckartprojection_torch(
                    hessian, coords_3d, self.atomsymbols
                )

                # Compute eigendecomposition
                eigvals, eigvecs = torch.linalg.eigh(hess_proj)

                # Determine number of rigid modes
                coords_cent = coords_3d.detach().to(torch.float64)
                coords_cent = coords_cent - coords_cent.mean(dim=0, keepdim=True)
                geom_rank = torch.linalg.matrix_rank(coords_cent.cpu(), tol=1e-8).item()
                expected_rigid = 5 if geom_rank <= 2 else 6
                total_modes = eigvals.shape[0]
                n_rigid = min(expected_rigid, max(0, total_modes - 2))

                # Remove rigid modes
                abs_sorted_idx = torch.argsort(torch.abs(eigvals))
                keep_idx = abs_sorted_idx[n_rigid:]
                vibrational_eigvals = eigvals[keep_idx]
                vibrational_eigvecs = eigvecs[:, keep_idx]

                # Count negative eigenvalues
                neg_count = (vibrational_eigvals < 0).sum().item()
                eig0 = vibrational_eigvals[0].item() if len(vibrational_eigvals) > 0 else None
                eig1 = vibrational_eigvals[1].item() if len(vibrational_eigvals) > 1 else None

                # Project forces onto POSITIVE eigenvalue subspace
                # Flatten forces to 1D for projection
                forces_flat = forces.flatten()  # (3N,)

                # Identify positive eigenvalue modes
                positive_mask = vibrational_eigvals > 0
                if positive_mask.sum() == 0:
                    # No positive modes - can't move! Just use forces
                    projected_forces = forces_flat
                    if self.verbose and iteration % 10 == 0:
                        print(f"    [ModeSelective] Warning: No positive eigenvalues at iter {iteration}")
                else:
                    # Project onto positive eigenvalue subspace
                    positive_eigvecs = vibrational_eigvecs[:, positive_mask]  # (3N, n_positive)
                    projection = positive_eigvecs @ (positive_eigvecs.T @ forces_flat)
                    projected_forces = projection

                projected_forces_3d = projected_forces.reshape(-1, 3)

                # Record trajectory
                force_rms = forces.norm(dim=1).mean().item()
                force_max = forces.norm(dim=1).max().item()
                proj_force_rms = projected_forces_3d.norm(dim=1).mean().item()

                self.trajectory["iteration"].append(iteration)
                self.trajectory["energy"].append(energy)
                self.trajectory["force_rms"].append(force_rms)
                self.trajectory["force_max"].append(force_max)
                self.trajectory["projected_force_rms"].append(proj_force_rms)
                self.trajectory["neg_eig_count"].append(neg_count)
                self.trajectory["eig0"].append(eig0)
                self.trajectory["eig1"].append(eig1)
                self.trajectory["dt"].append(self.dt)

                if self.verbose and iteration % 10 == 0:
                    print(f"    [ModeSelective] Iter {iteration:4d}: "
                          f"E={energy:12.6f} eV, |F|_rms={force_rms:.4f}, "
                          f"|F_proj|_rms={proj_force_rms:.4f}, "
                          f"neg_eig={neg_count}, dt={self.dt:.5f}")

                # Check convergence
                if neg_count <= self.target_neg_eig_count:
                    converged = True
                    stop_reason = "target_neg_eig_reached"
                    if self.verbose:
                        print(f"    [ModeSelective] Target reached: {neg_count} negative eigenvalues")
                    break

                # Take step with projected forces (backtracking line search)
                step_accepted = False
                trial_dt = self.dt

                for _ in range(10):
                    trial_coords = coords + trial_dt * projected_forces_3d

                    batch_trial = coord_atoms_to_torch_geometric(
                        trial_coords, self.atomic_nums, self.device
                    )
                    results_trial = self.calculator.predict(batch_trial, do_hessian=False)
                    trial_energy = results_trial["energy"].item()

                    if trial_energy < energy:
                        coords = trial_coords
                        step_accepted = True
                        self.dt = min(trial_dt * 1.2, self.dt_max)
                        break
                    else:
                        trial_dt *= self.alpha
                        if trial_dt < self.dt_min:
                            break

                if not step_accepted:
                    coords = coords + self.dt_min * projected_forces_3d
                    self.dt = self.dt_min
                    if self.verbose and iteration % 50 == 0:
                        print(f"    [ModeSelective] Warning: Backtracking failed at iter {iteration}")

                self._iteration_count = iteration + 1

        # Final evaluation
        final_eval = evaluate_geometry(
            self.calculator, coords, self.atomic_nums,
            self.atomsymbols, self.device
        )
        final_neg_count = final_eval["neg_eig_count"]
        final_energy = final_eval["energy"]

        if self.verbose:
            print(f"  [ModeSelective] Finished after {self._iteration_count} iterations")
            print(f"    Final energy: {final_energy:.6f} eV (Δ = {final_energy - initial_energy:+.6f})")
            print(f"    Final neg eigenvalues: {final_neg_count}")
            print(f"    Stop reason: {stop_reason}")

        return {
            "final_coords": coords,
            "initial_coords": initial_coords.reshape(-1, 3),
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
# FIRE Minimizer
# =============================================================================

class FIREMinimizer:
    """
    FIRE (Fast Inertial Relaxation Engine) minimizer.

    Very popular in atomistic simulations. Combines velocity-like dynamics
    with adaptive timestep. Often more robust than L-BFGS for rugged landscapes.

    Reference: Bitzek et al., PRL 97, 170201 (2006)
    """

    def __init__(
        self,
        calculator: EquiformerTorchCalculator,
        atomic_nums: torch.Tensor,
        device: str,
        max_iterations: int = 500,
        dt_init: float = 0.01,
        dt_max: float = 0.1,
        N_min: int = 5,  # Min steps before increasing dt
        f_inc: float = 1.1,  # Timestep increase factor
        f_dec: float = 0.5,  # Timestep decrease factor
        f_alpha: float = 0.99,  # Alpha decrease factor
        alpha_start: float = 0.1,  # Initial velocity mixing parameter
        target_neg_eig_count: int = 1,
        eigenvalue_check_freq: int = 5,
        verbose: bool = True,
    ):
        """
        Args:
            calculator: HIP calculator
            atomic_nums: Atomic numbers
            device: Device string
            max_iterations: Maximum iterations
            dt_init: Initial timestep
            dt_max: Maximum timestep
            N_min: Minimum steps before increasing timestep
            f_inc: Timestep increase factor
            f_dec: Timestep decrease factor
            f_alpha: Alpha decrease factor
            alpha_start: Initial mixing parameter
            target_neg_eig_count: Stop when neg_eig_count <= this
            eigenvalue_check_freq: Check eigenvalues every N iterations
            verbose: Print progress
        """
        self.calculator = calculator
        self.atomic_nums = torch.as_tensor(atomic_nums, dtype=torch.int64)
        self.atomsymbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in self.atomic_nums]
        self.device = device
        self.max_iterations = max_iterations
        self.dt = dt_init
        self.dt_max = dt_max
        self.N_min = N_min
        self.f_inc = f_inc
        self.f_dec = f_dec
        self.f_alpha = f_alpha
        self.alpha_start = alpha_start
        self.target_neg_eig_count = target_neg_eig_count
        self.eigenvalue_check_freq = eigenvalue_check_freq
        self.verbose = verbose

        self.trajectory: Dict[str, List] = {}
        self._iteration_count = 0

    def minimize(self, initial_coords: torch.Tensor) -> Dict[str, Any]:
        """Run FIRE minimization."""
        coords = initial_coords.clone().detach().reshape(-1, 3).to(self.device)
        self.trajectory = defaultdict(list)
        self._iteration_count = 0

        # Evaluate initial state
        initial_eval = evaluate_geometry(
            self.calculator, coords, self.atomic_nums,
            self.atomsymbols, self.device
        )
        initial_neg_count = initial_eval["neg_eig_count"]
        initial_energy = initial_eval["energy"]

        if self.verbose:
            print(f"  [FIRE] Starting minimization:")
            print(f"    Initial energy: {initial_energy:.6f} eV")
            print(f"    Initial neg eigenvalues: {initial_neg_count}")
            print(f"    Target: <= {self.target_neg_eig_count} negative eigenvalues")

        # Check if already at target
        if initial_neg_count <= self.target_neg_eig_count:
            if self.verbose:
                print(f"  [FIRE] Already at target. No minimization needed.")
            return {
                "final_coords": coords,
                "initial_coords": coords.clone(),
                "initial_neg_eig_count": initial_neg_count,
                "final_neg_eig_count": initial_neg_count,
                "initial_energy": initial_energy,
                "final_energy": initial_energy,
                "n_iterations": 0,
                "converged": True,
                "stop_reason": "already_at_target",
                "trajectory": dict(self.trajectory),
            }

        # FIRE state variables
        velocity = torch.zeros_like(coords)
        alpha = self.alpha_start
        N_pos = 0  # Number of steps with P > 0
        converged = False
        stop_reason = "max_iterations_reached"

        for iteration in range(self.max_iterations):
            with torch.no_grad():
                batch = coord_atoms_to_torch_geometric(coords, self.atomic_nums, self.device)
                results = self.calculator.predict(batch, do_hessian=True)

                energy = results["energy"].item()
                forces = results["forces"]  # (N, 3)
                hessian = results["hessian"].reshape(coords.numel(), coords.numel())

                # Check eigenvalues periodically
                if iteration % self.eigenvalue_check_freq == 0:
                    vib_eigvals, neg_count, _ = compute_vibrational_eigenvalues(
                        hessian, coords, self.atomsymbols
                    )
                    eig0 = vib_eigvals[0].item() if len(vib_eigvals) > 0 else None
                    eig1 = vib_eigvals[1].item() if len(vib_eigvals) > 1 else None

                    if neg_count <= self.target_neg_eig_count:
                        converged = True
                        stop_reason = "target_neg_eig_reached"
                        if self.verbose:
                            print(f"    [FIRE] Target reached: {neg_count} negative eigenvalues")
                else:
                    neg_count = None
                    eig0 = eig1 = None

                # Compute power P = F · v
                P = (forces * velocity).sum().item()

                # Mix velocity with force direction
                force_norm = forces.norm()
                velocity_norm = velocity.norm()

                if force_norm > 1e-10 and velocity_norm > 1e-10:
                    velocity = (1 - alpha) * velocity + alpha * (velocity_norm / force_norm) * forces
                else:
                    velocity = (1 - alpha) * velocity + alpha * forces

                # FIRE timestep adaptation
                if P > 0:
                    N_pos += 1
                    if N_pos > self.N_min:
                        self.dt = min(self.dt * self.f_inc, self.dt_max)
                        alpha = alpha * self.f_alpha
                else:
                    N_pos = 0
                    self.dt = self.dt * self.f_dec
                    velocity = torch.zeros_like(velocity)
                    alpha = self.alpha_start

                # Record trajectory
                force_rms = forces.norm(dim=1).mean().item()
                force_max = forces.norm(dim=1).max().item()
                velocity_rms = velocity.norm(dim=1).mean().item()

                self.trajectory["iteration"].append(iteration)
                self.trajectory["energy"].append(energy)
                self.trajectory["force_rms"].append(force_rms)
                self.trajectory["force_max"].append(force_max)
                self.trajectory["velocity_rms"].append(velocity_rms)
                self.trajectory["power_P"].append(P)
                self.trajectory["neg_eig_count"].append(neg_count)
                self.trajectory["eig0"].append(eig0)
                self.trajectory["eig1"].append(eig1)
                self.trajectory["dt"].append(self.dt)
                self.trajectory["alpha"].append(alpha)

                if self.verbose and iteration % 10 == 0:
                    neg_str = f"{neg_count}" if neg_count is not None else "?"
                    print(f"    [FIRE] Iter {iteration:4d}: "
                          f"E={energy:12.6f} eV, |F|_rms={force_rms:.4f}, "
                          f"P={P:+.3e}, neg_eig={neg_str}, dt={self.dt:.5f}")

                if converged:
                    break

                # Velocity Verlet integration
                # v(t+dt/2) = v(t) + (dt/2) * a(t)  [already done above via mixing]
                # x(t+dt) = x(t) + dt * v(t+dt/2)
                velocity = velocity + self.dt * forces
                coords = coords + self.dt * velocity

                self._iteration_count = iteration + 1

        # Final evaluation
        final_eval = evaluate_geometry(
            self.calculator, coords, self.atomic_nums,
            self.atomsymbols, self.device
        )
        final_neg_count = final_eval["neg_eig_count"]
        final_energy = final_eval["energy"]

        if self.verbose:
            print(f"  [FIRE] Finished after {self._iteration_count} iterations")
            print(f"    Final energy: {final_energy:.6f} eV (Δ = {final_energy - initial_energy:+.6f})")
            print(f"    Final neg eigenvalues: {final_neg_count}")
            print(f"    Stop reason: {stop_reason}")

        return {
            "final_coords": coords,
            "initial_coords": initial_coords.reshape(-1, 3),
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
# Eigenvector-Following Descent Minimizer
# =============================================================================

class EigenvectorFollowingMinimizer:
    """
    Eigenvector-following descent: systematically flip negative modes.

    At each step:
    1. Identify the most negative eigenvalue
    2. Move along its eigenvector to flip the mode
    3. Repeat until saddle order is reduced

    More targeted than blind minimization - directly addresses each
    negative eigenvalue in sequence.
    """

    def __init__(
        self,
        calculator: EquiformerTorchCalculator,
        atomic_nums: torch.Tensor,
        device: str,
        max_iterations: int = 500,
        step_size: float = 0.05,  # Step size along eigenvector
        target_neg_eig_count: int = 1,
        verbose: bool = True,
    ):
        """
        Args:
            calculator: HIP calculator
            atomic_nums: Atomic numbers
            device: Device string
            max_iterations: Maximum iterations
            step_size: Step size along eigenvector (Å)
            target_neg_eig_count: Stop when neg_eig_count <= this
            verbose: Print progress
        """
        self.calculator = calculator
        self.atomic_nums = torch.as_tensor(atomic_nums, dtype=torch.int64)
        self.atomsymbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in self.atomic_nums]
        self.device = device
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.target_neg_eig_count = target_neg_eig_count
        self.verbose = verbose

        self.trajectory: Dict[str, List] = {}
        self._iteration_count = 0

    def minimize(self, initial_coords: torch.Tensor) -> Dict[str, Any]:
        """Run eigenvector-following minimization."""
        coords = initial_coords.clone().detach().reshape(-1, 3).to(self.device)
        self.trajectory = defaultdict(list)
        self._iteration_count = 0

        # Evaluate initial state
        initial_eval = evaluate_geometry(
            self.calculator, coords, self.atomic_nums,
            self.atomsymbols, self.device
        )
        initial_neg_count = initial_eval["neg_eig_count"]
        initial_energy = initial_eval["energy"]

        if self.verbose:
            print(f"  [EigenvectorFollowing] Starting minimization:")
            print(f"    Initial energy: {initial_energy:.6f} eV")
            print(f"    Initial neg eigenvalues: {initial_neg_count}")
            print(f"    Target: <= {self.target_neg_eig_count} negative eigenvalues")

        # Check if already at target
        if initial_neg_count <= self.target_neg_eig_count:
            if self.verbose:
                print(f"  [EigenvectorFollowing] Already at target. No minimization needed.")
            return {
                "final_coords": coords,
                "initial_coords": coords.clone(),
                "initial_neg_eig_count": initial_neg_count,
                "final_neg_eig_count": initial_neg_count,
                "initial_energy": initial_energy,
                "final_energy": initial_energy,
                "n_iterations": 0,
                "converged": True,
                "stop_reason": "already_at_target",
                "trajectory": dict(self.trajectory),
            }

        converged = False
        stop_reason = "max_iterations_reached"

        for iteration in range(self.max_iterations):
            with torch.no_grad():
                batch = coord_atoms_to_torch_geometric(coords, self.atomic_nums, self.device)
                results = self.calculator.predict(batch, do_hessian=True)

                energy = results["energy"].item()
                forces = results["forces"]  # (N, 3)
                hessian = results["hessian"].reshape(coords.numel(), coords.numel())

                # Get vibrational eigenvalues and eigenvectors
                coords_3d = coords.reshape(-1, 3)
                hess_proj = massweigh_and_eckartprojection_torch(
                    hessian, coords_3d, self.atomsymbols
                )

                # Compute eigendecomposition
                eigvals, eigvecs = torch.linalg.eigh(hess_proj)

                # Determine number of rigid modes
                coords_cent = coords_3d.detach().to(torch.float64)
                coords_cent = coords_cent - coords_cent.mean(dim=0, keepdim=True)
                geom_rank = torch.linalg.matrix_rank(coords_cent.cpu(), tol=1e-8).item()
                expected_rigid = 5 if geom_rank <= 2 else 6
                total_modes = eigvals.shape[0]
                n_rigid = min(expected_rigid, max(0, total_modes - 2))

                # Remove rigid modes
                abs_sorted_idx = torch.argsort(torch.abs(eigvals))
                keep_idx = abs_sorted_idx[n_rigid:]
                vibrational_eigvals = eigvals[keep_idx]
                vibrational_eigvecs = eigvecs[:, keep_idx]

                # Count negative eigenvalues
                neg_count = (vibrational_eigvals < 0).sum().item()
                eig0 = vibrational_eigvals[0].item() if len(vibrational_eigvals) > 0 else None
                eig1 = vibrational_eigvals[1].item() if len(vibrational_eigvals) > 1 else None

                # Record trajectory
                force_rms = forces.norm(dim=1).mean().item()
                force_max = forces.norm(dim=1).max().item()

                self.trajectory["iteration"].append(iteration)
                self.trajectory["energy"].append(energy)
                self.trajectory["force_rms"].append(force_rms)
                self.trajectory["force_max"].append(force_max)
                self.trajectory["neg_eig_count"].append(neg_count)
                self.trajectory["eig0"].append(eig0)
                self.trajectory["eig1"].append(eig1)

                if self.verbose and iteration % 10 == 0:
                    print(f"    [EigenvectorFollowing] Iter {iteration:4d}: "
                          f"E={energy:12.6f} eV, neg_eig={neg_count}, "
                          f"eig0={eig0:.4e if eig0 else 'N/A'}")

                # Check convergence
                if neg_count <= self.target_neg_eig_count:
                    converged = True
                    stop_reason = "target_neg_eig_reached"
                    if self.verbose:
                        print(f"    [EigenvectorFollowing] Target reached: {neg_count} negative eigenvalues")
                    break

                # Follow the most negative eigenvector
                if neg_count > 0:
                    # Most negative eigenvalue is at index 0 (sorted ascending)
                    most_neg_eigvec = vibrational_eigvecs[:, 0]  # (3N,)
                    most_neg_eigval = vibrational_eigvals[0].item()

                    # Move along this eigenvector (in the positive direction to reduce the negative curvature)
                    step = self.step_size * most_neg_eigvec.reshape(-1, 3)
                    coords = coords + step

                    if self.verbose and iteration % 10 == 0:
                        print(f"      Following eigenvector with λ={most_neg_eigval:.4e}")
                else:
                    # No negative eigenvalues - shouldn't happen, but just in case
                    if self.verbose:
                        print(f"    [EigenvectorFollowing] Warning: No negative eigenvalues at iter {iteration}")
                    break

                self._iteration_count = iteration + 1

        # Final evaluation
        final_eval = evaluate_geometry(
            self.calculator, coords, self.atomic_nums,
            self.atomsymbols, self.device
        )
        final_neg_count = final_eval["neg_eig_count"]
        final_energy = final_eval["energy"]

        if self.verbose:
            print(f"  [EigenvectorFollowing] Finished after {self._iteration_count} iterations")
            print(f"    Final energy: {final_energy:.6f} eV (Δ = {final_energy - initial_energy:+.6f})")
            print(f"    Final neg eigenvalues: {final_neg_count}")
            print(f"    Stop reason: {stop_reason}")

        return {
            "final_coords": coords,
            "initial_coords": initial_coords.reshape(-1, 3),
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

def create_minimizer(
    method: str,
    calculator: EquiformerTorchCalculator,
    atomic_nums: torch.Tensor,
    device: str,
    max_iterations: int = 500,
    target_neg_eig_count: int = 1,
    eigenvalue_check_freq: int = 1,
    verbose: bool = True,
    **kwargs,
):
    """
    Factory function to create the appropriate minimizer based on method name.

    Args:
        method: Minimization method - one of:
            - 'lbfgs': L-BFGS-B (scipy)
            - 'steepest_descent': Simple steepest descent with line search
            - 'mode_selective': Mode-selective descent (only positive eigenvalue directions)
            - 'fire': FIRE (Fast Inertial Relaxation Engine)
            - 'eigenvector_following': Follow most negative eigenvector
        calculator: HIP calculator instance
        atomic_nums: Atomic numbers
        device: Device string
        max_iterations: Maximum iterations
        target_neg_eig_count: Stop when neg_eig_count <= this
        eigenvalue_check_freq: Check eigenvalues every N iterations
        verbose: Print progress
        **kwargs: Additional method-specific parameters

    Returns:
        Minimizer instance with .minimize(coords) method
    """
    method = method.lower()

    if method == 'lbfgs':
        return LBFGSEnergyMinimizer(
            calculator=calculator,
            atomic_nums=atomic_nums,
            device=device,
            max_iterations=max_iterations,
            max_step=kwargs.get('max_step', 0.5),
            target_neg_eig_count=target_neg_eig_count,
            eigenvalue_check_freq=eigenvalue_check_freq,
            verbose=verbose,
        )

    elif method == 'steepest_descent':
        return SteepestDescentMinimizer(
            calculator=calculator,
            atomic_nums=atomic_nums,
            device=device,
            max_iterations=max_iterations,
            dt_init=kwargs.get('dt_init', 0.01),
            dt_max=kwargs.get('dt_max', 0.1),
            dt_min=kwargs.get('dt_min', 1e-5),
            alpha=kwargs.get('alpha', 0.9),
            target_neg_eig_count=target_neg_eig_count,
            eigenvalue_check_freq=eigenvalue_check_freq,
            verbose=verbose,
        )

    elif method == 'mode_selective':
        return ModeSelectiveDescentMinimizer(
            calculator=calculator,
            atomic_nums=atomic_nums,
            device=device,
            max_iterations=max_iterations,
            dt_init=kwargs.get('dt_init', 0.01),
            dt_max=kwargs.get('dt_max', 0.1),
            dt_min=kwargs.get('dt_min', 1e-5),
            alpha=kwargs.get('alpha', 0.9),
            target_neg_eig_count=target_neg_eig_count,
            eigenvalue_check_freq=max(1, eigenvalue_check_freq),  # Must be 1 for mode-selective
            verbose=verbose,
        )

    elif method == 'fire':
        return FIREMinimizer(
            calculator=calculator,
            atomic_nums=atomic_nums,
            device=device,
            max_iterations=max_iterations,
            dt_init=kwargs.get('dt_init', 0.01),
            dt_max=kwargs.get('dt_max', 0.1),
            N_min=kwargs.get('N_min', 5),
            f_inc=kwargs.get('f_inc', 1.1),
            f_dec=kwargs.get('f_dec', 0.5),
            f_alpha=kwargs.get('f_alpha', 0.99),
            alpha_start=kwargs.get('alpha_start', 0.1),
            target_neg_eig_count=target_neg_eig_count,
            eigenvalue_check_freq=eigenvalue_check_freq,
            verbose=verbose,
        )

    elif method == 'eigenvector_following':
        return EigenvectorFollowingMinimizer(
            calculator=calculator,
            atomic_nums=atomic_nums,
            device=device,
            max_iterations=max_iterations,
            step_size=kwargs.get('step_size', 0.05),
            target_neg_eig_count=target_neg_eig_count,
            verbose=verbose,
        )

    else:
        raise ValueError(
            f"Unknown minimization method: '{method}'. "
            f"Choose from: 'lbfgs', 'steepest_descent', 'mode_selective', 'fire', 'eigenvector_following'"
        )


def lbfgs_energy_minimize(
    calculator: EquiformerTorchCalculator,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    device: str,
    max_iterations: int = 200,
    max_step: float = 0.5,
    target_neg_eig_count: int = 1,
    eigenvalue_check_freq: int = 1,
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
        max_step: Maximum displacement per atom (Å)
        target_neg_eig_count: Stop when neg_eig_count <= this (default: 1)
        eigenvalue_check_freq: Check eigenvalues every N iterations
        verbose: Print progress

    Returns:
        Result dictionary (see LBFGSEnergyMinimizer.minimize)

    Note:
        The ONLY convergence criterion is negative eigenvalue count.
    """
    minimizer = LBFGSEnergyMinimizer(
        calculator=calculator,
        atomic_nums=atomic_nums,
        device=device,
        max_iterations=max_iterations,
        max_step=max_step,
        target_neg_eig_count=target_neg_eig_count,
        eigenvalue_check_freq=eigenvalue_check_freq,
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
    axes[1].set_title("Force Magnitude (log scale) - for reference only")
    
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
        description="Energy Minimizer for pre-conditioning noisy geometries with multiple methods."
    )
    parser = add_common_args(parser)

    # Minimization method selection
    parser.add_argument("--method", type=str, default="lbfgs",
                        choices=['lbfgs', 'steepest_descent', 'mode_selective', 'fire', 'eigenvector_following'],
                        help="Minimization method (default: lbfgs). Options:\n"
                             "  lbfgs: L-BFGS-B (scipy) - good general purpose\n"
                             "  steepest_descent: Simple gradient descent - robust baseline\n"
                             "  mode_selective: Only move in positive eigenvalue directions - most direct\n"
                             "  fire: FIRE algorithm - popular in atomistic simulations\n"
                             "  eigenvector_following: Follow most negative eigenvector - targeted approach")

    # Common minimization arguments
    parser.add_argument("--start-from", type=str, default="reactant_noise2A",
                        help="Starting geometry: 'reactant', 'ts', 'midpoint_rt', "
                             "or with noise: 'reactant_noise0.5A', 'reactant_noise2A', etc.")
    parser.add_argument("--max-iterations", type=int, default=500,
                        help="Maximum minimization iterations (default: 500)")
    parser.add_argument("--target-neg-eig", type=int, default=1,
                        help="Target negative eigenvalue count to stop (default: 1, i.e., stop at TS or minimum). "
                             "This is the ONLY convergence criterion.")
    parser.add_argument("--eigenvalue-check-freq", type=int, default=1,
                        help="Check eigenvalues every N iterations (default: 1 = every iteration)")
    parser.add_argument("--noise-seed", type=int, default=42,
                        help="Random seed for noise generation (default: 42)")

    # L-BFGS specific arguments
    parser.add_argument("--max-step", type=float, default=0.5,
                        help="[L-BFGS only] Maximum displacement per atom in Å (default: 0.5)")

    # Gradient descent / FIRE / mode-selective specific arguments
    parser.add_argument("--dt-init", type=float, default=0.01,
                        help="[steepest_descent, mode_selective, fire] Initial timestep in Å (default: 0.01)")
    parser.add_argument("--dt-max", type=float, default=0.1,
                        help="[steepest_descent, mode_selective, fire] Maximum timestep in Å (default: 0.1)")

    # Eigenvector-following specific arguments
    parser.add_argument("--step-size", type=float, default=0.05,
                        help="[eigenvector_following] Step size along eigenvector in Å (default: 0.05)")
    
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
    loss_type_flags = f"{args.method}-maxiter{args.max_iterations}-target{args.target_neg_eig}neg-from-{args.start_from}"

    # Set up experiment logger
    logger = ExperimentLogger(
        base_dir=out_dir,
        script_name=f"{args.method}-minimize",
        loss_type_flags=loss_type_flags,
        max_graphs_per_transition=10,
        random_seed=42,
    )
    
    # Initialize W&B if requested
    if args.wandb:
        wandb_config = {
            "script": "energy_minimizer",
            "method": args.method,
            "start_from": args.start_from,
            "max_iterations": args.max_iterations,
            "max_step": args.max_step,
            "dt_init": args.dt_init,
            "dt_max": args.dt_max,
            "step_size": args.step_size,
            "target_neg_eig": args.target_neg_eig,
            "eigenvalue_check_freq": args.eigenvalue_check_freq,
            "max_samples": args.max_samples,
            "noise_seed": args.noise_seed,
        }
        init_wandb_run(
            project=args.wandb_project,
            name=f"{args.method}_{loss_type_flags}",
            config=wandb_config,
            entity=args.wandb_entity,
            tags=[args.start_from, args.method, "precondition"],
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
    
    print(f"Running Energy Minimization with {args.method.upper()}")
    print(f"  Method: {args.method}")
    print(f"  Start from: {args.start_from}")
    print(f"  Target: <= {args.target_neg_eig} negative eigenvalues (ONLY convergence criterion)")
    print(f"  Max iterations: {args.max_iterations}")
    if args.method == 'lbfgs':
        print(f"  Max step: {args.max_step} Å")
    elif args.method in ['steepest_descent', 'mode_selective', 'fire']:
        print(f"  Initial timestep: {args.dt_init} Å, Max timestep: {args.dt_max} Å")
    elif args.method == 'eigenvector_following':
        print(f"  Step size: {args.step_size} Å")
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
            
            # Create minimizer using factory function
            minimizer = create_minimizer(
                method=args.method,
                calculator=calculator,
                atomic_nums=batch.z,
                device=device,
                max_iterations=args.max_iterations,
                target_neg_eig_count=args.target_neg_eig,
                eigenvalue_check_freq=args.eigenvalue_check_freq,
                verbose=True,
                # Method-specific kwargs
                max_step=args.max_step,
                dt_init=args.dt_init,
                dt_max=args.dt_max,
                step_size=args.step_size,
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

