"""Core Sella TS refinement runner.

This module implements the main Sella transition state search workflow,
interfacing with your existing calculators and validation infrastructure.

Sella uses RS-P-RFO (Restricted-Step Partitioned Rational Function Optimization)
with internal coordinates for robust saddle point optimization.
"""
from __future__ import annotations

import os
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read

# On headless HPC nodes, ASE's visualization helper can try to spawn `ase gui`
# (which requires tkinter). Some versions of Sella import this helper for
# debugging. Make it a no-op unless explicitly enabled.
try:
    import ase.visualize as _ase_visualize

    if os.environ.get("SELLA_ENABLE_ASE_VIEW", "0") != "1" and not os.environ.get("DISPLAY"):
        _ase_visualize.view = lambda *args, **kwargs: None  # type: ignore[assignment]
except Exception:
    pass

# Sella import (will be pip installed on cluster)
try:
    from sella import Sella
    SELLA_AVAILABLE = True
except ImportError:
    SELLA_AVAILABLE = False
    Sella = None

from .ase_calculators import create_ase_calculator, create_hessian_function


def coords_to_ase_atoms(
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
) -> Atoms:
    """Convert coordinates and atomic numbers to ASE Atoms object.

    Args:
        coords: (N, 3) coordinates in Angstrom
        atomic_nums: (N,) atomic numbers

    Returns:
        ASE Atoms object
    """
    positions = coords.detach().cpu().numpy().reshape(-1, 3)
    numbers = atomic_nums.detach().cpu().numpy().flatten().astype(int)
    return Atoms(numbers=numbers, positions=positions)


def ase_atoms_to_coords(atoms: Atoms) -> torch.Tensor:
    """Convert ASE Atoms positions to torch tensor.

    Args:
        atoms: ASE Atoms object

    Returns:
        (N, 3) coordinates as torch tensor
    """
    return torch.tensor(atoms.get_positions(), dtype=torch.float32)


def parse_sella_trajectory(
    traj_path: str,
) -> Dict[str, List[Any]]:
    """Parse a Sella trajectory file and extract per-step metrics.

    Args:
        traj_path: Path to ASE trajectory file (.traj)

    Returns:
        Dictionary with lists of per-step metrics:
        - energy: Energy at each step
        - force_max: Maximum force at each step
        - force_mean: Mean force at each step
        - positions: Positions at each step (N, 3) arrays
        - disp_from_last: Mean atom displacement from previous step
        - disp_from_start: Mean atom displacement from starting geometry
    """
    trajectory: Dict[str, List[Any]] = {
        "energy": [],
        "force_max": [],
        "force_mean": [],
        "positions": [],
        "disp_from_last": [],
        "disp_from_start": [],
    }

    if not os.path.exists(traj_path):
        return trajectory

    try:
        frames = ase_read(traj_path, index=":")
        start_pos = None
        prev_pos = None

        for i, frame in enumerate(frames):
            energy = frame.get_potential_energy() if frame.calc is not None else None
            forces = frame.get_forces() if frame.calc is not None else None
            positions = frame.get_positions()

            if energy is not None:
                trajectory["energy"].append(float(energy))
            if forces is not None:
                force_norms = np.linalg.norm(forces, axis=1)
                trajectory["force_max"].append(float(np.max(force_norms)))
                trajectory["force_mean"].append(float(np.mean(force_norms)))

            # Store positions
            trajectory["positions"].append(positions.copy())

            # Compute displacements
            if start_pos is None:
                start_pos = positions.copy()
                trajectory["disp_from_start"].append(0.0)
            else:
                disp = np.linalg.norm(positions - start_pos, axis=1).mean()
                trajectory["disp_from_start"].append(float(disp))

            if prev_pos is None:
                trajectory["disp_from_last"].append(0.0)
            else:
                disp = np.linalg.norm(positions - prev_pos, axis=1).mean()
                trajectory["disp_from_last"].append(float(disp))

            prev_pos = positions.copy()
    except Exception:
        pass

    return trajectory


def run_sella_ts(
    calculator,
    calculator_type: str,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    fmax: float = 0.03,
    max_steps: int = 200,
    internal: bool = True,
    delta0: float = 0.048,
    order: int = 1,
    device: str = "cuda",
    save_trajectory: bool = True,
    trajectory_dir: Optional[str] = None,
    sample_index: int = 0,
    logfile: Optional[str] = None,
    verbose: bool = True,
    use_exact_hessian: bool = True,
    diag_every_n: int = 1,
    gamma: float = 0.0,
    # Trust radius parameters from Wander et al. (2024) arXiv:2410.01650v2
    rho_inc: float = 1.035,
    rho_dec: float = 5.0,
    sigma_inc: float = 1.15,
    sigma_dec: float = 0.65,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run Sella TS refinement on a starting geometry.

    This function:
    1. Creates ASE Atoms from the starting coordinates
    2. Attaches the appropriate ASE calculator wrapper
    3. Optionally creates a hessian_function for exact Hessian computation
    4. Runs Sella optimizer with P-RFO (Partitioned Rational Function Optimization)
    5. Returns final coordinates and optimization trajectory

    Trust radius parameters are from Wander et al. (2024) "Accessing Numerical
    Energy Hessians with Graph Neural Network Potentials" (arXiv:2410.01650v2).

    Args:
        calculator: HIP or SCINE calculator instance
        calculator_type: "hip" or "scine"
        coords0: (N, 3) starting coordinates in Angstrom
        atomic_nums: (N,) atomic numbers
        fmax: Force convergence threshold (eV/A)
        max_steps: Maximum optimization steps
        internal: Use internal coordinates (recommended for TS searches)
        delta0: Initial trust radius. Paper value: 4.8E-2 (0.048). Default: 0.048.
        order: Saddle order (1 for transition states)
        device: Device for HIP calculations
        save_trajectory: Whether to save the optimization trajectory
        trajectory_dir: Directory to save trajectory files
        sample_index: Sample index for naming trajectory files
        logfile: Path to log file (None for no logging, "-" for stdout)
        verbose: Print progress messages
        use_exact_hessian: If True, use exact Hessian from calculator at each step
            instead of Sella's iterative approximation. This is crucial for ML
            potentials like HIP which can provide exact Hessians directly.
            Default: True.
        diag_every_n: Recompute/update Hessian every N steps. Set to 1 to use
            fresh Hessian at every step (recommended when use_exact_hessian=True).
            Default: 1.
        gamma: Tolerance for iterative eigensolver (only used when
            use_exact_hessian=False). Set to 0 for tightest convergence.
            Default: 0.0.
        rho_inc: Threshold above which to increase trust radius. Paper value: 1.035.
        rho_dec: Threshold below which to decrease trust radius. Paper value: 5.0.
        sigma_inc: Factor by which to increase trust radius. Paper value: 1.15.
        sigma_dec: Factor by which to decrease trust radius. Paper value: 0.65.

    Returns:
        out_dict: Dictionary containing:
            - final_coords: Final optimized coordinates (torch tensor)
            - steps_taken: Number of optimization steps
            - converged: Whether fmax was reached
            - final_energy: Final energy (eV)
            - final_fmax: Final maximum force (eV/A)
            - trajectory: Per-step metrics dict
        aux: Dictionary containing:
            - trajectory_path: Path to saved trajectory file (if saved)
            - wall_time: Total wall clock time
    """
    if not SELLA_AVAILABLE:
        raise ImportError(
            "Sella is not installed. Install with: pip install sella"
        )

    t0 = time.time()

    # Convert to ASE Atoms
    atoms = coords_to_ase_atoms(coords0, atomic_nums)

    # Create and attach ASE calculator wrapper
    ase_calc = create_ase_calculator(
        calculator,
        calculator_type,
        device=device,
    )
    atoms.calc = ase_calc

    # Set up trajectory file
    traj_path = None
    if save_trajectory:
        if trajectory_dir is not None:
            os.makedirs(trajectory_dir, exist_ok=True)
            traj_path = os.path.join(trajectory_dir, f"sella_{sample_index:03d}.traj")
        else:
            # Use temp directory
            traj_path = tempfile.mktemp(suffix=".traj")

    # Create hessian_function if using exact Hessians
    hessian_fn = None
    if use_exact_hessian:
        hessian_fn = create_hessian_function(
            calculator,
            calculator_type,
            device=device,
        )
        if verbose:
            print(f"[Sella] Using exact Hessian from {calculator_type.upper()} at every step (diag_every_n={diag_every_n})")

    # Create Sella optimizer with P-RFO method
    # order=1 for TS (first-order saddle point)
    # internal=True uses internal coordinates (bonds/angles/dihedrals)
    # which helps with robustness
    # hessian_function: when provided, Sella uses exact Hessian instead of
    #   iterative approximation. For internal coords, Sella automatically
    #   converts Cartesian Hessian to internal coordinates.
    # diag_every_n: how often to recompute Hessian (1 = every step)
    # gamma: tolerance for iterative eigensolver (0 = tightest, only matters
    #   when hessian_function is not provided)
    # Trust radius params from Wander et al. (2024) arXiv:2410.01650v2:
    #   delta0=0.048, rho_inc=1.035, rho_dec=5.0, sigma_inc=1.15, sigma_dec=0.65
    opt = Sella(
        atoms,
        order=order,
        internal=internal,
        trajectory=traj_path,
        logfile=logfile,
        delta0=delta0,
        hessian_function=hessian_fn,
        diag_every_n=diag_every_n,
        gamma=gamma,
        # Trust radius management (paper-recommended values)
        rho_inc=rho_inc,
        rho_dec=rho_dec,
        sigma_inc=sigma_inc,
        sigma_dec=sigma_dec,
    )

    if verbose:
        hess_mode = "exact" if use_exact_hessian else "iterative"
        print(f"[Sella] Starting TS refinement (fmax={fmax}, max_steps={max_steps}, internal={internal}, hessian={hess_mode})")

    # Run optimization
    try:
        converged = opt.run(fmax=fmax, steps=max_steps)
    except Exception as e:
        if verbose:
            print(f"[Sella] Optimization failed: {e}")
        converged = False

    wall_time = time.time() - t0

    # Extract final state
    final_coords = ase_atoms_to_coords(atoms)
    steps_taken = opt.nsteps

    # Get final energy and forces
    try:
        final_energy = float(atoms.get_potential_energy())
        final_forces = atoms.get_forces()
        final_force_norms = np.linalg.norm(final_forces, axis=1)
        final_fmax = float(np.max(final_force_norms))
        final_force_mean = float(np.mean(final_force_norms))
    except Exception:
        final_energy = None
        final_fmax = None
        final_force_mean = None

    # Parse trajectory for per-step metrics
    trajectory = {}
    if traj_path and os.path.exists(traj_path):
        trajectory = parse_sella_trajectory(traj_path)

    if verbose:
        status = "converged" if converged else "not converged"
        print(f"[Sella] Finished: {steps_taken} steps, {status}, fmax={final_fmax:.4f} eV/A")

    out_dict = {
        "final_coords": final_coords,
        "steps_taken": steps_taken,
        "converged": converged,
        "final_energy": final_energy,
        "final_fmax": final_fmax,
        "final_force_mean": final_force_mean,
        "trajectory": trajectory,
    }

    aux = {
        "trajectory_path": traj_path,
        "wall_time": wall_time,
    }

    return out_dict, aux


def validate_ts_eigenvalues(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    scine_elements: Optional[list] = None,
) -> Dict[str, Any]:
    """Validate that final geometry is a true TS using Hessian eigenvalues.

    Uses your existing mass-weighted Eckart-projected vibrational analysis
    to verify the saddle order is exactly 1.

    Args:
        predict_fn: Prediction function that returns {energy, forces, hessian}
        coords: (N, 3) coordinates
        atomic_nums: (N,) atomic numbers
        scine_elements: SCINE element types (if using SCINE calculator)

    Returns:
        Dictionary containing:
        - neg_vib: Number of negative vibrational eigenvalues
        - eig0: Lowest vibrational eigenvalue
        - eig1: Second lowest vibrational eigenvalue
        - eig_product: Product of eig0 * eig1 (negative = TS signature)
        - is_ts: True if neg_vib == 1 (proper transition state)
    """
    from ...dependencies.hessian import vibrational_eigvals

    # Run prediction with Hessian
    out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)

    # Get vibrational eigenvalues (mass-weighted, Eckart-projected)
    vib_eigvals = vibrational_eigvals(
        out["hessian"],
        coords,
        atomic_nums,
        scine_elements=scine_elements,
    )

    # Count negative eigenvalues
    neg_vib = int((vib_eigvals < 0).sum().item())

    # Extract lowest two eigenvalues
    eig0 = float(vib_eigvals[0].item()) if vib_eigvals.numel() >= 1 else None
    eig1 = float(vib_eigvals[1].item()) if vib_eigvals.numel() >= 2 else None

    # Eigenvalue product (negative = TS signature: one negative, one positive)
    eig_product = float((vib_eigvals[0] * vib_eigvals[1]).item()) if vib_eigvals.numel() >= 2 else None

    return {
        "neg_vib": neg_vib,
        "eig0": eig0,
        "eig1": eig1,
        "eig_product": eig_product,
        "is_ts": neg_vib == 1,
    }
