import torch
import numpy as np
from typing import Tuple, List, Optional
from .dependencies.differentiable_projection import (
    differentiable_massweigh_and_eckartprojection_torch as massweigh_and_eckartprojection_torch
)

def _objective_and_grad_CORRECTED(self, x_flat: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Corrected implementation of objective_and_grad for LBFGSEnergyMinimizer.
    
    KEY FIXES:
    1. Projects BOTH Hessian and Forces using Eckart conditions.
    2. Ensures L-BFGS optimization occurs in the vibrational subspace.
    """
    # 1. Reconstruct coordinates
    coords = torch.tensor(
        x_flat.reshape(-1, 3),
        dtype=torch.float32,
        device=self.device,
    )
    
    with torch.no_grad():
        # 2. Compute Energy, Forces, Hessian
        batch = coord_atoms_to_torch_geometric(coords, self.atomic_nums, self.device)
        results = self.calculator.predict(batch, do_hessian=True)
        
        energy = results["energy"].item()
        forces_raw = results["forces"]  # (N, 3) -- Raw Cartesian
        hessian_raw = results["hessian"].reshape(coords.numel(), coords.numel())
        
        coords_3d = coords.reshape(-1, 3)
        
        # 3. CRITICAL FIX: Get Projection Operator from Hessian
        # This mass-weights and removes translation/rotation modes
        hess_proj = massweigh_and_eckartprojection_torch(
            hessian_raw, coords_3d, self.atomsymbols
        )
        
        # 4. Eigendecomposition to identify vibrational subspace
        eigvals_all, eigvecs_all = torch.linalg.eigh(hess_proj)
        
        # Determine rigid modes (5 for linear, 6 for non-linear)
        coords_cent = coords_3d.detach().to(torch.float64)
        coords_cent = coords_cent - coords_cent.mean(dim=0, keepdim=True)
        geom_rank = torch.linalg.matrix_rank(coords_cent.cpu(), tol=1e-8).item()
        expected_rigid = 5 if geom_rank <= 2 else 6
        
        # Sort and isolate vibrational modes
        # We assume the smallest |eigenvalues| correspond to rigid body modes
        total_modes = eigvals_all.shape[0]
        n_rigid = min(expected_rigid, max(0, total_modes - 2))
        
        abs_sorted_idx = torch.argsort(torch.abs(eigvals_all))
        keep_idx = abs_sorted_idx[n_rigid:]
        keep_idx_sorted, _ = torch.sort(keep_idx)
        
        eigvals_vib = eigvals_all[keep_idx_sorted]
        eigvecs_vib = eigvecs_all[:, keep_idx_sorted]  # Shape: (3N, n_vib)
        
        # 5. CRITICAL FIX: Project Forces onto Vibrational Subspace
        forces_flat = forces_raw.flatten() # (3N,)
        # Projection: F_proj = V * (V^T * F) where V is vibrational eigenvectors
        forces_proj_flat = eigvecs_vib @ (eigvecs_vib.T @ forces_flat)
        forces_proj_3d = forces_proj_flat.reshape(-1, 3)
        
        # 6. Prepare Gradient for Scipy
        # Scipy minimizes E, so grad = ∇E = -Forces
        # We use the PROJECTED forces
        grad = -forces_proj_flat.cpu().numpy().flatten().astype(np.float64)
        
        # 7. Update Trajectory & convergence check
        neg_count = (eigvals_vib < 0).sum().item()
        
        # Log statistics (using PROJECTED forces for consistency)
        force_rms_proj = forces_proj_3d.norm(dim=1).mean().item()
        
        if self._iteration_count % 10 == 0 and self.verbose:
            print(f"    [L-BFGS] Iter {self._iteration_count:4d}: "
                  f"E={energy:.6f}, |F_proj|={force_rms_proj:.4f}, neg_eig={neg_count}")

        # Convergence Check
        if neg_count <= self.target_neg_eig_count:
            self._should_stop = True

        self._iteration_count += 1
        
        return energy, grad
