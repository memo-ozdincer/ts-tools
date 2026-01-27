"""
SCINE-specific mass handling and frequency analysis.

This module provides mass-weighting and Eckart projection using NumPy/SciPy,
independent of HIP dependencies. Used when calculator="scine".
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import eigh, qr, svd
from typing import Tuple, Dict, Any

import scine_utilities


# ============================================================================
# Element Type to Mass Mapping (AMU)
# ============================================================================

SCINE_ELEMENT_MASSES = {
    scine_utilities.ElementType.H: 1.00784,
    scine_utilities.ElementType.He: 4.002602,
    scine_utilities.ElementType.Li: 6.941,
    scine_utilities.ElementType.Be: 9.012182,
    scine_utilities.ElementType.B: 10.811,
    scine_utilities.ElementType.C: 12.0107,
    scine_utilities.ElementType.N: 14.0067,
    scine_utilities.ElementType.O: 15.999,
    scine_utilities.ElementType.F: 18.9984,
    scine_utilities.ElementType.Ne: 20.1797,
    scine_utilities.ElementType.Na: 22.98977,
    scine_utilities.ElementType.Mg: 24.305,
    scine_utilities.ElementType.Al: 26.981538,
    scine_utilities.ElementType.Si: 28.0855,
    scine_utilities.ElementType.P: 30.973761,
    scine_utilities.ElementType.S: 32.065,
    scine_utilities.ElementType.Cl: 35.453,
    scine_utilities.ElementType.Ar: 39.948,
    scine_utilities.ElementType.K: 39.0983,
    scine_utilities.ElementType.Ca: 40.078,
    scine_utilities.ElementType.Sc: 44.955910,
    scine_utilities.ElementType.Ti: 47.867,
    scine_utilities.ElementType.V: 50.9415,
    scine_utilities.ElementType.Cr: 51.9961,
    scine_utilities.ElementType.Mn: 54.938049,
    scine_utilities.ElementType.Fe: 55.845,
    scine_utilities.ElementType.Co: 58.933200,
    scine_utilities.ElementType.Ni: 58.6934,
    scine_utilities.ElementType.Cu: 63.546,
    scine_utilities.ElementType.Zn: 65.39,
    scine_utilities.ElementType.Ga: 69.723,
    scine_utilities.ElementType.Ge: 72.64,
    scine_utilities.ElementType.As: 74.92160,
    scine_utilities.ElementType.Se: 78.96,
    scine_utilities.ElementType.Br: 79.904,
    scine_utilities.ElementType.Kr: 83.80,
    scine_utilities.ElementType.Rb: 85.4678,
    scine_utilities.ElementType.Sr: 87.62,
    scine_utilities.ElementType.Y: 88.90585,
    scine_utilities.ElementType.Zr: 91.224,
    scine_utilities.ElementType.Nb: 92.90638,
    scine_utilities.ElementType.Mo: 95.94,
    scine_utilities.ElementType.Tc: 98.0,
    scine_utilities.ElementType.Ru: 101.07,
    scine_utilities.ElementType.Rh: 102.90550,
    scine_utilities.ElementType.Pd: 106.42,
    scine_utilities.ElementType.Ag: 107.8682,
    scine_utilities.ElementType.Cd: 112.411,
    scine_utilities.ElementType.In: 114.818,
    scine_utilities.ElementType.Sn: 118.710,
    scine_utilities.ElementType.Sb: 121.760,
    scine_utilities.ElementType.Te: 127.60,
    scine_utilities.ElementType.I: 126.90447,
    scine_utilities.ElementType.Xe: 131.293,
}


# Atomic number to SCINE ElementType mapping (extended to cover more elements)
Z_TO_SCINE_ELEMENT = {
    1: scine_utilities.ElementType.H,
    2: scine_utilities.ElementType.He,
    3: scine_utilities.ElementType.Li,
    4: scine_utilities.ElementType.Be,
    5: scine_utilities.ElementType.B,
    6: scine_utilities.ElementType.C,
    7: scine_utilities.ElementType.N,
    8: scine_utilities.ElementType.O,
    9: scine_utilities.ElementType.F,
    10: scine_utilities.ElementType.Ne,
    11: scine_utilities.ElementType.Na,
    12: scine_utilities.ElementType.Mg,
    13: scine_utilities.ElementType.Al,
    14: scine_utilities.ElementType.Si,
    15: scine_utilities.ElementType.P,
    16: scine_utilities.ElementType.S,
    17: scine_utilities.ElementType.Cl,
    18: scine_utilities.ElementType.Ar,
    19: scine_utilities.ElementType.K,
    20: scine_utilities.ElementType.Ca,
    21: scine_utilities.ElementType.Sc,
    22: scine_utilities.ElementType.Ti,
    23: scine_utilities.ElementType.V,
    24: scine_utilities.ElementType.Cr,
    25: scine_utilities.ElementType.Mn,
    26: scine_utilities.ElementType.Fe,
    27: scine_utilities.ElementType.Co,
    28: scine_utilities.ElementType.Ni,
    29: scine_utilities.ElementType.Cu,
    30: scine_utilities.ElementType.Zn,
    31: scine_utilities.ElementType.Ga,
    32: scine_utilities.ElementType.Ge,
    33: scine_utilities.ElementType.As,
    34: scine_utilities.ElementType.Se,
    35: scine_utilities.ElementType.Br,
    36: scine_utilities.ElementType.Kr,
    37: scine_utilities.ElementType.Rb,
    38: scine_utilities.ElementType.Sr,
    39: scine_utilities.ElementType.Y,
    40: scine_utilities.ElementType.Zr,
    41: scine_utilities.ElementType.Nb,
    42: scine_utilities.ElementType.Mo,
    43: scine_utilities.ElementType.Tc,
    44: scine_utilities.ElementType.Ru,
    45: scine_utilities.ElementType.Rh,
    46: scine_utilities.ElementType.Pd,
    47: scine_utilities.ElementType.Ag,
    48: scine_utilities.ElementType.Cd,
    49: scine_utilities.ElementType.In,
    50: scine_utilities.ElementType.Sn,
    51: scine_utilities.ElementType.Sb,
    52: scine_utilities.ElementType.Te,
    53: scine_utilities.ElementType.I,
    54: scine_utilities.ElementType.Xe,
}


def get_scine_masses(elements: list) -> np.ndarray:
    """Get atomic masses in AMU for a list of SCINE ElementType objects.

    Args:
        elements: List of scine_utilities.ElementType

    Returns:
        masses_amu: Array of atomic masses in AMU, shape (N,)
    """
    masses = []
    for element in elements:
        if element not in SCINE_ELEMENT_MASSES:
            raise ValueError(
                f"Unsupported element {element}. "
                f"Add to SCINE_ELEMENT_MASSES in scine_masses.py"
            )
        masses.append(SCINE_ELEMENT_MASSES[element])
    return np.array(masses, dtype=np.float64)


# ============================================================================
# Frequency Analysis (NumPy/SciPy implementation)
# ============================================================================

class ScineFrequencyAnalyzer:
    """
    Mass-weighting and Eckart projection for SCINE Hessians.

    This implementation uses NumPy/SciPy and is independent of HIP.
    Based on the SVD projection method (numerically robust).
    """

    # Physical Constants
    EV_TO_JOULES = 1.602176634e-19
    ANGSTROM_TO_METERS = 1e-10
    AMU_TO_KG = 1.66053906660e-27
    C_LIGHT_CM_S = 2.99792458e10  # Speed of light in cm/s

    @staticmethod
    def _center_of_mass(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """Compute center of mass.

        Args:
            coords: (N, 3) array of positions
            masses: (N,) array of masses

        Returns:
            com: (3,) array of center of mass position
        """
        total_mass = np.sum(masses)
        return np.sum(coords * masses[:, None], axis=0) / total_mass

    @staticmethod
    def _get_inertia_tensor(coords: np.ndarray, masses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute inertia tensor for rotation vectors.

        Args:
            coords: (N, 3) array of positions
            masses: (N,) array of masses

        Returns:
            inertia: (3, 3) inertia tensor
            coords_centered: (N, 3) centered coordinates
        """
        # Center coords on COM
        com = ScineFrequencyAnalyzer._center_of_mass(coords, masses)
        coords_centered = coords - com

        inertia = np.zeros((3, 3))
        for i in range(len(masses)):
            r = coords_centered[i]
            m = masses[i]
            r_sq = np.dot(r, r)
            # I_ab = sum[ m * (r^2 delta_ab - r_a r_b) ]
            inertia += m * (r_sq * np.eye(3) - np.outer(r, r))

        return inertia, coords_centered

    def _get_vibrational_projector(
        self,
        coords: np.ndarray,
        masses: np.ndarray
    ) -> np.ndarray:
        """
        Build projection matrix P that maps 3N coords to (3N-6) vibrational coords.
        Uses SVD method for numerical robustness.

        Args:
            coords: (N, 3) positions in Angstrom
            masses: (N,) masses in AMU

        Returns:
            P: (3N-k, 3N) projection matrix where k is number of rigid modes (5 or 6)
        """
        n_atoms = len(masses)
        sqrt_masses = np.sqrt(masses)
        sqrt_masses_repeat = np.repeat(sqrt_masses, 3)

        # --- A. Translation Vectors (3 vectors) ---
        trans_vecs = []
        for axis in range(3):
            v = np.zeros((n_atoms, 3))
            v[:, axis] = 1.0
            v = v.flatten() * sqrt_masses_repeat
            v /= np.linalg.norm(v)
            trans_vecs.append(v)

        # --- B. Rotation Vectors (up to 3 vectors) ---
        inertia, coords_centered = self._get_inertia_tensor(coords, masses)
        # Eigenvectors of inertia tensor define principal axes
        _, principal_axes = eigh(inertia)  # columns are eigenvectors

        rot_vecs = []
        for axis in range(3):
            # Rotation axis direction
            u = principal_axes[:, axis]

            # r_i x u (cross product for each atom)
            displacements = np.cross(coords_centered, u)  # Shape (N, 3)

            v = displacements.flatten() * sqrt_masses_repeat
            norm = np.linalg.norm(v)

            # Check for linear molecules (norm ~ 0 for rotation along bond axis)
            if norm > 1e-6:
                v /= norm
                rot_vecs.append(v)

        # --- C. Orthonormalize (QR) ---
        # Stack TR vectors as rows (approx 6 x 3N, or 5 x 3N for linear)
        tr_space = np.vstack(trans_vecs + rot_vecs)

        # QR decomposition to orthonormalize
        q_ortho, _ = qr(tr_space.T, mode='economic')  # q_ortho is (3N, k) where k=5 or 6

        # --- D. SVD Projector ---
        # Get full SVD to find the null space of TR modes
        u_svd, _, _ = svd(q_ortho, full_matrices=True)
        # u_svd is (3N, 3N)
        # First k columns span TR space, remaining columns span vibrational space

        n_tr_modes = q_ortho.shape[1]

        # P = U[:, k:].T -> Shape (3N-k, 3N)
        P = u_svd[:, n_tr_modes:].T

        return P

    def project_hessian(
        self,
        elements: list,
        positions_angstrom: np.ndarray,
        hessian_ev_ang2: np.ndarray,
        apply_massweight: bool = True,
        apply_eckart: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mass-weight and/or Eckart-project a Hessian.

        Args:
            elements: List of scine_utilities.ElementType
            positions_angstrom: (N, 3) positions in Angstrom
            hessian_ev_ang2: (3N, 3N) Hessian in eV/Å²
            apply_massweight: If True, apply mass-weighting (M^{-1/2} H M^{-1/2}).
                Default True.
            apply_eckart: If True, apply Eckart projection to remove trans/rot modes.
                Default True.

        Returns:
            proj_hessian: Projected Hessian.
                - If both True: (3N-k, 3N-k) projected mass-weighted, eV/AMU units
                - If only massweight: (3N, 3N) mass-weighted, eV/AMU units
                - If only eckart: (3N-k, 3N-k) projected but NOT mass-weighted
                - If neither: (3N, 3N) raw Hessian
            masses_amu: (N,) masses in AMU (for reference)
        """
        n_atoms = len(elements)
        masses_amu = get_scine_masses(elements)

        # Start with raw Hessian
        hessian = hessian_ev_ang2.copy()

        # Step 1: Apply mass-weighting if requested
        if apply_massweight:
            # M^{-1/2} H M^{-1/2} where H is in eV/Å² and M is in AMU
            # Result has eigenvalues in eV/AMU
            m_sqrt = np.sqrt(masses_amu)
            m_sqrt_repeat = np.repeat(m_sqrt, 3)  # (3N,)
            inv_m_sqrt_mat = np.outer(1.0 / m_sqrt_repeat, 1.0 / m_sqrt_repeat)
            hessian = hessian * inv_m_sqrt_mat

        # Step 2: Apply Eckart projection if requested
        if apply_eckart:
            # Build projection matrix P (using positions in Angstrom, masses in AMU)
            P = self._get_vibrational_projector(positions_angstrom, masses_amu)

            # Project: P H P^T -> (3N-k, 3N-k)
            hessian = P @ hessian @ P.T

        # Symmetrize to remove numerical noise
        hessian = 0.5 * (hessian + hessian.T)

        return hessian, masses_amu

    def analyze(
        self,
        elements: list,
        positions_angstrom: np.ndarray,
        hessian_ev_ang2: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Full workflow: Mass-weight -> Eckart Project -> Diagonalize -> Frequencies.

        Args:
            elements: List of scine_utilities.ElementType
            positions_angstrom: (N, 3) positions in Angstrom
            hessian_ev_ang2: (3N, 3N) Hessian in eV/Å²

        Returns:
            Dictionary containing:
            - frequencies_cm: (3N-k,) vibrational frequencies in cm^-1
              (negative values indicate imaginary frequencies)
            - eigenvalues: (3N-k,) eigenvalues of projected Hessian
            - n_imaginary: Number of imaginary frequencies (TS order)
        """
        # Project Hessian
        proj_hessian, _ = self.project_hessian(elements, positions_angstrom, hessian_ev_ang2)

        # Diagonalize
        eigvals, _ = eigh(proj_hessian)

        # Convert eigenvalues to frequencies (cm^-1)
        frequencies = []
        for eig in eigvals:
            # omega = sqrt(eigenvalue)
            # nu = omega / (2 * pi)
            # wavenumber = nu / c
            if eig > 0:
                freq = np.sqrt(eig) / (2 * np.pi * self.C_LIGHT_CM_S)
            else:
                # Imaginary frequency (represented as negative)
                freq = -np.sqrt(np.abs(eig)) / (2 * np.pi * self.C_LIGHT_CM_S)
            frequencies.append(freq)

        frequencies = np.array(frequencies)
        n_imaginary = np.sum(frequencies < 0)

        return {
            "frequencies_cm": frequencies,
            "eigenvalues": eigvals,
            "n_imaginary": int(n_imaginary),
        }


# ============================================================================
# PyTorch interface (for consistency with existing codebase)
# ============================================================================

def scine_project_hessian_remove_rigid_modes(
    hessian_raw: torch.Tensor,
    coords: torch.Tensor,
    elements: list,  # List of scine_utilities.ElementType
    apply_massweight: bool = True,
    apply_eckart: bool = True,
) -> torch.Tensor:
    """
    Mass-weight + Eckart-project Hessian using SCINE masses (NumPy backend).

    This is the SCINE equivalent of the HIP version in hessian.py.
    Returns a PyTorch tensor for compatibility with the rest of the codebase.

    Args:
        hessian_raw: (3N, 3N) Hessian in eV/Å²
        coords: (N, 3) coordinates in Angstrom
        elements: List of scine_utilities.ElementType
        apply_massweight: If True, apply mass-weighting (M^{-1/2} H M^{-1/2}).
            Default True.
        apply_eckart: If True, apply Eckart projection to remove trans/rot modes.
            Default True.

    Returns:
        proj_hessian: Projected Hessian as PyTorch tensor.
            If both flags True: (3N-k, 3N-k)
            If only massweight: (3N, 3N) mass-weighted
            If neither: (3N, 3N) raw Hessian

    Raises:
        ValueError: If coords and elements have inconsistent atom counts
    """
    # Convert to NumPy
    hess_np = hessian_raw.detach().cpu().numpy()
    coords_np = coords.detach().cpu().numpy().reshape(-1, 3)

    # Validate atom count consistency
    n_atoms_coords = coords_np.shape[0]
    n_atoms_elements = len(elements)
    if n_atoms_coords != n_atoms_elements:
        raise ValueError(
            f"Atom count mismatch in SCINE Hessian projection: "
            f"coords has {n_atoms_coords} atoms but elements has {n_atoms_elements}. "
            f"This usually means trajectory positions don't match the original molecule. "
            f"Check that trajectory files aren't stale from a previous run."
        )

    # If no processing requested, return raw Hessian
    if not apply_massweight and not apply_eckart:
        return hessian_raw.clone()

    # Project using SCINE analyzer
    analyzer = ScineFrequencyAnalyzer()
    proj_hess_np, _ = analyzer.project_hessian(
        elements, coords_np, hess_np,
        apply_massweight=apply_massweight,
        apply_eckart=apply_eckart,
    )

    # Convert back to PyTorch
    return torch.from_numpy(proj_hess_np).to(
        device=hessian_raw.device,
        dtype=hessian_raw.dtype
    )


def scine_vibrational_eigvals(
    hessian_raw: torch.Tensor,
    coords: torch.Tensor,
    elements: list,  # List of scine_utilities.ElementType
    apply_massweight: bool = True,
    apply_eckart: bool = True,
) -> torch.Tensor:
    """
    Extract vibrational eigenvalues using SCINE mass-weighting.

    This is the SCINE equivalent of vibrational_eigvals in hessian.py.

    Args:
        hessian_raw: (3N, 3N) Hessian in eV/Å²
        coords: (N, 3) coordinates in Angstrom
        elements: List of scine_utilities.ElementType
        apply_massweight: If True, apply mass-weighting. Default True.
        apply_eckart: If True, apply Eckart projection. Default True.

    Returns:
        vib_eigvals: Eigenvalues as PyTorch tensor.
            If both flags True: (3N-k,) vibrational eigenvalues
            Otherwise: (3N,) all eigenvalues
    """
    proj_hess = scine_project_hessian_remove_rigid_modes(
        hessian_raw, coords, elements,
        apply_massweight=apply_massweight,
        apply_eckart=apply_eckart,
    )
    eigvals, _ = torch.linalg.eigh(proj_hess)
    return eigvals


# ============================================================================
# Vector projection functions for GAD (SCINE implementation)
# ============================================================================

def scine_get_vibrational_projector_full(
    coords: torch.Tensor,
    elements: list,
) -> torch.Tensor:
    """
    Get the full 3N x 3N vibrational projector P for SCINE.

    Unlike the reduced projector (3N-k, 3N), this returns a square projector
    that can be used to project vectors in 3N space.

    P = V V^T where V is (3N, 3N-k) vibrational basis

    Args:
        coords: (N, 3) coordinates in Angstrom
        elements: List of scine_utilities.ElementType

    Returns:
        P: (3N, 3N) vibrational projector
    """
    coords_np = coords.detach().cpu().numpy().reshape(-1, 3)
    masses_amu = get_scine_masses(elements)

    analyzer = ScineFrequencyAnalyzer()
    P_reduced = analyzer._get_vibrational_projector(coords_np, masses_amu)  # (3N-k, 3N)

    # P_full = P_reduced.T @ P_reduced gives us the full 3N x 3N vibrational projector
    P_full = P_reduced.T @ P_reduced  # (3N, 3N)

    return torch.from_numpy(P_full).to(device=coords.device, dtype=torch.float64)


def scine_project_vector_to_vibrational(
    vec: torch.Tensor,
    coords: torch.Tensor,
    elements: list,
    is_mass_weighted: bool = False,
) -> torch.Tensor:
    """
    Project a vector to remove translation/rotation components (SCINE version).

    For GAD, the gradient and guide vector v must be projected to prevent
    the dynamics from drifting into the null space.

    Args:
        vec: (3N,) or (N, 3) vector to project
        coords: (N, 3) coordinates in Angstrom
        elements: List of scine_utilities.ElementType
        is_mass_weighted: If True, vec is already in mass-weighted space.
                          If False, vec will be mass-weighted, projected, then un-weighted.

    Returns:
        vec_proj: Projected vector (3N,)
    """
    device = vec.device
    original_dtype = vec.dtype

    vec_flat = vec.reshape(-1).to(torch.float64)
    masses_amu = get_scine_masses(elements)

    sqrt_m = np.sqrt(masses_amu)
    sqrt_m_3n = np.repeat(sqrt_m, 3)

    # Get the full vibrational projector
    P = scine_get_vibrational_projector_full(coords, elements)  # (3N, 3N)

    vec_np = vec_flat.cpu().numpy()

    if is_mass_weighted:
        # vec is already in MW space, just project
        vec_proj_np = P.cpu().numpy() @ vec_np
    else:
        # Transform to MW space, project, transform back
        vec_mw = vec_np / sqrt_m_3n
        vec_mw_proj = P.cpu().numpy() @ vec_mw
        vec_proj_np = vec_mw_proj * sqrt_m_3n

    return torch.from_numpy(vec_proj_np).to(device=device, dtype=original_dtype)


def scine_project_guide_vector(
    v: torch.Tensor,
    coords: torch.Tensor,
    elements: list,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Project and optionally normalize the GAD guide vector v (SCINE version).

    The guide vector v should live in the mass-weighted vibrational subspace.

    Args:
        v: (3N,) guide vector (assumed to be in mass-weighted space)
        coords: (N, 3) coordinates in Angstrom
        elements: List of scine_utilities.ElementType
        normalize: If True, renormalize v after projection

    Returns:
        v_proj: Projected (and optionally normalized) guide vector (3N,)
    """
    # Guide vectors from eigenvector decomposition are already in MW space
    v_proj = scine_project_vector_to_vibrational(v, coords, elements, is_mass_weighted=True)

    if normalize:
        v_proj = v_proj / (v_proj.norm() + 1e-12)

    return v_proj


def scine_gad_dynamics_projected(
    coords: torch.Tensor,
    forces: torch.Tensor,
    v: torch.Tensor,
    elements: list,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Compute GAD dynamics with consistent Eckart projection (SCINE version).

    This function ensures that:
    1. The gradient (forces) is projected to remove TR components
    2. The guide vector v is projected to remove TR components
    3. The output dq/dt is projected to stay in vibrational space

    Args:
        coords: (N, 3) Cartesian coordinates
        forces: (N, 3) or (3N,) forces (negative gradient)
        v: (3N,) guide vector (eigenvector of projected Hessian)
        elements: List of scine_utilities.ElementType

    Returns:
        gad_vec: (N, 3) GAD direction in Cartesian space
        v_proj: (3N,) projected/normalized guide vector for tracking
        info: dict with diagnostic info
    """
    device = coords.device
    original_dtype = forces.dtype

    num_atoms = len(elements)
    f_flat = forces.reshape(-1).to(torch.float64)
    v_flat = v.reshape(-1).to(torch.float64)

    masses_amu = get_scine_masses(elements)
    sqrt_m = np.sqrt(masses_amu)
    sqrt_m_3n = np.repeat(sqrt_m, 3)
    sqrt_m_t = torch.from_numpy(sqrt_m_3n).to(device=device, dtype=torch.float64)

    # Get full projector
    P = scine_get_vibrational_projector_full(coords, elements)

    # ---- Project gradient (forces = -gradient) ----
    # Transform to MW space: grad_mw = M^{-1/2} @ (-forces)
    grad_mw = (-f_flat.cpu().numpy()) / sqrt_m_3n
    grad_mw_proj = P.cpu().numpy() @ grad_mw
    grad_mw_proj_t = torch.from_numpy(grad_mw_proj).to(device=device, dtype=torch.float64)

    # ---- Project guide vector v ----
    v_np = v_flat.cpu().numpy()
    v_proj_np = P.cpu().numpy() @ v_np
    v_proj_np = v_proj_np / (np.linalg.norm(v_proj_np) + 1e-12)
    v_proj_t = torch.from_numpy(v_proj_np).to(device=device, dtype=torch.float64)

    # ---- Compute GAD direction ----
    v_dot_grad = torch.dot(v_proj_t, grad_mw_proj_t)
    v_dot_v = torch.dot(v_proj_t, v_proj_t)

    dq_dt_mw = -grad_mw_proj_t + 2.0 * (v_dot_grad / (v_dot_v + 1e-12)) * v_proj_t

    # Project output
    dq_dt_mw_np = dq_dt_mw.cpu().numpy()
    dq_dt_mw_proj = P.cpu().numpy() @ dq_dt_mw_np
    dq_dt_mw_proj_t = torch.from_numpy(dq_dt_mw_proj).to(device=device, dtype=torch.float64)

    # Convert back to Cartesian space
    dq_dt_cart = sqrt_m_t * dq_dt_mw_proj_t

    gad_vec = dq_dt_cart.reshape(num_atoms, 3).to(original_dtype)

    info = {
        "v_dot_grad": float(v_dot_grad.item()),
        "grad_norm_mw": float(np.linalg.norm(grad_mw_proj)),
        "v_norm": float(np.linalg.norm(v_proj_np)),
    }

    return gad_vec, v_proj_t.to(v.dtype), info
