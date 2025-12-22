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
            coords: (N, 3) positions in meters (or consistent units)
            masses: (N,) masses in kg (or consistent units)

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mass-weight and Eckart-project a Hessian.

        Args:
            elements: List of scine_utilities.ElementType
            positions_angstrom: (N, 3) positions in Angstrom
            hessian_ev_ang2: (3N, 3N) Hessian in eV/Å²

        Returns:
            proj_hessian: (3N-k, 3N-k) projected mass-weighted Hessian
            masses_amu: (N,) masses in AMU (for reference)
        """
        n_atoms = len(elements)
        masses_amu = get_scine_masses(elements)

        # 1. Convert to SI units
        hessian_si = hessian_ev_ang2 * (self.EV_TO_JOULES / self.ANGSTROM_TO_METERS**2)
        masses_kg = masses_amu * self.AMU_TO_KG
        positions_m = positions_angstrom * self.ANGSTROM_TO_METERS

        # 2. Mass-weight Hessian: M^{-1/2} H M^{-1/2}
        m_sqrt = np.sqrt(masses_kg)
        m_sqrt_repeat = np.repeat(m_sqrt, 3)  # (3N,)
        inv_m_sqrt_mat = np.outer(1.0 / m_sqrt_repeat, 1.0 / m_sqrt_repeat)

        mw_hessian = hessian_si * inv_m_sqrt_mat

        # 3. Build projection matrix P
        P = self._get_vibrational_projector(positions_m, masses_kg)

        # 4. Project: P H_mw P^T
        proj_hessian = P @ mw_hessian @ P.T

        # Symmetrize to remove numerical noise
        proj_hessian = 0.5 * (proj_hessian + proj_hessian.T)

        return proj_hessian, masses_amu

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
) -> torch.Tensor:
    """
    Mass-weight + Eckart-project Hessian using SCINE masses (NumPy backend).

    This is the SCINE equivalent of the HIP version in hessian.py.
    Returns a PyTorch tensor for compatibility with the rest of the codebase.

    Args:
        hessian_raw: (3N, 3N) Hessian in eV/Å²
        coords: (N, 3) coordinates in Angstrom
        elements: List of scine_utilities.ElementType

    Returns:
        proj_hessian: (3N-k, 3N-k) projected Hessian as PyTorch tensor
    """
    # Convert to NumPy
    hess_np = hessian_raw.detach().cpu().numpy()
    coords_np = coords.detach().cpu().numpy().reshape(-1, 3)

    # Project using SCINE analyzer
    analyzer = ScineFrequencyAnalyzer()
    proj_hess_np, _ = analyzer.project_hessian(elements, coords_np, hess_np)

    # Convert back to PyTorch
    return torch.from_numpy(proj_hess_np).to(
        device=hessian_raw.device,
        dtype=hessian_raw.dtype
    )


def scine_vibrational_eigvals(
    hessian_raw: torch.Tensor,
    coords: torch.Tensor,
    elements: list,  # List of scine_utilities.ElementType
) -> torch.Tensor:
    """
    Extract vibrational eigenvalues using SCINE mass-weighting.

    This is the SCINE equivalent of vibrational_eigvals in hessian.py.

    Args:
        hessian_raw: (3N, 3N) Hessian in eV/Å²
        coords: (N, 3) coordinates in Angstrom
        elements: List of scine_utilities.ElementType

    Returns:
        vib_eigvals: (3N-k,) vibrational eigenvalues as PyTorch tensor
    """
    proj_hess = scine_project_hessian_remove_rigid_modes(hessian_raw, coords, elements)
    eigvals, _ = torch.linalg.eigh(proj_hess)
    return eigvals
