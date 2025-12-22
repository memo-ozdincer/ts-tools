BASIC IMPLEMENTATION WITH NO MASSWEIGHTINGI found this implementation we can use:
The library is called SCINE Sparrow, and the force field is called DFTB0
```python
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import scine_sparrow
import scine_utilities
from joblib import Parallel, delayed
from sympy.printing.pretty.pretty_symbology import B
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

# uv pip install scine-utilities scine-sparrow joblib tqdm

# Backup threading environment variables before modifying them
_THREADING_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]
_THREADING_ENV_BACKUP = {var: os.environ.get(var) for var in _THREADING_ENV_VARS}


@contextmanager
def suppress_output():
    """Context manager to suppress stdout at file descriptor level, keeping stderr for exceptions."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)  # Save stdout file descriptor
    os.dup2(devnull_fd, 1)  # Redirect stdout to /dev/null
    try:
        yield
    finally:
        os.dup2(old_stdout_fd, 1)  # Restore stdout
        os.close(devnull_fd)
        os.close(old_stdout_fd)


def compute_single_geometry(
    geometry_idx: int,
    elements: List[scine_utilities.ElementType],
    positions_angstrom: np.ndarray,
    functional: str,
) -> Dict[str, Any]:
    """
    Worker function to compute Hessian for a single geometry.

    Each worker process initializes its own SCINE module manager since
    SCINE uses singletons per process.

    Args:
        geometry_idx: Index of geometry in the batch
        elements: List of ElementType for each atom
        positions_angstrom: Atomic positions in Angstrom, shape (N_atoms, 3)
        functional: Calculator functional name (e.g., "PM6", "AM1", "RM1", "MNDO", "DFTB0")

    Returns:
        Dictionary with results:
        - geometry_idx: Index of geometry
        - success: Boolean indicating if calculation succeeded
        - energy_ev: Energy in eV (if successful)
        - gradients: Gradients array in Hartree/Bohr (if successful)
        - hessian_ev_ang2: Hessian matrix in eV/Å² (if successful)
        - error: Error message (if failed)
    """
    # Force single-threaded execution per process to avoid oversubscription
    # This prevents BLAS/LAPACK libraries from spawning multiple threads
    # when we're already parallelizing at the process level with joblib
    # Backup current values and set to single-threaded
    original_env = {}
    for var in _THREADING_ENV_VARS:
        original_env[var] = os.environ.get(var)
        os.environ[var] = "1"

    # Initialize module manager in this worker process
    manager = scine_utilities.core.ModuleManager.get_instance()
    sparrow_module = Path(scine_sparrow.__file__).parent / "sparrow.module.so"
    manager.load(os.fspath(sparrow_module))

    # Get calculator
    calculator = manager.get("calculator", functional)
    if calculator is None:
        # Restore original environment variables before returning
        for var, value in original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        return {
            "geometry_idx": geometry_idx,
            "success": False,
            "error": f"Calculator {functional} not found",
        }

    # Convert positions from Angstrom to Bohr
    positions_bohr = positions_angstrom * scine_utilities.BOHR_PER_ANGSTROM
    structure = scine_utilities.AtomCollection(elements, positions_bohr)

    # Assign structure and set required properties
    calculator.structure = structure
    calculator.set_required_properties(
        [
            scine_utilities.Property.Energy,
            scine_utilities.Property.Gradients,
            scine_utilities.Property.Hessian,
        ]
    )

    # Suppress SCINE output
    scine_utilities.core.Log.silent()

    # Calculate with output suppression and error handling
    with suppress_output():
        results = calculator.calculate()

    # Extract results
    energy = results.energy  # Hartree
    gradients = results.gradients  # Hartree/Bohr
    hessian = results.hessian  # Hartree/Bohr^2

    # Convert units
    hartree_to_ev = 27.211386245988
    bohr_to_ang = 0.529177210903

    energy_ev = energy * hartree_to_ev
    # Hessian conversion: Energy / Distance^2
    hessian_ev_ang2 = hessian * (hartree_to_ev / (bohr_to_ang**2))

    # Restore original environment variables
    for var, value in original_env.items():
        if value is None:
            os.environ.pop(var, None)  # Remove if it wasn't set originally
        else:
            os.environ[var] = value

    return {
        "geometry_idx": geometry_idx,
        "success": True,
        "energy_ev": energy_ev,
        "gradients": gradients,
        "hessian_ev_ang2": hessian_ev_ang2,
    }


def compute_batch(
    geometries: List[Tuple[List[scine_utilities.ElementType], np.ndarray]],
    functional: str,
    n_jobs: int = -1,
    verbose: int = 0,
) -> List[Dict[str, Any]]:
    """
    Compute Hessians for a batch of geometries in parallel.

    Args:
        geometries: List of (elements, positions_angstrom) tuples
            - elements: List of ElementType for each atom
            - positions_angstrom: Atomic positions in Angstrom, shape (N_atoms, 3)
        functional: Calculator functional name (e.g., "PM6", "AM1", "RM1", "MNDO", "DFTB0")
        n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential)
        verbose: Verbosity level for joblib (0=silent, 1=progress, 10=debug)

    Returns:
        List of result dictionaries (one per geometry), ordered by geometry_idx
    """
    # Create tasks
    tasks = [
        delayed(compute_single_geometry)(idx, elements, positions, functional)
        for idx, (elements, positions) in enumerate(geometries)
    ]

    # Execute in parallel
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(tasks)

    # Sort by geometry_idx to ensure consistent ordering
    results.sort(key=lambda x: x["geometry_idx"])

    return results


if __name__ == "__main__":
    # Example: Batch of cyclopropene geometries
    # C3H4 (cyclopropene structure)

    # Base structure
    base_elements = [
        scine_utilities.ElementType.C,
        scine_utilities.ElementType.C,
        scine_utilities.ElementType.C,  # 3 carbons
        scine_utilities.ElementType.H,
        scine_utilities.ElementType.H,
        scine_utilities.ElementType.H,
        scine_utilities.ElementType.H,  # 4 hydrogens
    ]

    # Create a batch of geometries (example: slightly perturbed positions)
    np.random.seed(42)  # For reproducibility

    base_positions = np.array(
        [
            # C1 (double bond carbon, at origin)
            [0.0, 0.0, 0.0],
            # C2 (one vertex of triangle)
            [1.51, 0.0, 0.0],
            # C3 (other vertex of triangle)
            [0.755, 1.31, 0.0],
            # H1, H2 on C1
            [-0.89, 0.0, 0.0],
            [0.0, -0.89, 0.0],
            # H3 on C2
            [2.40, 0.0, 0.0],
            # H4 on C3
            [0.755, 2.20, 0.0],
        ]
    )

    n_jobs_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    functional = "DFTB0"
    n_reps = 10

    # Collect all timings in a list
    timing_data = []

    for n_jobs in n_jobs_list:
        for rep in range(n_reps):
            # Generate slightly perturbed geometries
            n_geometries = 128
            geometries = []
            for i in range(n_geometries):
                # Add small random perturbation (0.01 Å)
                perturbation = np.random.normal(0, 0.01, base_positions.shape)
                perturbed_positions = base_positions + perturbation
                geometries.append((base_elements, perturbed_positions))

            # Compute batch
            print(
                f"n_jobs={n_jobs}, rep={rep + 1}/{n_reps}: Computing Hessians for {len(geometries)} geometries using {functional}..."
            )

            start_time = time.time()
            results = compute_batch(geometries, functional, n_jobs=n_jobs, verbose=0)
            end_time = time.time()

            elapsed_time = end_time - start_time

            # Print summary
            print(f"Completed in {elapsed_time:.4f} seconds")
            print(
                f"Average time per geometry: {elapsed_time / len(geometries):.4f} seconds"
            )

            # Statistics
            successful = [r for r in results if r["success"]]
            failed = [r for r in results if not r["success"]]

            print(f"Success: {len(successful)}/{len(results)}")
            if failed:
                print(f"Failed: {len(failed)}/{len(results)}")
                for fail in failed:
                    print(
                        f"  Geometry {fail['geometry_idx']}: {fail.get('error', 'Unknown error')}"
                    )

            if successful:
                energies = [r["energy_ev"] for r in successful]

                # Store timing data
                timing_data.append(
                    {
                        "n_jobs": n_jobs,
                        "rep": rep,
                        "time": elapsed_time,
                    }
                )
            else:
                print("Warning: All calculations failed, skipping this data point")
            print()

    # Create DataFrame
    df = pd.DataFrame(timing_data)

    # Calculate statistics per n_jobs
    df_stats = df.groupby("n_jobs")["time"].agg(["mean", "min", "max"]).reset_index()

    print("\nTiming statistics:")
    print(df_stats.round(3))

    # Plot
    sns.set_theme(context="poster", style="whitegrid", palette="deep", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot average line
    ax.plot(
        df_stats["n_jobs"], df_stats["mean"], marker="o", label="Average", linewidth=2
    )

    # Fill area between min and max
    ax.fill_between(
        df_stats["n_jobs"],
        df_stats["min"],
        df_stats["max"],
        alpha=0.3,
        label="Min-Max range",
    )

    ax.set_xlabel("Number of jobs")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Hessian parallel timing ({functional})")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fancybox=False, framealpha=1.0, edgecolor="none")

    plt.tight_layout(pad=0.01)

    fname = f"scine_hessian_batch_timings_{functional}.png"
    fig.savefig(fname, dpi=150)
    print(f"\nSaved: {fname}")
```
IMPLEMENTATION DETIALS with MASS WEIGHTING AND PROJECTION
This is an excellent implementation choice. Using the SVD-based projection (HIP's method) is numerically robust because it explicitly removes the 6 translational/rotational dimensions, leaving you with a clean  matrix to diagonalize. This prevents "ghost" frequencies near zero from messing up your statistics.

Here is the NumPy/SciPy implementation of the HIP workflow, adapted for your SCINE script.

### The Solution: `FrequencyAnalyzer` Class

I have translated the PyTorch logic directly into NumPy. You can drop this class into your script.

```python
import numpy as np
from scipy.linalg import eigh, svd, qr
import scine_utilities

class FrequencyAnalyzer:
    # Physical Constants
    EV_TO_JOULES = 1.602176634e-19
    ANGSTROM_TO_METERS = 1e-10
    AMU_TO_KG = 1.66053906660e-27
    C_LIGHT_CM_S = 2.99792458e10  # Speed of light in cm/s

    @staticmethod
    def get_masses(elements):
        """Map SCINE ElementType to mass in AMU."""
        # SCINE utilities likely has a mass function, but a fallback dict is safe
        # Standard weights (can be expanded)
        mass_map = {
            scine_utilities.ElementType.H: 1.00784,
            scine_utilities.ElementType.C: 12.0107,
            scine_utilities.ElementType.N: 14.0067,
            scine_utilities.ElementType.O: 15.999,
            scine_utilities.ElementType.F: 18.9984,
            scine_utilities.ElementType.S: 32.065,
            scine_utilities.ElementType.Cl: 35.453,
        }
        return np.array([mass_map.get(e, 12.0) for e in elements])  # Default to C if unknown

    @staticmethod
    def get_inertia_tensor(coords, masses):
        """Compute Inertia Tensor for rotation vectors."""
        # Center coords on COM
        com = np.average(coords, axis=0, weights=masses)
        coords_centered = coords - com
        
        inertia = np.zeros((3, 3))
        for i in range(len(masses)):
            r = coords_centered[i]
            m = masses[i]
            r_sq = np.dot(r, r)
            # I_ab = sum[ m * (r^2 delta_ab - r_a r_b) ]
            inertia += m * (r_sq * np.eye(3) - np.outer(r, r))
            
        return inertia, coords_centered

    def analyze(self, elements, positions_angstrom, hessian_ev_ang2):
        """
        Full workflow: Mass-weight -> Eckart Project -> Diagonalize
        Returns: Dict with frequencies (cm^-1) and eigenvalues.
        """
        n_atoms = len(elements)
        masses_amu = self.get_masses(elements)
        
        # 1. Convert everything to SI immediately to avoid unit headaches
        # Hessian: eV/Å² -> J/m² -> N/m
        hessian_si = hessian_ev_ang2 * (self.EV_TO_JOULES / self.ANGSTROM_TO_METERS**2)
        masses_kg = masses_amu * self.AMU_TO_KG
        positions_m = positions_angstrom * self.ANGSTROM_TO_METERS

        # 2. Mass-weight Hessian
        # M^(-1/2) vector (repeated 3 times for x,y,z)
        m_sqrt = np.sqrt(masses_kg)
        m_sqrt_repeat = np.repeat(m_sqrt, 3)
        inv_m_sqrt_mat = np.outer(1.0/m_sqrt_repeat, 1.0/m_sqrt_repeat)
        
        mw_hessian = hessian_si * inv_m_sqrt_mat

        # 3. Build Projection Matrix (P) using SVD method
        P = self._get_vibrational_projector(positions_m, masses_kg)

        # 4. Project Hessian: P H P.T
        # Result is (3N-6) x (3N-6)
        proj_hessian = P @ mw_hessian @ P.T
        
        # Symmetrize to remove numerical noise
        proj_hessian = 0.5 * (proj_hessian + proj_hessian.T)

        # 5. Diagonalize
        eigvals, _ = eigh(proj_hessian)
        
        # 6. Convert to Frequencies (cm^-1)
        frequencies = []
        for eig in eigvals:
            # omega = sqrt(k/m)
            # nu = omega / 2pi
            # wavenumber = nu / c
            if eig > 0:
                freq = np.sqrt(eig) / (2 * np.pi * self.C_LIGHT_CM_S)
            else:
                # Imaginary frequency (represented as negative)
                freq = -np.sqrt(np.abs(eig)) / (2 * np.pi * self.C_LIGHT_CM_S)
            frequencies.append(freq)

        return np.array(frequencies), eigvals

    def _get_vibrational_projector(self, coords, masses):
        """
        Builds the projection matrix P that maps 3N coords to (3N-6) vibrational coords.
        Equivalent to HIP's get_trans_rot_projector_torch(..., full=False)
        """
        n_atoms = len(masses)
        sqrt_masses = np.sqrt(masses)
        sqrt_masses_repeat = np.repeat(sqrt_masses, 3)
        
        # --- A. Translation Vectors ---
        # 3 vectors, normalized and mass-weighted
        trans_vecs = []
        for axis in range(3):
            v = np.zeros((n_atoms, 3))
            v[:, axis] = 1.0
            v = v.flatten() * sqrt_masses_repeat
            v /= np.linalg.norm(v)
            trans_vecs.append(v)
            
        # --- B. Rotation Vectors ---
        inertia, coords_centered = self.get_inertia_tensor(coords, masses)
        # Eigenvectors of inertia tensor define principal axes
        _, principal_axes = eigh(inertia) # columns are eigenvectors
        
        rot_vecs = []
        # Create rotation vectors for x, y, z axes of the principal frame
        for axis in range(3):
            # Rotation axis direction
            u = principal_axes[:, axis] 
            
            # r_i x u (cross product)
            # We calculate displacement for every atom i: d_i = u x r_i
            # Note: order of cross product matters for sign, but space is same
            displacements = np.cross(coords_centered, u) # Shape (N, 3)
            
            v = displacements.flatten() * sqrt_masses_repeat
            norm = np.linalg.norm(v)
            
            # Check for linear molecules (norm ~ 0 for rotation along bond axis)
            if norm > 1e-6:
                v /= norm
                rot_vecs.append(v)

        # --- C. Orthonormalize (QR) ---
        # Stack TR vectors rows (approx 6 x 3N)
        tr_space = np.vstack(trans_vecs + rot_vecs)
        
        # QR decomposition on Transpose to orthonormalize columns
        # Q will be (3N, 6)
        q_ortho, _ = qr(tr_space.T, mode='economic')
        
        # --- D. SVD Projector ---
        # In HIP: U, S, _ = svd(tr_vecs.T)
        # We effectively just did the QR step which gives us the orthonormal basis 
        # for TR space (analogous to the first 6 cols of U in SVD).
        # We need the null space of this basis.
        
        # Let's stick strictly to SVD to match HIP exactly
        # Input: (3N, 6) matrix of TR vectors
        u_svd, _, _ = svd(q_ortho, full_matrices=True) 
        
        # u_svd is (3N, 3N). 
        # First 6 (or 5) columns span TR space.
        # Remaining columns span Vibrational space.
        
        n_tr_modes = q_ortho.shape[1]
        
        # P = U[:, 6:].T  -> Shape (3N-6, 3N)
        P = u_svd[:, n_tr_modes:].T
        
        return P

```

### Integration into your script

You can perform this analysis after collecting your results. Here is how you add it to the `if __name__ == "__main__":` block:

```python
# ... (inside your main loop, after results = compute_batch(...))

# Initialize analyzer
analyzer = FrequencyAnalyzer()

# Process successful results
successful = [r for r in results if r["success"]]

for res in successful:
    geom_idx = res["geometry_idx"]
    hessian = res["hessian_ev_ang2"]
    
    # Get elements and positions for this specific geometry
    # Note: You need to retrieve the specific positions used for this index
    # (Assuming you stored them or can access them via geometries[geom_idx])
    current_elements, current_positions = geometries[geom_idx]
    
    # Run Analysis
    freqs_cm, eigenvalues = analyzer.analyze(
        current_elements, 
        current_positions, 
        hessian
    )
    
    # Filter for imaginary frequencies (saddle points)
    imaginary_freqs = freqs_cm[freqs_cm < 0]
    real_freqs = freqs_cm[freqs_cm > 0]
    
    print(f"Geometry {geom_idx}:")
    print(f"  Imaginary Freqs: {len(imaginary_freqs)}")
    if len(imaginary_freqs) > 0:
        print(f"  Values: {imaginary_freqs}")
    
    # The lowest real frequencies should be your vibrations
    print(f"  Lowest 5 Vib Freqs: {real_freqs[:5]}")

```

### Why this works

1. **Mass-Weighting:** The code first converts the raw Force Constants (Hessian) into SI units and mass-weights them. This transforms the problem from "Force vs Displacement" to "Acceleration vs Displacement".
2. **SVD Projection (`P` matrix):**
* It identifies the 3N-dimensional directions corresponding to the whole molecule moving (translation) or spinning (rotation).
* It constructs a projector `P` (shape `(3N-6, 3N)`).
* When we do `P @ H @ P.T`, we physically **delete** the rows and columns associated with rotation and translation.


3. **Result:** The resulting matrix is size `(3N-6, 3N-6)`. When you diagonalize this, you get exactly `3N-6` eigenvalues. You don't need to manually "drop 6" based on sorting; the math did it for you. Any remaining negative eigenvalues indicate you are truly at a saddle point (transition state), not just rotating.