import torch

from hip.masses import MASS_DICT

# ---- helpers --------------------------------------------------------------

def _to_torch_double(array_like, device=None):
    if isinstance(array_like, torch.Tensor):
        return array_like.to(dtype=torch.float64, device=device)
    return torch.as_tensor(array_like, dtype=torch.float64, device=device)


def mass_weigh_hessian_torch(hessian, masses3d):
    """M^{-1/2} H M^{-1/2} (all torch-autodiff friendly)."""
    h_t = _to_torch_double(hessian, device=hessian.device)
    m_t = _to_torch_double(masses3d, device=hessian.device)
    mm_sqrt_inv = torch.diag(1.0 / torch.sqrt(m_t))
    return mm_sqrt_inv @ h_t @ mm_sqrt_inv


def purify_hessian_sum_rules_torch(hessian: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """Enforce translational invariance sum rules on a Cartesian Hessian.

    For a translationally invariant PES, the Hessian satisfies:
        sum_j H[i,a; j,b] = 0   for all atom i, direction a.

    ML-predicted Hessians violate this, causing residual TR eigenvalues (~5e-5)
    after Eckart projection.  This function symmetrically distributes the
    row-sum error so the sum rules hold exactly, which makes projection clean.

    Fully differentiable (reshape / sum / subtract).

    Args:
        hessian: (3N, 3N) raw Cartesian Hessian
        n_atoms: number of atoms N

    Returns:
        (3N, 3N) purified Hessian satisfying translational sum rules
    """
    dtype = torch.float64
    H = hessian.to(dtype=dtype)
    dim3N = 3 * n_atoms

    # Reshape to block form (N, 3, N, 3)
    H_block = H.reshape(n_atoms, 3, n_atoms, 3)

    # For each row-block (i, a): compute sum over all (j, b)
    # row_sums[i, a] = sum_j sum_b H[i, a, j, b]
    row_sums = H_block.sum(dim=(2, 3))  # (N, 3)

    # Distribute correction uniformly across all column entries:
    # H[i, a, j, b] -= row_sums[i, a] / (3N)
    correction = row_sums[:, :, None, None] / dim3N  # (N, 3, 1, 1)
    H_block = H_block - correction

    H_purified = H_block.reshape(dim3N, dim3N)

    # Symmetrize (the correction may break exact symmetry slightly)
    H_purified = 0.5 * (H_purified + H_purified.transpose(0, 1))

    return H_purified


def _center_of_mass(coords3d, masses):
    total_mass = torch.sum(masses)
    return (coords3d * masses[:, None]).sum(dim=0) / total_mass


# ---- differentiable Eckart generators & projector ------------------------

def eckart_B_massweighted_torch(cart_coords, masses, eps=1e-12):
    """
    Build the 6 Eckart generators (3 translations, 3 rotations) in mass-weighted space:
      B \in R^{3N x 6}, columns are orthogonal to vibrations under the Euclidean metric.
    No eigen-decompositions, no QR -> smooth grads.
    """
    coords = _to_torch_double(cart_coords)
    masses = _to_torch_double(masses, device=coords.device)

    # shapes & mass factors
    xyz = coords.reshape(-1, 3)                                # (N, 3)
    N = xyz.shape[0]
    sqrt_m = torch.sqrt(masses)                                # (N,)
    sqrt_m3 = sqrt_m.repeat_interleave(3)                      # (3N,)

    # center geometry at mass-weighted COM (still differentiable)
    com = _center_of_mass(xyz, masses)                         # (3,)
    r = xyz - com[None, :]                                     # (N, 3)

    # --- 3 translations (mass-weighted unit directions) ---
    # For axis a, component-wise displacement is constant 1. After mass-weighting multiply by sqrt(m_i).
    ex = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=coords.device)
    ey = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=coords.device)
    ez = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=coords.device)
    Tcols = []
    for e in (ex, ey, ez):
        tiled = e.repeat(N)                                    # (3N,)
        col = sqrt_m3 * tiled                                  # mass-weighted translation
        # normalize to avoid scale pathologies (keeps smooth grads)
        col = col / (col.norm() + eps)
        Tcols.append(col)

    # --- 3 rotations (infinitesimal) ---
    # For axis ω = ex/ey/ez: δr_i = ω × r_i; then mass-weight by sqrt(m_i) per component.
    # ex × r = (0, -r_z, r_y); ey × r = (r_z, 0, -r_x); ez × r = (-r_y, r_x, 0)
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    R_ex = torch.stack([torch.zeros_like(rx), -rz, ry], dim=1)  # (N,3)
    R_ey = torch.stack([rz, torch.zeros_like(ry), -rx], dim=1)
    R_ez = torch.stack([-ry, rx, torch.zeros_like(rz)], dim=1)
    Rcols = []
    for Raxis in (R_ex, R_ey, R_ez):
        col = (Raxis * sqrt_m[:, None]).reshape(-1)             # mass-weighted
        # normalize each rotational generator for numerical stability
        col = col / (col.norm() + eps)
        Rcols.append(col)

    # Stack to B: (3N, 6)
    B = torch.stack(Tcols + Rcols, dim=1)
    return B  # columns span translations+rotations in MW metric


def eckartprojection_torch(cart_coords, masses, eps=1e-10):
    """
    Return the vibrational projector P \in R^{3N x 3N} in the *mass-weighted* space.
    P = I - B (B^T B + eps I)^{-1} B^T
    """
    B = eckart_B_massweighted_torch(cart_coords, masses, eps=eps)   # (3N, 6)
    G = B.transpose(0, 1) @ B                                       # (6, 6)
    # stabilized inverse; cholesky preferred when G is SPD (generic non-linear molecules)
    try:
        L = torch.linalg.cholesky(G + eps * torch.eye(6, dtype=G.dtype, device=G.device))
        Ginvt_Bt = torch.cholesky_solve(B.transpose(0, 1), L)       # (6, 3N)
    except RuntimeError:
        Ginvt_Bt = torch.linalg.solve(G + eps * torch.eye(6, dtype=G.dtype, device=G.device),
                                      B.transpose(0, 1))
    P = torch.eye(B.shape[0], dtype=B.dtype, device=B.device) - B @ Ginvt_Bt
    # make numerically symmetric/idempotent
    P = 0.5 * (P + P.transpose(0, 1))
    return P


# ---- reduced vibrational basis (Solution A) -----------------------------------

def build_vibrational_basis_torch(
    cart_coords: torch.Tensor,
    masses: torch.Tensor,
    eps: float = 1e-12,
    linear_tol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build orthonormal vibrational basis Q_vib by complementing the TR subspace.

    Unlike the projector P = I - B(B^TB)^{-1}B^T which keeps the full (3N, 3N)
    space with 6 near-zero eigenvalues, this constructs an explicit (3N, 3N-k)
    orthonormal basis for the vibrational subspace, where k = 5 (linear) or 6.

    The resulting Q_vib lets you project to a FULL-RANK (3N-k, 3N-k) Hessian
    with no zero eigenvalues and no threshold-based filtering.

    Args:
        cart_coords: (N, 3) Cartesian coordinates
        masses: (N,) atomic masses in AMU
        eps: regularization for B construction
        linear_tol: threshold on QR diagonal to detect degenerate TR modes
            (linear molecules have only 2 rotations → k=5)

    Returns:
        Q_vib: (3N, 3N-k) orthonormal columns spanning vibrational space
        Q_tr: (3N, k) orthonormal columns spanning TR space
        k: number of TR modes (5 or 6)
    """
    B = eckart_B_massweighted_torch(cart_coords, masses, eps=eps)  # (3N, 6)
    dim3N = B.shape[0]

    # QR decomposition to orthonormalize TR generators
    Q_full, R = torch.linalg.qr(B, mode="reduced")  # Q: (3N, 6), R: (6, 6)

    # Detect near-degenerate columns (linear molecules: one rotation has ~0 norm)
    diag_R = torch.abs(torch.diag(R))
    valid_mask = diag_R > linear_tol
    k = int(valid_mask.sum().item())
    k = max(k, 1)  # at least 1 TR mode (safety)

    # Keep only the valid TR columns
    Q_tr = Q_full[:, :k]  # (3N, k)

    # Build vibrational complement via SVD of Q_tr
    # U from SVD of Q_tr (full_matrices=True) gives complete orthonormal basis.
    # First k columns = TR space, remaining 3N-k columns = vibrational space.
    U, _, _ = torch.linalg.svd(Q_tr, full_matrices=True)  # U: (3N, 3N)
    Q_vib = U[:, k:]  # (3N, 3N-k)

    return Q_vib, Q_tr, k


def reduced_basis_hessian_torch(
    hessian: torch.Tensor,
    cart_coords: torch.Tensor,
    atomsymbols: list[str],
    purify: bool = False,
) -> dict[str, torch.Tensor]:
    """Mass-weight and project Hessian to the full-rank vibrational subspace.

    Returns a (3N-k, 3N-k) Hessian with NO zero eigenvalues — every eigenvalue
    is a genuine vibrational frequency.  This eliminates the need for
    threshold-based TR filtering and avoids the numerical issues of working
    with a rank-deficient (3N, 3N) projected Hessian.

    Args:
        hessian: (3N, 3N) raw Cartesian Hessian (eV/A^2)
        cart_coords: (N, 3) Cartesian coordinates (Angstrom)
        atomsymbols: list of element symbols ['C', 'H', ...]
        purify: if True, enforce translational sum rules before projecting

    Returns:
        dict with keys:
            H_red: (3N-k, 3N-k) full-rank vibrational Hessian
            Q_vib: (3N, 3N-k) orthonormal vibrational basis
            Q_tr:  (3N, k) orthonormal TR basis
            k_tr:  int, number of TR modes (5 or 6)
            H_mw:  (3N, 3N) mass-weighted Hessian (before reduction)
            masses: (N,) atomic masses
            sqrt_m: (3N,) sqrt(mass) repeated per coordinate
            sqrt_m_inv: (3N,) 1/sqrt(mass) repeated per coordinate
    """
    device = hessian.device
    dtype = torch.float64

    coords_3d = cart_coords.reshape(-1, 3).to(dtype)
    n_atoms = coords_3d.shape[0]

    masses_t, masses3d_t, sqrt_m, sqrt_m_inv = get_mass_weights_torch(
        atomsymbols, device=device, dtype=dtype
    )

    H = hessian.to(dtype=dtype)

    # Optionally purify sum rules
    if purify:
        H = purify_hessian_sum_rules_torch(H, n_atoms)

    # Mass-weight
    H_mw = mass_weigh_hessian_torch(H, masses3d_t)

    # Build vibrational basis
    Q_vib, Q_tr, k_tr = build_vibrational_basis_torch(coords_3d, masses_t)

    # Project to reduced space
    H_red = Q_vib.transpose(0, 1) @ H_mw @ Q_vib  # (3N-k, 3N-k)
    H_red = 0.5 * (H_red + H_red.transpose(0, 1))  # symmetrize

    return {
        "H_red": H_red,
        "Q_vib": Q_vib,
        "Q_tr": Q_tr,
        "k_tr": k_tr,
        "H_mw": H_mw,
        "masses": masses_t,
        "sqrt_m": sqrt_m,
        "sqrt_m_inv": sqrt_m_inv,
    }


# ---- vector projection functions for GAD ----------------------------------------

def get_mass_weights_torch(atomsymbols: list[str], device=None, dtype=torch.float64):
    """
    Get mass-weighting factors for a molecule.

    Args:
        atomsymbols: List of atom symbols ['C', 'H', 'O', ...]
        device: Torch device
        dtype: Torch dtype (default float64)

    Returns:
        masses: (N,) atomic masses in AMU
        masses3d: (3N,) masses repeated for each coordinate
        sqrt_m: (3N,) sqrt(masses) for mass-weighting
        sqrt_m_inv: (3N,) 1/sqrt(masses) for inverse mass-weighting
    """
    masses_t = torch.tensor([MASS_DICT[atom.lower()] for atom in atomsymbols],
                            dtype=dtype, device=device)
    masses3d_t = masses_t.repeat_interleave(3)
    sqrt_m = torch.sqrt(masses3d_t)
    sqrt_m_inv = 1.0 / sqrt_m
    return masses_t, masses3d_t, sqrt_m, sqrt_m_inv


def project_vector_to_vibrational_torch(
    vec: torch.Tensor,
    cart_coords: torch.Tensor,
    atomsymbols: list[str],
    eps: float = 1e-10,
    is_mass_weighted: bool = False,
) -> torch.Tensor:
    """
    Project a vector to remove translation/rotation components.

    For GAD, this is CRITICAL: the gradient and guide vector v must be projected
    to prevent the dynamics from drifting into the null space (eq. 7-9 in GAD paper).

    Args:
        vec: (3N,) or (N, 3) vector to project (gradient, guide vector, etc.)
        cart_coords: (N, 3) Cartesian coordinates
        atomsymbols: List of atom symbols
        eps: Regularization for projector construction
        is_mass_weighted: If True, vec is already in mass-weighted space.
                          If False, vec will be mass-weighted, projected, then un-weighted.

    Returns:
        vec_proj: Projected vector in the same space as input (3N,)
    """
    device = vec.device
    dtype = torch.float64

    vec_flat = vec.reshape(-1).to(dtype)
    coords_3d = cart_coords.reshape(-1, 3)

    masses_t, _, sqrt_m, sqrt_m_inv = get_mass_weights_torch(
        atomsymbols, device=device, dtype=dtype
    )

    # Build projector in mass-weighted space
    P = eckartprojection_torch(coords_3d, masses_t, eps=eps)

    if is_mass_weighted:
        # vec is already in MW space, just project
        vec_proj = P @ vec_flat
    else:
        # Transform to MW space, project, transform back
        # v_mw = M^{-1/2} @ v (for gradients: "raise index")
        vec_mw = sqrt_m_inv * vec_flat
        vec_mw_proj = P @ vec_mw
        # Transform back: v_proj = M^{1/2} @ v_mw_proj
        vec_proj = sqrt_m * vec_mw_proj

    return vec_proj.to(vec.dtype)


def project_guide_vector_torch(
    v: torch.Tensor,
    cart_coords: torch.Tensor,
    atomsymbols: list[str],
    eps: float = 1e-10,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Project and optionally normalize the GAD guide vector v.

    The guide vector v should live in the mass-weighted vibrational subspace.
    This function ensures v has no translation/rotation components.

    Args:
        v: (3N,) guide vector (assumed to be in mass-weighted space)
        cart_coords: (N, 3) Cartesian coordinates
        atomsymbols: List of atom symbols
        eps: Regularization for projector construction
        normalize: If True, renormalize v after projection

    Returns:
        v_proj: Projected (and optionally normalized) guide vector (3N,)
    """
    device = v.device
    dtype = torch.float64

    v_flat = v.reshape(-1).to(dtype)
    coords_3d = cart_coords.reshape(-1, 3)

    masses_t, _, _, _ = get_mass_weights_torch(
        atomsymbols, device=device, dtype=dtype
    )

    # Build projector in mass-weighted space
    P = eckartprojection_torch(coords_3d, masses_t, eps=eps)

    # Project v (already in MW space since it's an eigenvector of MW Hessian)
    v_proj = P @ v_flat

    # Renormalize
    if normalize:
        v_proj = v_proj / (v_proj.norm() + 1e-12)

    return v_proj.to(v.dtype)


def gad_dynamics_projected_torch(
    coords: torch.Tensor,
    forces: torch.Tensor,
    v: torch.Tensor,
    atomsymbols: list[str],
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Compute GAD dynamics (eqs 1-2) with consistent Eckart projection.

    This function ensures that:
    1. The gradient (forces) is projected to remove TR components
    2. The guide vector v is projected to remove TR components
    3. The output dq/dt is projected to stay in vibrational space

    This prevents "leakage" of the dynamics into the null space.

    Args:
        coords: (N, 3) Cartesian coordinates
        forces: (N, 3) or (3N,) forces (negative gradient)
        v: (3N,) guide vector (eigenvector of projected Hessian)
        atomsymbols: List of atom symbols
        eps: Regularization for projector construction

    Returns:
        gad_vec: (N, 3) GAD direction in Cartesian space
        v_proj: (3N,) projected/normalized guide vector for tracking
        info: dict with diagnostic info
    """
    device = coords.device
    dtype = torch.float64

    coords_3d = coords.reshape(-1, 3).to(dtype)
    f_flat = forces.reshape(-1).to(dtype)
    v_flat = v.reshape(-1).to(dtype)
    num_atoms = coords_3d.shape[0]

    masses_t, _, sqrt_m, sqrt_m_inv = get_mass_weights_torch(
        atomsymbols, device=device, dtype=dtype
    )

    # Build projector at current geometry
    P = eckartprojection_torch(coords_3d, masses_t, eps=eps)

    # ---- Project gradient (forces = -gradient) ----
    # Forces are in Cartesian space (eV/Angstrom), so we mass-weight, project, un-weight
    # grad_mw = M^{-1/2} @ (-forces)
    grad_mw = -sqrt_m_inv * f_flat
    grad_mw_proj = P @ grad_mw

    # ---- Project guide vector v ----
    # v is already in MW space (eigenvector of MW Hessian)
    v_proj = P @ v_flat
    v_proj = v_proj / (v_proj.norm() + 1e-12)

    # ---- Compute GAD direction (eq 1): dq/dt = -grad + 2(v·grad)/(v·v) * v ----
    v_dot_grad = torch.dot(v_proj, grad_mw_proj)
    v_dot_v = torch.dot(v_proj, v_proj)  # ~1 after normalization

    dq_dt_mw = -grad_mw_proj + 2.0 * (v_dot_grad / (v_dot_v + 1e-12)) * v_proj

    # ---- Project output to ensure it stays in vibrational space ----
    dq_dt_mw = P @ dq_dt_mw

    # ---- Convert back to Cartesian space ----
    # dq_cart = M^{1/2} @ dq_mw
    dq_dt_cart = sqrt_m * dq_dt_mw

    # Note: The standard GAD formulation uses forces directly (f + 2(v·(-f))v)
    # which is equivalent but avoids the mass-weighting in force space.
    # Here we implement the more rigorous version that works in MW space.

    gad_vec = dq_dt_cart.reshape(num_atoms, 3).to(forces.dtype)

    info = {
        "v_dot_grad": float(v_dot_grad.item()),
        "grad_norm_mw": float(grad_mw_proj.norm().item()),
        "v_norm": float(v_proj.norm().item()),
    }

    return gad_vec, v_proj.to(v.dtype), info


# ---- use this function ----------------------------

def differentiable_massweigh_and_eckartprojection_torch(
    hessian: torch.Tensor,
    cart_coords: torch.Tensor,
    atomsymbols: list[str],
    ev_thresh: float = -1e-6,   # kept for API compat; unused here
    apply_massweight: bool = True,
    apply_eckart: bool = True,
):
    """
    Eckart projection for a *non*-mass-weighted Hessian:
      1) mass-weight H (if apply_massweight=True)
      2) build MW projector P
      3) project: P H_mw P (if apply_eckart=True)
    Everything is differentiable w.r.t. coords, Hessian elements, and (if you wish) masses.

    NOTE: Returns Hessian in MASS-WEIGHTED space. For Cartesian output, use
    eckart_project_and_return_cartesian_torch() instead.

    Args:
        hessian: Raw Cartesian Hessian (3N, 3N)
        cart_coords: Cartesian coordinates (N, 3)
        atomsymbols: List of atom symbols ['C', 'H', 'O', ...]
        ev_thresh: Unused, kept for API compatibility
        apply_massweight: If True, apply mass-weighting. Default True.
        apply_eckart: If True, apply Eckart projection. Default True.

    Returns:
        Processed Hessian tensor.
        If both flags True: (3N, 3N) mass-weighted projected Hessian
        If only massweight: (3N, 3N) mass-weighted Hessian
        If only eckart: (3N, 3N) projected but NOT mass-weighted
        If neither: (3N, 3N) raw Hessian

    Raises:
        ValueError: If coords and atomsymbols have inconsistent atom counts
    """
    device = hessian.device
    dtype = torch.float64

    # Validate atom count consistency
    coords_3d = cart_coords.reshape(-1, 3)
    n_atoms_coords = coords_3d.shape[0]
    n_atoms_symbols = len(atomsymbols)
    if n_atoms_coords != n_atoms_symbols:
        raise ValueError(
            f"Atom count mismatch in HIP Hessian projection: "
            f"coords has {n_atoms_coords} atoms but atomsymbols has {n_atoms_symbols}. "
            f"This usually means trajectory positions don't match the original molecule. "
            f"Check that trajectory files aren't stale from a previous run."
        )

    # If no processing requested, return raw Hessian (converted to float64 for consistency)
    if not apply_massweight and not apply_eckart:
        return hessian.to(dtype=dtype)

    masses_t = torch.tensor([MASS_DICT[atom.lower()] for atom in atomsymbols],
                            dtype=dtype, device=device)
    masses3d_t = masses_t.repeat_interleave(3)

    # Start with Hessian (converted to float64)
    H = hessian.to(dtype=dtype)

    # 1) mass-weight the Hessian if requested
    if apply_massweight:
        H = mass_weigh_hessian_torch(H, masses3d_t)             # (3N,3N)

    # 2) project if requested
    if apply_eckart:
        P = eckartprojection_torch(cart_coords, masses_t)   # (3N,3N)
        H = P @ H @ P

    # Symmetrize
    H = 0.5 * (H + H.transpose(0, 1))
    return H


def eckart_project_and_return_cartesian_torch(
    hessian: torch.Tensor,
    cart_coords: torch.Tensor,
    atomsymbols: list[str],
):
    """
    Eckart projection that returns the Hessian in CARTESIAN coordinates.

    Full cycle:
      1) mass-weight H:         H_mw = M^{-1/2} H M^{-1/2}
      2) build MW projector P
      3) project in MW space:   H_mw_proj = P H_mw P
      4) UN-mass-weight:        H_cart_proj = M^{1/2} H_mw_proj M^{1/2}

    This removes the 6 translational/rotational modes while keeping the
    Hessian in Cartesian coordinates (required by Sella for internal coord conversion).

    Args:
        hessian: Raw Cartesian Hessian (3N, 3N)
        cart_coords: Cartesian coordinates (N, 3)
        atomsymbols: List of atom symbols ['C', 'H', 'O', ...]

    Returns:
        Eckart-projected Hessian in CARTESIAN coordinates (3N, 3N)
    """
    device = hessian.device
    dtype = torch.float64

    masses_t = torch.tensor([MASS_DICT[atom.lower()] for atom in atomsymbols],
                            dtype=dtype, device=device)
    masses3d_t = masses_t.repeat_interleave(3)

    # Mass-weighting matrices
    sqrt_m_inv = torch.diag(1.0 / torch.sqrt(masses3d_t))  # M^{-1/2}
    sqrt_m = torch.diag(torch.sqrt(masses3d_t))            # M^{1/2}

    # 1) mass-weight the Hessian: H_mw = M^{-1/2} H M^{-1/2}
    h_t = _to_torch_double(hessian, device=device)
    H_mw = sqrt_m_inv @ h_t @ sqrt_m_inv

    # 2) projector in the MW space
    P = eckartprojection_torch(cart_coords, masses_t)

    # 3) project in MW space: H_mw_proj = P H_mw P
    H_mw_proj = P @ H_mw @ P
    H_mw_proj = 0.5 * (H_mw_proj + H_mw_proj.transpose(0, 1))

    # 4) UN-mass-weight back to Cartesian: H_cart_proj = M^{1/2} H_mw_proj M^{1/2}
    H_cart_proj = sqrt_m @ H_mw_proj @ sqrt_m
    H_cart_proj = 0.5 * (H_cart_proj + H_cart_proj.transpose(0, 1))

    return H_cart_proj


# ---- compare with non-differentiable version ---------------------------------------------------

def compare_truncated_vs_projected(
    H_trunc: torch.Tensor,      # (3N-6, 3N-6) reduced (basis-truncated) Hessian
    H_proj: torch.Tensor,       # (3N, 3N) projected Hessian (with 6 near-zeros)
    V: torch.Tensor | None = None, # optional (3N, 3N-6) basis used for truncation
    zero_tol: float = 1e-8,     # threshold to drop rigid-body zeros in H_proj
):
    """
    Compute eigensystems and compare.
    Returns a dict with eigenvalue diffs and (if V provided) subspace/eigenvector alignment.
    """
    dtype = torch.float64
    H_trunc = H_trunc.to(dtype)
    H_proj  = H_proj.to(dtype)
    H_trunc = 0.5 * (H_trunc + H_trunc.T)
    H_proj  = 0.5 * (H_proj  + H_proj.T)

    # --- eigensystems ---
    # Reduced (vibrational) eigensystem
    eval_red, evec_red = torch.linalg.eigh(H_trunc)  # shapes: (K,), (K,K), K=3N-6

    # Projected: take vibrational (non-zero) spectrum only
    eval_full, evec_full = torch.linalg.eigh(H_proj)  # (3N,), (3N,3N)
    keep = torch.abs(eval_full) > zero_tol
    eval_vib = eval_full[keep]              # (K,)
    evec_vib = evec_full[:, keep]           # (3N, K)

    # Sort both vibrational spectra
    idx_red  = torch.argsort(eval_red)
    idx_vib  = torch.argsort(eval_vib)
    eval_red = eval_red[idx_red]
    evec_red = evec_red[:, idx_red]
    eval_vib = eval_vib[idx_vib]
    evec_vib = evec_vib[:, idx_vib]

    # --- eigenvalue comparison ---
    if eval_red.numel() != eval_vib.numel():
        raise ValueError(f"Vibrational eigenvalue count mismatch: "
                         f"{eval_red.numel()} vs {eval_vib.numel()} (check zero_tol)")

    ev_diff = eval_vib - eval_red
    ev_abs_err_max = torch.max(torch.abs(ev_diff)).item()
    ev_abs_err_mean = torch.mean(torch.abs(ev_diff)).item()
    ev_rel_err_mean = (torch.mean(torch.abs(ev_diff) / (torch.abs(eval_red) + 1e-30))).item()

    result = {
        "eigenvalues_reduced": eval_red,
        "eigenvalues_projected_vibrational": eval_vib,
        "eigenvalue_abs_err_max": ev_abs_err_max,
        "eigenvalue_abs_err_mean": ev_abs_err_mean,
        "eigenvalue_rel_err_mean": ev_rel_err_mean,
    }

    # --- eigenvector / subspace comparison (requires V) ---
    if V is not None:
        V = V.to(dtype)
        # Lift reduced eigenvectors to full space: U_red_full = V @ evec_red
        U_red_full = V @ evec_red              # (3N, K)
        # Orthonormalize numerically (good hygiene)
        U_red_full, _ = torch.linalg.qr(U_red_full, mode="reduced")  # (3N, K)

        # Orthonormalize evec_vib too (it should already be)
        U_vib = evec_vib
        # Subspace principal angles via SVD of cross-Gram
        C = U_vib.T @ U_red_full               # (K, K)
        svals = torch.linalg.svdvals(C)        # cosines of principal angles
        svals = torch.clamp(svals, 0.0, 1.0)
        principal_angles = torch.arccos(svals) # radians

        # Report permutation/degeneracy-invariant metrics
        result.update({
            "principal_angles_rad": principal_angles,
            "principal_angle_max_deg": torch.rad2deg(torch.max(principal_angles)).item(),
            "principal_angle_mean_deg": torch.rad2deg(torch.mean(principal_angles)).item(),
            "mean_vector_overlap": torch.mean(svals).item(),  # = mean cos(angle)
        })

        # Optional: per-mode overlaps after a cheap greedy matching (not assignment-optimal,
        # but quick and informative). Comment out if you prefer only subspace metrics.
        M = torch.abs(U_vib.T @ (V @ evec_red))  # (K,K) absolute overlaps
        # greedy: match each vib vector to the best reduced one
        perm = torch.argmax(M, dim=1)
        per_mode_overlap = M[torch.arange(M.shape[0]), perm]
        result.update({
            "per_mode_overlap_greedy": per_mode_overlap,
            "per_mode_overlap_mean_greedy": torch.mean(per_mode_overlap).item(),
            "per_mode_overlap_min_greedy": torch.min(per_mode_overlap).item(),
        })
    else:
        result["note"] = (
            "Eigenvector comparison requires the truncation basis V (shape 3N×(3N-6)). "
            "Eigenvalues are compared; subspace angles skipped."
        )

    return result

if __name__ == "__main__":

    from torch_geometric.data import Batch as TGBatch
    from torch_geometric.data import Data as TGData
    from hip.inference_utils import get_model_from_checkpoint
    import os
    from hip.frequency_analysis import eckart_projection_notmw_torch as projection_not_differentiable
    from hip.ff_lmdb import Z_TO_ATOM_SYMBOL
    
    # Load model with the specified checkpoint
    project_root = os.path.dirname(os.path.dirname(__file__))
    checkpoint_path = os.path.join(project_root, "ckpt/hip_v2.ckpt")
    
    # Load model directly
    model = get_model_from_checkpoint(checkpoint_path, device="cuda")
    
    # Example molecular system
    atoms = ["H", "O", "H", "C"]
    atomic_nums = [1, 8, 1, 6]  # H, O, H, C
    nat = len(atoms)
    coords3d = torch.randn(nat, 3)  # Random coordinates in Angstrom
    
    # Convert to torch tensors with gradients enabled
    coords_torch = torch.tensor(coords3d, dtype=torch.float64, requires_grad=True)
    
    # Ensure coordinates are a leaf on the model device with gradients enabled
    coords = torch.tensor(coords3d, dtype=torch.float32, device=model.device, requires_grad=True)
    # Create batch using the same code as in EquiformerTorchCalculator
    data = TGData(
        pos=coords, 
        z=torch.as_tensor(atomic_nums, dtype=torch.int64, device=model.device),
        charges=torch.as_tensor(atomic_nums, dtype=torch.int64, device=model.device),
        natoms=torch.tensor([len(atomic_nums)], dtype=torch.int64, device=model.device),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool, device=model.device),
    )
    batch = TGBatch.from_data_list([data])
    
    # Get hessian from model using autograd
    with torch.enable_grad():
        # batch.pos.requires_grad = True
        energy, forces, out = model.forward(
            batch,
            otf_graph=True,
        )
        hessian = out["hessian"]
        
        # Convert atomic numbers to symbols
        atomsymbols = [Z_TO_ATOM_SYMBOL[z] for z in atomic_nums]
        hessian = hessian.reshape(coords.numel(), coords.numel())
        hessian_proj = differentiable_massweigh_and_eckartprojection_torch(hessian, coords.reshape(-1, 3), atomsymbols)

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = torch.linalg.eigh(hessian_proj)
        
        # Compute product of the two smallest eigenvalues
        product = eigvals[0] * eigvals[1]
    
    # Compute gradient
    grad = torch.autograd.grad(
        product, 
        # batch.pos.reshape(-1, 3), 
        coords,
        retain_graph=True,
        create_graph=True
    )[0]
    
    # neg_inds = eigvals < ev_thresh
    # neg_eigvals = eigvals[neg_inds]
    # neg_num = torch.sum(neg_inds)
    
    print(f"\nProduct: {product}")
    print(f"Gradient: {grad}")
    print(f"Gradient norm: {torch.norm(grad):.6f}")
    
    # compare with non-differentiable version
    print("coords.shape: ", coords.shape)
    print("hessian.shape: ", hessian.shape)
    print("hessian_proj.shape: ", hessian_proj.shape)
    hessian_trunc, V = projection_not_differentiable(hessian.detach(), coords.detach(), atomsymbols, return_basis=True)
    print("hessian_trunc.shape: ", hessian_trunc.shape)
    
    # H_trunc: (3N-6,3N-6)   from basis truncation
    # H_proj:  (3N,3N)       from projector method
    # V:       (3N,3N-6)     the vibrational basis used to make H_trunc (optional but needed for vector tests)

    stats = compare_truncated_vs_projected(hessian_trunc, hessian_proj, V=V, zero_tol=1e-8)
    print("max |Δλ| =", stats["eigenvalue_abs_err_max"])
    if "principal_angle_max_deg" in stats:
        print("max principal angle (deg) =", stats["principal_angle_max_deg"])