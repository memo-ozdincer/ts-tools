# Code Snippets for December Experiments Document

---

## 1. L-BFGS Energy Minimizer

### Explanation

L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton optimization method that approximates the inverse Hessian using gradient history. We use it to minimize molecular energy while projecting gradients into the vibrational subspace (removing translation/rotation components).

**Key idea:** Since GAD converges 100% from non-minima, we can use L-BFGS to first reduce the saddle order (number of negative eigenvalues), then apply GAD.

**The algorithm:**
1. Project forces into vibrational subspace (mass-weighted Eckart projection)
2. Use projected forces as gradients for L-BFGS
3. Check eigenvalue count periodically
4. Stop when target saddle order (e.g., 1 negative eigenvalue) is reached

### Code Snippet

```python
class LBFGSEnergyMinimizer:
    """Energy minimizer using SciPy L-BFGS-B with vibrational projection."""

    def __init__(self, predict_fn, atomic_nums, *,
                 target_neg_eig_count=1, max_iterations=200):
        self.predict_fn = predict_fn
        self.atomic_nums = atomic_nums
        self.target_neg_eig_count = target_neg_eig_count
        self.max_iterations = max_iterations

    def _project_forces(self, coords, forces_raw):
        """Project forces into vibrational subspace (remove TR modes)."""
        # Build mass-weighted vibrational projector P
        P_mw = self._build_vibrational_projector(coords)

        # F_cart_proj = M^{1/2} * P * M^{-1/2} * F_cart
        f_mw = forces_raw / sqrt(masses)
        f_mw_proj = P_mw @ f_mw
        f_proj = f_mw_proj * sqrt(masses)
        return f_proj

    def _objective_and_grad(self, x_flat):
        """Return (energy, projected_gradient) for L-BFGS."""
        coords = x_flat.reshape(-1, 3)
        out = self.predict_fn(coords, do_hessian=False)

        forces_proj = self._project_forces(coords, out["forces"])
        grad = -forces_proj.flatten()  # gradient = -force

        return float(out["energy"]), grad

    def _callback(self, xk):
        """Check eigenvalues after each iteration."""
        coords = xk.reshape(-1, 3)
        out = self.predict_fn(coords, do_hessian=True)

        vib_eigvals = vibrational_eigvals(out["hessian"], coords)
        neg_count = (vib_eigvals < 0).sum()

        if neg_count <= self.target_neg_eig_count:
            raise EarlyStop()  # Target reached!

    def minimize(self, initial_coords):
        """Run L-BFGS minimization."""
        result = scipy.optimize.minimize(
            self._objective_and_grad,
            x0=initial_coords.flatten(),
            method="L-BFGS-B",
            jac=True,
            callback=self._callback,
            options={"maxiter": self.max_iterations}
        )
        return result.x.reshape(-1, 3)
```

---

## 2. Plateau Detection (Kicking)

### Explanation

During GAD optimization, the algorithm can get "stuck" at high-index saddle points where the step size becomes very small but the saddle order doesn't improve. Plateau detection monitors the number of negative eigenvalues and adaptively adjusts the step size.

**The algorithm:**
1. Track `best_neg_vib` (lowest saddle order seen)
2. If saddle order improves: reset step size to base `dt`
3. If saddle order worsens: shrink step size (more careful steps)
4. If saddle order unchanged for `patience` steps: boost step size (kick!)
5. Enforce a minimum step size floor to prevent getting stuck

### Code Snippet

```python
def run_gad_with_plateau_detection(
    predict_fn, coords, atomic_nums, *,
    n_steps, dt, dt_min, dt_max,
    plateau_patience=20,    # Steps before boosting
    plateau_boost=1.5,      # Multiply dt by this when stuck
    plateau_shrink=0.5,     # Multiply dt by this when regressing
):
    best_neg_vib = None
    no_improve = 0
    dt_eff = dt

    for step in range(n_steps):
        # Get energy, forces, Hessian
        out = predict_fn(coords, do_hessian=True)

        # Count negative vibrational eigenvalues
        vib_eigvals = vibrational_eigvals(out["hessian"], coords)
        neg_vib = (vib_eigvals < 0).sum()

        # === PLATEAU DETECTION ===
        if best_neg_vib is None:
            best_neg_vib = neg_vib
            no_improve = 0
        else:
            if neg_vib < best_neg_vib:
                # Improvement! Reset to base step size
                best_neg_vib = neg_vib
                no_improve = 0
                dt_eff = min(dt_eff, dt)
            elif neg_vib > best_neg_vib:
                # Regression - shrink step size
                dt_eff = max(dt_eff * plateau_shrink, dt_min)
                no_improve = 0
            else:
                # No change
                no_improve += 1

        # If stuck for too long, KICK (boost step size)
        if no_improve >= plateau_patience:
            dt_eff = min(dt_eff * plateau_boost, dt_max)
            no_improve = 0

        # Clamp to valid range
        dt_eff = np.clip(dt_eff, dt_min, dt_max)

        # Take GAD step
        gad_vec = compute_gad_vector(out["forces"], out["hessian"])
        coords = coords + dt_eff * gad_vec

    return coords
```

---

## 3. Higher-Order GAD (Multi-Mode Escape)

### Explanation

Standard GAD inverts forces along the lowest eigenvector (v1), driving the system toward a first-order saddle point. However, when starting from noisy geometries with many negative eigenvalues, GAD can converge to high-index saddles (order > 1) and get stuck.

**Key insight:** When stuck at a high-index saddle, the lowest eigenmode has converged but other negative modes remain. The solution is to perturb along the *second* eigenvector (v2) to escape, then resume GAD.

**The algorithm:**
1. Run GAD until displacement becomes tiny (plateau detected)
2. If converged to order-1 saddle: SUCCESS!
3. If stuck at order > 1: perturb along v2 (escape perturbation)
4. Resume GAD from perturbed geometry
5. Repeat until order-1 or max cycles reached

### Code Snippet

```python
def perform_escape_perturbation(predict_fn, coords, hessian, *,
                                 escape_delta=0.1):
    """Perturb geometry along v2 to escape high-index saddle."""

    # Get projected Hessian (Eckart + mass-weighted)
    hess_proj = get_projected_hessian(hessian, coords)

    # Eigendecomposition
    evals, evecs = torch.linalg.eigh(hess_proj)

    # Skip translation/rotation modes (near-zero eigenvalues)
    vib_mask = torch.abs(evals) > 1e-6
    vib_indices = torch.where(vib_mask)[0]

    # Get second vibrational eigenvector (v2)
    v2 = evecs[:, vib_indices[1]]
    v2 = v2 / v2.norm()
    lambda2 = evals[vib_indices[1]]

    # Adaptive delta based on curvature
    delta = escape_delta
    if lambda2 < -0.01:
        delta = escape_delta / sqrt(abs(lambda2))
        delta = min(delta, 1.0)  # Cap at 1 Angstrom

    # Try both directions, pick lower energy
    coords_plus = coords + delta * v2.reshape(-1, 3)
    coords_minus = coords - delta * v2.reshape(-1, 3)

    E_plus = predict_fn(coords_plus)["energy"]
    E_minus = predict_fn(coords_minus)["energy"]

    return coords_plus if E_plus < E_minus else coords_minus


def run_multi_mode_escape(predict_fn, coords, atomic_nums, *,
                          n_steps, escape_window=20,
                          escape_disp_threshold=5e-4,
                          max_escape_cycles=1000):
    """GAD with multi-mode escape mechanism."""

    disp_history = []
    escape_cycle = 0

    while escape_cycle < max_escape_cycles:
        # Run GAD step
        out = predict_fn(coords, do_hessian=True)
        gad_vec = compute_gad_vector(out["forces"], out["hessian"])

        # Track displacement
        disp = (coords_new - coords).norm(dim=1).mean()
        disp_history.append(disp)

        # Check for plateau (tiny displacements + stable neg_vib)
        if len(disp_history) >= escape_window:
            mean_disp = np.mean(disp_history[-escape_window:])
            neg_vib = count_negative_eigenvalues(out["hessian"])

            if mean_disp < escape_disp_threshold and neg_vib > 1:
                # STUCK at high-index saddle! Escape via v2
                coords = perform_escape_perturbation(
                    predict_fn, coords, out["hessian"]
                )
                disp_history.clear()
                escape_cycle += 1
                continue

        # Check for success (order-1 saddle)
        if eig_product < 0:  # lambda0 < 0 and lambda1 > 0
            return coords  # SUCCESS!

        # Normal GAD step
        coords = coords + dt * gad_vec

    return coords
```

---

## 4. Mode Tracking + Trust Radius

### Explanation

Two critical improvements that enable robust convergence:

**Mode Tracking:** When eigenvalues are close or degenerate, the ordering of eigenvectors from `torch.linalg.eigh` can swap between steps, causing oscillations. Mode tracking selects the eigenvector with maximum overlap with the previous step's direction, ensuring continuity.

**Trust Radius:** Limits the maximum displacement per step to prevent "explosions" where the geometry moves too far and enters unphysical regions. This is especially important after escape perturbations.

### Code Snippet: Mode Tracking

```python
def pick_tracked_mode(evecs, v_prev, *, k=8):
    """Pick eigenmode with maximum overlap to previous direction.

    Args:
        evecs: Eigenvectors from eigh, shape (3N, 3N)
        v_prev: Previous tracked mode (3N,) or None
        k: Search among lowest k eigenvectors

    Returns:
        v: Selected eigenvector (3N,)
        j: Index of selected mode (0 to k-1)
        overlap: |dot(v_prev, v)| (1.0 if v_prev is None)
    """
    if v_prev is None:
        # First step: use lowest eigenvector
        v = evecs[:, 0]
        return v / v.norm(), 0, 1.0

    # Consider lowest k eigenvectors
    V = evecs[:, :k]  # Shape: (3N, k)

    # Compute overlaps with previous direction
    overlaps = torch.abs(V.T @ v_prev)  # Shape: (k,)

    # Pick mode with maximum overlap
    j = torch.argmax(overlaps)
    v = V[:, j]
    overlap = overlaps[j]

    # Sign continuity: ensure dot(v, v_prev) >= 0
    if torch.dot(v, v_prev) < 0:
        v = -v

    return v / v.norm(), int(j), float(overlap)


def compute_gad_vector_tracked(forces, hessian, v_prev, *, k_track=8):
    """Compute GAD direction with mode tracking."""

    # Eigendecomposition
    evals, evecs = torch.linalg.eigh(hessian)

    # Pick stable mode (tracking)
    v, mode_idx, overlap = pick_tracked_mode(evecs, v_prev, k=k_track)

    # GAD formula: F_gad = F + 2*(F . v)*v
    f = forces.flatten()
    gad = f + 2.0 * torch.dot(-f, v) * v

    return gad.reshape(-1, 3), v, {"overlap": overlap, "mode_idx": mode_idx}
```

### Code Snippet: Trust Radius (Max Displacement Cap)

```python
def apply_trust_radius(gad_vec, dt, *, max_atom_disp=0.25):
    """Limit step size so no atom moves more than max_atom_disp.

    Args:
        gad_vec: GAD direction (N, 3)
        dt: Proposed step size
        max_atom_disp: Maximum allowed displacement per atom (Angstrom)

    Returns:
        dt_safe: Adjusted step size that respects trust radius
    """
    # Compute displacement that would result from this step
    step = dt * gad_vec

    # Per-atom displacement magnitudes
    atom_displacements = step.norm(dim=1)  # Shape: (N,)
    max_disp = atom_displacements.max()

    # Scale down if necessary
    if max_disp > max_atom_disp:
        dt_safe = dt * (max_atom_disp / max_disp)
    else:
        dt_safe = dt

    return dt_safe


# Usage in GAD loop:
def gad_step_with_trust_radius(coords, gad_vec, dt, max_atom_disp=0.25):
    """Take GAD step with trust radius enforcement."""

    dt_safe = apply_trust_radius(gad_vec, dt, max_atom_disp=max_atom_disp)

    new_coords = coords + dt_safe * gad_vec

    return new_coords, dt_safe
```

### Combined Effect

The combination of mode tracking and trust radius transforms the multi-mode GAD algorithm from 40% to 92% convergence (SCINE) / 71% (HIP):

1. **Mode tracking** eliminates oscillations caused by eigenvector reordering
2. **Trust radius** prevents geometry explosions after escape perturbations
3. Together, they enable stable traversal through the high-dimensional saddle landscape

```python
# Full GAD step with all improvements
def gad_step_robust(predict_fn, coords, atomic_nums, dt, v_prev, *,
                    max_atom_disp=0.25, k_track=8):
    """Single GAD step with mode tracking + trust radius."""

    out = predict_fn(coords, do_hessian=True)

    # 1. Compute GAD with mode tracking
    gad_vec, v_next, info = compute_gad_vector_tracked(
        out["forces"], out["hessian"], v_prev, k_track=k_track
    )

    # 2. Apply trust radius
    dt_safe = apply_trust_radius(gad_vec, dt, max_atom_disp=max_atom_disp)

    # 3. Take step
    new_coords = coords + dt_safe * gad_vec

    return new_coords, v_next, dt_safe, info
```

---

## Summary Table

| Method | Key Innovation | SCINE | HIP |
|--------|---------------|-------|-----|
| Pure GAD | Baseline (from TS geometry) | 100% | ~90% |
| L-BFGS | Energy minimization first | 10% | 0% |
| Plateau Detection | Adaptive step size boosting | 40% | 40% |
| Higher-Order GAD | v2 escape perturbation | 40% | 40% |
| + Mode Tracking + Trust Radius | Stable eigenvector tracking | **92%** | **71%** |

