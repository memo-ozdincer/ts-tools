# src/gad_rk45_search.py
import os
import json
import re
import argparse
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import numpy as np
from torch_geometric.data import Data as TGData, Batch as TGBatch

from .common_utils import setup_experiment, add_common_args, parse_starting_geometry
from hip.frequency_analysis import analyze_frequencies_torch
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from .experiment_logger import ExperimentLogger, RunResult, build_loss_type_flags

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- RK45 Solver Class (Modified to add max_steps) ---
class RK45:
    def __init__(self, f, t0, y0, t1, rtol=1e-6, atol=1e-9, h0=None, h_max=np.inf, safety=0.9, event_fn=None, max_steps=10000):
        self.f = f
        self.t = float(t0); self.t_end = float(t1)
        self.y = np.array(y0, dtype=float)
        self.rtol = rtol; self.atol = atol; self.h_max = h_max; self.safety = safety
        self.event_fn = event_fn
        self.max_steps = max_steps
        self.step_count = 0
        self.direction = np.sign(self.t_end - self.t) if self.t_end != self.t else 1.0
        if h0 is None:
            f0 = np.asarray(self.f(self.t, self.y, record_stats=False))
            scale = self.atol + np.abs(self.y) * self.rtol
            d0, d1 = np.linalg.norm(self.y / scale), np.linalg.norm(f0 / scale)
            h0 = 0.01 * d0 / d1 if d0 > 1e-5 and d1 > 1e-5 else 1e-6
        self.h = self.direction * min(abs(h0), self.h_max)
        self.c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1], dtype=float)
        self.a = [[], [1/5], [3/40, 9/40], [44/45, -56/15, 32/9], [19372/6561, -25360/2187, 64448/6561, -212/729], [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656], [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]]
        self.b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0], dtype=float)
        self.b4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], dtype=float)

    def step(self):
        t, y, h = self.t, self.y, self.h
        h = min(h, self.t_end - t) if self.direction > 0 else max(h, self.t_end - t)
        if abs(h) < 1e-15: return None
        k = []
        for i in range(7):
            yi = y.copy() + h * sum(a_ij * k[j] for j, a_ij in enumerate(self.a[i]))
            k.append(np.asarray(self.f(t + self.c[i] * h, yi, record_stats=(i==0))))
        k = np.array(k)
        y5 = y + h * np.tensordot(self.b5, k, axes=(0,0))
        y4 = y + h * np.tensordot(self.b4, k, axes=(0,0))
        err = y5 - y4
        scale = self.atol + np.maximum(np.abs(y), np.abs(y5)) * self.rtol
        err_norm = np.linalg.norm(err / scale) / np.sqrt(err.size)
        if err_norm <= 1.0:
            self.t, self.y = t + h, y5
            self.step_count += 1
            factor = self.safety * (1.0 / err_norm)**(1/5) if err_norm > 0 else 2.0
            self.h *= np.clip(factor, 0.2, 5.0)
            if self.event_fn is not None and self.event_fn(): return "event"
            return "accepted"
        else:
            self.h *= np.clip(self.safety * (1.0 / err_norm)**(1/5), 0.2, 5.0)
            return "rejected"

    def solve(self):
        while self.direction * (self.t - self.t_end) < 0:
            if self.step_count >= self.max_steps:
                print(f"[WARNING] Max steps ({self.max_steps}) reached. Stopping integration.")
                break
            status = self.step()
            if status == "event" or status is None: break

# --- Helper Functions (Unchanged) ---
def _prepare_hessian(hess, num_atoms): return hess.reshape(3*num_atoms, 3*num_atoms) if hess.dim()==1 else hess
def align_ordered_and_get_rmsd(A, B):
    if isinstance(A, np.ndarray): A = torch.from_numpy(A)
    if isinstance(B, np.ndarray): B = torch.from_numpy(B)
    A, B = A.float(), B.float()
    a_mean, b_mean = A.mean(dim=0), B.mean(dim=0)
    A_c, B_c = A - a_mean, B - b_mean
    H = A_c.T @ B_c
    U, _, Vt = torch.linalg.svd(H); R = Vt.T @ U.T
    if torch.linalg.det(R) < 0: Vt[-1, :] *= -1; R = Vt.T @ U.T
    t = b_mean - R @ a_mean
    A_aligned = (R @ A.T).T + t
    return torch.sqrt(((A_aligned - B) ** 2).sum(dim=1).mean()).item()
def _sanitize_formula(f): return re.sub(r"[^A-Za-z0-9_.-]+", "_", f).strip("_") or "sample"

def _extract_vibrational_eigvals(eigvals: torch.Tensor, coords: torch.Tensor) -> Tuple[Optional[torch.Tensor], int]:
    """
    Remove rigid-body zero modes from eigenvalue spectrum based on molecular geometry.
    Returns (vibrational_eigvals, rigid_modes_removed).
    """
    if not isinstance(eigvals, torch.Tensor) or eigvals.numel() == 0:
        return None, 0

    eigvals = eigvals.detach().to(torch.float64).flatten()
    coords3d = coords.detach().reshape(-1, 3).to(torch.float64)
    coords3d = coords3d - coords3d.mean(dim=0, keepdim=True)

    # Linear molecules have rank <= 2 → 5 rigid modes; otherwise 6
    geom_rank = torch.linalg.matrix_rank(coords3d.cpu(), tol=1e-8).item()
    rigid_modes = 5 if geom_rank <= 2 else 6

    total_modes = eigvals.numel()
    rigid_modes = min(rigid_modes, max(0, total_modes - 2))
    if rigid_modes == 0:
        vib_sorted, _ = torch.sort(eigvals)
        return vib_sorted, 0

    abs_sorted_idx = torch.argsort(torch.abs(eigvals))
    keep_idx = abs_sorted_idx[rigid_modes:]
    if keep_idx.numel() == 0:
        return None, rigid_modes

    keep_idx, _ = torch.sort(keep_idx)
    vibrational = eigvals[keep_idx]
    vibrational_sorted, _ = torch.sort(vibrational)
    return vibrational_sorted, rigid_modes

# --- MODIFIED Core Dynamics Function (With Optional Kick Mechanism) ---
class GADDynamics:
    def __init__(self, calculator, atomic_nums, stop_at_ts=False, kick_enabled=False,
                 kick_force_threshold=0.015, kick_magnitude=0.1,
                 min_eig_product_threshold=-1e-4, confirmation_steps=10,
                 stagnation_check_window=20, stagnation_variance_threshold=1e-6,
                 eigenvector_following=False, eigvec_follow_delay=5):
        """
        Args:
            kick_enabled: Enable kick mechanism to escape local minima
            kick_force_threshold: Force threshold in eV/Å below which kick is considered (default: 0.015)
            kick_magnitude: Magnitude of kick in Å (default: 0.1)
            min_eig_product_threshold: Minimum eigenvalue product to consider as TS candidate (default: -1e-4)
            confirmation_steps: Number of steps to wait for confirmation after TS candidate found (default: 10)
            stagnation_check_window: Window size for checking stagnation (default: 20)
            stagnation_variance_threshold: Variance threshold for detecting stagnation (default: 1e-6)
            eigenvector_following: Enable eigenvector-following refinement after TS candidate found (default: False)
            eigvec_follow_delay: Steps after TS candidate before switching to eigenvector following (default: 5)
        """
        self.calculator = calculator
        self.atomic_nums = torch.tensor(atomic_nums, dtype=torch.long)
        self.num_atoms = len(atomic_nums)
        self.stop_at_ts = stop_at_ts
        self.kick_enabled = kick_enabled
        self.kick_force_threshold = kick_force_threshold
        self.kick_magnitude = kick_magnitude
        self.min_eig_product_threshold = min_eig_product_threshold
        self.confirmation_steps = confirmation_steps
        self.stagnation_check_window = stagnation_check_window
        self.stagnation_variance_threshold = stagnation_variance_threshold
        self.eigenvector_following = eigenvector_following
        self.eigvec_follow_delay = eigvec_follow_delay
        self.device = next(calculator.potential.parameters()).device
        self.trajectory = defaultdict(list)
        self.ts_found = False
        self.ts_candidate_step = None
        self.eigvec_follow_active = False
        self.num_kicks = 0

    def __call__(self, t, y, record_stats=True):
        coords = torch.from_numpy(y).float().reshape(self.num_atoms, 3).to(self.device)
        batch = TGBatch.from_data_list([TGData(pos=coords, z=self.atomic_nums, natoms=torch.tensor([self.num_atoms]))]).to(self.device)
        results = self.calculator.predict(batch, do_hessian=True)
        forces = results["forces"]

        # Calculate eigenvalues and eigenvectors
        hessian = _prepare_hessian(results["hessian"], self.num_atoms)
        freq_info = analyze_frequencies_torch(results["hessian"], coords, self.atomic_nums)
        eigvals_raw = freq_info.get("eigvals")
        vibrational_eigvals, rigid_modes_removed = _extract_vibrational_eigvals(eigvals_raw, coords)

        eigvals_for_tracking: Optional[torch.Tensor] = None
        using_vibrational = False
        if vibrational_eigvals is not None and vibrational_eigvals.numel() >= 2:
            eigvals_for_tracking = vibrational_eigvals
            using_vibrational = True
        elif isinstance(eigvals_raw, torch.Tensor) and eigvals_raw.numel() >= 2:
            eigvals_for_tracking, _ = torch.sort(eigvals_raw.detach().to(torch.float64).flatten())
            using_vibrational = False

        eig0 = eig1 = eig_prod = None
        neg_vibrational = None
        if eigvals_for_tracking is not None and eigvals_for_tracking.numel() >= 2:
            eig0 = float(eigvals_for_tracking[0].item())
            eig1 = float(eigvals_for_tracking[1].item())
            eig_prod = eig0 * eig1
            if using_vibrational:
                neg_vibrational = int((eigvals_for_tracking < 0).sum().item())

        # Get unprojected eigenvalues for full Hessian (for GAD direction)
        evals_full, evecs_full = torch.linalg.eigh(hessian)

        # Extract eigenvalue information
        force_magnitude = forces.norm(dim=1).mean().item()

        # Check if we should apply a kick (escape local minimum or stagnation)
        apply_kick = False
        kick_reason = ""

        if self.kick_enabled and eig_prod is not None:
            # Calculate stagnation metrics if we have enough history
            eig_prod_variance = None
            if len(self.trajectory["eig_product"]) >= self.stagnation_check_window:
                recent_eig_products = [ep for ep in self.trajectory["eig_product"][-self.stagnation_check_window:] if ep is not None]
                if len(recent_eig_products) >= self.stagnation_check_window:
                    eig_prod_variance = np.var(recent_eig_products)

            # KICK Criterion 1: Stagnation (low variance) AND not near TS
            if (eig_prod_variance is not None and
                eig_prod_variance < self.stagnation_variance_threshold and
                eig_prod > self.min_eig_product_threshold):
                apply_kick = True
                kick_reason = f"Stagnation (var={eig_prod_variance:.2e}, eig_prod={eig_prod:.4e})"

            # KICK Criterion 2: Force small AND not near TS (stuck in flat region)
            elif (force_magnitude < self.kick_force_threshold and eig_prod > 0):
                apply_kick = True
                kick_reason = f"Low force in non-TS region (|F|={force_magnitude:.4f}, eig_prod={eig_prod:.4e})"

            if apply_kick:
                self.num_kicks += 1
                if record_stats:
                    print(f"  [KICK {self.num_kicks}] {kick_reason}")

        if apply_kick:
            # Apply kick: random perturbation scaled by kick_magnitude
            kick_direction = torch.randn_like(coords)
            kick_direction = kick_direction / (kick_direction.norm() + 1e-12)
            velocity = kick_direction * self.kick_magnitude
        else:
            # Check if we should switch to eigenvector-following mode
            current_step = len(self.trajectory["time"]) + 1
            if (self.eigenvector_following and
                self.ts_candidate_step is not None and
                current_step >= self.ts_candidate_step + self.eigvec_follow_delay and
                not self.eigvec_follow_active):
                self.eigvec_follow_active = True
                if record_stats:
                    print(f"  [MODE SWITCH] Activating eigenvector-following refinement at step {current_step}")

            if self.eigvec_follow_active:
                # Eigenvector-following mode: maximize uphill motion along lowest eigenvector
                # while minimizing motion in all other directions
                # Get projected eigenvectors (from frequency analysis)
                hessian_proj = _prepare_hessian(freq_info["hessian_proj"], self.num_atoms)
                _, evecs_proj = torch.linalg.eigh(hessian_proj)
                min_mode = evecs_proj[:, 0].to(forces.dtype)  # Lowest eigenvector (most negative)
                min_mode = min_mode.reshape(self.num_atoms, 3)
                min_mode /= (torch.norm(min_mode) + 1e-12)

                # Project forces onto the minimum mode
                f_flat = forces.reshape(-1)
                min_mode_flat = min_mode.reshape(-1)
                projection_onto_mode = torch.dot(f_flat, min_mode_flat)

                # Move uphill along minimum mode (reverse force projection), downhill in all others
                # velocity = -F + 2*(F·v)v  (ascent along v, descent in orthogonal space)
                velocity_flat = -f_flat + 2.0 * projection_onto_mode * min_mode_flat
                velocity = velocity_flat.reshape(self.num_atoms, 3)
            else:
                # Standard GAD velocity
                v = evecs_full[:, 0].to(forces.dtype)
                v /= (torch.norm(v) + 1e-12)
                f_flat = forces.reshape(-1)
                # Equation: x_dot = F + 2(-F.v)v (since F = -grad(V))
                gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
                velocity = gad_flat.reshape(self.num_atoms, 3)

        # Record trajectory statistics
        if record_stats:
            self.trajectory["time"].append(t)
            self.trajectory["energy"].append(results["energy"].item())
            self.trajectory["force_mean"].append(forces.norm(dim=1).mean().item())
            self.trajectory["gad_mean"].append(velocity.norm(dim=1).mean().item())
            self.trajectory["eig_product"].append(eig_prod)
            self.trajectory["eig0"].append(eig0)  # Track individual eigenvalues
            self.trajectory["eig1"].append(eig1)
            self.trajectory["neg_num"].append(freq_info.get("neg_num", -1))
            self.trajectory["neg_vibrational"].append(neg_vibrational)
            self.trajectory["rigid_modes_removed"].append(rigid_modes_removed)
            self.trajectory["using_vibrational"].append(1 if using_vibrational else 0)
            self.trajectory["kick_applied"].append(1 if apply_kick else 0)
            self.trajectory["eigvec_follow_mode"].append(1 if self.eigvec_follow_active else 0)

            # Multi-stage early stopping logic
            if self.stop_at_ts and eig_prod is not None:
                current_step = len(self.trajectory["time"])

                # Stage 1: Check if eigenvalue product is meaningfully negative
                if eig_prod < self.min_eig_product_threshold:
                    if self.ts_candidate_step is None:
                        self.ts_candidate_step = current_step
                        eig0_str = f"{eig0:.6f}" if eig0 is not None else "nan"
                        eig1_str = f"{eig1:.6f}" if eig1 is not None else "nan"
                        print(f"  [TS CANDIDATE] Found at step {current_step}, t={t:.4f}: λ₀*λ₁={eig_prod:.6e} (λ₀={eig0_str}, λ₁={eig1_str})")

                    # Stage 2: Confirm TS by checking if eigenvalue product magnitude is increasing
                    if current_step >= self.ts_candidate_step + self.confirmation_steps:
                        candidate_eig_prod = self.trajectory["eig_product"][self.ts_candidate_step - 1]
                        if candidate_eig_prod is not None:
                            magnitude_increase = abs(eig_prod) / (abs(candidate_eig_prod) + 1e-20)
                        else:
                            magnitude_increase = None

                        if magnitude_increase is not None and magnitude_increase > 1.2:  # 20% increase in magnitude
                            self.ts_found = True
                            print(f"  [TS CONFIRMED] at step {current_step}: λ₀*λ₁={eig_prod:.6e} (increased by {magnitude_increase:.2f}x)")
                        else:
                            # Check if λ₀ is becoming MORE negative
                            candidate_eig0 = self.trajectory["eig0"][self.ts_candidate_step - 1]
                            if (eig0 is not None and candidate_eig0 is not None and
                                    eig0 < candidate_eig0 * 1.1):  # λ₀ became 10% more negative
                                self.ts_found = True
                                print(f"  [TS CONFIRMED] at step {current_step}: λ₀={eig0:.6f} (more negative than {candidate_eig0:.6f})")
                else:
                    # Reset candidate if we drift back to positive or less negative
                    if self.ts_candidate_step is not None and current_step < self.ts_candidate_step + self.confirmation_steps:
                        print(f"  [TS CANDIDATE RESET] Drifted back: λ₀*λ₁={eig_prod:.6e}")
                    self.ts_candidate_step = None

        return velocity.cpu().numpy().flatten()

    def event_function(self):
        return self.ts_found

# --- Plotting ---
def plot_trajectory(
    trajectory,
    sample_index,
    formula,
    start_from,
    initial_neg_num,
    final_neg_num,
    initial_neg_vibrational,
    final_neg_vibrational,
):
    """
    Plot GAD trajectory.

    Returns:
        Tuple of (matplotlib Figure, suggested filename)
    """
    timesteps = np.array(trajectory.get("time", []))
    def _nanify(key): return np.array([v if v is not None else np.nan for v in trajectory.get(key, [])], dtype=float)

    # Create 5 subplots to include individual eigenvalues
    fig, axes = plt.subplots(5, 1, figsize=(8, 15), sharex=True)
    fig.suptitle(f"GAD Trajectory for Sample {sample_index}: {formula}", fontsize=14)

    # Energy
    axes[0].plot(timesteps, _nanify("energy"), marker=".", lw=1.2)
    axes[0].set_ylabel("Energy (eV)")
    axes[0].set_title("Energy")

    # Force magnitude
    axes[1].plot(timesteps, _nanify("force_mean"), marker=".", color="tab:orange", lw=1.2)
    axes[1].set_ylabel("Mean |F| (eV/Å)")
    axes[1].set_title("Force Magnitude")

    # Individual eigenvalues
    eig0, eig1 = _nanify("eig0"), _nanify("eig1")
    axes[2].plot(timesteps, eig0, marker=".", color="tab:red", lw=1.2, label="λ₀ (most negative)")
    axes[2].plot(timesteps, eig1, marker=".", color="tab:blue", lw=1.2, label="λ₁ (2nd smallest)")
    axes[2].axhline(0, color='grey', ls='--', lw=1)
    axes[2].set_ylabel("Eigenvalue (eV/Å²)")
    axes[2].set_title("Individual Projected Eigenvalues")
    axes[2].legend(loc='best')
    if len(eig0) > 0 and not np.isnan(eig0[-1]):
        axes[2].text(0.98, 0.05, f"Final λ₀={eig0[-1]:.6f}", transform=axes[2].transAxes, ha='right', va='bottom', fontsize=9)

    # Eigenvalue product
    eig_prod = _nanify("eig_product")
    axes[3].plot(timesteps, eig_prod, marker=".", color="tab:purple", lw=1.2, label="λ₀ * λ₁")
    axes[3].axhline(0, color='grey', ls='--', lw=1)
    axes[3].axhline(-1e-4, color='red', ls=':', lw=1, label="TS threshold")
    if len(eig_prod) > 0 and not np.isnan(eig_prod[0]):
        axes[3].text(0.02, 0.95, f"Start: {eig_prod[0]:.4e}", transform=axes[3].transAxes, ha='left', va='top', c='purple', fontsize=9)
    if len(eig_prod) > 0 and not np.isnan(eig_prod[-1]):
        axes[3].text(0.98, 0.95, f"End: {eig_prod[-1]:.4e}", transform=axes[3].transAxes, ha='right', va='top', c='purple', fontsize=9)
    axes[3].set_ylabel("Eigenvalue Product")
    axes[3].set_title("Product of Two Smallest Projected Eigenvalues")
    axes[3].legend(loc='best')

    # GAD magnitude
    axes[4].plot(timesteps, _nanify("gad_mean"), marker=".", color="tab:green", lw=1.2, label="Mean |GAD|")
    axes[4].set_ylabel("Mean |GAD| (Å)")
    axes[4].set_xlabel("Time (a.u.)")
    axes[4].set_title("GAD Vector Magnitude")

    neg_vib = _nanify("neg_vibrational")
    if neg_vib.size > 0 and not np.all(np.isnan(neg_vib)):
        ax4b = axes[4].twinx()
        ax4b.step(timesteps, neg_vib, where="post", color="tab:red", lw=1.2, label="# neg vib eig")
        ax4b.set_ylabel("# Neg Vibrational")
        valid_counts = neg_vib[~np.isnan(neg_vib)]
        if valid_counts.size > 0:
            ax4b.set_ylim(-0.5, max(valid_counts.max() + 0.5, 1.5))
        handles, labels = axes[4].get_legend_handles_labels()
        h2, l2 = ax4b.get_legend_handles_labels()
        axes[4].legend(handles + h2, labels + l2, loc='best', fontsize=8)
    else:
        axes[4].legend(loc='best', fontsize=8)

    axes[4].text(
        0.98, 0.05,
        (
            f"neg eig (freq): {initial_neg_num} → {final_neg_num}\n"
            f"neg eig (vib): {initial_neg_vibrational} → {final_neg_vibrational}"
        ),
        transform=axes[4].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    # Add visual indicators for mode switches
    if "eigvec_follow_mode" in trajectory:
        eigvec_mode = np.array(trajectory["eigvec_follow_mode"])
        if np.any(eigvec_mode > 0):
            # Find where mode switches to eigenvector-following
            mode_switch_idx = np.where(np.diff(eigvec_mode) > 0)[0]
            if len(mode_switch_idx) > 0:
                switch_time = timesteps[mode_switch_idx[0] + 1]
                for ax in axes:
                    ax.axvline(switch_time, color='orange', ls=':', lw=2, alpha=0.7, label='Eigvec Follow')
                axes[0].legend(loc='best', fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f"traj_{sample_index:03d}_{_sanitize_formula(formula)}_from_{start_from}_{initial_neg_num}to{final_neg_num}.png"
    return fig, filename

# --- Main script execution block (With Kick args) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GAD-RK45 search for transition states.")
    parser = add_common_args(parser)
    parser.add_argument("--t-end", type=float, default=2.0, help="Total integration time for the solver.")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum number of integration steps.")
    parser.add_argument("--start-from", type=str, default="reactant",
                        help="Starting geometry: 'reactant', 'ts', 'midpoint_rt', 'three_quarter_rt', "
                             "or add noise: 'reactant_noise0.5A', 'reactant_noise1A', 'reactant_noise2A', 'reactant_noise10A', etc.")
    parser.add_argument("--stop-at-ts", action="store_true", help="Stop simulation as soon as a TS is found.")

    # Kick mechanism arguments
    parser.add_argument("--enable-kick", action="store_true", help="Enable kick mechanism to escape local minima.")
    parser.add_argument("--kick-force-threshold", type=float, default=0.015,
                        help="Force threshold in eV/Å below which kick is considered (default: 0.015).")
    parser.add_argument("--kick-magnitude", type=float, default=0.1,
                        help="Magnitude of kick displacement in Å (default: 0.1).")

    # Multi-stage early stopping arguments
    parser.add_argument("--min-eig-product-threshold", type=float, default=-1e-4,
                        help="Minimum eigenvalue product to consider as TS candidate (default: -1e-4).")
    parser.add_argument("--confirmation-steps", type=int, default=10,
                        help="Number of steps to wait for TS confirmation (default: 10).")
    parser.add_argument("--stagnation-check-window", type=int, default=20,
                        help="Window size for stagnation detection (default: 20).")
    parser.add_argument("--stagnation-variance-threshold", type=float, default=1e-6,
                        help="Variance threshold for stagnation detection (default: 1e-6).")

    # Eigenvector-following refinement
    parser.add_argument("--eigenvector-following", action="store_true",
                        help="Enable eigenvector-following refinement after TS candidate found.")
    parser.add_argument("--eigvec-follow-delay", type=int, default=5,
                        help="Steps after TS candidate before switching to eigenvector following (default: 5).")

    # W&B arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="gad-ts-search",
                        help="W&B project name (default: gad-ts-search)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity/username (optional)")

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)

    # Set up experiment logger
    loss_type_flags = build_loss_type_flags(args)

    # Prepare W&B config
    wandb_config = {
        "script": "gad_rk45_search",
        "start_from": args.start_from,
        "t_end": args.t_end,
        "max_steps": args.max_steps,
        "stop_at_ts": args.stop_at_ts,
        "enable_kick": args.enable_kick,
        "kick_force_threshold": args.kick_force_threshold if args.enable_kick else None,
        "kick_magnitude": args.kick_magnitude if args.enable_kick else None,
        "eigenvector_following": args.eigenvector_following,
        "max_samples": args.max_samples,
    }

    logger = ExperimentLogger(
        base_dir=out_dir,
        script_name="gad-rk45",
        loss_type_flags=loss_type_flags,
        max_graphs_per_transition=10,
        random_seed=42,  # For reproducible sampling
        use_wandb=args.wandb,
        wandb_project=args.wandb_project if args.wandb else None,
        wandb_entity=args.wandb_entity,
        wandb_tags=[args.start_from, "rk45"],
        wandb_config=wandb_config,
    )

    print(f"Running GAD-RK45 Search. Start: {args.start_from.upper()}, Stop at TS: {args.stop_at_ts}")
    print(f"Output directory: {logger.run_dir}")
    print(f"Kick enabled: {args.enable_kick}")
    if args.enable_kick:
        print(f"  Force threshold: {args.kick_force_threshold} eV/Å, Kick magnitude: {args.kick_magnitude} Å")
    if args.stop_at_ts:
        print(f"Multi-stage TS detection:")
        print(f"  Min eigenvalue product: {args.min_eig_product_threshold:.2e}")
        print(f"  Confirmation steps: {args.confirmation_steps}")
    if args.eigenvector_following:
        print(f"Eigenvector-following refinement: ENABLED")
        print(f"  Activation delay: {args.eigvec_follow_delay} steps after TS candidate")
    for i, batch in enumerate(dataloader):
        if i >= args.max_samples: break
        print(f"\n--- Processing Sample {i} (Formula: {batch.formula[0]}) ---")
        try:
            # Use parse_starting_geometry to handle both standard and noisy starting points
            initial_coords = parse_starting_geometry(args.start_from, batch, noise_seed=42)

            dynamics_fn = GADDynamics(
                calculator, batch.z.numpy(),
                stop_at_ts=args.stop_at_ts,
                kick_enabled=args.enable_kick,
                kick_force_threshold=args.kick_force_threshold,
                kick_magnitude=args.kick_magnitude,
                min_eig_product_threshold=args.min_eig_product_threshold,
                confirmation_steps=args.confirmation_steps,
                stagnation_check_window=args.stagnation_check_window,
                stagnation_variance_threshold=args.stagnation_variance_threshold,
                eigenvector_following=args.eigenvector_following,
                eigvec_follow_delay=args.eigvec_follow_delay
            )
            solver = RK45(f=dynamics_fn, t0=0.0, y0=initial_coords.numpy().flatten(),
                         t1=args.t_end, h_max=0.1, event_fn=dynamics_fn.event_function,
                         max_steps=args.max_steps)
            
            dynamics_fn(0.0, initial_coords.numpy().flatten(), record_stats=True)
            solver.solve()
            
            final_coords = torch.from_numpy(solver.y).reshape(-1, 3)

            # Extract final, definitive results from trajectory
            traj = dynamics_fn.trajectory
            initial_neg_num = traj["neg_num"][0] if traj["neg_num"] else -1
            final_neg_num = traj["neg_num"][-1] if traj["neg_num"] else -1
            neg_vibrational_series = traj.get("neg_vibrational", [])
            def _first_defined(seq):
                for value in seq:
                    if value is not None and not (isinstance(value, float) and np.isnan(value)):
                        return value
                return None
            initial_neg_vibrational = _first_defined(neg_vibrational_series)
            final_neg_vibrational = _first_defined(reversed(neg_vibrational_series)) if neg_vibrational_series else None
            rigid_modes_series = [rm for rm in traj.get("rigid_modes_removed", []) if rm is not None]
            rigid_modes_removed = int(rigid_modes_series[0]) if rigid_modes_series else None

            # Compute steps_to_ts: use ts_candidate_step if TS was found
            steps_to_ts = dynamics_fn.ts_candidate_step if dynamics_fn.ts_found else None

            # Create RunResult
            result = RunResult(
                sample_index=i,
                formula=batch.formula[0],
                initial_neg_eigvals=initial_neg_num,
                final_neg_eigvals=final_neg_num,
                initial_neg_vibrational=initial_neg_vibrational,
                final_neg_vibrational=final_neg_vibrational,
                steps_taken=len(traj["time"]) - 1,
                steps_to_ts=steps_to_ts,
                final_time=solver.t,
                final_eig0=traj["eig0"][-1] if traj["eig0"] else None,
                final_eig1=traj["eig1"][-1] if traj["eig1"] else None,
                final_eig_product=traj["eig_product"][-1],
                final_loss=None,  # Not applicable for RK45
                rmsd_to_known_ts=None,  # Could compute if needed
                stop_reason="ts_confirmed" if dynamics_fn.ts_found else None,
                plot_path=None,  # Will be set below
                extra_data={
                    "initial_eig_product": traj["eig_product"][0],
                    "initial_eig0": traj["eig0"][0] if traj["eig0"] else None,
                    "initial_eig1": traj["eig1"][0] if traj["eig1"] else None,
                    "rigid_modes_removed": rigid_modes_removed,
                    "num_kicks": dynamics_fn.num_kicks,
                    "ts_candidate_step": dynamics_fn.ts_candidate_step,
                    "ts_confirmed": dynamics_fn.ts_found,
                    "eigvec_follow_activated": dynamics_fn.eigvec_follow_active,
                    "final_mean_force": traj["force_mean"][-1],
                    "final_mean_gad": traj["gad_mean"][-1],
                }
            )

            # Add result to logger
            logger.add_result(result)

            # Create plot
            fig, filename = plot_trajectory(
                trajectory=traj,
                sample_index=i,
                formula=batch.formula[0],
                start_from=args.start_from,
                initial_neg_num=initial_neg_num,
                final_neg_num=final_neg_num,
                initial_neg_vibrational=initial_neg_vibrational,
                final_neg_vibrational=final_neg_vibrational,
            )

            # Save plot using logger (handles sampling)
            plot_path = logger.save_graph(result, fig, filename)
            if plot_path:
                result.plot_path = plot_path
                print(f"  Saved plot to: {plot_path}")
            else:
                print(f"  Skipped plot (max samples for {result.transition_key} reached)")
            plt.close(fig)

            print("Result:")
            print(f"  Transition: {result.transition_key}")
            print(f"  Solver Steps: {result.steps_taken}, Final Time: {result.final_time:.3f}")
            print(f"  Neg Eigs: {result.initial_neg_eigvals} -> {result.final_neg_eigvals}")
            if initial_neg_vibrational is not None or final_neg_vibrational is not None:
                print(f"  Neg Vibrational Eigs: {initial_neg_vibrational} -> {final_neg_vibrational}")
            if args.enable_kick:
                print(f"  Kicks applied: {dynamics_fn.num_kicks}")

        except Exception as e:
            print(f"[ERROR] Sample {i} failed: {e}"); import traceback; traceback.print_exc()

    # Save all results and aggregate statistics
    all_runs_path, aggregate_path = logger.save_all_results()
    print(f"\nSaved all runs to: {all_runs_path}")
    print(f"Saved aggregate stats to: {aggregate_path}")

    # Print summary
    logger.print_summary()

    # Finish W&B run
    logger.finish()
