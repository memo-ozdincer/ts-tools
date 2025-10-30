# src/gad_rk45_search.py
import os
import json
import re
import argparse
from typing import Any, Dict, List, Optional
from collections import defaultdict

import torch
import numpy as np
from torch_geometric.data import Data as TGData, Batch as TGBatch

from .common_utils import setup_experiment, add_common_args
from hip.frequency_analysis import analyze_frequencies_torch
from hip.equiformer_torch_calculator import EquiformerTorchCalculator

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

# --- MODIFIED Core Dynamics Function (With Optional Kick Mechanism) ---
class GADDynamics:
    def __init__(self, calculator, atomic_nums, stop_at_ts=False, kick_enabled=False,
                 kick_force_threshold=0.015, kick_magnitude=0.1):
        """
        Args:
            kick_enabled: Enable kick mechanism to escape local minima
            kick_force_threshold: Force threshold in eV/Å below which kick is considered (default: 0.015)
            kick_magnitude: Magnitude of kick in Å (default: 0.1)
        """
        self.calculator = calculator
        self.atomic_nums = torch.tensor(atomic_nums, dtype=torch.long)
        self.num_atoms = len(atomic_nums)
        self.stop_at_ts = stop_at_ts
        self.kick_enabled = kick_enabled
        self.kick_force_threshold = kick_force_threshold
        self.kick_magnitude = kick_magnitude
        self.device = next(calculator.potential.parameters()).device
        self.trajectory = defaultdict(list)
        self.ts_found = False
        self.num_kicks = 0

    def __call__(self, t, y, record_stats=True):
        coords = torch.from_numpy(y).float().reshape(self.num_atoms, 3).to(self.device)
        batch = TGBatch.from_data_list([TGData(pos=coords, z=self.atomic_nums, natoms=torch.tensor([self.num_atoms]))]).to(self.device)
        results = self.calculator.predict(batch, do_hessian=True)
        forces = results["forces"]

        # Calculate eigenvalues and eigenvectors
        hessian = _prepare_hessian(results["hessian"], self.num_atoms)
        freq_info = analyze_frequencies_torch(results["hessian"], coords, self.atomic_nums)
        eigvals = freq_info["eigvals"]  # These are the projected eigenvalues

        # Get unprojected eigenvalues for full Hessian (for GAD direction)
        evals_full, evecs_full = torch.linalg.eigh(hessian)

        # Check if we should apply a kick (escape local minimum)
        apply_kick = False
        if self.kick_enabled:
            force_magnitude = forces.norm(dim=1).mean().item()
            # Check if force is below threshold AND both smallest projected eigenvalues are positive (local min)
            if force_magnitude < self.kick_force_threshold and eigvals.numel() >= 2:
                eig0, eig1 = eigvals[0].item(), eigvals[1].item()
                if eig0 > 0 and eig1 > 0:  # Local minimum
                    apply_kick = True
                    self.num_kicks += 1

        if apply_kick:
            # Apply kick: random perturbation scaled by kick_magnitude
            kick_direction = torch.randn_like(coords)
            kick_direction = kick_direction / (kick_direction.norm() + 1e-12)
            velocity = kick_direction * self.kick_magnitude
            if record_stats:
                print(f"  [KICK {self.num_kicks}] at t={t:.3f}: |F|={force_magnitude:.4f} eV/Å, λ₀={eig0:.6f}, λ₁={eig1:.6f}")
        else:
            # Standard GAD velocity
            v = evecs_full[:, 0].to(forces.dtype)
            v /= (torch.norm(v) + 1e-12)
            f_flat = forces.reshape(-1)
            # Equation: x_dot = F + 2(-F.v)v (since F = -grad(V))
            gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
            velocity = gad_flat.reshape(self.num_atoms, 3)

        if record_stats:
            self.trajectory["time"].append(t)
            self.trajectory["energy"].append(results["energy"].item())
            self.trajectory["force_mean"].append(forces.norm(dim=1).mean().item())
            self.trajectory["gad_mean"].append(velocity.norm(dim=1).mean().item())
            eig_prod = (eigvals[0] * eigvals[1]).item() if eigvals.numel() >= 2 else None
            self.trajectory["eig_product"].append(eig_prod)
            self.trajectory["neg_num"].append(freq_info.get("neg_num", -1))
            self.trajectory["kick_applied"].append(1 if apply_kick else 0)
            if self.stop_at_ts and eig_prod is not None and eig_prod < -1e-5:
                self.ts_found = True

        return velocity.cpu().numpy().flatten()

    def event_function(self):
        return self.ts_found

# --- Plotting (Unchanged) ---
def plot_trajectory(trajectory, sample_index, formula, out_dir):
    # ... (code is unchanged)
    timesteps = np.array(trajectory.get("time", []))
    def _nanify(key): return np.array([v if v is not None else np.nan for v in trajectory.get(key, [])], dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(f"GAD Trajectory for Sample {sample_index}: {formula}", fontsize=14)
    axes[0].plot(timesteps, _nanify("energy"), marker=".", lw=1.2); axes[0].set_ylabel("Energy (eV)"); axes[0].set_title("Energy")
    axes[1].plot(timesteps, _nanify("force_mean"), marker=".", color="tab:orange", lw=1.2); axes[1].set_ylabel("Mean |F| (eV/Å)"); axes[1].set_title("Force Magnitude")
    eig_ax, eig_prod = axes[2], _nanify("eig_product")
    eig_ax.plot(timesteps, eig_prod, marker=".", color="tab:purple", lw=1.2, label="λ_0 * λ_1"); eig_ax.axhline(0, color='grey', ls='--', lw=1)
    if len(eig_prod) > 0 and not np.isnan(eig_prod[0]): eig_ax.text(0.02, 0.95, f"Start: {eig_prod[0]:.4f}", transform=eig_ax.transAxes, ha='left', va='top', c='purple', fontsize=9)
    if len(eig_prod) > 0 and not np.isnan(eig_prod[-1]): eig_ax.text(0.98, 0.95, f"End: {eig_prod[-1]:.4f}", transform=eig_ax.transAxes, ha='right', va='top', c='purple', fontsize=9)
    eig_ax.set_ylabel("Eigenvalue Product"); eig_ax.set_title("Product of Two Smallest Hessian Eigenvalues"); eig_ax.legend(loc='best')
    axes[3].plot(timesteps, _nanify("gad_mean"), marker=".", color="tab:green", lw=1.2); axes[3].set_ylabel("Mean |GAD| (Å)"); axes[3].set_xlabel("Time (a.u.)"); axes[3].set_title("GAD Vector Magnitude")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f"rgd1_gad_traj_{sample_index:03d}_{_sanitize_formula(formula)}.png"
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=200); plt.close(fig)
    return out_path

# --- Main script execution block (With Kick args) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GAD-RK45 search for transition states.")
    parser = add_common_args(parser)
    parser.add_argument("--t-end", type=float, default=2.0, help="Total integration time for the solver.")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum number of integration steps.")
    parser.add_argument("--start-from", type=str, default="reactant", choices=["ts", "reactant", "midpoint_rt", "three_quarter_rt"])
    parser.add_argument("--stop-at-ts", action="store_true", help="Stop simulation as soon as a TS is found.")

    # Kick mechanism arguments
    parser.add_argument("--enable-kick", action="store_true", help="Enable kick mechanism to escape local minima.")
    parser.add_argument("--kick-force-threshold", type=float, default=0.015,
                        help="Force threshold in eV/Å below which kick is considered (default: 0.015).")
    parser.add_argument("--kick-magnitude", type=float, default=0.1,
                        help="Magnitude of kick displacement in Å (default: 0.1).")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)

    print(f"Running GAD-RK45 Search. Start: {args.start_from.upper()}, Stop at TS: {args.stop_at_ts}")
    print(f"Kick enabled: {args.enable_kick}")
    if args.enable_kick:
        print(f"  Force threshold: {args.kick_force_threshold} eV/Å, Kick magnitude: {args.kick_magnitude} Å")

    results_summary = []
    for i, batch in enumerate(dataloader):
        if i >= args.max_samples: break
        print(f"\n--- Processing Sample {i} (Formula: {batch.formula[0]}) ---")
        try:
            if args.start_from == "reactant": initial_coords = batch.pos_reactant
            elif args.start_from == "midpoint_rt": initial_coords = 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition
            elif args.start_from == "three_quarter_rt": initial_coords = 0.25 * batch.pos_reactant + 0.75 * batch.pos_transition
            else: initial_coords = batch.pos_transition

            dynamics_fn = GADDynamics(
                calculator, batch.z.numpy(),
                stop_at_ts=args.stop_at_ts,
                kick_enabled=args.enable_kick,
                kick_force_threshold=args.kick_force_threshold,
                kick_magnitude=args.kick_magnitude
            )
            solver = RK45(f=dynamics_fn, t0=0.0, y0=initial_coords.numpy().flatten(),
                         t1=args.t_end, h_max=0.1, event_fn=dynamics_fn.event_function,
                         max_steps=args.max_steps)
            
            dynamics_fn(0.0, initial_coords.numpy().flatten(), record_stats=True)
            solver.solve()
            
            final_coords = torch.from_numpy(solver.y).reshape(-1, 3)
            
            summary = {
                "sample_index": i, "formula": batch.formula[0],
                "steps_taken": len(dynamics_fn.trajectory["time"]) - 1,
                "final_time": solver.t,
                "initial_eig_product": dynamics_fn.trajectory["eig_product"][0],
                "final_eig_product": dynamics_fn.trajectory["eig_product"][-1],
                "initial_neg_eigvals": dynamics_fn.trajectory["neg_num"][0],
                "final_neg_eigvals": dynamics_fn.trajectory["neg_num"][-1],
                "rmsd_to_start": align_ordered_and_get_rmsd(initial_coords, final_coords),
                "num_kicks": dynamics_fn.num_kicks,
            }
            results_summary.append(summary)

            plot_path = plot_trajectory(dynamics_fn.trajectory, i, batch.formula[0], out_dir)

            print("Result:")
            print(f"  Solver Steps: {summary['steps_taken']}, Final Time: {summary['final_time']:.3f}")
            print(f"  Neg Eigs: {summary['initial_neg_eigvals']} -> {summary['final_neg_eigvals']}")
            if args.enable_kick:
                print(f"  Kicks applied: {dynamics_fn.num_kicks}")

        except Exception as e:
            print(f"[ERROR] Sample {i} failed: {e}"); import traceback; traceback.print_exc()

    kick_suffix = "_kick" if args.enable_kick else ""
    out_json = os.path.join(out_dir, f"gad_rk45_{args.start_from}{'_stopts' if args.stop_at_ts else ''}{kick_suffix}_{len(results_summary)}.json")
    with open(out_json, "w") as f: json.dump(results_summary, f, indent=2)
    print(f"\nSaved summary to {out_json}")