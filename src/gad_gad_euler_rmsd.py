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

class RK45:
    """
    Adaptive Dormand–Prince 5(4) solver for y' = f(t, y).
    """
    def __init__(self, f, t0, y0, t1,
                 rtol=1e-6, atol=1e-9,
                 h0=None,
                 h_min=1e-12,
                 h_max=np.inf,
                 safety=0.9):
        self.f = f
        self.t = float(t0)
        self.t_end = float(t1)
        self.y = np.array(y0, dtype=float)
        self.rtol = rtol
        self.atol = atol
        self.h_min = h_min
        self.h_max = h_max
        self.safety = safety
        self.direction = np.sign(self.t_end - self.t) if self.t_end != self.t else 1.0
        if h0 is None:
            f0 = np.asarray(self.f(self.t, self.y))
            scale = self.atol + np.abs(self.y) * self.rtol
            denom = np.maximum(scale, 1e-14)
            h_guess = 0.01 * np.linalg.norm(self.y / denom) / (np.linalg.norm(f0 / denom) + 1e-14)
            if not np.isfinite(h_guess) or h_guess == 0.0:
                h_guess = 1e-3
            self.h = self.direction * min(abs(h_guess), self.h_max)
        else:
            self.h = self.direction * min(abs(h0), self.h_max)
        self.c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1], dtype=float)
        self.a = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84]
        ]
        self.b5 = np.array([35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0], dtype=float)
        self.b4 = np.array([5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], dtype=float)
        self.err_exponent = 1.0 / 5.0

    def step(self):
        t, y, h = self.t, self.y, self.h
        if self.direction > 0: h = min(h, self.t_end - t)
        else: h = max(h, self.t_end - t)
        if h == 0: return True, 0.0
        k = []
        for i in range(7):
            yi = y.copy()
            for j, a_ij in enumerate(self.a[i]):
                yi += h * a_ij * k[j]
            k.append(np.asarray(self.f(t + self.c[i] * h, yi)))
        k = np.array(k)
        y5 = y + h * np.tensordot(self.b5, k, axes=(0, 0))
        y4 = y + h * np.tensordot(self.b4, k, axes=(0, 0))
        err = y5 - y4
        scale = self.atol + np.maximum(np.abs(y), np.abs(y5)) * self.rtol
        err_norm = np.linalg.norm(err / scale) / np.sqrt(err.size)
        if err_norm <= 1.0:
            self.t, self.y = t + h, y5
            factor = 2.0 if err_norm == 0.0 else self.safety * (1.0 / err_norm) ** self.err_exponent
            h_new = np.clip(factor, 0.2, 5.0) * h
            if self.direction > 0: h_new = min(abs(h_new), self.h_max) * self.direction
            else: h_new = min(abs(h_new), self.h_max) * self.direction
            self.h = h_new
            return True, h_new
        else:
            factor = np.clip(factor, 0.2, 5.0) * self.safety * (1.0 / err_norm) ** self.err_exponent
            h_new = factor * h
            if abs(h_new) < self.h_min: raise RuntimeError("Step size underflow.")
            self.h = h_new
            return False, h_new

    def solve(self, dense=False):
        if dense: T, Y = [self.t], [self.y.copy()]
        while True:
            if self.direction > 0 and self.t >= self.t_end: break
            if self.direction < 0 and self.t <= self.t_end: break
            accepted, _ = self.step()
            if accepted and dense: T.append(self.t); Y.append(self.y.copy())
        return (np.array(T), np.array(Y)) if dense else (self.t, self.y)

# --- Helper functions for geometry and analysis ---
def _prepare_hessian(hess: torch.Tensor, num_atoms: int) -> torch.Tensor:
    if hess.dim() == 1: hess = hess.reshape(3 * num_atoms, 3 * num_atoms)
    elif hess.dim() == 3 and hess.shape[0] == 1: hess = hess[0]
    return hess

def _extract_eig_product(freq_info: Dict[str, Any]) -> Optional[float]:
    eigvals = freq_info.get("eigvals")
    if eigvals is None or not isinstance(eigvals, torch.Tensor) or eigvals.numel() < 2: return None
    return (eigvals[0] * eigvals[1]).item()

def align_ordered_and_get_rmsd(A, B):
    # ... (unchanged)
    if isinstance(A, np.ndarray): A = torch.from_numpy(A)
    if isinstance(B, np.ndarray): B = torch.from_numpy(B)
    A, B = A.float(), B.float()
    a_mean, b_mean = A.mean(dim=0), B.mean(dim=0)
    A_c, B_c = A - a_mean, B - b_mean
    H = A_c.T @ B_c
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    if torch.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = b_mean - R @ a_mean
    A_aligned = (R @ A.T).T + t
    return torch.sqrt(((A_aligned - B) ** 2).sum(dim=1).mean()).item()

# --- The Core Dynamics Function for the RK45 Solver ---
class GADDynamics:
    def __init__(self, calculator, atomic_nums, force_thresh, kick_scale):
        self.calculator = calculator
        self.atomic_nums = torch.tensor(atomic_nums, dtype=torch.long)
        self.num_atoms = len(atomic_nums)
        self.force_thresh = force_thresh
        self.kick_scale = kick_scale
        self.device = next(calculator.potential.parameters()).device

    def __call__(self, t, y):
        """ This is the f(t, y) for the ODE solver. y is the flattened coordinates. """
        coords = torch.from_numpy(y).float().reshape(self.num_atoms, 3).to(self.device)
        
        # Prepare batch and get predictions
        batch = TGBatch.from_data_list([TGData(pos=coords, z=self.atomic_nums, natoms=torch.tensor([self.num_atoms]))]).to(self.device)
        results = self.calculator.predict(batch, do_hessian=True)
        forces = results["forces"]
        
        # Check if we are at a minimum
        rms_force = torch.norm(forces) / np.sqrt(self.num_atoms)
        if rms_force < self.force_thresh:
            print(f"  (Low force detected: RMS|F|={rms_force:.4f}. Applying eigenvector kick.)")
            hessian = _prepare_hessian(results["hessian"], self.num_atoms)
            _, evecs = torch.linalg.eigh(hessian)
            v_kick = evecs[:, 0].to(forces.dtype)
            v_kick /= (torch.norm(v_kick) + 1e-12)
            velocity = self.kick_scale * v_kick
        else:
            # Standard GAD velocity
            hessian = _prepare_hessian(results["hessian"], self.num_atoms)
            _, evecs = torch.linalg.eigh(hessian)
            v = evecs[:, 0].to(forces.dtype)
            v /= (torch.norm(v) + 1e-12)
            
            f_flat = forces.reshape(-1)
            # The equation from the image: x_dot = F - 2(F.v)v, since F = -grad(V)
            gad_flat = f_flat - 2.0 * torch.dot(f_flat, v) * v
            velocity = gad_flat.reshape(self.num_atoms, 3)
            
        return velocity.cpu().numpy().flatten()

# --- Main script execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GAD-RK45 search for transition states.")
    parser = add_common_args(parser)
    parser.add_argument("--t-end", type=float, default=1.0, help="Total integration time for the solver.")
    parser.add_argument("--force-kick-thresh", type=float, default=0.05, help="RMS force (eV/Å) below which to apply the eigenvector kick.")
    parser.add_argument("--kick-scale", type=float, default=0.2, help="Scaling factor for the eigenvector kick velocity.")
    parser.add_argument("--start-from", type=str, default="reactant", choices=["ts", "reactant", "midpoint_rt", "three_quarter_rt"])
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    
    print("Running GAD-RK45 Search")
    print(f"Starting from: {args.start_from.upper()}, Integration Time: {args.t_end}, Kick Threshold: {args.force_kick_thresh}")

    results_summary = []
    for i, batch in enumerate(dataloader):
        if i >= args.max_samples: break
        print(f"\n--- Processing Sample {i} (Formula: {batch.formula[0]}) ---")
        try:
            # Select starting coordinates
            if args.start_from == "reactant": initial_coords = batch.pos_reactant
            elif args.start_from == "midpoint_rt": initial_coords = 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition
            elif args.start_from == "three_quarter_rt": initial_coords = 0.25 * batch.pos_reactant + 0.75 * batch.pos_transition
            else: initial_coords = batch.pos_transition

            # Set up the dynamics function for this molecule
            dynamics_fn = GADDynamics(
                calculator=calculator,
                atomic_nums=batch.z.numpy(),
                force_thresh=args.force_kick_thresh,
                kick_scale=args.kick_scale
            )

            # Set up and run the RK45 solver
            solver = RK45(
                f=dynamics_fn,
                t0=0.0,
                y0=initial_coords.numpy().flatten(),
                t1=args.t_end,
                h_max=0.1 # Limit max step to avoid jumping too far
            )
            T, Y_traj = solver.solve(dense=True)
            final_coords = torch.from_numpy(Y_traj[-1]).reshape(-1, 3).to(device)

            # Analyze initial and final states
            initial_results = calculator.predict(TGBatch.from_data_list([TGData(pos=initial_coords.to(device), z=batch.z, natoms=batch.natoms)]).to(device), do_hessian=True)
            initial_freq = analyze_frequencies_torch(initial_results["hessian"], initial_coords, batch.z)
            final_results = calculator.predict(TGBatch.from_data_list([TGData(pos=final_coords, z=batch.z, natoms=batch.natoms)]).to(device), do_hessian=True)
            final_freq = analyze_frequencies_torch(final_results["hessian"], final_coords, batch.z)

            summary = {
                "sample_index": i, "formula": batch.formula[0],
                "steps_taken": len(T) - 1,
                "initial_eig_product": _extract_eig_product(initial_freq),
                "final_eig_product": _extract_eig_product(final_freq),
                "initial_neg_eigvals": initial_freq["neg_num"],
                "final_neg_eigvals": final_freq["neg_num"],
                "rmsd_to_start": align_ordered_and_get_rmsd(initial_coords, final_coords),
            }
            results_summary.append(summary)
            
            print("Result:")
            print(f"  Solver Steps: {summary['steps_taken']}")
            print(f"  Eig Product: {summary['initial_eig_product']:.5f} -> {summary['final_eig_product']:.5f}")
            print(f"  Neg Eigs: {summary['initial_neg_eigvals']} -> {summary['final_neg_eigvals']}")

        except Exception as e:
            print(f"[ERROR] Sample {i} failed: {e}")
            import traceback
            traceback.print_exc()

    out_json = os.path.join(out_dir, f"gad_rk45_{args.start_from}_{len(results_summary)}.json")
    with open(out_json, "w") as f: json.dump(results_summary, f, indent=2)
    print(f"\nSaved summary to {out_json}")