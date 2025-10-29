# src/gad_projected_gradient_descent.py
import os
import json
import argparse
from typing import Any, Dict, List
from collections import defaultdict

import torch
import numpy as np
import scipy.constants as spc
from torch_geometric.data import Data as TGData, Batch as TGBatch

from .common_utils import setup_experiment, add_common_args
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.masses import MASS_DICT
from nets.prediction_utils import Z_TO_ATOM_SYMBOL # Assuming this is in your project path

# --- Helper functions for RMSD (unchanged) ---
def find_rigid_alignment(A, B):
    if isinstance(A, torch.Tensor): A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor): B = B.detach().cpu().numpy()
    a_mean = A.mean(axis=0); b_mean = B.mean(axis=0)
    A_c = A - a_mean; B_c = B - b_mean
    H = A_c.T @ B_c
    U, _, Vt = np.linalg.svd(H); V = Vt.T
    R = V @ U.T
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    t = b_mean - R @ a_mean
    return R, t

def get_rmsd(A, B): return float(np.sqrt(((A - B) ** 2).sum(axis=1).mean()))

def align_ordered_and_get_rmsd(A, B):
    if A.shape != B.shape: return float("inf")
    R, t = find_rigid_alignment(A, B)
    A_aligned = (R @ A.T).T + t
    return get_rmsd(A_aligned, B)

# --- YOUR PROVIDED DIFFERENTIABLE PROJECTION FUNCTIONS ---
def _to_torch_double(array_like, device=None):
    if isinstance(array_like, torch.Tensor):
        return array_like.to(dtype=torch.float64, device=device)
    return torch.as_tensor(array_like, dtype=torch.float64, device=device)

def inertia_tensor_torch(coords3d, masses):
    coords3d_t = _to_torch_double(coords3d); masses_t = _to_torch_double(masses)
    x, y, z = coords3d_t.T
    squares = torch.sum(coords3d_t**2 * masses_t[:, None], dim=0)
    I_xx, I_yy, I_zz = squares[1] + squares[2], squares[0] + squares[2], squares[0] + squares[1]
    I_xy = -torch.sum(masses_t * x * y); I_xz = -torch.sum(masses_t * x * z); I_yz = -torch.sum(masses_t * y * z)
    return torch.stack([torch.stack([I_xx, I_xy, I_xz]), torch.stack([I_xy, I_yy, I_yz]), torch.stack([I_xz, I_yz, I_zz])])

def get_trans_rot_vectors_torch(cart_coords, masses):
    cart_coords_t = _to_torch_double(cart_coords); masses_t = _to_torch_double(masses)
    coords3d = cart_coords_t.reshape(-1, 3)
    com = (coords3d * masses_t[:, None]).sum(dim=0) / torch.sum(masses_t)
    coords3d_centered = coords3d - com[None, :]
    _, Iv = torch.linalg.eigh(inertia_tensor_torch(coords3d_centered, masses_t)); Iv = Iv.T
    masses_rep = masses_t.repeat_interleave(3); sqrt_masses = torch.sqrt(masses_rep)
    trans_vecs = []
    for vec in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        v = sqrt_masses * _to_torch_double(vec, device=cart_coords_t.device).repeat(masses_t.numel())
        trans_vecs.append(v / torch.linalg.norm(v))
    rot_vecs = torch.zeros((3, cart_coords_t.numel()), dtype=torch.float64, device=cart_coords_t.device)
    for i in range(masses_t.size(0)):
        p_vec = Iv @ coords3d_centered[i]
        for ix in range(3):
            rot_vecs[0, 3 * i + ix] = Iv[2, ix] * p_vec[1] - Iv[1, ix] * p_vec[2]
            rot_vecs[1, 3 * i + ix] = Iv[2, ix] * p_vec[0] - Iv[0, ix] * p_vec[2]
            rot_vecs[2, 3 * i + ix] = Iv[0, ix] * p_vec[1] - Iv[1, ix] * p_vec[0]
    rot_vecs = rot_vecs * sqrt_masses[None, :]
    rot_vecs = rot_vecs[torch.linalg.norm(rot_vecs, dim=1) > 1e-6]
    tr_vecs = torch.cat([torch.stack(trans_vecs), rot_vecs], dim=0)
    Q, _ = torch.linalg.qr(tr_vecs.T)
    return Q.T

def get_trans_rot_projector_torch(cart_coords, masses, full=False):
    tr_vecs = get_trans_rot_vectors_torch(cart_coords, masses=masses)
    if full:
        n = tr_vecs.size(1)
        P = torch.eye(n, dtype=tr_vecs.dtype, device=tr_vecs.device)
        for tr_vec in tr_vecs: P = P - torch.outer(tr_vec, tr_vec)
        return P
    else:
        U, S, _ = torch.linalg.svd(tr_vecs.T, full_matrices=True)
        return U[:, S.numel() :].T

def mass_weigh_hessian_torch(hessian, masses3d):
    h_t = _to_torch_double(hessian, device=hessian.device)
    m_t = _to_torch_double(masses3d, device=hessian.device)
    mm_sqrt_inv = torch.diag(1.0 / torch.sqrt(m_t))
    return mm_sqrt_inv @ h_t @ mm_sqrt_inv

def massweigh_and_eckartprojection_torch(
    hessian: torch.Tensor, cart_coords: torch.Tensor, atomsymbols: list[str]
):
    """ Your new, differentiable Eckart projection function. """
    masses_t = _to_torch_double([MASS_DICT[atom.lower()] for atom in atomsymbols], device=hessian.device)
    masses3d_t = masses_t.repeat_interleave(3)
    mw_hessian_t = mass_weigh_hessian_torch(hessian, masses3d_t)
    P_t = get_trans_rot_projector_torch(cart_coords, masses=masses_t, full=False)
    proj_hessian_t = P_t @ mw_hessian_t @ P_t.T
    return (proj_hessian_t + proj_hessian_t.T) / 2.0

# --- Your core logic, implemented as a function ---
def get_gradient_of_projected_eigprod(
    potential: torch.nn.Module, coords: torch.Tensor, atomic_nums: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    batch = coord_atoms_to_torch_geometric(coords, atomic_nums).to(potential.device)
    with torch.enable_grad():
        batch.pos.requires_grad_(True)
        _, _, out = potential.forward(batch, otf_graph=True)
        hess_raw = out["hessian"].reshape(len(atomic_nums)*3, len(atomic_nums)*3)
        
        atomsymbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in atomic_nums]
        
        # Use the NEW function inside the gradient calculation
        projected_hessian = massweigh_and_eckartprojection_torch(hess_raw, batch.pos, atomsymbols)
        
        eigvals, _ = torch.linalg.eigh(projected_hessian)
        prod = eigvals[0] * eigvals[1]
        
        # Use YOUR specified gradient calculation method
        grad = torch.autograd.grad(prod, batch.pos)[0]
        
    return prod.detach(), grad

# --- The optimization loop built around YOUR logic ---
def run_projected_gradient_descent(
    calculator: EquiformerTorchCalculator, initial_coords: torch.Tensor, atomic_nums: torch.Tensor,
    n_steps: int, lr: float
) -> Dict[str, Any]:
    potential = calculator.potential
    coords = initial_coords.clone().to(potential.device)
    history = defaultdict(list)

    for step in range(n_steps):
        eig_prod, grad = get_gradient_of_projected_eigprod(potential, coords, atomic_nums)
        
        history["eig_product"].append(eig_prod.item())
        if step % 10 == 0:
            print(f"  Step {step:03d}: Projected Eig Product = {eig_prod.item():.5f}")

        with torch.no_grad():
            coords -= lr * grad
            
    # Final analysis
    final_coords = coords.clone()
    with torch.no_grad():
        final_batch = coord_atoms_to_torch_geometric(final_coords, atomic_nums).to(potential.device)
        _, _, final_out = potential.forward(final_batch, otf_graph=True)
        hess_final = final_out["hessian"].reshape(len(atomic_nums)*3, len(atomic_nums)*3)
        atomsymbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in atomic_nums]
        final_proj_hess = massweigh_and_eckartprojection_torch(hess_final, final_coords, atomsymbols)
        final_eigvals, _ = torch.linalg.eigh(final_proj_hess)
        final_neg_num = torch.sum(final_eigvals < -1e-6).item()

    return {
        "final_coords": final_coords.cpu(), "history": history,
        "final_eig_product": (final_eigvals[0] * final_eigvals[1]).item(),
        "final_neg_eigvals": final_neg_num,
    }

# --- Main script execution block ---
if __name__ == "__main__":
    # ... (argparse and data loading are standard)
    parser = argparse.ArgumentParser(description="Run DIRECT gradient descent on PROJECTED Hessian eigenvalues.")
    parser = add_common_args(parser)
    parser.add_argument("--n-steps-opt", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--start-from", type=str, default="reactant", 
                        choices=["reactant", "ts", "midpoint_rt", "three_quarter_rt"])
    args = parser.parse_args()
    
    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    
    print(f"Running DIRECT Projected Eigenvalue Gradient Descent.")
    print(f"Starting From: {args.start_from.upper()}, Steps: {args.n_steps_opt}, LR: {args.lr}")

    results_summary = []
    for i, batch in enumerate(dataloader):
        if i >= args.max_samples: break
        print(f"\n--- Processing Sample {i} (Formula: {batch.formula[0]}) ---")
        try:
            if args.start_from == "reactant": initial_coords = batch.pos_reactant
            elif args.start_from == "midpoint_rt": initial_coords = 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition
            elif args.start_from == "three_quarter_rt": initial_coords = 0.25 * batch.pos_reactant + 0.75 * batch.pos_transition
            else: initial_coords = batch.pos_transition

            opt_results = run_projected_gradient_descent(
                calculator=calculator, initial_coords=initial_coords, atomic_nums=batch.z,
                n_steps=args.n_steps_opt, lr=args.lr,
            )
            
            # Initial analysis for comparison
            with torch.no_grad():
                initial_batch = coord_atoms_to_torch_geometric(initial_coords, batch.z).to(device)
                _, _, initial_out = calculator.potential.forward(initial_batch, otf_graph=True)
                hess_initial = initial_out["hessian"].reshape(len(batch.z)*3, len(batch.z)*3)
                atomsymbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in batch.z]
                initial_proj_hess = massweigh_and_eckartprojection_torch(hess_initial, initial_coords, atomsymbols)
                initial_eigvals, _ = torch.linalg.eigh(initial_proj_hess)
                initial_neg_num = torch.sum(initial_eigvals < -1e-6).item()

            summary = {
                "initial_eig_product": (initial_eigvals[0] * initial_eigvals[1]).item(),
                "final_eig_product": opt_results["final_eig_product"],
                "initial_neg_eigvals": initial_neg_num,
                "final_neg_eigvals": opt_results["final_neg_eigvals"],
            }
            results_summary.append(summary)
            
            print("Result:")
            print(f"  Projected Eig Product: {summary['initial_eig_product']:.4f} -> {summary['final_eig_product']:.4f}")
            print(f"  Projected Neg Eigs: {summary['initial_neg_eigvals']} -> {summary['final_neg_eigvals']}")
            
        except Exception as e:
            print(f"[ERROR] Sample {i} failed: {e}")
            import traceback
            traceback.print_exc()

    out_json = os.path.join(out_dir, f"direct_proj_grad_descent_{args.start_from}_{len(results_summary)}.json")
    with open(out_json, "w") as f: json.dump(results_summary, f, indent=2)
    print(f"\nSaved summary to {out_json}")