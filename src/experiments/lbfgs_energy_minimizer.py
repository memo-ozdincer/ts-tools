from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
import torch
try:
    from scipy.optimize import minimize as scipy_minimize
except Exception:  # pragma: no cover
    scipy_minimize = None

from ..dependencies.common_utils import add_common_args, parse_starting_geometry, setup_experiment
from ..dependencies.experiment_logger import (
    ExperimentLogger,
    RunResult,
    build_loss_type_flags,
)
from ..dependencies.hessian import (
    get_scine_elements_from_predict_output,
    vibrational_eigvals,
)
from ..logging import finish_wandb, init_wandb_run, log_sample, log_summary
from ..runners._predict import make_predict_fn_from_calculator


def _sanitize_wandb_name(s: str) -> str:
    s = str(s)
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s[:128] if len(s) > 128 else s


def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().reshape(-1)[0].item())
    return float(x)


def _force_mean(forces: torch.Tensor) -> float:
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    f = forces.reshape(-1, 3)
    return float(f.norm(dim=1).mean().item())


class _EarlyStop(Exception):
    pass


@dataclass
class _EvalDiagnostics:
    energy: float
    forces_raw: torch.Tensor  # (N,3)
    forces_proj: torch.Tensor  # (N,3)
    neg_vib: Optional[int] = None
    eig0: Optional[float] = None
    eig1: Optional[float] = None


class LBFGSEnergyMinimizer:
    """Energy minimizer using SciPy L-BFGS-B.

    Key property: gradients returned to the optimizer are *projected* to the
    vibrational subspace consistent with mass-weighting + Eckart projection.

    This is implemented for both HIP and SCINE calculators via the shared
    `predict_fn` interface and SCINE element extraction.
    """

    def __init__(
        self,
        predict_fn,
        atomic_nums: torch.Tensor,
        *,
        device: str,
        target_neg_eig_count: int = 1,
        max_iterations: int = 200,
        max_step: float = 0.5,
        eigenvalue_check_freq: int = 1,
        verbose: bool = True,
    ):
        self.predict_fn = predict_fn
        self.atomic_nums = atomic_nums.detach().clone().to(torch.long)
        self.device = str(device)

        self.target_neg_eig_count = int(target_neg_eig_count)
        self.max_iterations = int(max_iterations)
        self.max_step = float(max_step)
        self.eigenvalue_check_freq = int(max(1, eigenvalue_check_freq))
        self.verbose = bool(verbose)

        self._iteration = 0
        self._stop_x: Optional[Any] = None
        self._latest_diag: Optional[_EvalDiagnostics] = None

        self.trajectory: Dict[str, list] = defaultdict(list)

    def _device_atomic_nums(self) -> torch.Tensor:
        if self.device == "cpu":
            return self.atomic_nums.cpu()
        return self.atomic_nums.to(self.device)

    def _project_forces(
        self,
        *,
        coords: torch.Tensor,
        forces_raw: torch.Tensor,
        scine_elements: Optional[list],
    ) -> torch.Tensor:
        """Project Cartesian forces into vibrational subspace, then map back.

        We work in mass-weighted coordinates:
          F_mw = M^{-1/2} F_cart
          F_mw_proj = P_mw F_mw
          F_cart_proj = M^{1/2} F_mw_proj

        For HIP, P_mw comes from `eckartprojection_torch`.
        For SCINE, we use the SVD-based projector in `ScineFrequencyAnalyzer`.
        """

        coords3d = coords.reshape(-1, 3)
        forces3d = forces_raw
        if forces3d.dim() == 3 and forces3d.shape[0] == 1:
            forces3d = forces3d[0]
        forces3d = forces3d.reshape(-1, 3)

        if scine_elements is None:
            from hip.masses import MASS_DICT
            from hip.ff_lmdb import Z_TO_ATOM_SYMBOL

            atom_symbols = [Z_TO_ATOM_SYMBOL[int(z)] for z in self.atomic_nums.detach().cpu().tolist()]
            masses = torch.tensor(
                [MASS_DICT[s.lower()] for s in atom_symbols],
                dtype=torch.float64,
                device=coords3d.device,
            )
            masses3 = masses.repeat_interleave(3)

            P_mw = self._hip_vibrational_projector_mw(coords3d.to(dtype=torch.float64), masses)
            f = forces3d.reshape(-1).to(dtype=torch.float64)
            f_mw = f / torch.sqrt(masses3)
            f_mw_proj = P_mw @ f_mw
            f_proj = f_mw_proj * torch.sqrt(masses3)
            return f_proj.to(dtype=forces3d.dtype).reshape(-1, 3)

        # SCINE path: build full-space MW projector using the vibrational basis P (rows orthonormal)
        from ..dependencies.scine_masses import ScineFrequencyAnalyzer, get_scine_masses

        masses_np = get_scine_masses(scine_elements)
        masses = torch.tensor(masses_np, dtype=torch.float64, device=coords3d.device)
        masses3 = masses.repeat_interleave(3)

        coords_np = coords3d.detach().cpu().numpy().reshape(-1, 3)
        analyzer = ScineFrequencyAnalyzer()
        P_red = analyzer._get_vibrational_projector(coords_np, masses_np)  # (3N-k, 3N)
        P_full = P_red.T @ P_red  # (3N, 3N)

        P_mw = torch.from_numpy(P_full).to(device=coords3d.device, dtype=torch.float64)
        f = forces3d.reshape(-1).to(dtype=torch.float64)
        f_mw = f / torch.sqrt(masses3)
        f_mw_proj = P_mw @ f_mw
        f_proj = f_mw_proj * torch.sqrt(masses3)
        return f_proj.to(dtype=forces3d.dtype).reshape(-1, 3)

    def _hip_vibrational_projector_mw(
        self,
        coords3d: torch.Tensor,
        masses: torch.Tensor,
        *,
        rot_tol: float = 1e-6,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Build a mass-weighted vibrational projector that handles linear molecules.

        We construct translational + rotational generators in mass-weighted space,
        drop any near-zero rotation (linear molecules -> 2 rotations), then form:
          P = I - Q Q^T
        where Q is an orthonormal basis for the TR subspace.
        """

        xyz = coords3d.reshape(-1, 3)
        n = int(xyz.shape[0])
        if n <= 1:
            return torch.eye(3 * n, dtype=torch.float64, device=xyz.device)

        masses = masses.reshape(-1).to(dtype=torch.float64, device=xyz.device)
        sqrt_m = torch.sqrt(masses)
        sqrt_m3 = sqrt_m.repeat_interleave(3)

        # Center on COM
        total_mass = masses.sum()
        com = (xyz * masses[:, None]).sum(dim=0) / (total_mass + eps)
        r = xyz - com[None, :]

        # --- Translation generators (3) ---
        tcols = []
        for axis in range(3):
            v = torch.zeros((n, 3), dtype=torch.float64, device=xyz.device)
            v[:, axis] = 1.0
            col = (v.reshape(-1) * sqrt_m3)
            col = col / (col.norm() + eps)
            tcols.append(col)

        # --- Rotation generators (up to 3; drop near-zero for linear molecules) ---
        # Inertia tensor in AMU*Å^2 (mass units consistent with Hessian MW convention)
        rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
        Ixx = torch.sum(masses * (ry * ry + rz * rz))
        Iyy = torch.sum(masses * (rx * rx + rz * rz))
        Izz = torch.sum(masses * (rx * rx + ry * ry))
        Ixy = -torch.sum(masses * (rx * ry))
        Ixz = -torch.sum(masses * (rx * rz))
        Iyz = -torch.sum(masses * (ry * rz))
        inertia = torch.stack(
            [
                torch.stack([Ixx, Ixy, Ixz]),
                torch.stack([Ixy, Iyy, Iyz]),
                torch.stack([Ixz, Iyz, Izz]),
            ],
            dim=0,
        )

        # Principal axes
        _, axes = torch.linalg.eigh(inertia)

        rcols = []
        for axis in range(3):
            u = axes[:, axis]
            disp = torch.cross(r, u[None, :].expand_as(r), dim=1)  # (N,3)
            col = (disp * sqrt_m[:, None]).reshape(-1)
            norm = col.norm()
            if float(norm.item()) > float(rot_tol):
                col = col / (norm + eps)
                rcols.append(col)

        # Stack TR generators -> (3N, k)
        B = torch.stack(tcols + rcols, dim=1) if (tcols or rcols) else torch.zeros((3 * n, 0), dtype=torch.float64, device=xyz.device)

        # Orthonormalize TR subspace
        if B.shape[1] == 0:
            return torch.eye(3 * n, dtype=torch.float64, device=xyz.device)

        Q, _ = torch.linalg.qr(B, mode="reduced")
        P = torch.eye(Q.shape[0], dtype=torch.float64, device=xyz.device) - (Q @ Q.T)
        P = 0.5 * (P + P.T)
        return P

    def _eval_energy_forces(
        self,
        coords: torch.Tensor,
        *,
        do_hessian: bool,
    ) -> Tuple[Dict[str, Any], _EvalDiagnostics]:
        atomic_nums = self._device_atomic_nums()
        out = self.predict_fn(coords, atomic_nums, do_hessian=do_hessian, require_grad=False)

        energy = _to_float(out.get("energy"))
        forces_raw = out.get("forces")
        if not isinstance(forces_raw, torch.Tensor):
            raise RuntimeError("predict_fn output missing 'forces' tensor")

        scine_elements = get_scine_elements_from_predict_output(out)
        forces_proj = self._project_forces(coords=coords, forces_raw=forces_raw, scine_elements=scine_elements)

        diag = _EvalDiagnostics(
            energy=float(energy),
            forces_raw=forces_raw.detach(),
            forces_proj=forces_proj.detach(),
        )

        if do_hessian:
            hessian = out.get("hessian")
            if not isinstance(hessian, torch.Tensor):
                raise RuntimeError("predict_fn output missing 'hessian' tensor")
            vib = vibrational_eigvals(hessian, coords, atomic_nums, scine_elements=scine_elements)
            neg = int((vib < 0).sum().item()) if vib.numel() else -1
            diag.neg_vib = neg
            diag.eig0 = float(vib[0].item()) if vib.numel() >= 1 else None
            diag.eig1 = float(vib[1].item()) if vib.numel() >= 2 else None

        return out, diag

    def _objective_and_grad(self, x_flat: Any) -> Tuple[float, Any]:
        coords = torch.tensor(
            x_flat.reshape(-1, 3),
            dtype=torch.float32,
            device=torch.device(self.device),
        )

        _, diag = self._eval_energy_forces(coords, do_hessian=False)
        grad = (-diag.forces_proj.reshape(-1)).detach().cpu().numpy().astype(np.float64)
        self._latest_diag = diag
        return float(diag.energy), grad

    def _callback(self, xk: Any) -> None:
        self._iteration += 1

        coords = torch.tensor(
            np.asarray(xk, dtype=np.float64).reshape(-1, 3),
            dtype=torch.float32,
            device=torch.device(self.device),
        )

        do_hessian = (
            self._iteration % self.eigenvalue_check_freq == 0
            or self._iteration == 1
        )

        _, diag = self._eval_energy_forces(coords, do_hessian=do_hessian)

        self.trajectory["iteration"].append(int(self._iteration))
        self.trajectory["energy"].append(float(diag.energy))
        self.trajectory["force_mean_raw"].append(_force_mean(diag.forces_raw))
        self.trajectory["force_mean_proj"].append(_force_mean(diag.forces_proj))
        self.trajectory["neg_vib"].append(int(diag.neg_vib) if diag.neg_vib is not None else None)
        self.trajectory["eig0"].append(diag.eig0)
        self.trajectory["eig1"].append(diag.eig1)

        if self.verbose and (self._iteration == 1 or self._iteration % 10 == 0):
            neg_str = str(diag.neg_vib) if diag.neg_vib is not None else "?"
            print(
                f"    [L-BFGS] Iter {self._iteration:4d}: E={diag.energy:12.6f} eV, "
                f"|F|_mean={_force_mean(diag.forces_proj):.4f}, neg_vib={neg_str}"
            )

        if do_hessian and diag.neg_vib is not None and diag.neg_vib >= 0:
            if int(diag.neg_vib) <= int(self.target_neg_eig_count):
                self._stop_x = np.asarray(xk, dtype=np.float64).copy()
                raise _EarlyStop()

    def minimize(self, initial_coords: torch.Tensor) -> Dict[str, Any]:
        if np is None or scipy_minimize is None:
            raise ImportError(
                "This module requires numpy + scipy to run L-BFGS-B. "
                "Install them in your runtime environment (e.g., on the cluster)."
            )
        self._iteration = 0
        self._stop_x = None
        self._latest_diag = None
        self.trajectory = defaultdict(list)

        coords0 = initial_coords.detach().clone().reshape(-1, 3).to(torch.float32)
        coords0 = coords0.to(torch.device(self.device))

        # Initial diagnostics (includes Hessian + eigenvalues)
        _, init_diag = self._eval_energy_forces(coords0, do_hessian=True)

        x0 = coords0.detach().cpu().numpy().reshape(-1).astype(np.float64)

        if init_diag.neg_vib is not None and init_diag.neg_vib >= 0:
            if int(init_diag.neg_vib) <= int(self.target_neg_eig_count):
                return {
                    "initial_coords": coords0.detach().cpu(),
                    "final_coords": coords0.detach().cpu(),
                    "initial_energy": float(init_diag.energy),
                    "final_energy": float(init_diag.energy),
                    "initial_neg_vib": int(init_diag.neg_vib),
                    "final_neg_vib": int(init_diag.neg_vib),
                    "initial_eig0": init_diag.eig0,
                    "initial_eig1": init_diag.eig1,
                    "final_eig0": init_diag.eig0,
                    "final_eig1": init_diag.eig1,
                    "final_eig_product": (float(init_diag.eig0) * float(init_diag.eig1)) if (init_diag.eig0 is not None and init_diag.eig1 is not None) else None,
                    "n_iterations": 0,
                    "converged": True,
                    "stop_reason": "already_at_target",
                    "trajectory": dict(self.trajectory),
                }

        bounds = None
        if self.max_step is not None and self.max_step > 0:
            lower = x0 - float(self.max_step)
            upper = x0 + float(self.max_step)
            bounds = list(zip(lower.tolist(), upper.tolist()))

        converged = False
        stop_reason = "max_iterations"
        x_final = None

        if self.verbose:
            print("  [L-BFGS] Starting minimization:")
            print(f"    Initial energy: {init_diag.energy:.6f} eV")
            print(f"    Initial neg vibrational: {init_diag.neg_vib}")
            print(f"    Target: <= {self.target_neg_eig_count} negative vibrational eigenvalues")

        try:
            res = scipy_minimize(
                self._objective_and_grad,
                x0=x0,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                callback=self._callback,
                options={
                    "maxiter": int(self.max_iterations),
                    # Disable SciPy convergence criteria; we stop by eigenvalue criterion.
                    "gtol": 0.0,
                    "ftol": 0.0,
                },
            )
            x_final = res.x
        except _EarlyStop:
            converged = True
            stop_reason = "target_neg_vib"
            x_final = self._stop_x if self._stop_x is not None else x0
        except Exception as e:
            converged = False
            stop_reason = f"error:{type(e).__name__}"
            x_final = self._stop_x if self._stop_x is not None else x0
            if self.verbose:
                print(f"  [L-BFGS] Optimization error: {e}")

        if x_final is None:
            x_final = x0

        coords_final = torch.tensor(
            np.asarray(x_final, dtype=np.float64).reshape(-1, 3),
            dtype=torch.float32,
            device=torch.device(self.device),
        )

        # Final diagnostics (includes Hessian + eigenvalues)
        _, final_diag = self._eval_energy_forces(coords_final, do_hessian=True)

        if not converged and final_diag.neg_vib is not None and final_diag.neg_vib >= 0:
            if int(final_diag.neg_vib) <= int(self.target_neg_eig_count):
                converged = True
                stop_reason = "target_neg_vib"

        return {
            "initial_coords": coords0.detach().cpu(),
            "final_coords": coords_final.detach().cpu(),
            "initial_energy": float(init_diag.energy),
            "final_energy": float(final_diag.energy),
            "initial_neg_vib": int(init_diag.neg_vib) if init_diag.neg_vib is not None else None,
            "final_neg_vib": int(final_diag.neg_vib) if final_diag.neg_vib is not None else None,
            "initial_eig0": init_diag.eig0,
            "initial_eig1": init_diag.eig1,
            "final_eig0": final_diag.eig0,
            "final_eig1": final_diag.eig1,
            "final_eig_product": (float(final_diag.eig0) * float(final_diag.eig1)) if (final_diag.eig0 is not None and final_diag.eig1 is not None) else None,
            "n_iterations": int(self._iteration),
            "converged": bool(converged),
            "stop_reason": str(stop_reason),
            "trajectory": dict(self.trajectory),
        }


def main(
    argv: Optional[list[str]] = None,
    *,
    default_calculator: Optional[str] = None,
    enforce_calculator: bool = False,
    script_name_prefix: str = "exp-lbfgs-precondition",
) -> None:
    parser = argparse.ArgumentParser(description="L-BFGS energy minimizer (HIP + SCINE).")
    parser = add_common_args(parser)

    parser.add_argument(
        "--start-from",
        type=str,
        default="reactant_noise2A",
        help=(
            "Starting geometry: reactant/ts/midpoint_rt/three_quarter_rt, "
            "or with noise: reactant_noise0.5A, reactant_noise2A, etc."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--max-step", type=float, default=0.5)
    parser.add_argument("--target-neg-eig", type=int, default=1)
    parser.add_argument("--eigenvalue-check-freq", type=int, default=1)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lbfgs-precondition")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)

    args = parser.parse_args(argv)

    if default_calculator is not None:
        args.calculator = default_calculator
    if enforce_calculator and default_calculator is not None:
        args.calculator = default_calculator

    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)

    calculator_type = getattr(args, "calculator", "hip").lower()
    if calculator_type == "scine":
        device = "cpu"

    predict_fn = make_predict_fn_from_calculator(calculator, calculator_type)

    loss_type_flags = build_loss_type_flags(args)
    run_tag = f"lbfgs-maxiter{args.max_iterations}-maxstep{args.max_step}-target{args.target_neg_eig}neg-from-{args.start_from}"

    logger = ExperimentLogger(
        base_dir=out_dir,
        script_name=f"{script_name_prefix}-{calculator_type}",
        loss_type_flags=f"{run_tag}__{loss_type_flags}",
        max_graphs_per_transition=10,
        random_seed=42,
    )

    if args.wandb:
        wandb_config = {
            "script": "lbfgs_energy_minimizer",
            "calculator": calculator_type,
            "start_from": args.start_from,
            "max_iterations": int(args.max_iterations),
            "max_step": float(args.max_step),
            "target_neg_eig": int(args.target_neg_eig),
            "eigenvalue_check_freq": int(args.eigenvalue_check_freq),
            "max_samples": int(args.max_samples),
            "noise_seed": getattr(args, "noise_seed", None),
        }

        name = args.wandb_name
        if not name:
            name = _sanitize_wandb_name(f"{script_name_prefix}__{calculator_type}__{run_tag}")

        init_wandb_run(
            project=args.wandb_project,
            name=str(name),
            config=wandb_config,
            entity=args.wandb_entity,
            tags=[calculator_type, "lbfgs", str(args.start_from)],
            run_dir=str(logger.run_dir),
        )

    all_metrics: Dict[str, list] = {
        "wallclock_time": [],
        "n_iterations": [],
        "initial_neg_vib": [],
        "final_neg_vib": [],
        "initial_energy": [],
        "final_energy": [],
        "converged": [],
    }

    print("Running L-BFGS energy minimization")
    print(f"  Calculator: {calculator_type}")
    print(f"  Start from: {args.start_from}")
    print(f"  Target: <= {args.target_neg_eig} negative vibrational eigenvalues")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Max step: {args.max_step} Å (per-coordinate bounds)")
    print(f"  Output directory: {logger.run_dir}")
    print("-" * 60)

    for i, batch in enumerate(dataloader):
        if i >= int(args.max_samples):
            break

        formula = getattr(batch, "formula", ["sample"])[0]
        print(f"\n--- Sample {i} (Formula: {formula}) ---")
        t0 = time.time()

        atomic_nums = batch.z.detach().cpu()
        start_coords = parse_starting_geometry(
            args.start_from,
            batch,
            noise_seed=getattr(args, "noise_seed", None),
            sample_index=i,
        ).detach()

        start_coords = start_coords.to(torch.device(device))

        minimizer = LBFGSEnergyMinimizer(
            predict_fn,
            atomic_nums,
            device=device,
            max_iterations=int(args.max_iterations),
            max_step=float(args.max_step),
            target_neg_eig_count=int(args.target_neg_eig),
            eigenvalue_check_freq=int(args.eigenvalue_check_freq),
            verbose=True,
        )

        out = minimizer.minimize(start_coords)
        wall = time.time() - t0

        all_metrics["wallclock_time"].append(float(wall))
        all_metrics["n_iterations"].append(int(out.get("n_iterations", 0)))
        all_metrics["initial_neg_vib"].append(out.get("initial_neg_vib"))
        all_metrics["final_neg_vib"].append(out.get("final_neg_vib"))
        all_metrics["initial_energy"].append(out.get("initial_energy"))
        all_metrics["final_energy"].append(out.get("final_energy"))
        all_metrics["converged"].append(bool(out.get("converged", False)))

        sample_metrics = {
            "wallclock_time": float(wall),
            "n_iterations": int(out.get("n_iterations", 0)),
            "initial_neg_vib": out.get("initial_neg_vib"),
            "final_neg_vib": out.get("final_neg_vib"),
            "initial_energy": out.get("initial_energy"),
            "final_energy": out.get("final_energy"),
            "final_eig0": out.get("final_eig0"),
            "final_eig1": out.get("final_eig1"),
            "final_eig_product": out.get("final_eig_product"),
            "converged": bool(out.get("converged", False)),
            "stop_reason": out.get("stop_reason"),
        }

        log_sample(i, sample_metrics, fig=None, plot_name=None)

        # Persist per-sample JSON (ExperimentLogger tracks only RunResult summaries)
        sample_path = os.path.join(str(logger.run_dir), f"sample_{i:03d}.json")
        with open(sample_path, "w") as f:
            json.dump(
                {
                    "formula": str(formula),
                    "sample_index": int(i),
                    "start_from": str(args.start_from),
                    "calculator": str(calculator_type),
                    "result": out,
                    "wallclock_time": float(wall),
                },
                f,
                indent=2,
            )

        # Add structured result for aggregate stats
        init_neg = out.get("initial_neg_vib")
        final_neg = out.get("final_neg_vib")
        steps_taken = int(out.get("n_iterations", 0) or 0)
        rr = RunResult(
            sample_index=int(i),
            formula=str(formula),
            initial_neg_eigvals=int(init_neg) if init_neg is not None else -1,
            final_neg_eigvals=int(final_neg) if final_neg is not None else -1,
            initial_neg_vibrational=int(init_neg) if init_neg is not None else None,
            final_neg_vibrational=int(final_neg) if final_neg is not None else None,
            steps_taken=steps_taken,
            steps_to_ts=None,
            final_time=float(wall),
            final_eig0=out.get("final_eig0"),
            final_eig1=out.get("final_eig1"),
            final_eig_product=out.get("final_eig_product"),
            final_loss=None,
            rmsd_to_known_ts=None,
            stop_reason=str(out.get("stop_reason")),
            plot_path=None,
            extra_data={
                "initial_energy": out.get("initial_energy"),
                "final_energy": out.get("final_energy"),
                "converged": bool(out.get("converged", False)),
            },
        )
        logger.add_result(rr)

        print(
            f"  Done in {wall:.2f}s | iters={out.get('n_iterations')} | "
            f"neg_vib {out.get('initial_neg_vib')} → {out.get('final_neg_vib')} | "
            f"converged={out.get('converged')} ({out.get('stop_reason')})"
        )

    # Summary
    total = len(all_metrics["converged"])
    conv = int(np.sum(np.asarray(all_metrics["converged"], dtype=np.int32))) if total else 0
    summary = {
        "total_samples": int(total),
        "converged_count": int(conv),
        "converged_rate": float(conv / total) if total else 0.0,
        "avg_wallclock_time": float(np.mean(all_metrics["wallclock_time"])) if total else None,
        "avg_iterations": float(np.mean(all_metrics["n_iterations"])) if total else None,
    }

    log_summary(summary)

    # Save canonical ExperimentLogger outputs
    logger.save_all_results()

    if args.wandb:
        finish_wandb()


if __name__ == "__main__":
    main()
