from __future__ import annotations

"""Multi-mode escape for GAD using GAD VECTOR NORMALIZATION instead of v2 kicks.

Test of Andreas's suggestion: normalize the GAD vector when |gad| < threshold
so we always step by a unit vector with FIXED dt.

Key differences from v2 kick approach:
1. No discrete jumps along eigenvectors
2. Always uses GAD dynamics (stays on GAD manifold)
3. When |gad| is small: step = dt * (gad / |gad|)  → always move distance dt
4. When |gad| is large: step = dt * gad             → normal GAD
5. FIXED dt only - NO adaptive dt control

Algorithm:
1. Compute GAD vector using Eckart-projected mass-weighted Hessian
2. If |gad| < threshold: normalize gad → gad / |gad|
3. Step: coords += dt * gad_normalized
4. Repeat until index = 1 or max steps reached
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.core_algos_gad import pick_tracked_mode
from ..dependencies.common_utils import add_common_args, parse_starting_geometry, setup_experiment
from ..dependencies.experiment_logger import ExperimentLogger, RunResult, build_loss_type_flags
from ..dependencies.hessian import (
    vibrational_eigvals,
    get_scine_elements_from_predict_output,
    project_hessian_remove_rigid_modes,
    prepare_hessian,
)
from ..logging import finish_wandb, init_wandb_run, log_sample, log_summary
from ..logging.plotly_utils import plot_gad_trajectory_interactive
from ..runners._predict import make_predict_fn_from_calculator

# SCINE projection (may not be available)
try:
    from ..dependencies.scine_masses import (
        ScineFrequencyAnalyzer,
        get_scine_masses,
    )
    SCINE_PROJECTION_AVAILABLE = True
except ImportError:
    SCINE_PROJECTION_AVAILABLE = False


def _sanitize_wandb_name(s: str) -> str:
    s = str(s)
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s[:128] if len(s) > 128 else s


def _auto_wandb_name(*, script: str, loss_type_flags: str, args: argparse.Namespace) -> str:
    calculator = getattr(args, "calculator", "hip")
    start_from = getattr(args, "start_from", "unknown")
    method = getattr(args, "method", "euler")
    normalize_target = getattr(args, "normalize_target", "gad")
    norm_threshold = getattr(args, "norm_threshold", None)
    n_steps = getattr(args, "n_steps", None)
    noise_seed = getattr(args, "noise_seed", None)
    job_id = os.environ.get("SLURM_JOB_ID")

    parts = [
        script,
        str(calculator),
        str(start_from),
        str(method),
        f"norm-{normalize_target}",
        f"thresh{norm_threshold}" if norm_threshold is not None else None,
        f"steps{n_steps}" if n_steps is not None else None,
        f"seed{noise_seed}" if noise_seed is not None else None,
        f"job{job_id}" if job_id else None,
        str(loss_type_flags),
    ]
    parts = [p for p in parts if p]
    return _sanitize_wandb_name("__".join(parts))


def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().reshape(-1)[0].item())
    return float(x)


def _force_mean(forces: torch.Tensor) -> float:
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    f = forces.reshape(-1, 3)
    return float(f.norm(dim=1).mean().item())


def _force_norm(forces: torch.Tensor) -> float:
    """Compute Frobenius norm of force vector."""
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    return float(forces.reshape(-1).norm().item())


def _mean_atom_norm(x: torch.Tensor) -> float:
    if x.dim() == 3 and x.shape[0] == 1:
        x = x[0]
    return float(x.reshape(-1, 3).norm(dim=1).mean().item())


def _max_atom_norm(x: torch.Tensor) -> float:
    if x.dim() == 3 and x.shape[0] == 1:
        x = x[0]
    return float(x.reshape(-1, 3).norm(dim=1).max().item())


def _vib_mask_from_evals(evals: torch.Tensor, *, tr_threshold: float) -> torch.Tensor:
    """Mask out translation/rotation (near-zero) modes."""
    if tr_threshold <= 0:
        return torch.ones_like(evals, dtype=torch.bool)
    return evals.abs() > float(tr_threshold)


def get_projected_hessian(
    hessian_raw: torch.Tensor,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    scine_elements: Optional[list] = None,
) -> torch.Tensor:
    """Get Eckart-projected, mass-weighted Hessian (3N x 3N)."""
    if scine_elements is not None:
        if not SCINE_PROJECTION_AVAILABLE:
            raise RuntimeError("SCINE projection requested but scine_masses not available")
        return _scine_project_hessian_full(hessian_raw, coords, scine_elements)
    return project_hessian_remove_rigid_modes(hessian_raw, coords, atomic_nums)


def _scine_project_hessian_full(
    hessian_raw: torch.Tensor,
    coords: torch.Tensor,
    elements: list,
) -> torch.Tensor:
    """SCINE Eckart projection returning full 3N x 3N Hessian."""
    import numpy as np
    from scipy.linalg import eigh

    hess_np = hessian_raw.detach().cpu().numpy()
    coords_np = coords.detach().cpu().numpy().reshape(-1, 3)
    n_atoms = len(elements)

    masses_amu = get_scine_masses(elements)
    m_sqrt = np.sqrt(masses_amu)
    m_sqrt_3n = np.repeat(m_sqrt, 3)

    inv_m_sqrt_mat = np.outer(1.0 / m_sqrt_3n, 1.0 / m_sqrt_3n)
    hess_np_2d = hess_np.reshape(3 * n_atoms, 3 * n_atoms)
    H_mw = hess_np_2d * inv_m_sqrt_mat

    analyzer = ScineFrequencyAnalyzer()
    P_reduced = analyzer._get_vibrational_projector(coords_np, masses_amu)
    P_full = P_reduced.T @ P_reduced

    H_proj = P_full @ H_mw @ P_full
    H_proj = 0.5 * (H_proj + H_proj.T)

    return torch.from_numpy(H_proj).to(device=hessian_raw.device, dtype=hessian_raw.dtype)


def compute_gad_vector_projected_tracked(
    forces: torch.Tensor,
    hessian_proj: torch.Tensor,
    v_prev: torch.Tensor | None,
    *,
    k_track: int = 8,
    beta: float = 1.0,
    tr_threshold: float = 1e-6,
    norm_threshold: float = 0.01,
    normalize_target: str = "gad",  # "gad" or "force"
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """GAD direction using projected Hessian with mode tracking and normalization.
    
    Args:
        normalize_target: What to normalize when small.
            - "gad": Normalize GAD vector when |gad| < threshold
            - "force": Normalize force when |f| < threshold, then compute GAD
    
    When normalization triggers:
    - gad mode: gad = gad / |gad| → step = dt * unit_gad
    - force mode: f = f / |f| → gad = f_norm + 2*(-f_norm·v)*v
    """
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    forces = forces.reshape(-1, 3)
    num_atoms = int(forces.shape[0])

    hess = hessian_proj
    if hess.dim() != 2 or hess.shape[0] != 3 * num_atoms:
        hess = prepare_hessian(hess, num_atoms)

    evals, evecs = torch.linalg.eigh(hess)

    vib_mask = torch.abs(evals) > float(tr_threshold)
    if not vib_mask.any():
        candidate_indices = torch.arange(min(int(k_track), int(evecs.shape[1])), device=evecs.device)
    else:
        vib_indices = torch.where(vib_mask)[0]
        candidate_indices = vib_indices[: int(min(int(k_track), int(vib_indices.numel())))]

    V = evecs[:, candidate_indices].to(device=forces.device, dtype=forces.dtype)
    v_prev_local = v_prev.to(device=forces.device, dtype=forces.dtype).reshape(-1) if v_prev is not None else None
    v_new, j, overlap = pick_tracked_mode(V, v_prev_local, k=int(V.shape[1]))
    if v_prev_local is not None and float(beta) < 1.0:
        v = (1.0 - float(beta)) * v_prev_local + float(beta) * v_new
        v = v / (v.norm() + 1e-12)
    else:
        v = v_new

    f_flat = forces.reshape(-1)
    f_norm_val = f_flat.norm()
    was_normalized = False
    norm_val_before = 0.0
    
    if normalize_target == "force":
        # Force normalization: normalize f before computing GAD
        norm_val_before = float(f_norm_val.item())
        if f_norm_val < norm_threshold and f_norm_val > 1e-12:
            f_flat = f_flat / f_norm_val
            was_normalized = True
        # Compute GAD with (possibly normalized) force
        gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
    else:
        # GAD normalization: compute GAD first, then normalize
        gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
        gad_norm_val = gad_flat.norm()
        norm_val_before = float(gad_norm_val.item())
        if gad_norm_val < norm_threshold and gad_norm_val > 1e-12:
            gad_flat = gad_flat / gad_norm_val
            was_normalized = True

    gad_vec = gad_flat.view(num_atoms, 3)
    v_next = v.detach().clone().reshape(-1)
    
    return gad_vec, v_next, {
        "mode_overlap": float(overlap),
        "mode_index": float(j),
        "was_normalized": was_normalized,
        "norm_val_before": norm_val_before,
        "normalize_target": normalize_target,
    }


def gad_euler_step_projected(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    dt: float,
    out: Optional[Dict[str, Any]] = None,
    scine_elements: Optional[list] = None,
    v_prev: torch.Tensor | None = None,
    k_track: int = 8,
    beta: float = 1.0,
    norm_threshold: float = 0.01,
    normalize_target: str = "gad",
) -> Dict[str, Any]:
    """GAD Euler step using PROJECTED Hessian with normalization."""
    if coords.dim() == 3 and coords.shape[0] == 1:
        coords = coords[0]
    coords = coords.reshape(-1, 3)

    if out is None:
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)

    forces = out["forces"]
    hessian = out["hessian"]

    hessian_proj = get_projected_hessian(hessian, coords, atomic_nums, scine_elements)

    gad_vec, v_next, info = compute_gad_vector_projected_tracked(
        forces,
        hessian_proj,
        v_prev,
        k_track=k_track,
        beta=beta,
        norm_threshold=norm_threshold,
        normalize_target=normalize_target,
    )
    new_coords = coords + dt * gad_vec

    return {
        "new_coords": new_coords,
        "gad_vec": gad_vec,
        "out": out,
        "hessian_proj": hessian_proj,
        "v_next": v_next,
        **info,
    }


def _check_plateau_convergence(
    disp_history: list[float],
    neg_vib_history: list[int],
    current_neg_vib: int,
    *,
    window: int,
    disp_threshold: float,
    neg_vib_std_threshold: float,
) -> bool:
    """Check if we've converged to a plateau based on displacement history."""
    if len(disp_history) < window:
        return False

    recent_disp = disp_history[-window:]
    recent_neg_vib = neg_vib_history[-window:]

    mean_disp = float(np.mean(recent_disp))
    std_neg_vib = float(np.std(recent_neg_vib))

    return (
        mean_disp < disp_threshold
        and std_neg_vib <= neg_vib_std_threshold
        and current_neg_vib > 1
    )


def run_multi_mode_force_norm(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    n_steps: int,
    dt: float,
    stop_at_ts: bool,
    ts_eps: float,
    # Normalization parameters
    norm_threshold: float,
    normalize_target: str = "gad",  # "gad" or "force"
    profile_every: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run GAD with normalization when |gad| or |f| is small.
    
    Uses FIXED dt only.
    - normalize_target="gad": normalize GAD vector when |gad| < threshold
    - normalize_target="force": normalize force when |f| < threshold
    """
    coords = coords0.detach().clone().to(torch.float32)

    trajectory = {k: [] for k in [
        "energy",
        "force_mean",
        "force_norm",
        "eig0",
        "eig1",
        "eig_product",
        "neg_vib",
        "disp_from_last",
        "disp_from_start",
        "gad_norm",
        "norm_val_before",
        "was_normalized",
    ]}

    start_pos = coords.clone()
    prev_pos = coords.clone()

    steps_to_ts: Optional[int] = None
    total_steps = 0
    normalization_events: list[Dict[str, Any]] = []

    # Track the ascent eigenmode across steps
    v_prev: torch.Tensor | None = None

    while total_steps < n_steps:
        t_predict0 = time.time() if profile_every and (total_steps % profile_every == 0) else None
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        t_predict1 = time.time() if t_predict0 is not None else None
        
        energy = out.get("energy")
        forces = out.get("forces")
        hessian = out.get("hessian")

        energy_value = _to_float(energy)
        force_mean = _force_mean(forces)
        force_norm_val = _force_norm(forces)

        scine_elements = get_scine_elements_from_predict_output(out)

        # Get eigenvalues
        vib = vibrational_eigvals(hessian, coords, atomic_nums, scine_elements=scine_elements)
        eig0 = float(vib[0].item()) if vib.numel() >= 1 else float("nan")
        eig1 = float(vib[1].item()) if vib.numel() >= 2 else float("nan")
        eig_prod = float((vib[0] * vib[1]).item()) if vib.numel() >= 2 else float("inf")
        neg_vib = int((vib < 0).sum().item()) if vib.numel() > 0 else -1

        # Compute GAD vector with normalization
        step_out = gad_euler_step_projected(
            predict_fn,
            coords,
            atomic_nums,
            dt=dt,
            out=out,
            scine_elements=scine_elements,
            v_prev=v_prev,
            k_track=8,
            beta=1.0,
            norm_threshold=norm_threshold,
            normalize_target=normalize_target,
        )
        gad_vec = step_out["gad_vec"]
        new_coords = step_out["new_coords"]
        v_prev = step_out.get("v_next")
        mode_overlap = float(step_out.get("mode_overlap", 1.0))
        mode_index = int(step_out.get("mode_index", 0.0))
        was_normalized = step_out.get("was_normalized", False)
        norm_val_before = step_out.get("norm_val_before", 0.0)

        disp_from_last = float((coords - prev_pos).norm(dim=1).mean().item()) if total_steps > 0 else 0.0
        disp_from_start = float((coords - start_pos).norm(dim=1).mean().item()) if total_steps > 0 else 0.0

        gad_norm = _mean_atom_norm(gad_vec)

        if t_predict0 is not None and t_predict1 is not None:
            print(
                f"[profile] step={total_steps} predict={t_predict1 - t_predict0:.4f}s "
                f"norm_val_before={norm_val_before:.6f} normalized={was_normalized} target={normalize_target}"
            )

        # Track normalization events
        if was_normalized:
            normalization_events.append({
                "step": total_steps,
                "norm_val_before": norm_val_before,
                "neg_vib": neg_vib,
                "eig_product": eig_prod,
                "normalize_target": normalize_target,
            })

        trajectory["energy"].append(energy_value)
        trajectory["force_mean"].append(force_mean)
        trajectory["force_norm"].append(force_norm_val)
        trajectory["eig0"].append(eig0)
        trajectory["eig1"].append(eig1)
        trajectory["eig_product"].append(eig_prod)
        trajectory["neg_vib"].append(int(neg_vib))
        trajectory["disp_from_last"].append(disp_from_last)
        trajectory["disp_from_start"].append(disp_from_start)
        trajectory["gad_norm"].append(gad_norm)
        trajectory["norm_val_before"].append(norm_val_before)
        trajectory["was_normalized"].append(int(was_normalized))

        trajectory.setdefault("mode_overlap", []).append(float(mode_overlap))
        trajectory.setdefault("mode_index", []).append(int(mode_index))

        # Check for TS (index = 1)
        if stop_at_ts and steps_to_ts is None and np.isfinite(eig_prod) and eig_prod < -abs(ts_eps):
            steps_to_ts = total_steps
            break

        # Take GAD step with FIXED dt
        prev_pos = coords.clone()
        coords = new_coords.detach()
        total_steps += 1

    # Pad trajectories to same length
    while len(trajectory["gad_norm"]) < len(trajectory["energy"]):
        trajectory["gad_norm"].append(float("nan"))

    # Final vibrational analysis
    final_neg_vib = -1
    try:
        final_out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
        scine_elements = get_scine_elements_from_predict_output(final_out)
        final_vib_eigvals = vibrational_eigvals(final_out["hessian"], coords, atomic_nums, scine_elements=scine_elements)
        final_neg_vib = int((final_vib_eigvals < 0).sum().item())
    except Exception:
        pass

    aux = {
        "steps_to_ts": steps_to_ts,
        "normalization_events": normalization_events,
        "total_steps": total_steps,
    }

    final_out_dict = {
        "final_coords": coords.detach().cpu(),
        "trajectory": trajectory,
        "steps_taken": total_steps,
        "steps_to_ts": steps_to_ts,
        "final_eig0": trajectory["eig0"][-1] if trajectory["eig0"] else None,
        "final_eig1": trajectory["eig1"][-1] if trajectory["eig1"] else None,
        "final_eig_product": trajectory["eig_product"][-1] if trajectory["eig_product"] else None,
        "final_neg_vibrational": final_neg_vib,
    }

    return final_out_dict, aux


def _save_trajectory_json(logger: ExperimentLogger, result: RunResult, trajectory: Dict[str, Any], normalization_events: list) -> Optional[str]:
    transition_dir = logger.run_dir / result.transition_key
    transition_dir.mkdir(parents=True, exist_ok=True)
    path = transition_dir / f"trajectory_{result.sample_index:03d}.json"
    try:
        data = {
            "trajectory": trajectory,
            "normalization_events": normalization_events,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return str(path)
    except Exception:
        return None


def main(
    argv: Optional[list[str]] = None,
    *,
    default_calculator: Optional[str] = None,
    enforce_calculator: bool = False,
    script_name_prefix: str = "norm",
) -> None:
    parser = argparse.ArgumentParser(
        description="Normalization runner: Multi-mode escape via GAD or force normalization (Andreas's suggestion)."
    )
    parser = add_common_args(parser)

    if default_calculator is not None:
        parser.set_defaults(calculator=str(default_calculator))

    parser.add_argument("--method", type=str, default="euler", choices=["euler"])
    parser.add_argument("--n-steps", type=int, default=1500, help="Total max GAD steps")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--start-from", type=str, default="midpoint_rt")
    parser.add_argument("--stop-at-ts", action="store_true")
    parser.add_argument("--ts-eps", type=float, default=1e-5)

    # Normalization parameters (FIXED dt only)
    parser.add_argument(
        "--normalize-target",
        type=str,
        default="gad",
        choices=["gad", "force"],
        help="What to normalize: 'gad' = normalize GAD vector, 'force' = normalize force before GAD.",
    )
    parser.add_argument(
        "--norm-threshold",
        type=float,
        default=0.01,
        help="Norm threshold. Normalize to unit length when |target| < this value.",
    )
    parser.add_argument("--profile-every", type=int, default=0)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="norm-multi-mode")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)

    args = parser.parse_args(argv)

    if enforce_calculator and default_calculator is not None:
        if str(getattr(args, "calculator", "")).lower() != str(default_calculator).lower():
            raise ValueError(
                f"This entrypoint enforces --calculator={default_calculator}. "
                f"Got --calculator={getattr(args, 'calculator', None)}."
            )

    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    calculator_type = getattr(args, "calculator", "hip").lower()
    if calculator_type == "scine":
        device = "cpu"

    predict_fn = make_predict_fn_from_calculator(calculator, calculator_type)

    loss_type_flags = build_loss_type_flags(args)
    script_name = f"{script_name_prefix}-{args.method}"
    logger = ExperimentLogger(
        base_dir=out_dir,
        script_name=script_name,
        loss_type_flags=loss_type_flags,
        max_graphs_per_transition=10,
        random_seed=42,
    )

    if args.wandb:
        wandb_config = {
            "script": script_name,
            "method": args.method,
            "start_from": args.start_from,
            "stop_at_ts": bool(args.stop_at_ts),
            "calculator": getattr(args, "calculator", "hip"),
            "dt": args.dt,
            "n_steps": args.n_steps,
            "normalize_target": args.normalize_target,
            "norm_threshold": args.norm_threshold,
        }
        wandb_name = args.wandb_name
        if not wandb_name:
            wandb_name = _auto_wandb_name(script=script_name, loss_type_flags=loss_type_flags, args=args)

        init_wandb_run(
            project=args.wandb_project,
            name=str(wandb_name),
            config=wandb_config,
            entity=args.wandb_entity,
            tags=[args.start_from, args.method, f"norm-{args.normalize_target}", "fixed-dt", "noisy"],
            run_dir=out_dir,
        )

    all_metrics = defaultdict(list)

    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break

        batch = batch.to(device)
        atomic_nums = batch.z.detach().cpu().to(device)
        formula = getattr(batch, "formula", "sample")

        start_coords = parse_starting_geometry(
            args.start_from,
            batch,
            noise_seed=getattr(args, "noise_seed", None),
            sample_index=i,
        ).detach().to(device)

        # Initial vibrational order
        try:
            init_out = predict_fn(start_coords, atomic_nums, do_hessian=True, require_grad=False)
            init_scine_elements = get_scine_elements_from_predict_output(init_out)
            init_vib = vibrational_eigvals(init_out["hessian"], start_coords, atomic_nums, scine_elements=init_scine_elements)
            initial_neg = int((init_vib < 0).sum().item())
        except Exception:
            initial_neg = -1

        t0 = time.time()
        try:
            out_dict, aux = run_multi_mode_force_norm(
                predict_fn,
                start_coords,
                atomic_nums,
                n_steps=int(args.n_steps),
                dt=float(args.dt),
                stop_at_ts=bool(args.stop_at_ts),
                ts_eps=float(args.ts_eps),
                norm_threshold=float(args.norm_threshold),
                normalize_target=str(args.normalize_target),
                profile_every=int(args.profile_every),
            )
            wall = time.time() - t0
        except Exception as e:
            wall = time.time() - t0
            stop_reason = f"{type(e).__name__}: {e}"
            print(f"[WARN] Sample {i} failed during run: {stop_reason}")

            result = RunResult(
                sample_index=i,
                formula=str(formula),
                initial_neg_eigvals=int(initial_neg),
                final_neg_eigvals=-1,
                initial_neg_vibrational=None,
                final_neg_vibrational=None,
                steps_taken=0,
                steps_to_ts=None,
                final_time=float(wall),
                final_eig0=None,
                final_eig1=None,
                final_eig_product=None,
                final_loss=None,
                rmsd_to_known_ts=None,
                stop_reason=stop_reason,
                plot_path=None,
                extra_data={
                    "method": f"norm_{args.normalize_target}",
                    "normalize_target": args.normalize_target,
                    "norm_threshold": float(args.norm_threshold),
                },
            )
            logger.add_result(result)
            if args.wandb:
                log_sample(i, {"steps_taken": 0, "wallclock_s": wall, "stop_reason": stop_reason, "failed": 1})
            continue

        final_neg = out_dict.get("final_neg_vibrational", -1)

        result = RunResult(
            sample_index=i,
            formula=str(formula),
            initial_neg_eigvals=initial_neg,
            final_neg_eigvals=int(final_neg) if final_neg is not None else -1,
            initial_neg_vibrational=None,
            final_neg_vibrational=int(final_neg) if final_neg is not None else None,
            steps_taken=int(out_dict["steps_taken"]),
            steps_to_ts=aux.get("steps_to_ts"),
            final_time=float(wall),
            final_eig0=out_dict.get("final_eig0"),
            final_eig1=out_dict.get("final_eig1"),
            final_eig_product=out_dict.get("final_eig_product"),
            final_loss=None,
            rmsd_to_known_ts=None,
            stop_reason=None,
            plot_path=None,
            extra_data={
                "method": f"norm_{args.normalize_target}",
                "normalize_target": args.normalize_target,
                "norm_threshold": float(args.norm_threshold),
                "normalization_events": aux.get("normalization_events"),
            },
        )

        logger.add_result(result)

        # Generate interactive figure
        fig_interactive = plot_gad_trajectory_interactive(
            out_dict["trajectory"],
            sample_index=i,
            formula=str(formula),
            start_from=args.start_from,
            initial_neg_num=initial_neg,
            final_neg_num=int(final_neg) if final_neg is not None else -1,
            steps_to_ts=aux.get("steps_to_ts"),
        )

        html_path = Path(out_dir) / f"traj_{i:03d}.html"
        fig_interactive.write_html(str(html_path))
        result.plot_path = str(html_path)

        _save_trajectory_json(logger, result, out_dict["trajectory"], aux.get("normalization_events", []))

        metrics = {
            "steps_taken": result.steps_taken,
            "steps_to_ts": result.steps_to_ts,
            "final_eig0": result.final_eig0,
            "final_eig1": result.final_eig1,
            "final_eig_product": result.final_eig_product,
            "final_neg_vibrational": result.final_neg_vibrational,
            "wallclock_s": result.final_time,
            "normalization_count": len(aux.get("normalization_events", [])),
        }

        for k, v in metrics.items():
            if v is not None:
                all_metrics[k].append(v)

        if args.wandb:
            log_sample(i, metrics, fig=fig_interactive, plot_name="trajectory_interactive")

    all_runs_path, aggregate_stats_path = logger.save_all_results()
    summary = logger.compute_aggregate_stats()
    logger.print_summary()

    if args.wandb:
        log_summary(summary)
        finish_wandb()

    print(f"Saved results: {all_runs_path}")
    print(f"Saved stats:   {aggregate_stats_path}")


if __name__ == "__main__":
    main(default_calculator="hip", enforce_calculator=False, script_name_prefix="gad-norm")
