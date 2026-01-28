"""Systematic comparison of kick strategies.

This runner executes GAD with different kick strategies and compares:
1. Success rate (converging to index-1 TS)
2. Number of escape cycles needed
3. Total steps to convergence
4. Energy landscape traversal patterns

Strategies compared:
- v2 (current): Kick along second vibrational mode
- v1: Kick along first vibrational mode
- random: Random direction (control)
- random_ortho_v1: Random direction orthogonal to v₁
- gradient_descent: Take N gradient descent steps
- ortho_v1_grad_descent: Gradient descent constrained to subspace orthogonal to v₁
- higher_modes: Try v₂, v₃, v₄ sequentially
- adaptive_k_reflect: Reflect along full unstable subspace
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Import kick strategies
from ..kick_experiments.kick_strategies import KICK_STRATEGIES, get_kick_strategy
from ..logging import TrajectoryLogger, create_escape_event

# Import core GAD components
from src.noisy.multi_mode_eckartmw import (
    get_projected_hessian,
    _step_metrics_from_projected_hessian,
    _force_mean,
    _to_float,
    _min_interatomic_distance,
    compute_gad_vector_projected_tracked,
    _atomic_nums_to_symbols,
)

from src.dependencies.hessian import (
    vibrational_eigvals,
    get_scine_elements_from_predict_output,
)


@dataclass
class KickComparisonResult:
    """Results from a single run with a specific kick strategy."""
    strategy: str
    sample_id: str
    formula: str

    # Outcome
    success: bool
    final_morse_index: int
    total_steps: int
    escape_cycles: int
    wall_time: float

    # Trajectory summaries
    energies: List[float] = field(default_factory=list)
    morse_indices: List[int] = field(default_factory=list)
    escape_events: List[Dict] = field(default_factory=list)

    # Error info
    error: Optional[str] = None


def _check_plateau_convergence(
    disp_history: List[float],
    neg_vib_history: List[int],
    current_neg_vib: int,
    *,
    window: int,
    disp_threshold: float,
    neg_vib_std_threshold: float,
) -> bool:
    """Check if stuck using displacement-based detection."""
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


def _eig_metrics_from_projected_hessian(
    hessian_proj: torch.Tensor,
    *,
    tr_threshold: float,
    eigh_device: str = "cpu",
) -> Tuple[float, float, float, int]:
    """Compute eig0/eig1/eig_product/neg_vib from projected Hessian."""
    hess = hessian_proj
    if hess.dim() != 2:
        side = int(hess.numel() ** 0.5)
        hess = hess.reshape(side, side)

    if str(eigh_device).lower() == "cpu":
        hess_eigh = hess.detach().to(device=torch.device("cpu"))
    else:
        hess_eigh = hess

    evals, _ = torch.linalg.eigh(hess_eigh)
    evals_local = evals.to(dtype=torch.float32)
    vib_mask = torch.abs(evals_local) > float(tr_threshold)
    evals_vib = evals_local[vib_mask] if vib_mask.any() else evals_local

    if int(evals_vib.numel()) >= 2:
        eig0 = float(evals_vib[0].item())
        eig1 = float(evals_vib[1].item())
        eig_product = float((evals_vib[0] * evals_vib[1]).item())
    else:
        eig0, eig1, eig_product = float("nan"), float("nan"), float("inf")

    neg_vib = int((evals_vib < -float(tr_threshold)).sum().item()) if int(evals_vib.numel()) > 0 else -1

    return eig0, eig1, eig_product, neg_vib


def run_gad_with_kick_strategy(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    kick_strategy_name: str,
    *,
    n_steps: int = 5000,
    dt: float = 0.005,
    dt_min: float = 1e-6,
    dt_max: float = 0.08,
    max_atom_disp: float = 0.35,
    # Plateau detection
    escape_window: int = 10,
    escape_disp_threshold: float = 1e-4,
    escape_neg_vib_std: float = 0.5,
    # Kick parameters
    escape_delta: float = 0.3,
    max_escape_cycles: int = 500,
    min_interatomic_dist: float = 0.5,
    force_escape_after: Optional[int] = None,
    # Convergence
    ts_eps: float = 1e-5,
    stop_at_ts: bool = True,
    # TR tracking / projection
    tr_threshold: float = 1e-6,
    project_gradient_and_v: bool = False,
    # Tracking
    sample_id: str = "unknown",
    formula: str = "",
    scine_elements=None,
    log_dir: Optional[str] = None,
) -> KickComparisonResult:
    """Run GAD with specified kick strategy.

    This mirrors the multi_mode_eckartmw.py logic but swaps out the
    kick strategy for systematic comparison.
    """
    t0 = time.time()

    try:
        coords = coords0.detach().clone().to(torch.float32)
        if coords.dim() == 3 and coords.shape[0] == 1:
            coords = coords[0]
        coords = coords.reshape(-1, 3)

        # Get kick function
        kick_fn = get_kick_strategy(kick_strategy_name)

        # Logging
        logger = TrajectoryLogger(sample_id=sample_id, formula=formula) if log_dir else None

        # Tracking variables
        energies = []
        morse_indices = []
        escape_events = []
        disp_history = []
        neg_vib_history = []

        start_pos = coords.clone()
        prev_pos = coords.clone()

        total_steps = 0
        escape_cycle = 0
        dt_eff = dt
        v_prev = None
        atomsymbols = _atomic_nums_to_symbols(atomic_nums) if project_gradient_and_v else None

        # GAD state
        best_neg_vib = None
        no_improve = 0
        plateau_patience = 20
        last_escape_step = -1

        while escape_cycle < max_escape_cycles and total_steps < n_steps:
            # Get energy, forces, Hessian
            out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=False)
            energy = _to_float(out["energy"])
            forces = out["forces"]
            hessian = out["hessian"]

            if scine_elements is None:
                scine_elements = get_scine_elements_from_predict_output(out)

            # Get projected Hessian
            hess_proj = get_projected_hessian(hessian, coords, atomic_nums, scine_elements)

            # Compute GAD and eigenvalues
            if project_gradient_and_v:
                gad_vec, v_prev, mode_info = compute_gad_vector_projected_tracked(
                    forces=forces,
                    hessian_proj=hess_proj,
                    v_prev=v_prev,
                    k_track=8,
                    beta=1.0,
                    tr_threshold=tr_threshold,
                    coords=coords,
                    atomsymbols=atomsymbols,
                    project_gradient_and_v=True,
                )
                eig0, eig1, eig_product, neg_vib = _eig_metrics_from_projected_hessian(
                    hess_proj,
                    tr_threshold=tr_threshold,
                    eigh_device="cpu",
                )
                mode_overlap = float(mode_info.get("mode_overlap", float("nan")))
                mode_index = int(mode_info.get("mode_index", 0))
            else:
                gad_vec, eig0, eig1, eig_product, neg_vib, v_prev, mode_overlap, mode_index = _step_metrics_from_projected_hessian(
                    forces=forces,
                    hessian_proj=hess_proj,
                    tr_threshold=tr_threshold,
                    eigh_device="cpu",
                    v_prev=v_prev,
                    k_track=8,
                    beta=1.0,
                )

            energies.append(energy)
            morse_indices.append(neg_vib)

            disp_from_last = float((coords - prev_pos).norm(dim=1).mean().item()) if total_steps > 0 else 0.0
            if total_steps > 0:
                disp_history.append(disp_from_last)
                neg_vib_history.append(neg_vib)

            x_disp_window = float(np.mean(disp_history[-escape_window:])) if disp_history else float("nan")

            if logger is not None:
                logger.log_step(
                    step=total_steps,
                    coords=coords,
                    energy=energy,
                    forces=forces,
                    hessian_proj=hess_proj,
                    gad_vec=gad_vec,
                    dt_eff=dt_eff,
                    coords_prev=prev_pos if total_steps > 0 else None,
                    energy_prev=energies[-2] if len(energies) > 1 else None,
                    mode_index=mode_index,
                    x_disp_window=x_disp_window,
                    tr_threshold=tr_threshold,
                )

            # Check for TS (index = 1)
            if stop_at_ts and np.isfinite(eig_product) and eig_product < -abs(ts_eps):
                result = KickComparisonResult(
                    strategy=kick_strategy_name,
                    sample_id=sample_id,
                    formula=formula,
                    success=True,
                    final_morse_index=neg_vib,
                    total_steps=total_steps,
                    escape_cycles=escape_cycle,
                    wall_time=time.time() - t0,
                    energies=energies,
                    morse_indices=morse_indices,
                    escape_events=escape_events,
                )
                if logger is not None:
                    logger.finalize(
                        final_coords=coords,
                        final_morse_index=neg_vib,
                        converged_to_ts=True,
                    )
                    logger.save(log_dir, prefix=kick_strategy_name)
                return result

            # Check for plateau
            is_plateau = _check_plateau_convergence(
                disp_history,
                neg_vib_history,
                neg_vib,
                window=escape_window,
                disp_threshold=escape_disp_threshold,
                neg_vib_std_threshold=escape_neg_vib_std,
            )
            trigger_reason = "displacement_plateau"

            if (
                not is_plateau
                and force_escape_after is not None
                and total_steps - last_escape_step >= int(force_escape_after)
                and int(neg_vib) > 1
            ):
                is_plateau = True
                trigger_reason = "forced_escape_after"

            if is_plateau:
                # Perform escape using specified strategy
                new_coords, escape_info = kick_fn(
                    predict_fn,
                    coords,
                    atomic_nums,
                    hess_proj,
                    delta=escape_delta,
                    min_interatomic_dist=min_interatomic_dist,
                )

                # Compute post-escape state for logging
                post_out = predict_fn(new_coords, atomic_nums, do_hessian=True, require_grad=False)
                post_hess_proj = get_projected_hessian(
                    post_out["hessian"], new_coords, atomic_nums, scine_elements
                )
                post_forces = post_out["forces"]
                post_energy = _to_float(post_out["energy"])

                # Map kick metadata
                kick_mode = 0
                kick_lambda = float("nan")
                if kick_strategy_name == "v1":
                    kick_mode = 1
                    kick_lambda = float(escape_info.get("lambda1", float("nan")))
                elif kick_strategy_name == "v2":
                    kick_mode = 2
                    kick_lambda = float(escape_info.get("lambda2", float("nan")))
                elif kick_strategy_name == "higher_modes":
                    kick_mode = int(escape_info.get("mode_used", 0))
                    kick_lambda = float(escape_info.get("lambda", float("nan")))

                kick_direction = escape_info.get("direction")
                if isinstance(kick_direction, (int, float)):
                    kick_direction = f"+{kick_strategy_name}" if kick_direction > 0 else f"-{kick_strategy_name}"
                kick_direction = str(kick_direction) if kick_direction is not None else kick_strategy_name

                displacement_magnitude = float((new_coords - coords).norm(dim=1).mean().item())
                min_dist_after = _min_interatomic_distance(new_coords)

                if logger is not None:
                    escape_event = create_escape_event(
                        step=total_steps,
                        escape_cycle=escape_cycle,
                        trigger_reason=trigger_reason,
                        pre_hessian_proj=hess_proj,
                        pre_forces=forces,
                        pre_energy=energy,
                        kick_mode=kick_mode,
                        kick_direction=kick_direction,
                        kick_delta_base=escape_delta,
                        kick_delta_effective=float(escape_info.get("delta_used", float("nan"))),
                        kick_lambda=kick_lambda,
                        post_hessian_proj=post_hess_proj,
                        post_forces=post_forces,
                        post_energy=post_energy,
                        accepted=bool(escape_info.get("success", False)),
                        rejection_reason=escape_info.get("reason"),
                        displacement_magnitude=displacement_magnitude,
                        min_dist_after=min_dist_after,
                        mean_disp_at_trigger=float(np.mean(disp_history[-escape_window:])) if len(disp_history) >= escape_window else 0.0,
                        neg_vib_std_at_trigger=float(np.std(neg_vib_history[-escape_window:])) if len(neg_vib_history) >= escape_window else 0.0,
                        tr_threshold=tr_threshold,
                    )
                    logger.log_escape(escape_event)

                escape_info["step"] = total_steps
                escape_info["cycle"] = escape_cycle
                escape_info["pre_morse_index"] = neg_vib
                escape_events.append(escape_info)

                coords = new_coords.detach()
                escape_cycle += 1
                last_escape_step = total_steps
                dt_eff = dt  # Reset dt
                disp_history.clear()
                neg_vib_history.clear()
                prev_pos = coords.clone()
                v_prev = None  # Reset mode tracking
                continue

            # Take GAD step
            step_disp = dt_eff * gad_vec
            max_disp = float(step_disp.norm(dim=1).max().item())
            if max_disp > max_atom_disp and max_disp > 0:
                scale = max_atom_disp / max_disp
                step_disp = scale * step_disp
                dt_eff = max(dt_eff * 0.8, dt_min)
            else:
                dt_eff = min(dt_eff * 1.05, dt_max)

            new_coords = coords + step_disp

            # Validate geometry
            if _min_interatomic_distance(new_coords) < min_interatomic_dist:
                # Reduce step size
                step_disp = step_disp * 0.5
                new_coords = coords + step_disp
                dt_eff = max(dt_eff * 0.5, dt_min)

            # Adaptive dt control based on Morse index
            if best_neg_vib is None or neg_vib < best_neg_vib:
                best_neg_vib = neg_vib
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= plateau_patience:
                    dt_eff = min(dt_eff * 1.5, dt_max)
                    no_improve = 0

            prev_pos = coords.clone()
            coords = new_coords.detach()
            total_steps += 1

        # Did not converge
        final_neg_vib = morse_indices[-1] if morse_indices else -1

        result = KickComparisonResult(
            strategy=kick_strategy_name,
            sample_id=sample_id,
            formula=formula,
            success=(final_neg_vib == 1),
            final_morse_index=final_neg_vib,
            total_steps=total_steps,
            escape_cycles=escape_cycle,
            wall_time=time.time() - t0,
            energies=energies,
            morse_indices=morse_indices,
            escape_events=escape_events,
        )
        if logger is not None:
            logger.finalize(
                final_coords=coords,
                final_morse_index=final_neg_vib,
                converged_to_ts=bool(final_neg_vib == 1),
            )
            logger.save(log_dir, prefix=kick_strategy_name)

        return result

    except Exception as e:
        return KickComparisonResult(
            strategy=kick_strategy_name,
            sample_id=sample_id,
            formula=formula,
            success=False,
            final_morse_index=-1,
            total_steps=0,
            escape_cycles=0,
            wall_time=time.time() - t0,
            error=str(e),
        )


def run_kick_comparison_experiment(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    strategies: List[str] = None,
    *,
    sample_id: str = "unknown",
    formula: str = "",
    out_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, KickComparisonResult]:
    """Run comparison of all (or specified) kick strategies on a single sample.

    Args:
        predict_fn: Energy/force/Hessian prediction function
        coords0: Starting coordinates
        atomic_nums: Atomic numbers
        strategies: List of strategies to compare (None = all)
        sample_id: Sample identifier
        formula: Chemical formula
        out_dir: Output directory for results
        **kwargs: Arguments passed to run_gad_with_kick_strategy

    Returns:
        Dictionary mapping strategy name to results
    """
    if strategies is None:
        strategies = list(KICK_STRATEGIES.keys())

    results = {}

    for strategy in strategies:
        print(f"  Running {strategy}...")
        result = run_gad_with_kick_strategy(
            predict_fn,
            coords0.clone(),
            atomic_nums,
            strategy,
            sample_id=sample_id,
            formula=formula,
            **kwargs,
        )
        results[strategy] = result
        print(f"    → Success: {result.success}, Steps: {result.total_steps}, Escapes: {result.escape_cycles}")

    # Save results if output directory specified
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        summary = {
            strategy: {
                "success": r.success,
                "final_morse_index": r.final_morse_index,
                "total_steps": r.total_steps,
                "escape_cycles": r.escape_cycles,
                "wall_time": r.wall_time,
                "error": r.error,
            }
            for strategy, r in results.items()
        }

        with open(out_path / f"kick_comparison_{sample_id}.json", "w") as f:
            json.dump(summary, f, indent=2)

    return results


def aggregate_comparison_results(
    all_results: List[Dict[str, KickComparisonResult]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate results across multiple samples.

    Returns statistics for each strategy:
    - success_rate
    - mean_steps (when successful)
    - mean_escapes (when successful)
    - mean_wall_time
    """
    # Collect results by strategy
    by_strategy = {}
    for sample_results in all_results:
        for strategy, result in sample_results.items():
            if strategy not in by_strategy:
                by_strategy[strategy] = []
            by_strategy[strategy].append(result)

    aggregated = {}
    for strategy, results in by_strategy.items():
        successes = [r for r in results if r.success]
        aggregated[strategy] = {
            "n_samples": len(results),
            "n_success": len(successes),
            "success_rate": len(successes) / len(results) if results else 0.0,
            "mean_steps_success": np.mean([r.total_steps for r in successes]) if successes else float("nan"),
            "std_steps_success": np.std([r.total_steps for r in successes]) if successes else float("nan"),
            "mean_escapes_success": np.mean([r.escape_cycles for r in successes]) if successes else float("nan"),
            "mean_wall_time": np.mean([r.wall_time for r in results]),
            "final_index_distribution": {},
        }

        # Final index distribution
        for r in results:
            idx = r.final_morse_index
            if idx not in aggregated[strategy]["final_index_distribution"]:
                aggregated[strategy]["final_index_distribution"][idx] = 0
            aggregated[strategy]["final_index_distribution"][idx] += 1

    return aggregated
