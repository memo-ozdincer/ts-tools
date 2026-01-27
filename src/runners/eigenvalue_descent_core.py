from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Any, Dict, Optional

import torch

from ..dependencies.common_utils import add_common_args, parse_starting_geometry, setup_experiment
from ..dependencies.experiment_logger import ExperimentLogger, RunResult, build_loss_type_flags
from ..logging import finish_wandb, init_wandb_run, log_sample, log_summary
from src.core_algos.signenforcer import sign_enforcer_loss
from ..dependencies.hessian import vibrational_eigvals
from ._predict import make_predict_fn_from_calculator


def run_eigenvalue_descent_core(
    predict_fn,
    coords0: torch.Tensor,
    atomic_nums: torch.Tensor,
    *,
    n_steps: int,
    lr: float,
    loss_type: str,
    sign_neg_target: float,
    sign_pos_floor: float,
    max_disp: float,
) -> Dict[str, Any]:
    coords = coords0.clone().detach().to(torch.float32)
    coords.requires_grad = True

    history = defaultdict(list)
    stop_reason: Optional[str] = None

    for step in range(n_steps):
        out = predict_fn(coords, atomic_nums, do_hessian=True, require_grad=True)
        hessian = out.get("hessian")
        vib = vibrational_eigvals(hessian, coords, atomic_nums)

        if vib.numel() < 2:
            stop_reason = "insufficient_vibrational_eigvals"
            break

        eig0 = vib[0]
        eig1 = vib[1]
        eig_prod = eig0 * eig1
        neg_vib = int((vib < 0).sum().item())

        if loss_type == "eig_product":
            loss = eig_prod
        elif loss_type == "sign_enforcer":
            loss, _ = sign_enforcer_loss(vib, sign_neg_target=sign_neg_target, sign_pos_floor=sign_pos_floor)
        else:
            raise ValueError(f"Unsupported loss_type for core runner: {loss_type}")

        grad = torch.autograd.grad(loss, coords, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            update = lr * grad
            update_per_atom = update.reshape(-1, 3)
            atom_disp = torch.norm(update_per_atom, dim=1)
            max_atom_disp = float(atom_disp.max().item()) if atom_disp.numel() else 0.0
            if max_atom_disp > max_disp and max_atom_disp > 0:
                update = update * (max_disp / max_atom_disp)
                max_atom_disp = float(max_disp)
            coords = (coords - update).detach()
            coords.requires_grad = True

        history["loss"].append(float(loss.detach().cpu().item()))
        history["eig0"].append(float(eig0.detach().cpu().item()))
        history["eig1"].append(float(eig1.detach().cpu().item()))
        history["eig_product"].append(float(eig_prod.detach().cpu().item()))
        history["neg_vibrational"].append(int(neg_vib))
        history["max_atom_disp"].append(float(max_atom_disp))
        history["step"].append(int(step))

        if loss_type == "eig_product" and float(eig_prod.detach().cpu().item()) < -1e-5:
            stop_reason = "eig_product_negative"
            break
        if loss_type == "sign_enforcer" and neg_vib == 1:
            stop_reason = "exactly_one_negative"
            break

    return {
        "final_coords": coords.detach().cpu(),
        "history": history,
        "final_loss": history["loss"][-1] if history["loss"] else float("inf"),
        "final_eig0": history["eig0"][-1] if history["eig0"] else float("nan"),
        "final_eig1": history["eig1"][-1] if history["eig1"] else float("nan"),
        "final_eig_product": history["eig_product"][-1] if history["eig_product"] else float("inf"),
        "final_neg_vibrational": history["neg_vibrational"][-1] if history["neg_vibrational"] else -1,
        "stop_reason": stop_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Core eigenvalue-descent runner (refactored entrypoint).")
    parser = add_common_args(parser)

    parser.add_argument("--n-steps-opt", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--start-from", type=str, default="reactant_noise2A")
    parser.add_argument("--loss-type", type=str, default="eig_product", choices=["eig_product", "sign_enforcer"])

    parser.add_argument("--sign-neg-target", type=float, default=-5e-3)
    parser.add_argument("--sign-pos-floor", type=float, default=1e-3)

    parser.add_argument("--max-disp", type=float, default=0.2, help="Max per-atom step (Å)")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gad-noise-experiments")
    parser.add_argument("--wandb-entity", type=str, default=None)

    args = parser.parse_args()

    if getattr(args, "calculator", "hip").lower() == "scine":
        raise NotImplementedError(
            "`src.runners.eigenvalue_descent_core` requires autograd through the calculator/Hessian. "
            "SCINE is CPU-only and not autograd-differentiable in this repo. Use HIP for this runner, "
            "or keep using the legacy `src.gad_eigenvalue_descent` if you have a SCINE-specific method there."
        )

    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    # setup_experiment() disables grads globally; this runner needs autograd.
    torch.set_grad_enabled(True)
    predict_fn = make_predict_fn_from_calculator(calculator, getattr(args, "calculator", "hip"))

    loss_type_flags = build_loss_type_flags(args)
    logger = ExperimentLogger(
        base_dir=out_dir,
        script_name="eigdescent-core",
        loss_type_flags=loss_type_flags,
        max_graphs_per_transition=10,
        random_seed=42,
    )

    if args.wandb:
        wandb_config = {
            "script": "eigenvalue_descent_core",
            "loss_type": args.loss_type,
            "start_from": args.start_from,
            "n_steps_opt": args.n_steps_opt,
            "lr": args.lr,
            "max_disp": args.max_disp,
            "calculator": getattr(args, "calculator", "hip"),
        }
        init_wandb_run(
            project=args.wandb_project,
            name=f"eigdescent-core_{loss_type_flags}",
            config=wandb_config,
            entity=args.wandb_entity,
            tags=[args.start_from, args.loss_type, "core"],
            run_dir=out_dir,
        )

    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break

        batch = batch.to(device)
        atomic_nums = batch.z.detach().cpu().to(device)
        formula = getattr(batch, "formula", "sample")

        start_coords = parse_starting_geometry(args.start_from, batch, noise_seed=getattr(args, "noise_seed", None), sample_index=i)
        start_coords = start_coords.detach().to(device)

        # Initial saddle order (vibrational), for proper transition bucketing.
        try:
            init_out = predict_fn(start_coords, atomic_nums, do_hessian=True, require_grad=False)
            init_vib = vibrational_eigvals(init_out["hessian"], start_coords, atomic_nums)
            initial_neg = int((init_vib < 0).sum().item())
        except Exception:
            initial_neg = -1

        t0 = time.time()
        out = run_eigenvalue_descent_core(
            predict_fn,
            start_coords,
            atomic_nums,
            n_steps=args.n_steps_opt,
            lr=args.lr,
            loss_type=args.loss_type,
            sign_neg_target=args.sign_neg_target,
            sign_pos_floor=args.sign_pos_floor,
            max_disp=args.max_disp,
        )
        wall = time.time() - t0

        result = RunResult(
            sample_index=i,
            formula=str(formula),
            initial_neg_eigvals=initial_neg,
            final_neg_eigvals=int(out["final_neg_vibrational"]),
            initial_neg_vibrational=None,
            final_neg_vibrational=int(out["final_neg_vibrational"]),
            steps_taken=len(out["history"].get("step", [])),
            steps_to_ts=None,
            final_time=float(wall),
            final_eig0=float(out["final_eig0"]),
            final_eig1=float(out["final_eig1"]),
            final_eig_product=float(out["final_eig_product"]),
            final_loss=float(out["final_loss"]),
            rmsd_to_known_ts=None,
            stop_reason=str(out.get("stop_reason")),
            plot_path=None,
        )
        logger.add_result(result)

        metrics = {
            "final_loss": result.final_loss,
            "final_eig0": result.final_eig0,
            "final_eig1": result.final_eig1,
            "final_eig_product": result.final_eig_product,
            "final_neg_vibrational": result.final_neg_vibrational,
            "steps_taken": result.steps_taken,
            "wallclock_s": result.final_time,
            "stop_reason": result.stop_reason,
        }

        if args.wandb:
            log_sample(i, metrics, fig=None, plot_name=None)

    all_runs_path, aggregate_stats_path = logger.save_all_results()
    summary = logger.compute_aggregate_stats()
    logger.print_summary()

    if args.wandb:
        log_summary(summary)
        finish_wandb()

    print(f"Saved results: {all_runs_path}")
    print(f"Saved stats:   {aggregate_stats_path}")


if __name__ == "__main__":
    main()
