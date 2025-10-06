import os
import json
import re
from typing import Any, Dict, List, Optional

import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data as TGDData, Batch
from torch_geometric.loader import DataLoader

from transition1x import Dataloader as T1xDataloader

from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.frequency_analysis import analyze_frequencies_torch
from ocpmodels.hessian_graph_transform import HessianGraphTransform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Transition1xDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        split: str = "test",
        max_samples: Optional[int] = None,
        transform=None,
    ):
        self.transform = transform
        self.samples: List[TGDData] = []

        loader = T1xDataloader(h5_path, datasplit=split, only_final=True)

        for idx, mol in enumerate(loader):
            if max_samples is not None and len(self.samples) >= max_samples:
                break
            try:
                ts = mol["transition_state"]

                data = TGDData(
                    z=torch.tensor(ts["atomic_numbers"], dtype=torch.long),
                    pos_transition=torch.tensor(ts["positions"], dtype=torch.float),
                    energy=torch.tensor(ts["wB97x_6-31G(d).energy"], dtype=torch.float),
                    forces=torch.tensor(ts["wB97x_6-31G(d).forces"], dtype=torch.float),
                    rxn=ts["rxn"],
                    formula=ts["formula"],
                )

                self.samples.append(data)
            except Exception as e:
                print(f"[WARN] Skipping idx={idx} due to error: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TGDData:
        data = self.samples[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data


# -----------------------------
# Alignment + RMSD utilities
# -----------------------------
def find_rigid_alignment(A: np.ndarray, B: np.ndarray):
    """Kabsch (no masses), handles reflection."""
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    H = A_c.T @ B_c
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    t = b_mean - R @ a_mean
    return R, t


def get_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(((A - B) ** 2).sum(axis=1).mean()))


def align_ordered_and_get_rmsd(A, B) -> float:
    """Rigid-align A to B and compute RMSD. A,B: (N,3), same ordering."""
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor):
        B = B.detach().cpu().numpy()
    if A.shape != B.shape:
        return float("inf")
    R, t = find_rigid_alignment(A, B)
    A_aligned = (R @ A.T).T + t
    return get_rmsd(A_aligned, B)


# -----------------------------
# Pre-transform: use TS coords
# -----------------------------
class UsePos:
    def __init__(self, attr: str = "pos_transition"):
        self.attr = attr

    def __call__(self, data: TGDData) -> TGDData:
        pos = getattr(data, self.attr, None)
        if pos is None:
            raise ValueError(f"Data missing '{self.attr}'. Keys: {list(data.keys())}")
        data.pos = pos
        return data


# -----------------------------
# GAD (Euler) loop
# -----------------------------
def _scalar_from(results: Dict[str, Any], key: str) -> Optional[float]:
    value = results.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().cpu().view(-1)[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prepare_hessian(hess: torch.Tensor, num_atoms: int) -> torch.Tensor:
    if hess.dim() == 1:
        side = int(hess.numel() ** 0.5)
        hess = hess.view(side, side)
    elif hess.dim() == 3 and hess.shape[0] == 1:
        hess = hess[0]
    elif hess.dim() > 2:
        hess = hess.reshape(3 * num_atoms, 3 * num_atoms)
    return hess


def _extract_smallest_eigs(
    hessian: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
):
    """Return the two smallest eigenvalues and their product via frequency analysis."""
    try:
        freq_info = analyze_frequencies_torch(hessian, positions, atomic_numbers)
    except Exception as exc:
        print(f"[WARN] Frequency analysis failed: {exc}")
        return None, None, None

    eigvals = freq_info.get("eigvals")
    if eigvals is None:
        return None, None, None
    if not isinstance(eigvals, torch.Tensor):
        eigvals = torch.as_tensor(eigvals)

    eigvals = eigvals.detach()
    if eigvals.device.type != "cpu":
        eigvals = eigvals.cpu()
    eigvals = eigvals.reshape(-1)
    if eigvals.numel() < 2:
        return None, None, None

    eigvals_sorted, _ = torch.sort(eigvals)
    smallest = float(eigvals_sorted[0].item())
    second_smallest = float(eigvals_sorted[1].item())
    return smallest, second_smallest, smallest * second_smallest


def _mean_vector_magnitude(vec: torch.Tensor) -> float:
    vec = vec.detach()
    if vec.device.type != "cpu":
        vec = vec.cpu()
    return float(vec.norm(dim=1).mean().item())


def run_gad_euler_on_batch(
    calculator: EquiformerTorchCalculator,
    batch: Batch,
    n_steps: int = 200,
    dt: float = 0.01,
) -> Dict[str, Any]:
    """
    Runs GAD Euler updates on positions in `batch` using predicted Hessians.
    Returns dict with RMSD, forces, natoms.
    """
    assert int(batch.batch.max().item()) + 1 == 1, "Use batch_size=1."
    start_pos = batch.pos.detach().clone()

    results = calculator.predict(batch, do_hessian=True)
    energy_start = _scalar_from(results, "energy")

    trajectory = {
        "energy": [],
        "force_mean": [],
        "gad_mean": [],
        "eig_min1": [],
        "eig_min2": [],
        "eig_product": [],
    }

    atomic_numbers = batch.z

    def _record_step(predictions: Dict[str, Any], gad_vec: Optional[torch.Tensor]):
        trajectory["energy"].append(_scalar_from(predictions, "energy"))
        trajectory["force_mean"].append(_mean_vector_magnitude(predictions["forces"]))
        if gad_vec is None:
            trajectory["gad_mean"].append(None)
        else:
            trajectory["gad_mean"].append(_mean_vector_magnitude(gad_vec))

        hessian = predictions.get("hessian")
        if hessian is None:
            trajectory["eig_min1"].append(None)
            trajectory["eig_min2"].append(None)
            trajectory["eig_product"].append(None)
        else:
            eig0, eig1, eig_prod = _extract_smallest_eigs(hessian, batch.pos, atomic_numbers)
            trajectory["eig_min1"].append(eig0)
            trajectory["eig_min2"].append(eig1)
            trajectory["eig_product"].append(eig_prod)

    _record_step(results, gad_vec=None)

    for _ in range(n_steps):
        forces = results["forces"]  # (N,3)
        N = forces.shape[0]
        hess = _prepare_hessian(results["hessian"], N)

        # smallest eigenvector
        evals, evecs = torch.linalg.eigh(hess)
        v = evecs[:, 0]
        v = v / (v.norm() + 1e-12)

        # GAD velocity: F + 2(-F·v)v   (since F = -∇V)
        f_flat = forces.reshape(-1)
        gad_flat = f_flat + 2.0 * torch.dot(-f_flat, v) * v
        gad = gad_flat.view(N, 3)

        # Euler update
        batch.pos = batch.pos + dt * gad

        results = calculator.predict(batch, do_hessian=True)
        _record_step(results, gad)

    end_pos = batch.pos.detach().clone()
    rmsd_val = align_ordered_and_get_rmsd(start_pos, end_pos)

    forces_end = results["forces"]
    rms_force = forces_end.pow(2).mean().sqrt().item()
    max_atom_force = forces_end.norm(dim=1).max().item()
    energy_end = _scalar_from(results, "energy")

    def _to_float(value: Optional[float]) -> float:
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    min_eig_start = _to_float(trajectory["eig_min1"][0] if trajectory["eig_min1"] else None)
    min_eig_end = _to_float(trajectory["eig_min1"][-1] if trajectory["eig_min1"] else None)
    eig_product_start = _to_float(trajectory["eig_product"][0] if trajectory["eig_product"] else None)
    eig_product_end = _to_float(trajectory["eig_product"][-1] if trajectory["eig_product"] else None)

    displacement = (end_pos - start_pos).norm(dim=1)

    return {
        "rmsd": float(rmsd_val),
        "rms_force_end": float(rms_force),
        "max_atom_force_end": float(max_atom_force),
        "natoms": int(start_pos.shape[0]),
        "energy_start": energy_start,
        "energy_end": energy_end,
        "min_hess_eig_start": min_eig_start,
        "min_hess_eig_end": min_eig_end,
        "eig_product_start": eig_product_start,
        "eig_product_end": eig_product_end,
        "mean_displacement": float(displacement.mean().item()),
        "max_displacement": float(displacement.max().item()),
        "trajectory": {
            "energy": trajectory["energy"],
            "force_mean": trajectory["force_mean"],
            "gad_mean": trajectory["gad_mean"],
            "eig_min1": trajectory["eig_min1"],
            "eig_min2": trajectory["eig_min2"],
            "eig_product": trajectory["eig_product"],
        },
    }


def _sanitize_formula(formula: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", formula)
    safe = safe.strip("_")
    return safe or "sample"


def plot_trajectory(
    trajectory: Dict[str, List[Optional[float]]],
    sample_index: int,
    formula: str,
    out_dir: str,
) -> str:
    """Create and save a 4-panel trajectory plot. Returns path to image."""
    num_steps = len(trajectory.get("energy", []))
    timesteps = np.arange(num_steps)

    def _nanify(values: Optional[List[Optional[float]]]) -> np.ndarray:
        values = list(values) if values is not None else []
        if len(values) < num_steps:
            values = values + [None] * (num_steps - len(values))
        return np.array([np.nan if v is None else v for v in values], dtype=float)

    energies = _nanify(trajectory.get("energy"))
    force_mean = _nanify(trajectory.get("force_mean"))
    gad_mean = _nanify(trajectory.get("gad_mean"))
    eig_min1 = _nanify(trajectory.get("eig_min1"))
    eig_min2 = _nanify(trajectory.get("eig_min2"))
    eig_product = _nanify(trajectory.get("eig_product"))

    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)

    axes[0].plot(timesteps, energies, marker="o", linewidth=1.2)
    axes[0].set_ylabel("Energy (eV)")
    axes[0].set_title("Energy over time")

    axes[1].plot(timesteps, force_mean, marker="o", color="tab:orange", linewidth=1.2)
    axes[1].set_ylabel("Mean |F| (eV/Å)")
    axes[1].set_title("Force magnitude over time")

    eig_ax = axes[2]
    eig_ax.plot(timesteps, eig_min1, marker="o", color="tab:blue", linewidth=1.2, label="lambda_0")
    eig_ax.plot(timesteps, eig_min2, marker="o", color="tab:red", linewidth=1.2, label="lambda_1")
    eig_ax.set_ylabel("Eigenvalues")
    eig_ax.set_title("Smallest Hessian eigenvalues")

    prod_ax = eig_ax.twinx()
    prod_ax.plot(
        timesteps,
        eig_product,
        marker="o",
        color="tab:purple",
        linewidth=1.2,
        linestyle="--",
        label="lambda_0*lambda_1",
    )
    prod_ax.set_ylabel("Product")

    for step_idx, (t, prod_val, eig0_val, eig1_val) in enumerate(
        zip(timesteps, eig_product, eig_min1, eig_min2)
    ):
        if step_idx % 5 != 0:
            continue
        if any(np.isnan(val) for val in (prod_val, eig0_val, eig1_val)):
            continue
        label = f"λ0={eig0_val:.3f}\nλ1={eig1_val:.3f}"
        prod_ax.annotate(
            label,
            xy=(t, prod_val),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize="x-small",
            color="tab:purple",
        )

    eig_lines, eig_labels = eig_ax.get_legend_handles_labels()
    prod_lines, prod_labels = prod_ax.get_legend_handles_labels()
    eig_ax.legend(eig_lines + prod_lines, eig_labels + prod_labels, loc="best", fontsize="small")

    axes[3].plot(timesteps, gad_mean, marker="o", color="tab:green", linewidth=1.2)
    axes[3].set_ylabel("Mean |GAD| (Å)")
    axes[3].set_xlabel("Step")
    axes[3].set_title("GAD vector magnitude over time")

    fig.tight_layout()

    safe_formula = _sanitize_formula(formula)
    filename = f"rgd1_gad_traj_{sample_index:03d}_{safe_formula}.png"
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    HOME = os.path.expanduser("~")
    PROJECT = "/project/memo"  # big files here

    MAX_SAMPLES = 20  # number of unique samples to process
    DATASET_LOAD_MULTIPLIER = 5  # load more candidates so we can pick unique formulas
    DATASET_MAX_SAMPLES = MAX_SAMPLES * DATASET_LOAD_MULTIPLIER
    SELECT_UNIQUE_FORMULAS = True
    T1X_SPLIT = "test"
    N_STEPS = 50
    DT = 0.005

    # I/O layout
    checkpoint_path = os.path.join(PROJECT, "large-files", "ckpt", "hesspred_v1.ckpt")
    h5_path = os.path.join(
        PROJECT,
        "large-files",
        "data",
        "transition1x.h5",
    )
    out_dir = os.path.join(HOME, "large-files", "out")
    os.makedirs(out_dir, exist_ok=True)

    calculator = EquiformerTorchCalculator(
        checkpoint_path=checkpoint_path,
        hessian_method="predict",
    )

    use_pos_tf = UsePos("pos_transition")
    # hess_tf = HessianGraphTransform(
    #     cutoff=calculator.potential.cutoff,
    #     max_neighbors=calculator.potential.max_neighbors,
    #     use_pbc=getattr(calculator.potential, "use_pbc", False),
    # )
    # composed_tf = lambda d: hess_tf(use_pos_tf(d))

    dataset = Transition1xDataset(
        h5_path=h5_path,
        split=T1X_SPLIT,
        max_samples=DATASET_MAX_SAMPLES,
        transform=use_pos_tf,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    results_summary: List[Dict[str, Any]] = []

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset:    {h5_path} (split={T1X_SPLIT})")
    print(f"Device:     {device}")
    print(f"Loaded samples: {len(dataset)}")
    print(f"Processing up to {MAX_SAMPLES} samples with GAD-Euler (steps={N_STEPS}, dt={DT})")

    if len(dataset) == 0:
        raise RuntimeError("No Transition1x samples loaded. Check h5 path and split.")

    plot_dir = os.path.join(out_dir, "gad_trajectories")
    os.makedirs(plot_dir, exist_ok=True)

    seen_formulas = set()
    processed = 0

    for dataset_idx, batch in enumerate(dataloader):
        if processed >= MAX_SAMPLES:
            break
        try:
            formula_attr = getattr(batch, "formula", "")
            if isinstance(formula_attr, (list, tuple)):
                formula = formula_attr[0]
            elif isinstance(formula_attr, torch.Tensor):
                if formula_attr.numel() == 1:
                    formula = formula_attr.item()
                else:
                    formula = formula_attr[0].item()
            else:
                formula = formula_attr
            if isinstance(formula, bytes):
                formula = formula.decode("utf-8", errors="ignore")
            formula = str(formula)

            if SELECT_UNIQUE_FORMULAS and formula in seen_formulas:
                continue
            seen_formulas.add(formula)

            batch.natoms=torch.tensor([batch.pos.shape[0]], dtype=torch.long)
            batch = batch.to(device)
            out = run_gad_euler_on_batch(calculator, batch, n_steps=N_STEPS, dt=DT)
            sample_idx = processed
            processed += 1
            plot_path = plot_trajectory(out["trajectory"], sample_idx, formula, plot_dir)
            plot_path_rel = os.path.relpath(plot_path, out_dir)

            result = {
                "dataset_index": dataset_idx,
                "sample_order": sample_idx,
                "formula": formula,
                "plot_path": plot_path_rel,
                **out,
            }
            results_summary.append(result)
            energy_start = out["energy_start"]
            energy_end = out["energy_end"]
            if energy_start is not None and energy_end is not None:
                delta_e_str = f"ΔE={energy_end - energy_start:.5f} eV"
            else:
                delta_e_str = "ΔE=NA"
            print(
                f"[sample {sample_idx}] N={out['natoms']}, RMSD={out['rmsd']:.4f} Å, {delta_e_str}, "
                f"RMS|F|_end={out['rms_force_end']:.3f} eV/Å, "
                f"max|F_i|_end={out['max_atom_force_end']:.3f} eV/Å, "
                f"λmin_start={out['min_hess_eig_start']:.5f}, "
                f"λmin_end={out['min_hess_eig_end']:.5f}, "
                f"mean disp={out['mean_displacement']:.4f} Å, "
                f"max disp={out['max_displacement']:.4f} Å, "
                f"formula={formula}"
            )
        except Exception as e:
            print(f"[{dataset_idx}] ERROR: {e}")

    out_json = os.path.join(out_dir, f"rgd1_gad_rmsd_{len(results_summary)}.json")
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(
        f"\nSaved GAD RMSD summary for {len(results_summary)} unique samples → {out_json}"
    )
