from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sanitize_formula(formula: str) -> str:
    import re

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(formula))
    return (safe.strip("_") or "sample")


def plot_gad_trajectory_3x2(
    trajectory: Dict[str, List[Optional[float]]],
    sample_index: int,
    formula: str,
    start_from: str,
    initial_neg_num: int,
    final_neg_num: int,
    *,
    steps_to_ts: Optional[int] = None,
    mode_switch_step: Optional[int] = None,
) -> Tuple[plt.Figure, str]:
    """Standard 3x2 GAD trajectory plot.

    This is copied from `plot_trajectory_new` in `src/gad_gad_euler_rmsd.py`
    to provide a stable, reusable logging primitive.

    Returns:
        (fig, suggested_filename)
    """

    num_steps = len(trajectory.get("energy", []))
    timesteps = np.arange(num_steps)

    def _nanify(values: List[Optional[float]]) -> np.ndarray:
        return np.array([v if v is not None else np.nan for v in values], dtype=float)

    noise_info = ""
    if "_noise" in start_from:
        noise_str = start_from.split("_noise")[1]
        noise_info = f" (noise {noise_str})"

    # Add transition info to title (e.g., "17→4" for 17 neg eigs to 4 neg eigs)
    transition_info = f" [{initial_neg_num}→{final_neg_num}]"

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f"Sample {sample_index}: {formula}{noise_info}{transition_info}", fontsize=14)

    # Panel 1: Energy
    ax = axes[0, 0]
    ax.plot(timesteps, _nanify(trajectory.get("energy", [])), marker=".", lw=1.2, markersize=3)
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Energy")
    ax.set_xlabel("Step")
    if mode_switch_step is not None:
        ax.axvline(mode_switch_step, color="red", linestyle=":", linewidth=2, alpha=0.7, label=f"→MIN @ {mode_switch_step}")

    # Panel 2: Force magnitude
    ax = axes[0, 1]
    ax.plot(timesteps, _nanify(trajectory.get("force_mean", [])), marker=".", color="tab:orange", lw=1.2, markersize=3)
    ax.set_ylabel("Mean |F| (eV/Å)")
    ax.set_title("Force Magnitude")
    ax.set_xlabel("Step")

    # Panel 3: Eigenvalue product
    ax = axes[1, 0]
    eig_product = _nanify(trajectory.get("eig_product", []))
    ax.plot(timesteps, eig_product, marker=".", color="tab:purple", lw=1.2, markersize=3, label="λ₀ * λ₁")
    ax.axhline(0, color="grey", linestyle="--", linewidth=1, zorder=1)
    if len(eig_product) > 0 and not np.isnan(eig_product[0]):
        ax.text(0.02, 0.95, f"Start: {eig_product[0]:.4f}", transform=ax.transAxes, ha="left", va="top", color="tab:purple", fontsize=9)
    if len(eig_product) > 0 and not np.isnan(eig_product[-1]):
        ax.text(0.98, 0.95, f"End: {eig_product[-1]:.4f}", transform=ax.transAxes, ha="right", va="top", color="tab:purple", fontsize=9)
    if steps_to_ts is not None:
        ax.axvline(steps_to_ts, color="green", linestyle="--", linewidth=2, alpha=0.7, label=f"TS @ {steps_to_ts}")
    if mode_switch_step is not None:
        ax.axvline(mode_switch_step, color="red", linestyle=":", linewidth=2, alpha=0.7, label=f"→MIN @ {mode_switch_step}")
    ax.set_ylabel("Eigenvalue Product")
    ax.set_title("Eigenvalue Product (λ₀ * λ₁)")
    ax.set_xlabel("Step")
    ax.legend(loc="best", fontsize=8)

    late_start = 200
    if len(eig_product) > late_start + 50:
        late_eig = eig_product[late_start:]
        late_steps = timesteps[late_start:]
        late_valid = late_eig[~np.isnan(late_eig)]
        if len(late_valid) > 0:
            ax_inset = ax.inset_axes([0.55, 0.1, 0.4, 0.35])
            ax_inset.plot(late_steps, late_eig, marker=".", color="tab:purple", lw=1.0, markersize=2)
            ax_inset.axhline(0, color="grey", linestyle="--", linewidth=0.5, zorder=1)
            y_min, y_max = np.nanmin(late_eig), np.nanmax(late_eig)
            y_range = y_max - y_min
            if y_range > 0:
                ax_inset.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            ax_inset.set_xlabel("Step", fontsize=7)
            ax_inset.set_ylabel("λ₀·λ₁", fontsize=7)
            ax_inset.tick_params(labelsize=6)
            ax_inset.set_title(f"Steps {late_start}+", fontsize=7, pad=2)
            ax_inset.patch.set_alpha(0.9)
            if steps_to_ts is not None and steps_to_ts >= late_start:
                ax_inset.axvline(steps_to_ts, color="green", linestyle="--", linewidth=1, alpha=0.7)
            if mode_switch_step is not None and mode_switch_step >= late_start:
                ax_inset.axvline(mode_switch_step, color="red", linestyle=":", linewidth=1, alpha=0.7)

    # Panel 4: eigenvalues
    ax = axes[1, 1]
    eig0 = _nanify(trajectory.get("eig0", []))
    eig1 = _nanify(trajectory.get("eig1", []))
    ax.plot(timesteps, eig0, marker=".", color="tab:red", lw=1.2, markersize=3, label="λ₀")
    ax.plot(timesteps, eig1, marker=".", color="tab:green", lw=1.2, markersize=3, label="λ₁")
    ax.axhline(0, color="grey", linestyle=":", linewidth=1, zorder=1)
    ax.set_ylabel("Eigenvalue (eV/Å²)")
    ax.set_title("Smallest Eigenvalues (λ₀, λ₁)")
    ax.set_xlabel("Step")
    ax.legend(loc="best", fontsize=8)
    if len(eig0) > 0 and not np.isnan(eig0[-1]):
        ax.text(0.98, 0.05, f"Final λ₀={eig0[-1]:.4f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="tab:red")
    if mode_switch_step is not None:
        ax.axvline(mode_switch_step, color="red", linestyle=":", linewidth=2, alpha=0.7)

    # Panel 5: displacement from last
    ax = axes[2, 0]
    disp_last = _nanify(trajectory.get("disp_from_last", []))
    ax.plot(timesteps, disp_last, marker=".", color="tab:red", lw=1.2, markersize=3)
    ax.set_ylabel("Mean Disp (Å)")
    ax.set_title("Displacement from Last Step")
    ax.set_xlabel("Step")
    if len(disp_last) > 10:
        later_disp = disp_last[10:]
        later_disp_valid = later_disp[~np.isnan(later_disp)]
        if len(later_disp_valid) > 0:
            y_max = np.percentile(later_disp_valid, 99) * 1.2
            y_max = max(y_max, 0.01)
            ax.set_ylim(0, y_max)
            ax.text(0.98, 0.95, f"Avg (>10): {np.mean(later_disp_valid):.4f} Å", transform=ax.transAxes, ha="right", va="top", fontsize=9)

    # Panel 6: displacement from start
    ax = axes[2, 1]
    disp_start = _nanify(trajectory.get("disp_from_start", []))
    ax.plot(timesteps, disp_start, marker=".", color="tab:blue", lw=1.2, markersize=3)
    ax.set_ylabel("Mean Disp (Å)")
    ax.set_title("Displacement from Start")
    ax.set_xlabel("Step")
    if len(disp_start) > 10:
        later_disp_start = disp_start[10:]
        later_disp_start_valid = later_disp_start[~np.isnan(later_disp_start)]
        if len(later_disp_start_valid) > 0:
            y_min = np.percentile(later_disp_start_valid, 1) * 0.9
            y_max = np.percentile(later_disp_start_valid, 99) * 1.1
            ax.set_ylim(max(0, y_min), y_max)
    if len(disp_start) > 0 and not np.isnan(disp_start[-1]):
        ax.text(0.98, 0.95, f"Final: {disp_start[-1]:.4f} Å", transform=ax.transAxes, ha="right", va="top", fontsize=9)

    summary_parts = [f"neg eig: {initial_neg_num} → {final_neg_num}"]
    if steps_to_ts is not None:
        summary_parts.append(f"TS @ {steps_to_ts}")
    if mode_switch_step is not None:
        summary_parts.append(f"→MIN @ {mode_switch_step}")

    axes[2, 1].text(
        0.02,
        0.05,
        "\n".join(summary_parts),
        transform=axes[2, 1].transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Keep filenames consistent with the legacy convention so it's easy to
    # visually bucket outputs by saddle-order transition.
    filename = (
        f"traj_{sample_index:03d}_{_sanitize_formula(formula)}_from_{start_from}_"
        f"{initial_neg_num}to{final_neg_num}.png"
    )
    return fig, filename
