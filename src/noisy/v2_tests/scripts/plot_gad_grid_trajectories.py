#!/usr/bin/env python3
"""Plot trajectory statistics for GAD grid runs.

Reads *_trajectory.json files produced by the GAD grid search (saved by
TrajectoryLogger) and creates per-sample diagnostic plots covering:
  - Eigenvalue evolution (eig_0, eig_1) toward TS index-1
  - Trust radius (step_size_eff) and actual step displacement (x_disp_step) over time
  - Negative vibrational mode count / Morse index progress
  - Mode tracking overlap between consecutive steps

Note: TrajectoryLogger saves trajectory data as a column-oriented dict of lists
(field_name -> [val_step0, val_step1, ...]), NOT a list of dicts.
Field names come directly from ExtendedMetrics.to_dict():
  step_size_eff, x_disp_step, eig_0, eig_1, morse_index, grad_norm,
  mode_overlap, eig_gap_01, energy, energy_delta, ...
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def _col(data: dict, key: str) -> np.ndarray:
    """Extract a column from the column-oriented trajectory dict."""
    vals = data.get(key, [])
    return np.array(vals, dtype=float) if vals else np.array([], dtype=float)


def plot_trajectory(traj_path: Path, output_dir: Path) -> None:
    with open(traj_path) as f:
        data = json.load(f)

    # TrajectoryLogger saves column-oriented: {field: [val_per_step, ...]}
    if not data or "step" not in data:
        return

    steps       = _col(data, "step")
    if len(steps) == 0:
        return

    sample_id   = traj_path.stem.replace("_trajectory", "")

    eig_0        = _col(data, "eig_0")
    eig_1        = _col(data, "eig_1")
    eig_product  = eig_0 * eig_1
    morse_index  = _col(data, "morse_index")
    trust_radius = _col(data, "step_size_eff")   # dt_eff stored as step_size_eff
    disp_step    = _col(data, "x_disp_step")     # actual per-step displacement
    grad_norm    = _col(data, "grad_norm")
    overlap      = _col(data, "mode_overlap")
    eig_gap      = _col(data, "eig_gap_01")
    energy_delta = _col(data, "energy_delta")

    # Infer convergence: any step where eig_0 < 0 and eig_1 > 0 counts
    ts_mask = (eig_0 < 0) & (eig_1 > 0)
    converged = bool(ts_mask.any())
    title_suffix = "CONVERGED TO TS" if converged else "not converged"

    fig, axs = plt.subplots(4, 1, figsize=(11, 16), sharex=True)
    fig.suptitle(f"GAD Trajectory: {sample_id}  [{title_suffix}]", fontsize=13)

    # ---- Panel 1: Eigenvalues and TS criterion ----
    ax = axs[0]
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.plot(steps, eig_0, label="eig_0 (lowest vib)", color="tab:blue")
    ax.plot(steps, eig_1, label="eig_1 (2nd lowest)", color="tab:orange", linestyle="--")
    if ts_mask.any():
        ylo, yhi = ax.get_ylim()
        ax.fill_between(steps, ylo, yhi, where=ts_mask,
                        alpha=0.12, color="green", label="TS region (index-1)")
    ax.set_ylabel("Eigenvalue (eV/Å²)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Hessian Eigenvalues  (eig_0 < 0, eig_1 > 0  →  TS index-1)")

    # ---- Panel 2: eig_product and Morse index ----
    ax2 = axs[1]
    finite_ep = eig_product[np.isfinite(eig_product)]
    if len(finite_ep):
        clip = abs(finite_ep).max() * 2
        clipped = np.clip(eig_product, -clip, clip)
    else:
        clipped = eig_product
    ax2.plot(steps, clipped, color="tab:red", label="eig_0 × eig_1")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("eig_0 × eig_1  (eV/Å²)²", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax2_r = ax2.twinx()
    ax2_r.step(steps, morse_index, color="tab:purple", alpha=0.8, label="Morse index")
    ax2_r.set_ylabel("Morse index (neg vib modes)", color="tab:purple")
    ax2_r.tick_params(axis="y", labelcolor="tab:purple")
    max_mi = float(np.nanmax(morse_index)) if np.any(np.isfinite(morse_index)) else 2
    ax2_r.set_ylim(-0.5, max(max_mi, 2) + 0.5)
    ax2.set_title("TS Criterion (eig_product < 0) and Morse Index")
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: Trust radius vs actual step displacement (mirrors NR plot) ----
    ax3 = axs[2]
    ax3.plot(steps, trust_radius, label="Trust Radius (step_size_eff)", color="tab:blue", linestyle="--")
    ax3.plot(steps, disp_step,   label="Actual Step Displacement (x_disp_step)", color="tab:green")
    ax3.plot(steps, grad_norm,   label="Gradient norm", color="tab:brown", alpha=0.5)

    # Highlight steps where trust radius was fully used (disp ≈ trust_radius)
    hit_mask = (
        np.isfinite(trust_radius) & np.isfinite(disp_step) &
        (trust_radius > 0) &
        (disp_step >= trust_radius * 0.98)
    )
    if hit_mask.any():
        ax3.scatter(steps[hit_mask], disp_step[hit_mask],
                    color="red", zorder=5, s=12, label="Hit Trust Radius")

    ax3.set_ylabel("Displacement / Norm (Å or eV/Å)")
    try:
        ax3.set_yscale("log")
    except Exception:
        pass
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Trust Radius and Step Size Evolution")

    # ---- Panel 4: Mode tracking overlap ----
    ax4 = axs[3]
    ax4.plot(steps, overlap, color="tab:cyan", label="|<v_t | v_{t-1}>|")
    ax4.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax4.set_ylabel("|<v_t | v_{t-1}>|")
    ax4.set_xlabel("Step")
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, alpha=0.3)
    ax4.set_title("Mode Tracking Overlap  (1.0 = perfect continuity, drops = mode flip)")

    plt.tight_layout()

    out_file = output_dir / f"{traj_path.stem}.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid_summary(grid_dir: Path, result_glob: str, output_dir: Path) -> None:
    """Bar-chart summary of success rates across combos."""
    import re

    COMBO_RE = re.compile(
        r"mad(?P<mad>[^_]+)_tr(?P<tr>[^_]+)_bl(?P<bl>[^_]+)_pg(?P<pg>true|false)$"
    )

    tags, rates, steps_list = [], [], []
    for result_path in sorted(grid_dir.glob(result_glob)):
        if not result_path.is_file():
            continue
        combo_tag = result_path.parent.name
        if not COMBO_RE.fullmatch(combo_tag):
            continue
        with open(result_path) as f:
            payload = json.load(f)
        m = payload.get("metrics", {})
        tags.append(combo_tag)
        rates.append(float(m.get("success_rate", 0.0)))
        s = m.get("mean_steps_when_success")
        steps_list.append(float(s) if s is not None and math.isfinite(float(s)) else float("nan"))

    if not tags:
        return

    order = sorted(range(len(tags)), key=lambda i: -rates[i])
    tags = [tags[i] for i in order]
    rates = [rates[i] for i in order]
    steps_list = [steps_list[i] for i in order]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(12, len(tags) * 0.6), 10))
    fig.suptitle("GAD Grid Search Summary", fontsize=14)

    x = np.arange(len(tags))
    bars = ax_top.bar(x, rates, color="steelblue", edgecolor="black", linewidth=0.5)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(tags, rotation=75, ha="right", fontsize=7)
    ax_top.set_ylabel("Success Rate")
    ax_top.set_ylim(0, 1.05)
    ax_top.set_title("Success Rate by Configuration (sorted)")
    ax_top.grid(True, axis="y", alpha=0.3)
    for bar, rate in zip(bars, rates):
        ax_top.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{rate:.2f}", ha="center", va="bottom", fontsize=6)

    finite_steps = [(s if math.isfinite(s) else 0) for s in steps_list]
    colors = ["steelblue" if math.isfinite(s) else "lightgray" for s in steps_list]
    ax_bot.bar(x, finite_steps, color=colors, edgecolor="black", linewidth=0.5)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(tags, rotation=75, ha="right", fontsize=7)
    ax_bot.set_ylabel("Mean Steps to TS (when successful)")
    ax_bot.set_title("Steps to TS by Configuration (gray = no successes)")
    ax_bot.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "gad_grid_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved grid summary plot to {output_dir / 'gad_grid_summary.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GAD grid trajectory diagnostics")
    parser.add_argument("--grid-dir", required=True, help="Grid directory with combo subdirs")
    parser.add_argument("--output-dir", required=True, help="Directory to save PNG plots")
    parser.add_argument(
        "--result-glob",
        default="*/gad_*_parallel_*_results.json",
        help="Glob for per-combo result JSON files",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="Plot every N-th trajectory file (default 10) to limit output volume",
    )
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary bar chart across combos
    plot_grid_summary(grid_dir, args.result_glob, output_dir)

    # Per-sample trajectory plots
    traj_files = sorted(grid_dir.rglob("*_trajectory.json"))
    if not traj_files:
        print(f"No trajectory files found under {grid_dir}")
        return

    print(f"Found {len(traj_files)} trajectory files. Plotting every {args.sample_every}-th.")
    plotted = 0
    for i, traj_file in enumerate(traj_files):
        if i % args.sample_every != 0:
            continue

        rel_path = traj_file.relative_to(grid_dir)
        combo_tag = rel_path.parts[0] if len(rel_path.parts) > 1 else "unknown_combo"
        combo_out_dir = output_dir / combo_tag
        combo_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            plot_trajectory(traj_file, combo_out_dir)
            plotted += 1
        except Exception as e:
            print(f"  Warning: failed to plot {traj_file}: {e}")

        if plotted % 20 == 0 and plotted > 0:
            print(f"  Plotted {plotted} files...")

    print(f"Done! {plotted} trajectory plots saved to {output_dir}")


if __name__ == "__main__":
    main()
