#!/usr/bin/env python3
"""Plot trajectory statistics for GAD grid runs.

Reads *_trajectory.json files produced by the GAD grid search and creates
per-sample diagnostic plots covering:
  - Eigenvalue evolution (eig0, eig1, eig_product) toward TS index-1
  - Trust radius (dt_eff) and actual step displacement over time
  - Negative vibrational mode count (Morse index progress)
  - Mode tracking overlap between consecutive steps
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def _nan_array(values: list, key: str) -> np.ndarray:
    return np.array([v.get(key, float("nan")) if isinstance(v, dict) else float("nan") for v in values], dtype=float)


def plot_trajectory(traj_path: Path, output_dir: Path) -> None:
    with open(traj_path) as f:
        data = json.load(f)

    trajectory = data.get("trajectory", [])
    if not trajectory:
        return

    sample_id = data.get("sample_id", "unknown_sample")
    converged = data.get("converged_to_ts", False)
    title_suffix = "CONVERGED" if converged else "not converged"

    steps = np.array([t.get("step", i) for i, t in enumerate(trajectory)])

    eig0         = _nan_array(trajectory, "eig0")
    eig1         = _nan_array(trajectory, "eig1")
    eig_product  = _nan_array(trajectory, "eig_product")
    neg_vib      = _nan_array(trajectory, "neg_vib")
    dt_eff       = _nan_array(trajectory, "dt_eff")
    disp_last    = _nan_array(trajectory, "disp_from_last")
    gad_norm     = _nan_array(trajectory, "gad_norm")
    overlap      = _nan_array(trajectory, "mode_overlap")

    fig, axs = plt.subplots(4, 1, figsize=(11, 16), sharex=True)
    fig.suptitle(f"GAD Trajectory: {sample_id}  [{title_suffix}]", fontsize=13)

    # ---- Panel 1: Eigenvalues and TS criterion ----
    ax = axs[0]
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.plot(steps, eig0, label="eig0 (lowest)", color="tab:blue")
    ax.plot(steps, eig1, label="eig1 (2nd lowest)", color="tab:orange", linestyle="--")
    # Shade TS region (eig0 < 0, eig1 > 0)
    ts_mask = (eig0 < 0) & (eig1 > 0)
    if ts_mask.any():
        ax.fill_between(steps, ax.get_ylim()[0], ax.get_ylim()[1],
                        where=ts_mask, alpha=0.12, color="green", label="TS region")
    ax.set_ylabel("Eigenvalue (eV/Å²)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Hessian Eigenvalues (eig0 < 0, eig1 > 0 → TS index-1)")

    # ---- Panel 2: eig_product and Morse index ----
    ax2 = axs[1]
    finite_ep = eig_product[np.isfinite(eig_product)]
    if len(finite_ep):
        clipped = np.clip(eig_product, -abs(finite_ep).max() * 2, abs(finite_ep).max() * 2)
    else:
        clipped = eig_product
    ax2.plot(steps, clipped, color="tab:red", label="eig0 × eig1")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("eig0 × eig1 (eV/Å²)²", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax2_r = ax2.twinx()
    ax2_r.step(steps, neg_vib, color="tab:purple", alpha=0.8, label="neg_vib count")
    ax2_r.set_ylabel("Neg. vib modes (Morse index)", color="tab:purple")
    ax2_r.tick_params(axis="y", labelcolor="tab:purple")
    ax2_r.set_ylim(-0.5, max(float(np.nanmax(neg_vib)) if np.any(np.isfinite(neg_vib)) else 2, 2) + 0.5)

    ax2.set_title("TS Criterion (eig_product < 0) and Morse Index")
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: Trust radius and step displacement ----
    ax3 = axs[2]
    ax3.plot(steps, dt_eff, label="Trust Radius (dt_eff)", color="tab:blue", linestyle="--")
    ax3.plot(steps, disp_last, label="Actual Disp from Last Step", color="tab:green")
    ax3.plot(steps, gad_norm, label="GAD vector norm", color="tab:brown", alpha=0.5)
    ax3.set_ylabel("Displacement / Norm (Å)")
    try:
        ax3.set_yscale("log")
    except Exception:
        pass
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Trust Radius and Step Size Evolution")

    # ---- Panel 4: Mode tracking overlap ----
    ax4 = axs[3]
    ax4.plot(steps, overlap, color="tab:cyan")
    ax4.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax4.set_ylabel("|<v_t | v_{t-1}>|")
    ax4.set_xlabel("Step")
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, alpha=0.3)
    ax4.set_title("Mode Tracking Overlap (1.0 = perfect continuity)")

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
