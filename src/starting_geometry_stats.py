"""
Generate and analyze starting geometries with various noise levels.

This script tests different starting positions by adding noise and computes statistics
on the number of negative eigenvalues (imaginary frequencies) at each geometry.
The goal is to find noise levels that produce a good distribution of higher-order
saddle points (multiple negative eigenvalues).
"""

import os
import json
import argparse
from typing import Any, Dict, List, Optional
from collections import defaultdict, Counter

import torch
import numpy as np
from torch_geometric.data import Data as TGData, Batch as TGBatch

from .common_utils import setup_experiment, add_common_args
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from .differentiable_projection import differentiable_massweigh_and_eckartprojection_torch as massweigh_and_eckartprojection_torch
from hip.ff_lmdb import Z_TO_ATOM_SYMBOL

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# W&B import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def coord_atoms_to_torch_geometric(coords, atomic_nums, device):
    """Convert coordinates and atomic numbers to a PyG batch for model inference."""
    if isinstance(coords, torch.Tensor) and coords.dim() == 1:
        coords = coords.reshape(-1, 3)

    if isinstance(coords, torch.Tensor):
        if torch.isnan(coords).any() or torch.isinf(coords).any():
            raise ValueError("Invalid coordinates detected (NaN or Inf)")

    data = TGData(
        pos=torch.as_tensor(coords, dtype=torch.float32).reshape(-1, 3),
        z=torch.as_tensor(atomic_nums, dtype=torch.int64),
        charges=torch.as_tensor(atomic_nums, dtype=torch.int64),
        natoms=torch.tensor([len(atomic_nums)], dtype=torch.int64),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool),
    )
    return TGBatch.from_data_list([data]).to(device)


def add_gaussian_noise_to_coords(coords: torch.Tensor, noise_rms_angstrom: float) -> torch.Tensor:
    """
    Add Gaussian noise to coordinates with specified RMS amplitude.

    Args:
        coords: (N, 3) or (3N,) tensor of coordinates
        noise_rms_angstrom: Target RMS displacement in Angstroms

    Returns:
        Noisy coordinates with same shape as input
    """
    original_shape = coords.shape
    coords_flat = coords.reshape(-1, 3)

    # Generate Gaussian noise with std = noise_rms_angstrom
    # This gives RMS displacement ≈ noise_rms_angstrom
    noise = torch.randn_like(coords_flat) * noise_rms_angstrom

    noisy_coords = coords_flat + noise
    return noisy_coords.reshape(original_shape)


def compute_eigenvalue_statistics(
    calculator: EquiformerTorchCalculator,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    device: str,
) -> Dict[str, Any]:
    """
    Compute Hessian eigenvalues at a given geometry and return statistics.

    Returns dict with:
        - vibrational_eigvals: eigenvalues after removing rigid-body modes
        - neg_count: number of negative eigenvalues
        - neg_eigvals: list of negative eigenvalues
        - pos_eigvals: list of positive eigenvalues
        - all_eigvals: all eigenvalues (including near-zero rigid modes)
    """
    model = calculator.potential
    atomsymbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in atomic_nums]

    try:
        with torch.no_grad():
            # Ensure coords is on the correct device before using it
            coords = coords.to(device)
            batch = coord_atoms_to_torch_geometric(coords, atomic_nums, device)
            _, _, out = model.forward(batch, otf_graph=True)
            hess_raw = out["hessian"].reshape(coords.numel(), coords.numel())
            hess_proj = massweigh_and_eckartprojection_torch(hess_raw, coords, atomsymbols)
            eigvals, _ = torch.linalg.eigh(hess_proj)

            # Remove rigid-body modes (6 smallest by absolute value)
            coords_cent = coords.detach().reshape(-1, 3).to(torch.float64)
            coords_cent = coords_cent - coords_cent.mean(dim=0, keepdim=True)
            geom_rank = torch.linalg.matrix_rank(coords_cent.cpu(), tol=1e-8).item()
            expected_rigid = 5 if geom_rank <= 2 else 6
            total_modes = eigvals.shape[0]
            expected_rigid = min(expected_rigid, max(0, total_modes - 2))

            abs_sorted_idx = torch.argsort(torch.abs(eigvals))
            keep_idx = abs_sorted_idx[expected_rigid:]
            keep_idx, _ = torch.sort(keep_idx)
            vibrational_eigvals = eigvals[keep_idx]

            neg_mask = vibrational_eigvals < 0
            neg_count = neg_mask.sum().item()
            neg_eigvals = vibrational_eigvals[neg_mask].cpu().numpy().tolist()
            pos_eigvals = vibrational_eigvals[~neg_mask].cpu().numpy().tolist()

            return {
                "vibrational_eigvals": vibrational_eigvals.cpu().numpy().tolist(),
                "neg_count": neg_count,
                "neg_eigvals": neg_eigvals,
                "pos_eigvals": pos_eigvals,
                "all_eigvals": eigvals.cpu().numpy().tolist(),
                "num_rigid_removed": expected_rigid,
                "success": True,
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "neg_count": -1,
        }


def generate_starting_geometries(
    batch,
    noise_levels: List[float],
    n_samples_per_noise: int = 5,
) -> List[Dict[str, Any]]:
    """
    Generate a list of starting geometries with various noise levels.

    Args:
        batch: Data batch containing pos_reactant, pos_transition, etc.
        noise_levels: List of RMS noise levels in Angstroms
        n_samples_per_noise: Number of random samples per noise level

    Returns:
        List of dicts with keys: {name, coords, noise_level}
    """
    geometries = []

    # Base geometries (no noise)
    base_geoms = [
        ("reactant", batch.pos_reactant),
        ("ts", batch.pos_transition),
        ("midpoint_rt", 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition),
        ("quarter_rt", 0.75 * batch.pos_reactant + 0.25 * batch.pos_transition),
        ("three_quarter_rt", 0.25 * batch.pos_reactant + 0.75 * batch.pos_transition),
    ]

    for name, coords in base_geoms:
        geometries.append({
            "name": name,
            "coords": coords.clone(),
            "noise_level": 0.0,
        })

    # Add noise to each base geometry
    for noise_rms in noise_levels:
        for base_name, base_coords in base_geoms:
            for sample_idx in range(n_samples_per_noise):
                noisy_coords = add_gaussian_noise_to_coords(base_coords.clone(), noise_rms)
                geometries.append({
                    "name": f"{base_name}_noise{noise_rms:.1f}A_s{sample_idx}",
                    "coords": noisy_coords,
                    "noise_level": noise_rms,
                    "base_geometry": base_name,
                    "sample_idx": sample_idx,
                })

    return geometries


def plot_eigenvalue_distribution(
    results: List[Dict[str, Any]],
    out_dir: str,
    sample_index: int,
    formula: str,
) -> Optional[str]:
    """
    Plot distribution of negative eigenvalue counts vs noise level.
    """
    # Organize data by noise level and base geometry
    noise_levels = sorted(set(r["noise_level"] for r in results))

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle(f"Eigenvalue Statistics (Sample {sample_index}): {formula}", fontsize=14)

    # Plot 1: Histogram of negative eigenvalue counts
    neg_counts = [r["neg_count"] for r in results if r["success"]]
    if neg_counts:
        axes[0].hist(neg_counts, bins=range(min(neg_counts), max(neg_counts) + 2),
                     align='left', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel("Number of Negative Eigenvalues")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Distribution of Negative Eigenvalue Counts")
        axes[0].grid(True, alpha=0.3)

    # Plot 2: Negative eigenvalue count vs noise level
    noise_vs_neg = defaultdict(list)
    for r in results:
        if r["success"]:
            noise_vs_neg[r["noise_level"]].append(r["neg_count"])

    if noise_vs_neg:
        noise_sorted = sorted(noise_vs_neg.keys())
        means = [np.mean(noise_vs_neg[n]) for n in noise_sorted]
        stds = [np.std(noise_vs_neg[n]) for n in noise_sorted]

        axes[1].errorbar(noise_sorted, means, yerr=stds, marker='o', capsize=5,
                        linewidth=2, markersize=8, label='Mean ± Std')

        # Show individual points
        for noise in noise_sorted:
            counts = noise_vs_neg[noise]
            x_jitter = noise + np.random.normal(0, noise * 0.02, len(counts))
            axes[1].scatter(x_jitter, counts, alpha=0.3, s=20, color='gray')

        axes[1].set_xlabel("Noise Level (RMS Angstrom)")
        axes[1].set_ylabel("Number of Negative Eigenvalues")
        axes[1].set_title("Negative Eigenvalues vs Noise Level")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    filename = f"starting_geom_stats_{sample_index:03d}_{formula}.png"
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_summary_statistics(
    all_results: List[Dict[str, Any]],
    out_dir: str,
) -> Optional[str]:
    """
    Plot summary statistics across all samples.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Summary Statistics Across All Samples", fontsize=14)

    # Collect data
    noise_levels = sorted(set(r["noise_level"] for r in all_results))

    # Plot 1: Overall distribution of negative eigenvalues
    neg_counts = [r["neg_count"] for r in all_results if r["success"]]
    if neg_counts:
        counter = Counter(neg_counts)
        counts_sorted = sorted(counter.keys())
        frequencies = [counter[c] for c in counts_sorted]

        axes[0, 0].bar(counts_sorted, frequencies, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel("Number of Negative Eigenvalues")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Overall Distribution of Negative Eigenvalues")
        axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Average negative count vs noise level
    noise_vs_neg = defaultdict(list)
    for r in all_results:
        if r["success"]:
            noise_vs_neg[r["noise_level"]].append(r["neg_count"])

    if noise_vs_neg:
        noise_sorted = sorted(noise_vs_neg.keys())
        means = [np.mean(noise_vs_neg[n]) for n in noise_sorted]
        stds = [np.std(noise_vs_neg[n]) for n in noise_sorted]

        axes[0, 1].errorbar(noise_sorted, means, yerr=stds, marker='o', capsize=5,
                           linewidth=2, markersize=8)
        axes[0, 1].set_xlabel("Noise Level (RMS Angstrom)")
        axes[0, 1].set_ylabel("Mean Number of Negative Eigenvalues")
        axes[0, 1].set_title("Mean Negative Eigenvalues vs Noise Level")
        axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Violin plot of negative counts by noise level
    if noise_vs_neg:
        data_for_violin = [noise_vs_neg[n] for n in noise_sorted]
        positions = list(range(len(noise_sorted)))

        parts = axes[1, 0].violinplot(data_for_violin, positions=positions,
                                       showmeans=True, showextrema=True)
        axes[1, 0].set_xticks(positions)
        axes[1, 0].set_xticklabels([f"{n:.1f}" for n in noise_sorted])
        axes[1, 0].set_xlabel("Noise Level (RMS Angstrom)")
        axes[1, 0].set_ylabel("Number of Negative Eigenvalues")
        axes[1, 0].set_title("Distribution by Noise Level (Violin Plot)")
        axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Percentage with >1 negative eigenvalue vs noise
    if noise_vs_neg:
        percentages = []
        for n in noise_sorted:
            counts = noise_vs_neg[n]
            pct_gt1 = 100 * sum(1 for c in counts if c > 1) / len(counts)
            percentages.append(pct_gt1)

        axes[1, 1].plot(noise_sorted, percentages, marker='o', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel("Noise Level (RMS Angstrom)")
        axes[1, 1].set_ylabel("% with >1 Negative Eigenvalue")
        axes[1, 1].set_title("Higher-Order Saddle Points vs Noise")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(50, color='red', linestyle='--', alpha=0.5, label='50%')
        axes[1, 1].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    filename = "starting_geom_summary_stats.png"
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and analyze starting geometries with various noise levels."
    )
    parser = add_common_args(parser)

    parser.add_argument("--noise-levels", type=float, nargs="+",
                       default=[0.5, 1.0, 2.0, 5.0, 10.0],
                       help="List of RMS noise levels in Angstroms (default: 0.5 1.0 2.0 5.0 10.0)")
    parser.add_argument("--n-samples-per-noise", type=int, default=5,
                       help="Number of random samples per noise level (default: 5)")

    # W&B arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="gad-ts-search",
                        help="W&B project name (default: gad-ts-search)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity/username (optional)")

    args = parser.parse_args()

    calculator, dataloader, device, out_dir = setup_experiment(args, shuffle=False)
    stats_out_dir = os.path.join(out_dir, "starting_geometry_stats")
    os.makedirs(stats_out_dir, exist_ok=True)

    # Initialize W&B if requested
    wandb_run = None
    use_wandb = args.wandb and WANDB_AVAILABLE

    if use_wandb:
        if not WANDB_AVAILABLE:
            print("[WARNING] W&B requested but not installed. Install with: pip install wandb")
            use_wandb = False
        else:
            # Prepare W&B config
            wandb_config = {
                "script": "starting_geometry_stats",
                "noise_levels": args.noise_levels,
                "n_samples_per_noise": args.n_samples_per_noise,
                "max_samples": args.max_samples,
            }

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name="starting-geom-stats",
                tags=["starting-geometry", "eigenvalue-stats"],
                config=wandb_config,
                dir=str(stats_out_dir),
            )
            print(f"[W&B] Initialized run: {wandb_run.name} ({wandb_run.url})")

    print("=" * 70)
    print("STARTING GEOMETRY EIGENVALUE STATISTICS")
    print("=" * 70)
    print(f"Noise levels (RMS Angstroms): {args.noise_levels}")
    print(f"Samples per noise level: {args.n_samples_per_noise}")
    print(f"Output directory: {stats_out_dir}")
    print("=" * 70)

    all_results = []
    all_samples_summary = []

    for mol_idx, batch in enumerate(dataloader):
        if mol_idx >= args.max_samples:
            break

        print(f"\n{'='*70}")
        print(f"Processing Sample {mol_idx}: {batch.formula[0]}")
        print(f"{'='*70}")

        # Generate all starting geometries for this molecule
        geometries = generate_starting_geometries(
            batch,
            args.noise_levels,
            args.n_samples_per_noise
        )

        print(f"Generated {len(geometries)} starting geometries")

        sample_results = []
        for geom_idx, geom in enumerate(geometries):
            print(f"  [{geom_idx+1}/{len(geometries)}] Analyzing: {geom['name']}")

            stats = compute_eigenvalue_statistics(
                calculator=calculator,
                coords=geom["coords"],
                atomic_nums=batch.z,
                device=device,
            )

            result = {
                "mol_index": mol_idx,
                "formula": batch.formula[0],
                "geometry_name": geom["name"],
                "noise_level": geom["noise_level"],
                "success": stats["success"],
                "neg_count": stats.get("neg_count", -1),
            }

            if stats["success"]:
                result.update({
                    "neg_eigvals": stats["neg_eigvals"],
                    "pos_eigvals": stats["pos_eigvals"][:5],  # Only store first 5 positive
                    "num_rigid_removed": stats["num_rigid_removed"],
                })
                print(f"      Negative eigenvalues: {stats['neg_count']}")
            else:
                print(f"      FAILED: {stats.get('error', 'Unknown error')}")

            sample_results.append(result)
            all_results.append(result)

            # Log to W&B
            if use_wandb and wandb_run is not None and stats["success"]:
                wandb.log({
                    "geometry/mol_index": mol_idx,
                    "geometry/noise_level": geom["noise_level"],
                    "geometry/neg_count": stats["neg_count"],
                    "geometry/geometry_name": geom["name"],
                })

        # Generate per-molecule plot
        plot_path = plot_eigenvalue_distribution(
            sample_results,
            stats_out_dir,
            mol_idx,
            batch.formula[0]
        )
        if plot_path:
            print(f"\nSaved plot: {os.path.basename(plot_path)}")

            # Upload plot to W&B
            if use_wandb and wandb_run is not None:
                wandb.log({
                    f"plots/molecule_{mol_idx}": wandb.Image(plot_path),
                    "mol_index": mol_idx,
                })

        # Compute per-molecule summary
        successful = [r for r in sample_results if r["success"]]
        if successful:
            neg_counts = [r["neg_count"] for r in successful]
            counter = Counter(neg_counts)

            sample_summary = {
                "mol_index": mol_idx,
                "formula": batch.formula[0],
                "total_geometries": len(sample_results),
                "successful": len(successful),
                "mean_neg_count": np.mean(neg_counts),
                "std_neg_count": np.std(neg_counts),
                "min_neg_count": min(neg_counts),
                "max_neg_count": max(neg_counts),
                "distribution": dict(counter),
                "pct_higher_order": 100 * sum(1 for c in neg_counts if c > 1) / len(neg_counts),
            }
            all_samples_summary.append(sample_summary)

            print(f"\nSample {mol_idx} Summary:")
            print(f"  Negative eigenvalue count: {sample_summary['mean_neg_count']:.2f} ± {sample_summary['std_neg_count']:.2f}")
            print(f"  Range: [{sample_summary['min_neg_count']}, {sample_summary['max_neg_count']}]")
            print(f"  Distribution: {dict(counter)}")
            print(f"  Higher-order saddles (>1 neg): {sample_summary['pct_higher_order']:.1f}%")

    # Save all results to JSON
    results_json = os.path.join(stats_out_dir, "all_geometry_stats.json")
    with open(results_json, "w") as f:
        json.dump({
            "config": {
                "noise_levels": args.noise_levels,
                "n_samples_per_noise": args.n_samples_per_noise,
                "max_samples": args.max_samples,
            },
            "per_geometry_results": all_results,
            "per_molecule_summary": all_samples_summary,
        }, f, indent=2)
    print(f"\nSaved detailed results to: {results_json}")

    # Generate summary plot
    summary_plot = plot_summary_statistics(all_results, stats_out_dir)
    if summary_plot:
        print(f"Saved summary plot: {os.path.basename(summary_plot)}")

        # Upload summary plot to W&B
        if use_wandb and wandb_run is not None:
            wandb.log({"plots/summary": wandb.Image(summary_plot)})

    # Print final summary
    print("\n" + "=" * 70)
    print(" " * 20 + "FINAL SUMMARY")
    print("=" * 70)

    if all_samples_summary:
        overall_mean = np.mean([s["mean_neg_count"] for s in all_samples_summary])
        overall_std = np.mean([s["std_neg_count"] for s in all_samples_summary])
        overall_pct_higher = np.mean([s["pct_higher_order"] for s in all_samples_summary])

        print(f"\nOverall statistics (averaged across {len(all_samples_summary)} molecules):")
        print(f"  Mean negative eigenvalue count: {overall_mean:.2f} ± {overall_std:.2f}")
        print(f"  Higher-order saddles (>1 neg): {overall_pct_higher:.1f}%")

        # Distribution by noise level
        print(f"\nStatistics by noise level:")
        for noise in sorted(set(args.noise_levels + [0.0])):
            noise_results = [r for r in all_results if r["success"] and r["noise_level"] == noise]
            if noise_results:
                neg_counts = [r["neg_count"] for r in noise_results]
                pct_higher = 100 * sum(1 for c in neg_counts if c > 1) / len(neg_counts)
                print(f"  {noise:5.1f} Å: {np.mean(neg_counts):5.2f} ± {np.std(neg_counts):4.2f}  "
                      f"(higher-order: {pct_higher:5.1f}%)")

        # Overall distribution
        all_neg_counts = [r["neg_count"] for r in all_results if r["success"]]
        counter = Counter(all_neg_counts)
        print(f"\nOverall distribution of negative eigenvalues:")
        for count in sorted(counter.keys()):
            pct = 100 * counter[count] / len(all_neg_counts)
            print(f"  {count} neg eig: {counter[count]:4d} geometries ({pct:5.1f}%)")

        # Log aggregate statistics to W&B
        if use_wandb and wandb_run is not None:
            # Summary metrics
            wandb_run.summary["summary/total_molecules"] = len(all_samples_summary)
            wandb_run.summary["summary/total_geometries"] = len(all_results)
            wandb_run.summary["summary/mean_neg_count"] = overall_mean
            wandb_run.summary["summary/std_neg_count"] = overall_std
            wandb_run.summary["summary/pct_higher_order"] = overall_pct_higher

            # Per-noise-level statistics
            for noise in sorted(set(args.noise_levels + [0.0])):
                noise_results = [r for r in all_results if r["success"] and r["noise_level"] == noise]
                if noise_results:
                    neg_counts = [r["neg_count"] for r in noise_results]
                    pct_higher = 100 * sum(1 for c in neg_counts if c > 1) / len(neg_counts)
                    noise_key = f"noise_{noise:.1f}A"
                    wandb_run.summary[f"by_noise/{noise_key}/mean_neg_count"] = np.mean(neg_counts)
                    wandb_run.summary[f"by_noise/{noise_key}/std_neg_count"] = np.std(neg_counts)
                    wandb_run.summary[f"by_noise/{noise_key}/pct_higher_order"] = pct_higher

            # Upload JSON files as artifacts
            artifact = wandb.Artifact(
                name="starting-geometry-stats",
                type="results",
                description="Starting geometry eigenvalue statistics"
            )
            artifact.add_file(results_json, name="all_geometry_stats.json")
            wandb_run.log_artifact(artifact)

    print("=" * 70)

    # Finish W&B run
    if use_wandb and wandb_run is not None:
        wandb.finish()
        print("[W&B] Run finished")
