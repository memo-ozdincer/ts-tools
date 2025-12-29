from __future__ import annotations

"""Simple eigenvalue classification experiment.

Compares how HIP and SCINE classify starting geometries by counting negative
eigenvalues of the Eckart-projected, mass-weighted Hessian.

Starting geometries tested:
- midpoint_rt: Midpoint between reactant and TS
- reactant: Reactant geometry
- product: Product geometry  
- noise1A: Midpoint + 1Å Gaussian noise
- noise2A: Midpoint + 2Å Gaussian noise

For each calculator (HIP, SCINE), we compute:
- Number of negative vibrational eigenvalues (after TR projection)
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as TGDData
from transition1x import Dataloader as T1xDataloader

from ..dependencies.common_utils import (
    add_gaussian_noise_to_coords,
    Transition1xDataset,
    UsePos,
)
from ..dependencies.hessian import (
    vibrational_eigvals,
    get_scine_elements_from_predict_output,
    project_hessian_remove_rigid_modes,
)
from ..runners._predict import make_predict_fn_from_calculator


# --- Extended Dataset Class with Product ---
class Transition1xDatasetWithProduct(torch.utils.data.Dataset):
    """Loads transition state data including product geometry."""
    
    def __init__(
        self,
        h5_path: str,
        split: str = "test",
        max_samples: Optional[int] = None,
        transform=None,
    ):
        self.transform = transform
        self.samples = []
        loader = T1xDataloader(h5_path, datasplit=split, only_final=True)

        for idx, mol in enumerate(loader):
            if max_samples is not None and len(self.samples) >= max_samples:
                break
            try:
                ts = mol["transition_state"]
                reactant = mol["reactant"]
                product = mol["product"]

                # Ensure all have same atom count
                if len(ts["atomic_numbers"]) != len(reactant["atomic_numbers"]):
                    continue
                if len(ts["atomic_numbers"]) != len(product["atomic_numbers"]):
                    continue
                
                data = TGDData(
                    z=torch.tensor(ts["atomic_numbers"], dtype=torch.long),
                    pos_transition=torch.tensor(ts["positions"], dtype=torch.float),
                    pos_reactant=torch.tensor(reactant["positions"], dtype=torch.float),
                    pos_product=torch.tensor(product["positions"], dtype=torch.float),
                    rxn=ts["rxn"],
                    formula=ts["formula"],
                )
                self.samples.append(data)
            except Exception as e:
                print(f"[WARN] Skipping idx={idx} due to error: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data


def get_starting_coords(batch, start_from: str, noise_seed: int = 42, sample_index: int = 0):
    """Get starting coordinates for a given geometry specification."""
    
    # Base geometries
    if start_from == "midpoint_rt":
        coords = 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition
    elif start_from == "reactant":
        coords = batch.pos_reactant.clone()
    elif start_from == "product":
        coords = batch.pos_product.clone()
    elif start_from == "midpoint_rt_noise1A":
        coords = 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition
        torch.manual_seed(noise_seed + sample_index)
        coords = add_gaussian_noise_to_coords(coords, 1.0)
    elif start_from == "midpoint_rt_noise2A":
        coords = 0.5 * batch.pos_reactant + 0.5 * batch.pos_transition
        torch.manual_seed(noise_seed + sample_index)
        coords = add_gaussian_noise_to_coords(coords, 2.0)
    else:
        raise ValueError(f"Unknown start_from: {start_from}")
    
    return coords


def count_negative_eigenvalues(
    predict_fn,
    coords: torch.Tensor,
    atomic_nums: torch.Tensor,
    calculator_type: str,
    tr_threshold: float = 1e-4,
) -> tuple[int, list]:
    """Count negative vibrational eigenvalues using proper projection.
    
    Returns:
        (n_negative, eigenvalues_list)
    """
    result = predict_fn(coords, atomic_nums, do_hessian=True)
    hessian = result["hessian"]
    
    # Get SCINE elements if applicable
    scine_elements = get_scine_elements_from_predict_output(result)
    
    # Compute vibrational eigenvalues with proper projection
    evals = vibrational_eigvals(hessian, coords, atomic_nums, scine_elements=scine_elements)
    
    # Count negatives (excluding near-zero TR modes)
    n_neg = int((evals < -tr_threshold).sum().item())
    
    return n_neg, evals.tolist()


def run_classification(args):
    """Run eigenvalue classification for both calculators."""
    
    torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Define starting geometries to test
    start_from_list = ["midpoint_rt", "reactant", "product", "midpoint_rt_noise1A", "midpoint_rt_noise2A"]
    
    # Load dataset
    print(f"Loading dataset: {args.h5_path} (split={args.split})")
    dataset = Transition1xDatasetWithProduct(
        h5_path=args.h5_path,
        split=args.split,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Loaded {len(dataset)} samples.")
    
    # Results storage
    results = {
        "hip": defaultdict(list),
        "scine": defaultdict(list),
    }
    
    # Setup calculators
    calculators = {}
    
    # HIP calculator
    print("Loading HIP calculator...")
    from hip.equiformer_torch_calculator import EquiformerTorchCalculator
    hip_calc = EquiformerTorchCalculator(
        checkpoint_path=args.checkpoint_path,
        hessian_method="predict",
    )
    calculators["hip"] = {
        "calculator": hip_calc,
        "predict_fn": make_predict_fn_from_calculator(hip_calc, "hip"),
    }
    
    # SCINE calculator
    print("Loading SCINE calculator...")
    from ..dependencies.scine_calculator import create_scine_calculator
    scine_calc = create_scine_calculator(
        functional=args.scine_functional,
        device="cpu",
    )
    calculators["scine"] = {
        "calculator": scine_calc,
        "predict_fn": make_predict_fn_from_calculator(scine_calc, "scine"),
    }
    
    print("-" * 50)
    print(f"Running classification on {len(dataset)} samples...")
    print(f"Starting geometries: {start_from_list}")
    print("-" * 50)
    
    # Process each sample
    for i, batch in enumerate(dataloader):
        if i >= args.max_samples:
            break
            
        print(f"\nSample {i+1}/{min(len(dataset), args.max_samples)} ({batch.formula[0]})")
        
        atomic_nums = batch.z
        
        for start_from in start_from_list:
            coords = get_starting_coords(batch, start_from, noise_seed=args.noise_seed, sample_index=i)
            
            for calc_name, calc_info in calculators.items():
                try:
                    # Move coords to appropriate device
                    device = args.device if calc_name == "hip" else "cpu"
                    coords_dev = coords.to(device)
                    atomic_nums_dev = atomic_nums.to(device)
                    
                    n_neg, evals = count_negative_eigenvalues(
                        calc_info["predict_fn"],
                        coords_dev,
                        atomic_nums_dev,
                        calc_name,
                    )
                    
                    results[calc_name][start_from].append({
                        "sample_idx": i,
                        "formula": batch.formula[0],
                        "n_negative": n_neg,
                        "eigenvalues": evals[:10],  # Store first 10 eigenvalues
                    })
                    
                    print(f"  {calc_name:>5} | {start_from:<20} | neg_eigs = {n_neg}")
                    
                except Exception as e:
                    print(f"  {calc_name:>5} | {start_from:<20} | ERROR: {e}")
                    results[calc_name][start_from].append({
                        "sample_idx": i,
                        "formula": batch.formula[0],
                        "n_negative": -1,
                        "error": str(e),
                    })
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Mean negative eigenvalues by geometry and calculator")
    print("=" * 60)
    
    summary = {}
    for calc_name in ["hip", "scine"]:
        summary[calc_name] = {}
        print(f"\n{calc_name.upper()}:")
        for start_from in start_from_list:
            data = results[calc_name][start_from]
            valid = [d["n_negative"] for d in data if d["n_negative"] >= 0]
            if valid:
                mean_neg = sum(valid) / len(valid)
                # Count distribution
                counts = defaultdict(int)
                for v in valid:
                    counts[v] += 1
                dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
                print(f"  {start_from:<20}: mean={mean_neg:.2f} (n={len(valid)}) | distribution: {dist_str}")
                summary[calc_name][start_from] = {
                    "mean": mean_neg,
                    "count": len(valid),
                    "distribution": dict(counts),
                }
            else:
                print(f"  {start_from:<20}: no valid samples")
                summary[calc_name][start_from] = {"mean": None, "count": 0}
    
    # Comparison table
    print("\n" + "=" * 60)
    print("COMPARISON: HIP vs SCINE agreement")
    print("=" * 60)
    
    for start_from in start_from_list:
        hip_data = results["hip"][start_from]
        scine_data = results["scine"][start_from]
        
        agreements = 0
        total = 0
        for h, s in zip(hip_data, scine_data):
            if h["n_negative"] >= 0 and s["n_negative"] >= 0:
                total += 1
                if h["n_negative"] == s["n_negative"]:
                    agreements += 1
        
        if total > 0:
            print(f"  {start_from:<20}: {agreements}/{total} agree ({100*agreements/total:.1f}%)")
        else:
            print(f"  {start_from:<20}: no valid comparisons")
    
    # Save results
    output_file = Path(args.out_dir) / "eigenvalue_classification_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": summary,
            "detailed": {k: dict(v) for k, v in results.items()},
            "args": vars(args),
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")
    
    return results, summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare HIP and SCINE eigenvalue classification of starting geometries"
    )
    
    PROJECT = "/project/memo"
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples to process.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--h5-path", type=str, default=os.path.join(PROJECT, "large-files", "data", "transition1x.h5"))
    parser.add_argument("--checkpoint-path", type=str, default=os.path.join(PROJECT, "large-files", "ckpt", "hip_v2.ckpt"))
    parser.add_argument("--out-dir", type=str, default=os.path.join(PROJECT, "large-files", "graphs", "eigenvalue_classification"))
    parser.add_argument("--scine-functional", type=str, default="DFTB0", help="SCINE functional (DFTB0, PM6, AM1, etc.)")
    parser.add_argument("--noise-seed", type=int, default=42, help="Seed for reproducible noise.")
    
    args = parser.parse_args(argv)
    run_classification(args)


if __name__ == "__main__":
    main()
