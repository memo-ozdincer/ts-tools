#!/usr/bin/env python3
"""Validate best HPO hyperparameters on full dataset.

After finding optimal hyperparameters via HPO, this script validates them
on the complete dataset (not just difficult samples) to ensure:
1. The optimized parameters improve overall convergence rate
2. Performance on already-converging samples is maintained
3. The improvement generalizes beyond the training set

Usage (from ts-tools root):
    python scripts/killarney/experiments/2025/validate_hpo_params.py \
        --json path/to/hpo_results.json --calculator hip \
        --h5-path /path/to/data.h5 --checkpoint-path /path/to/ckpt \
        --out-dir ./validation_output

Alternative: Use the main experiment scripts directly with the best parameters
    (copy values from hpo_results.json into the SLURM script)
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add ts-tools root to path for imports
script_dir = Path(__file__).resolve().parent
ts_tools_root = script_dir.parents[4]  # Go up 4 levels: 2025 -> experiments -> killarney -> scripts -> ts-tools
if str(ts_tools_root) not in sys.path:
    sys.path.insert(0, str(ts_tools_root))


def get_experiment_main(calculator: str):
    """Dynamically import the experiment main function.
    
    Python doesn't allow package names starting with numbers, so we use importlib.
    """
    # The package is literally named "2025", which requires importlib
    package_path = f"src.experiments.2025.{calculator}_multi_mode_eckartmw"
    module = importlib.import_module(package_path)
    return module.main


def load_best_params(json_path: str) -> Dict[str, Any]:
    """Load best hyperparameters from HPO results JSON."""
    with open(json_path) as f:
        results = json.load(f)
    return results["best_params"]


def params_to_args(params: Dict[str, Any]) -> list[str]:
    """Convert hyperparameter dict to command-line arguments."""
    args = []
    
    # Map param names to CLI flags
    param_mapping = {
        "dt_min": "--dt-min",
        "dt_max": "--dt-max",
        "plateau_patience": "--plateau-patience",
        "plateau_boost": "--plateau-boost",
        "plateau_shrink": "--plateau-shrink",
        "escape_disp_threshold": "--escape-disp-threshold",
        "escape_window": "--escape-window",
        "escape_neg_vib_std": "--escape-neg-vib-std",
        "escape_delta": "--escape-delta",
        "adaptive_delta_scale": None,  # Special handling
        "trust_radius_max": "--max-atom-disp",
    }
    
    for param_name, param_value in params.items():
        cli_flag = param_mapping.get(param_name)
        
        if cli_flag is None:
            if param_name == "adaptive_delta_scale":
                # Convert scale to boolean + magnitude
                if param_value > 0:
                    args.extend(["--adaptive-delta"])
                    # Scale escape_delta by adaptive_delta_scale
                    # This is handled in the objective function
                else:
                    args.extend(["--no-adaptive-delta"])
            continue
        
        args.extend([cli_flag, str(param_value)])
    
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Validate best HPO hyperparameters on full dataset."
    )
    
    parser.add_argument(
        "--calculator",
        type=str,
        required=True,
        choices=["hip", "scine"],
        help="Calculator to use (hip or scine)",
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Path to hpo_results.json containing best parameters",
    )
    parser.add_argument(
        "--params-json",
        type=str,
        help="JSON string with hyperparameters (alternative to --json file)",
    )
    
    # Experiment parameters
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, help="HIP checkpoint (required for HIP)")
    parser.add_argument("--scine-functional", type=str, default="DFTB0", help="SCINE functional")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=15000)
    parser.add_argument("--start-from", type=str, default="midpoint_rt_noise1.0A")
    parser.add_argument("--noise-seed", type=int, default=42)
    
    # Baseline comparison
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run baseline (default params) for comparison",
    )
    
    args = parser.parse_args()
    
    # Load hyperparameters
    if args.json:
        params = load_best_params(args.json)
        print(f"Loaded best hyperparameters from: {args.json}")
    elif args.params_json:
        params = json.loads(args.params_json)
        print("Loaded hyperparameters from JSON string")
    else:
        parser.error("Must provide either --json or --params-json")
    
    print("\nHyperparameters to validate:")
    for key, value in params.items():
        print(f"  {key:25s}: {value}")
    
    # Convert to CLI args
    hpo_args = params_to_args(params)
    
    # Base experiment args
    base_args = [
        "--h5-path", args.h5_path,
        "--out-dir", str(Path(args.out_dir) / "hpo_validation"),
        "--max-samples", str(args.max_samples),
        "--n-steps", str(args.n_steps),
        "--start-from", args.start_from,
        "--noise-seed", str(args.noise_seed),
        "--stop-at-ts",
    ]
    
    if args.calculator == "hip":
        if not args.checkpoint_path:
            parser.error("--checkpoint-path required for HIP calculator")
        base_args.extend(["--checkpoint-path", args.checkpoint_path])
    else:  # scine
        base_args.extend(["--scine-functional", args.scine_functional])
    
    # Run with HPO parameters
    print("\n" + "=" * 80)
    print("RUNNING VALIDATION WITH HPO PARAMETERS")
    print("=" * 80)
    
    full_args = base_args + hpo_args
    
    # Get the appropriate main function dynamically
    experiment_main = get_experiment_main(args.calculator)
    experiment_main(argv=full_args)
    
    # Optionally compare with baseline
    if args.compare_baseline:
        print("\n" + "=" * 80)
        print("RUNNING BASELINE FOR COMPARISON")
        print("=" * 80)
        
        baseline_args = base_args.copy()
        baseline_args[baseline_args.index("--out-dir") + 1] = str(Path(args.out_dir) / "baseline")
        
        experiment_main(argv=baseline_args)
        
        print("\n" + "=" * 80)
        print("COMPARISON COMPLETE")
        print("=" * 80)
        print(f"HPO results:      {Path(args.out_dir) / 'hpo_validation' / 'all_results.json'}")
        print(f"Baseline results: {Path(args.out_dir) / 'baseline' / 'all_results.json'}")
        print("\nCompare convergence rates in the JSON files or aggregate_stats.json")


if __name__ == "__main__":
    main()
