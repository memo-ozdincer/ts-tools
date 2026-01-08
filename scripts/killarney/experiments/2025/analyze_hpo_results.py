#!/usr/bin/env python3
"""Analyze and visualize Optuna HPO results for multi-mode Eckart-MW GAD.

This script loads an Optuna study from SQLite database and generates
comprehensive visualizations to understand:
- Which hyperparameters matter most
- How optimization progressed over time
- Relationships between parameters
- Trade-offs between objectives

Usage:
    python analyze_hpo_results.py --storage sqlite:///path/to/hip_hpo_study.db --study-name hip-multi-mode-eckartmw-hpo-12345
    python analyze_hpo_results.py --json path/to/hpo_results.json
"""

import argparse
import json
from pathlib import Path

import optuna


def analyze_from_storage(storage: str, study_name: str, output_dir: str = "./hpo_analysis"):
    """Analyze HPO results from Optuna database."""
    
    # Load study
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    print("=" * 80)
    print("OPTUNA STUDY ANALYSIS")
    print("=" * 80)
    print(f"Study name: {study_name}")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best convergence rate: {study.best_trial.user_attrs.get('convergence_rate', 'N/A'):.3f}")
    print(f"Best mean steps: {study.best_trial.user_attrs.get('mean_steps_to_converge', 'N/A'):.1f}")
    
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key:25s}: {value}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations in: {output_path}")
    
    # 1. Optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(str(output_path / "optimization_history.html"))
    print("  ✓ optimization_history.html")
    
    # 2. Parameter importances
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(str(output_path / "param_importances.html"))
        print("  ✓ param_importances.html")
    except Exception as e:
        print(f"  ✗ param_importances.html: {e}")
    
    # 3. Parallel coordinate plot
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(str(output_path / "parallel_coordinate.html"))
        print("  ✓ parallel_coordinate.html")
    except Exception as e:
        print(f"  ✗ parallel_coordinate.html: {e}")
    
    # 4. Slice plot (parameter effects)
    try:
        fig = optuna.visualization.plot_slice(study)
        fig.write_html(str(output_path / "slice_plot.html"))
        print("  ✓ slice_plot.html")
    except Exception as e:
        print(f"  ✗ slice_plot.html: {e}")
    
    # 5. Contour plots for key parameter pairs
    key_pairs = [
        ("dt_max", "escape_delta"),
        ("plateau_patience", "plateau_boost"),
        ("escape_disp_threshold", "escape_window"),
        ("adaptive_delta_scale", "trust_radius_max"),
        ("dt_min", "dt_max"),
    ]
    
    for param1, param2 in key_pairs:
        try:
            fig = optuna.visualization.plot_contour(study, params=[param1, param2])
            safe_name = f"contour_{param1}_vs_{param2}.html"
            fig.write_html(str(output_path / safe_name))
            print(f"  ✓ {safe_name}")
        except Exception as e:
            print(f"  ✗ contour_{param1}_vs_{param2}.html: {e}")
    
    # 6. Timeline plot
    try:
        fig = optuna.visualization.plot_timeline(study)
        fig.write_html(str(output_path / "timeline.html"))
        print("  ✓ timeline.html")
    except Exception as e:
        print(f"  ✗ timeline.html: {e}")
    
    # 7. Export summary statistics
    summary = {
        "study_name": study_name,
        "n_trials": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_score": study.best_value,
        "best_params": study.best_params,
        "best_convergence_rate": study.best_trial.user_attrs.get("convergence_rate"),
        "best_mean_steps": study.best_trial.user_attrs.get("mean_steps_to_converge"),
        "top_10_trials": [
            {
                "trial": t.number,
                "score": t.value,
                "params": t.params,
                "convergence_rate": t.user_attrs.get("convergence_rate"),
                "mean_steps": t.user_attrs.get("mean_steps_to_converge"),
            }
            for t in sorted(study.trials, key=lambda x: x.value if x.value is not None else -float('inf'), reverse=True)[:10]
        ],
    }
    
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  ✓ summary.json")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


def analyze_from_json(json_path: str, output_dir: str = "./hpo_analysis"):
    """Analyze HPO results from JSON file (limited analysis)."""
    
    with open(json_path) as f:
        results = json.load(f)
    
    print("=" * 80)
    print("HPO RESULTS ANALYSIS (from JSON)")
    print("=" * 80)
    print(f"Best trial: #{results['best_trial']}")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Best convergence rate: {results['best_convergence_rate']:.3f}")
    print(f"Best mean steps: {results['best_mean_steps']:.1f}")
    
    print("\nBest Hyperparameters:")
    for key, value in results['best_params'].items():
        print(f"  {key:25s}: {value}")
    
    # Sort trials by score
    trials = sorted(results['all_trials'], key=lambda x: x['score'] if x['score'] is not None else -float('inf'), reverse=True)
    
    print("\nTop 10 Trials:")
    print(f"{'Trial':>6} {'Score':>8} {'Conv Rate':>10} {'Mean Steps':>11}")
    print("-" * 40)
    for t in trials[:10]:
        score = t['score'] if t['score'] is not None else float('nan')
        conv_rate = t.get('convergence_rate', 0)
        mean_steps = t.get('mean_steps', 0)
        print(f"{t['number']:6d} {score:8.4f} {conv_rate:10.3f} {mean_steps:11.1f}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save top trials
    with open(output_path / "top_trials.json", "w") as f:
        json.dump(trials[:20], f, indent=2)
    
    print(f"\nSaved top 20 trials to: {output_path / 'top_trials.json'}")
    print("\nNote: For full visualization, use --storage option with Optuna database")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize Optuna HPO results for multi-mode Eckart-MW GAD."
    )
    
    parser.add_argument(
        "--storage",
        type=str,
        help="Optuna storage URL (e.g., sqlite:///path/to/study.db)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        help="Optuna study name (required if using --storage)",
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Path to hpo_results.json (alternative to --storage)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./hpo_analysis",
        help="Output directory for visualizations (default: ./hpo_analysis)",
    )
    
    args = parser.parse_args()
    
    if args.storage and args.study_name:
        analyze_from_storage(args.storage, args.study_name, args.output_dir)
    elif args.json:
        analyze_from_json(args.json, args.output_dir)
    else:
        parser.error("Must provide either (--storage and --study-name) or --json")


if __name__ == "__main__":
    main()
