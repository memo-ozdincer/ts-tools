#!/usr/bin/env python3
"""
HPO Results Analysis Script

Analyzes Optuna SQLite databases from four experimental setups:
1. HIP + Sella (GPU)
2. SCINE + Sella (CPU)
3. HIP + Multi-Mode Eckart-MW (GPU)
4. SCINE + Multi-Mode Eckart-MW (CPU)

Generates figures for LaTeX document.

Usage:
    python analyze_hpo_results.py
"""

import sqlite3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Any


# Configuration
HPO_DIR = Path(__file__).parent
OUTPUT_DIR = HPO_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Database files
DATABASES = {
    "hip_sella": HPO_DIR / "hip_sella_hpo_job201956.db",
    "scine_sella": HPO_DIR / "scine_sella_hpo_job833394.db",
    "hip_multimode": HPO_DIR / "hip_multi_mode_hpo_201955.db",
    "scine_multimode": HPO_DIR / "scine_multi_mode_hpo_825749.db",
}

# Matplotlib style
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


@dataclass
class StudyResults:
    """Container for study results."""
    name: str
    n_trials: int
    n_complete: int
    n_pruned: int
    params_df: pd.DataFrame
    attrs_df: pd.DataFrame
    distributions: dict[str, dict]
    best_trial: dict[str, Any] | None


def load_study(db_path: Path) -> StudyResults | None:
    """Load study results from SQLite database."""
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return None

    conn = sqlite3.connect(db_path)

    # Get trial counts
    trials_df = pd.read_sql_query(
        "SELECT trial_id, number, state FROM trials", conn
    )
    n_trials = len(trials_df)
    if n_trials == 0:
        conn.close()
        return StudyResults(
            name=db_path.stem,
            n_trials=0,
            n_complete=0,
            n_pruned=0,
            params_df=pd.DataFrame(),
            attrs_df=pd.DataFrame(),
            distributions={},
            best_trial=None,
        )

    n_complete = len(trials_df[trials_df["state"] == "COMPLETE"])
    n_pruned = len(trials_df[trials_df["state"] == "PRUNED"])

    # Get parameters
    params_df = pd.read_sql_query("""
        SELECT
            t.trial_id,
            t.number,
            t.state,
            tp.param_name,
            tp.param_value,
            tp.distribution_json,
            tv.value as objective_value
        FROM trials t
        LEFT JOIN trial_params tp ON t.trial_id = tp.trial_id
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
    """, conn)

    # Get user attributes
    attrs_df = pd.read_sql_query("""
        SELECT
            t.trial_id,
            t.number,
            t.state,
            tua.key,
            tua.value_json
        FROM trials t
        LEFT JOIN trial_user_attributes tua ON t.trial_id = tua.trial_id
    """, conn)

    # Parse distributions
    distributions = {}
    if len(params_df) > 0:
        for _, row in params_df[params_df["distribution_json"].notna()].drop_duplicates("param_name").iterrows():
            distributions[row["param_name"]] = json.loads(row["distribution_json"])

    # Get best trial
    best_trial = None
    if n_complete > 0:
        best_row = params_df[params_df["state"] == "COMPLETE"].sort_values(
            "objective_value", ascending=False
        ).iloc[0]
        best_trial = {"trial_id": best_row["trial_id"], "objective": best_row["objective_value"]}

    conn.close()

    return StudyResults(
        name=db_path.stem,
        n_trials=n_trials,
        n_complete=n_complete,
        n_pruned=n_pruned,
        params_df=params_df,
        attrs_df=attrs_df,
        distributions=distributions,
        best_trial=best_trial,
    )


def pivot_params(results: StudyResults) -> pd.DataFrame:
    """Pivot parameter data to wide format."""
    if results.params_df.empty:
        return pd.DataFrame()

    df = results.params_df.copy()
    pivot = df.pivot_table(
        index=["trial_id", "number", "state", "objective_value"],
        columns="param_name",
        values="param_value",
        aggfunc="first"
    ).reset_index()

    return pivot


def pivot_attrs(results: StudyResults) -> pd.DataFrame:
    """Pivot user attributes to wide format."""
    if results.attrs_df.empty:
        return pd.DataFrame()

    df = results.attrs_df.copy()
    df = df[df["key"].notna()]

    # Parse JSON values
    def parse_json(x):
        if pd.isna(x):
            return np.nan
        try:
            return json.loads(x)
        except:
            return x

    df["value"] = df["value_json"].apply(parse_json)

    pivot = df.pivot_table(
        index=["trial_id", "number", "state"],
        columns="key",
        values="value",
        aggfunc="first"
    ).reset_index()

    return pivot


def plot_optimization_history(results: StudyResults, ax: plt.Axes):
    """Plot optimization history."""
    df = pivot_params(results)
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    df = df.sort_values("number")

    # Plot all trials
    colors = df["state"].map({"COMPLETE": "C0", "PRUNED": "C3", "RUNNING": "C2"})
    ax.scatter(df["number"], df["objective_value"], c=colors, alpha=0.6, s=20)

    # Plot best so far line (complete trials only)
    complete = df[df["state"] == "COMPLETE"].copy()
    if not complete.empty:
        complete = complete.sort_values("number")
        complete["best_so_far"] = complete["objective_value"].cummax()
        ax.plot(complete["number"], complete["best_so_far"], "k-", linewidth=1.5, label="Best")

    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective Value")
    ax.legend()


def plot_param_importance(results: StudyResults, ax: plt.Axes):
    """Plot parameter correlation with objective (simple importance proxy)."""
    df = pivot_params(results)
    if df.empty or "objective_value" not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    complete = df[df["state"] == "COMPLETE"].copy()
    if len(complete) < 5:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        return

    # Calculate correlations
    param_cols = [c for c in complete.columns if c not in ["trial_id", "number", "state", "objective_value"]]
    correlations = {}
    for col in param_cols:
        if complete[col].notna().sum() > 3:
            correlations[col] = abs(complete[col].astype(float).corr(complete["objective_value"]))

    if not correlations:
        ax.text(0.5, 0.5, "No correlations", ha="center", va="center", transform=ax.transAxes)
        return

    # Sort and plot
    sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in sorted_params]
    values = [p[1] for p in sorted_params]

    ax.barh(range(len(names)), values, color="C0", alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("|Correlation| with Objective")
    ax.set_xlim(0, 1)


def plot_param_distributions(results: StudyResults, param_name: str, ax: plt.Axes):
    """Plot parameter value distribution with objective coloring."""
    df = pivot_params(results)
    if df.empty or param_name not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    complete = df[df["state"] == "COMPLETE"].copy()
    if len(complete) < 3:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        return

    x = complete[param_name].astype(float)
    y = complete["objective_value"]

    # Check if log scale
    dist = results.distributions.get(param_name, {})
    is_log = dist.get("attributes", {}).get("log", False)

    scatter = ax.scatter(x, y, c=y, cmap="viridis", alpha=0.7, s=30)

    if is_log:
        ax.set_xscale("log")

    ax.set_xlabel(param_name)
    ax.set_ylabel("Objective")

    # Add range info
    low = dist.get("attributes", {}).get("low", x.min())
    high = dist.get("attributes", {}).get("high", x.max())
    ax.axvline(low, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(high, color="gray", linestyle="--", alpha=0.3)


def plot_user_attr_distribution(results: StudyResults, attr_name: str, ax: plt.Axes):
    """Plot distribution of a user attribute."""
    df = pivot_attrs(results)
    if df.empty or attr_name not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    complete = df[df["state"] == "COMPLETE"].copy()
    if len(complete) < 3:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        return

    values = complete[attr_name].dropna().astype(float)
    ax.hist(values, bins=min(15, len(values)), alpha=0.7, color="C0", edgecolor="black")
    ax.axvline(values.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {values.mean():.3f}")
    ax.axvline(values.max(), color="green", linestyle=":", linewidth=2, label=f"Max: {values.max():.3f}")
    ax.set_xlabel(attr_name)
    ax.set_ylabel("Count")
    ax.legend()


def generate_scine_sella_figures(results: StudyResults):
    """Generate all figures for SCINE Sella study."""
    print(f"\n=== Generating SCINE Sella figures ===")
    print(f"Trials: {results.n_trials} (complete: {results.n_complete}, pruned: {results.n_pruned})")

    params_wide = pivot_params(results)
    attrs_wide = pivot_attrs(results)

    # Figure 1: Optimization history
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_optimization_history(results, ax)
    ax.set_title("SCINE Sella: Optimization History")
    fig.savefig(OUTPUT_DIR / "scine_sella_history.pdf")
    plt.close(fig)

    # Figure 2: Parameter importance
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_param_importance(results, ax)
    ax.set_title("SCINE Sella: Parameter Correlations")
    fig.savefig(OUTPUT_DIR / "scine_sella_importance.pdf")
    plt.close(fig)

    # Figure 3: Key parameter distributions
    sella_params = ["delta0", "fmax", "rho_dec", "sigma_dec", "sigma_inc", "apply_eckart"]
    available_params = [p for p in sella_params if p in params_wide.columns]

    n_params = len(available_params)
    if n_params > 0:
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows))
        axes = axes.flatten() if n_params > 1 else [axes]

        for i, param in enumerate(available_params):
            plot_param_distributions(results, param, axes[i])

        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle("SCINE Sella: Parameter vs Objective", y=1.02)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "scine_sella_params.pdf")
        plt.close(fig)

    # Figure 4: Key metrics histograms
    key_attrs = ["eigenvalue_ts_rate", "sella_convergence_rate", "both_rate", "avg_steps"]
    available_attrs = [a for a in key_attrs if a in attrs_wide.columns]

    if available_attrs:
        n_attrs = len(available_attrs)
        fig, axes = plt.subplots(1, n_attrs, figsize=(4 * n_attrs, 3))
        axes = axes if n_attrs > 1 else [axes]

        for i, attr in enumerate(available_attrs):
            plot_user_attr_distribution(results, attr, axes[i])

        fig.suptitle("SCINE Sella: Metric Distributions", y=1.05)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "scine_sella_metrics.pdf")
        plt.close(fig)

    # Table: Best trials
    if not params_wide.empty:
        complete = params_wide[params_wide["state"] == "COMPLETE"].sort_values("objective_value", ascending=False)
        best = complete.head(5)
        print("\n=== Top 5 Trials ===")
        print(best.to_string(index=False))
        best.to_csv(OUTPUT_DIR / "scine_sella_best_trials.csv", index=False)

    # Table: Optimal ranges
    print("\n=== Optimal Parameter Ranges (Top 25%) ===")
    if not params_wide.empty:
        complete = params_wide[params_wide["state"] == "COMPLETE"].copy()
        threshold = complete["objective_value"].quantile(0.75)
        top_25 = complete[complete["objective_value"] >= threshold]

        optimal_ranges = {}
        for col in available_params:
            if col in top_25.columns:
                values = top_25[col].astype(float)
                optimal_ranges[col] = {
                    "min": values.min(),
                    "max": values.max(),
                    "mean": values.mean(),
                    "search_low": results.distributions.get(col, {}).get("attributes", {}).get("low"),
                    "search_high": results.distributions.get(col, {}).get("attributes", {}).get("high"),
                }

        ranges_df = pd.DataFrame(optimal_ranges).T
        print(ranges_df.to_string())
        ranges_df.to_csv(OUTPUT_DIR / "scine_sella_optimal_ranges.csv")


def generate_scine_multimode_figures(results: StudyResults):
    """Generate figures for SCINE Multi-Mode study."""
    print(f"\n=== Generating SCINE Multi-Mode figures ===")
    print(f"Trials: {results.n_trials} (complete: {results.n_complete}, pruned: {results.n_pruned})")

    if results.n_complete < 2:
        print("Insufficient data for detailed analysis")
        params_wide = pivot_params(results)
        attrs_wide = pivot_attrs(results)

        if not params_wide.empty:
            print("\n=== Available Data ===")
            print(params_wide.to_string())
            params_wide.to_csv(OUTPUT_DIR / "scine_multimode_data.csv", index=False)

        if not attrs_wide.empty:
            print(attrs_wide.to_string())
        return

    # Similar figures as SCINE Sella
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_optimization_history(results, ax)
    ax.set_title("SCINE Multi-Mode: Optimization History")
    fig.savefig(OUTPUT_DIR / "scine_multimode_history.pdf")
    plt.close(fig)


def generate_summary_table():
    """Generate summary table across all studies."""
    summary = []

    for name, db_path in DATABASES.items():
        results = load_study(db_path)
        if results is None:
            continue

        row = {
            "Study": name,
            "Total Trials": results.n_trials,
            "Complete": results.n_complete,
            "Pruned": results.n_pruned,
            "Best Objective": results.best_trial["objective"] if results.best_trial else None,
        }
        summary.append(row)

    df = pd.DataFrame(summary)
    print("\n=== Study Summary ===")
    print(df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "study_summary.csv", index=False)
    return df


def main():
    """Main analysis function."""
    print("=" * 60)
    print("HPO Results Analysis")
    print("=" * 60)

    # Generate summary
    generate_summary_table()

    # Analyze each study
    for name, db_path in DATABASES.items():
        results = load_study(db_path)
        if results is None or results.n_trials == 0:
            print(f"\n=== {name}: No data ===")
            continue

        if "sella" in name and "scine" in name:
            generate_scine_sella_figures(results)
        elif "multimode" in name and "scine" in name:
            generate_scine_multimode_figures(results)
        else:
            print(f"\n=== {name}: {results.n_trials} trials ({results.n_complete} complete) ===")

    print("\n" + "=" * 60)
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
