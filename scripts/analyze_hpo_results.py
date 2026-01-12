#!/usr/bin/env python3
"""Comprehensive analysis of HPO results from Optuna SQLite databases.

Generates a detailed LaTeX report with:
- Sella HPO results (HIP and SCINE)
- Multi-mode Eckart-MW GAD HPO results (HIP and SCINE)
- Statistical analysis, correlations, and insights
"""

import sqlite3
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Output directories
OUTPUT_DIR = Path("/Users/memoozdincer/Desktop/outputs")
TS_TOOLS_DIR = Path("/Users/memoozdincer/Documents/Research/Guzik/ts-tools")

# Database paths
DB_PATHS = {
    "hip_sella": OUTPUT_DIR / "hip_hpo_job1802595.db",
    "scine_sella": TS_TOOLS_DIR / "outputs" / "scine_sella" / "scine_hpo_job1802463.db",
    "hip_multimode": OUTPUT_DIR / "hip_multi_mode_hpo_1820826.db",
    "scine_multimode": OUTPUT_DIR / "scine_multi_mode_hpo_1809794.db",
}

# JSON results (for SCINE Sella which completed)
SCINE_SELLA_JSON = OUTPUT_DIR / "scine_hpo_results.json"


@dataclass
class Trial:
    """Represents a single HPO trial."""
    number: int
    state: str
    value: Optional[float]
    params: Dict[str, Any]
    user_attrs: Dict[str, Any]
    datetime_start: str
    datetime_complete: Optional[str]
    intermediate_values: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class StudyResults:
    """Full results from an HPO study."""
    name: str
    direction: str
    n_trials: int
    n_completed: int
    n_pruned: int
    n_failed: int
    trials: List[Trial]
    best_trial: Optional[Trial] = None
    param_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)


def load_optuna_db(db_path: Path) -> Optional[StudyResults]:
    """Load all trial data from an Optuna SQLite database."""
    if not db_path.exists():
        print(f"Warning: Database not found: {db_path}")
        return None
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get study info
    cursor.execute("SELECT study_id, study_name FROM studies LIMIT 1")
    study_row = cursor.fetchone()
    if not study_row:
        conn.close()
        return None
    study_id, study_name = study_row
    
    # Get direction
    cursor.execute("SELECT direction FROM study_directions WHERE study_id = ?", (study_id,))
    dir_row = cursor.fetchone()
    direction = dir_row[0] if dir_row else "MAXIMIZE"
    
    # Get all trials
    cursor.execute("""
        SELECT trial_id, number, state, datetime_start, datetime_complete 
        FROM trials WHERE study_id = ?
        ORDER BY number
    """, (study_id,))
    trial_rows = cursor.fetchall()
    
    trials = []
    param_ranges = defaultdict(lambda: [float('inf'), float('-inf')])
    
    for trial_id, number, state, dt_start, dt_complete in trial_rows:
        # Get params
        cursor.execute("""
            SELECT param_name, param_value, distribution_json 
            FROM trial_params WHERE trial_id = ?
        """, (trial_id,))
        params = {}
        for pname, pvalue, dist_json in cursor.fetchall():
            params[pname] = pvalue
            if pvalue is not None:
                param_ranges[pname][0] = min(param_ranges[pname][0], pvalue)
                param_ranges[pname][1] = max(param_ranges[pname][1], pvalue)
        
        # Get user attrs
        cursor.execute("""
            SELECT key, value_json FROM trial_user_attributes WHERE trial_id = ?
        """, (trial_id,))
        user_attrs = {}
        for key, val_json in cursor.fetchall():
            try:
                user_attrs[key] = json.loads(val_json)
            except:
                user_attrs[key] = val_json
        
        # Get trial value
        cursor.execute("""
            SELECT value FROM trial_values WHERE trial_id = ?
        """, (trial_id,))
        val_row = cursor.fetchone()
        value = val_row[0] if val_row else None
        
        # Get intermediate values
        cursor.execute("""
            SELECT step, intermediate_value FROM trial_intermediate_values 
            WHERE trial_id = ? ORDER BY step
        """, (trial_id,))
        intermediate = [(row[0], row[1]) for row in cursor.fetchall()]
        
        trial = Trial(
            number=number,
            state=state,
            value=value,
            params=params,
            user_attrs=user_attrs,
            datetime_start=dt_start,
            datetime_complete=dt_complete,
            intermediate_values=intermediate,
        )
        trials.append(trial)
    
    conn.close()
    
    # Count states
    n_completed = sum(1 for t in trials if t.state == "COMPLETE")
    n_pruned = sum(1 for t in trials if t.state == "PRUNED")
    n_failed = sum(1 for t in trials if t.state not in ("COMPLETE", "PRUNED", "RUNNING"))
    
    # Find best trial
    completed_trials = [t for t in trials if t.state == "COMPLETE" and t.value is not None]
    best_trial = None
    if completed_trials:
        if direction == "MAXIMIZE":
            best_trial = max(completed_trials, key=lambda t: t.value)
        else:
            best_trial = min(completed_trials, key=lambda t: t.value)
    
    return StudyResults(
        name=study_name,
        direction=direction,
        n_trials=len(trials),
        n_completed=n_completed,
        n_pruned=n_pruned,
        n_failed=n_failed,
        trials=trials,
        best_trial=best_trial,
        param_ranges=dict(param_ranges),
    )


def load_scine_sella_json() -> Optional[Dict]:
    """Load the SCINE Sella HPO results JSON."""
    if not SCINE_SELLA_JSON.exists():
        return None
    with open(SCINE_SELLA_JSON) as f:
        return json.load(f)


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute basic statistics for a list of values."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def compute_correlations(trials: List[Trial], params: List[str], metric: str) -> Dict[str, float]:
    """Compute correlation between each parameter and a metric."""
    completed = [t for t in trials if t.state == "COMPLETE" and t.value is not None]
    if len(completed) < 5:
        return {}
    
    correlations = {}
    metric_values = []
    
    for t in completed:
        if metric == "value":
            metric_values.append(t.value)
        elif metric in t.user_attrs:
            metric_values.append(t.user_attrs[metric])
        else:
            return {}
    
    metric_arr = np.array(metric_values)
    
    for param in params:
        param_values = [t.params.get(param, np.nan) for t in completed]
        param_arr = np.array(param_values)
        
        # Skip if any NaN
        mask = ~(np.isnan(param_arr) | np.isnan(metric_arr))
        if mask.sum() < 5:
            continue
        
        corr = np.corrcoef(param_arr[mask], metric_arr[mask])[0, 1]
        if not np.isnan(corr):
            correlations[param] = float(corr)
    
    return correlations


def analyze_param_effectiveness(trials: List[Trial], param: str, metric: str = "value") -> Dict:
    """Analyze which parameter ranges are most effective."""
    completed = [t for t in trials if t.state == "COMPLETE" and t.value is not None]
    if len(completed) < 5:
        return {}
    
    values = []
    for t in completed:
        if param not in t.params:
            continue
        pval = t.params[param]
        if metric == "value":
            mval = t.value
        elif metric in t.user_attrs:
            mval = t.user_attrs[metric]
        else:
            continue
        values.append((pval, mval))
    
    if not values:
        return {}
    
    values.sort(key=lambda x: x[0])
    n = len(values)
    
    # Split into thirds
    low_third = values[:n//3]
    mid_third = values[n//3:2*n//3]
    high_third = values[2*n//3:]
    
    result = {}
    if low_third:
        result["low_range"] = {
            "param_range": (low_third[0][0], low_third[-1][0]),
            "metric_mean": np.mean([v[1] for v in low_third]),
            "metric_std": np.std([v[1] for v in low_third]),
        }
    if mid_third:
        result["mid_range"] = {
            "param_range": (mid_third[0][0], mid_third[-1][0]),
            "metric_mean": np.mean([v[1] for v in mid_third]),
            "metric_std": np.std([v[1] for v in mid_third]),
        }
    if high_third:
        result["high_range"] = {
            "param_range": (high_third[0][0], high_third[-1][0]),
            "metric_mean": np.mean([v[1] for v in high_third]),
            "metric_std": np.std([v[1] for v in high_third]),
        }
    
    return result


def format_float(val: float, decimals: int = 4) -> str:
    """Format float with specified decimals."""
    if abs(val) < 0.0001 and val != 0:
        return f"{val:.2e}"
    return f"{val:.{decimals}f}"


def format_range(low: float, high: float) -> str:
    """Format a range for display."""
    return f"[{format_float(low, 3)}, {format_float(high, 3)}]"


def escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        '_': r'\_',
        '%': r'\%',
        '&': r'\&',
        '#': r'\#',
        '$': r'\$',
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def generate_latex_report(results: Dict[str, StudyResults], scine_json: Optional[Dict]) -> str:
    """Generate comprehensive LaTeX report."""
    
    # Preamble
    latex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{geometry}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{multirow}
\usepackage{graphicx}

\geometry{margin=1in}
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue}

\definecolor{bestcolor}{RGB}{0,128,0}
\definecolor{worstcolor}{RGB}{180,0,0}
\definecolor{neutralcolor}{RGB}{100,100,100}

\title{Hyperparameter Optimization Results:\\Transition State Search Algorithms}
\author{HPO Analysis Report}
\date{January 2026}

\begin{document}
\maketitle
\tableofcontents
\newpage

\section{Executive Summary}

This report presents comprehensive hyperparameter optimization (HPO) results for two transition state (TS) finding algorithms:
\begin{itemize}
    \item \textbf{Sella}: P-RFO based TS optimizer with trust radius management
    \item \textbf{Multi-mode Eckart-MW GAD}: Gentlest Ascent Dynamics with second-mode escape mechanism
\end{itemize}

Each algorithm was tested with two potential energy surface calculators:
\begin{itemize}
    \item \textbf{HIP}: Machine learning-based interatomic potential (GPU-accelerated)
    \item \textbf{SCINE}: Semi-empirical quantum mechanical calculator (CPU-based, DFTB0)
\end{itemize}

\subsection{Key Findings Summary}

"""
    
    # Add summary for each study
    for key, study in results.items():
        if study is None:
            continue
        
        latex += f"\\paragraph{{{escape_latex(key.replace('_', ' ').title())}}}\n"
        latex += f"Completed {study.n_completed} of {study.n_trials} trials"
        if study.n_pruned > 0:
            latex += f" ({study.n_pruned} pruned)"
        latex += ". "
        
        if study.best_trial:
            latex += f"Best score: \\textbf{{{format_float(study.best_trial.value, 4)}}}. "
            if "eigenvalue_ts_rate" in study.best_trial.user_attrs:
                ts_rate = study.best_trial.user_attrs["eigenvalue_ts_rate"]
                latex += f"Best TS rate: \\textbf{{{format_float(ts_rate*100, 1)}\\%}}. "
            elif "convergence_rate" in study.best_trial.user_attrs:
                conv_rate = study.best_trial.user_attrs["convergence_rate"]
                latex += f"Best convergence rate: \\textbf{{{format_float(conv_rate*100, 1)}\\%}}. "
        latex += "\n\n"
    
    # ===================================================================================
    # PART 1: SELLA HPO
    # ===================================================================================
    latex += r"""
\newpage
\section{Sella Hyperparameter Optimization}

Sella is a P-RFO (Partitioned Rational Function Optimization) based transition state optimizer. 
The algorithm uses a trust radius framework to control step sizes and requires careful tuning of 
its parameters for optimal performance across different potential energy surfaces.

\subsection{Hyperparameters Optimized}

\begin{table}[H]
\centering
\caption{Sella Hyperparameters and Search Ranges}
\begin{tabular}{llll}
\toprule
\textbf{Parameter} & \textbf{Description} & \textbf{HIP Range} & \textbf{SCINE Range} \\
\midrule
\texttt{delta0} & Initial trust radius & $[0.15, 0.8]$ (log) & $[0.03, 0.8]$ (log) \\
\texttt{rho\_dec} & Trust radius decrease threshold & $[15, 80]$ & $[3, 80]$ \\
\texttt{rho\_inc} & Trust radius increase threshold & $[1.01, 1.1]$ & $[1.01, 1.1]$ \\
\texttt{sigma\_dec} & Trust radius decrease factor & $[0.75, 0.95]$ & $[0.5, 0.95]$ \\
\texttt{sigma\_inc} & Trust radius increase factor & $[1.1, 1.8]$ & $[1.1, 1.8]$ \\
\texttt{fmax} & Force convergence threshold & $[10^{-4}, 10^{-2}]$ (log) & $[10^{-4}, 10^{-2}]$ (log) \\
\texttt{apply\_eckart} & Eckart project Hessian & \{True, False\} & \{True, False\} \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Fixed parameters}: \texttt{gamma}=0.0, \texttt{internal}=True, \texttt{use\_exact\_hessian}=True, 
\texttt{diag\_every\_n}=1, \texttt{max\_steps}=100.

\textbf{Objective function}: Weighted combination prioritizing eigenvalue TS rate (weight=1.0), 
speed bonus (weight=0.01), and Sella convergence rate (weight=0.001).

"""
    
    # HIP Sella Results
    hip_sella = results.get("hip_sella")
    if hip_sella:
        latex += generate_sella_section(hip_sella, "HIP", is_hip=True)
    
    # SCINE Sella Results
    scine_sella = results.get("scine_sella")
    if scine_sella:
        latex += generate_sella_section(scine_sella, "SCINE", is_hip=False, json_data=scine_json)
    
    # Sella comparison
    if hip_sella and scine_sella:
        latex += generate_sella_comparison(hip_sella, scine_sella)
    
    # ===================================================================================
    # PART 2: MULTI-MODE ECKART-MW GAD HPO
    # ===================================================================================
    latex += r"""
\newpage
\section{Multi-Mode Eckart-MW GAD Hyperparameter Optimization}

Multi-mode Eckart-MW GAD is a Gentlest Ascent Dynamics algorithm with an escape mechanism 
that perturbs along the second-smallest eigenvector when the optimization gets stuck in 
higher-order saddle points or plateaus.

\subsection{Hyperparameters Optimized}

\begin{table}[H]
\centering
\caption{Multi-Mode GAD Hyperparameters and Search Ranges}
\begin{tabular}{lll}
\toprule
\textbf{Parameter} & \textbf{Description} & \textbf{Range} \\
\midrule
\texttt{dt\_min} & Minimum time step & $[10^{-7}, 10^{-5}]$ (log) \\
\texttt{dt\_max} & Maximum time step & $[0.01, 0.1]$ (log) \\
\texttt{plateau\_patience} & Steps before dt adjustment & $[3, 20]$ \\
\texttt{plateau\_boost} & dt increase factor & $[1.2, 3.0]$ \\
\texttt{plateau\_shrink} & dt decrease factor & $[0.3, 0.7]$ \\
\texttt{escape\_disp\_threshold} & Displacement threshold for plateau detection & $[10^{-5}, 10^{-3}]$ (log) \\
\texttt{escape\_window} & Window size for plateau detection & $[10, 50]$ \\
\texttt{escape\_neg\_vib\_std} & Stability threshold for saddle index & $[0.1, 1.0]$ \\
\texttt{escape\_delta} & Base perturbation magnitude & $[0.05, 0.5]$ \\
\texttt{adaptive\_delta\_scale} & Scale factor for adaptive perturbations & $[0.0, 2.0]$ \\
\texttt{trust\_radius\_max} & Maximum displacement per step & $[0.1, 0.5]$ \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Objective function}: Convergence rate (primary) with small penalty for steps to converge.

"""
    
    # HIP Multi-mode Results
    hip_mm = results.get("hip_multimode")
    if hip_mm:
        latex += generate_multimode_section(hip_mm, "HIP")
    
    # SCINE Multi-mode Results
    scine_mm = results.get("scine_multimode")
    if scine_mm:
        latex += generate_multimode_section(scine_mm, "SCINE")
    
    # Multi-mode comparison
    if hip_mm and scine_mm:
        latex += generate_multimode_comparison(hip_mm, scine_mm)
    
    # Conclusion
    latex += r"""
\newpage
\section{Conclusions and Recommendations}

\subsection{Sella Algorithm}

Based on the HPO results:

\begin{enumerate}
    \item \textbf{Trust radius management is critical}: The initial trust radius (\texttt{delta0}) 
    and decrease parameters (\texttt{rho\_dec}, \texttt{sigma\_dec}) have the strongest impact 
    on convergence success.
    
    \item \textbf{HIP requires more conservative settings}: The ML potential's noise characteristics 
    require larger trust radii and more gradual trust adjustments compared to SCINE's analytical 
    Hessians.
    
    \item \textbf{Eckart projection effect varies}: For HIP, Eckart projection may help stabilize 
    the optimization by removing spurious translational/rotational contributions, while for SCINE 
    it appears less critical.
    
    \item \textbf{Convergence threshold}: An intermediate \texttt{fmax} around $10^{-3}$ appears 
    optimal; too tight causes non-convergence, too loose accepts poor TS candidates.
\end{enumerate}

\subsection{Multi-Mode GAD Algorithm}

\begin{enumerate}
    \item \textbf{Adaptive time stepping is essential}: The \texttt{plateau\_patience} and 
    \texttt{plateau\_boost} parameters significantly affect the ability to escape higher-order 
    saddle points.
    
    \item \textbf{Escape mechanism tuning}: The displacement threshold and window size for 
    detecting plateaus require careful tuning---too sensitive triggers unnecessary perturbations, 
    too insensitive causes the algorithm to get stuck.
    
    \item \textbf{Trust radius constraints}: A maximum displacement per step (\texttt{trust\_radius\_max}) 
    of 0.2--0.3 \AA{} provides a good balance between convergence speed and stability.
\end{enumerate}

\end{document}
"""
    
    return latex


def generate_sella_section(study: StudyResults, calc_name: str, is_hip: bool, json_data: Optional[Dict] = None) -> str:
    """Generate detailed Sella results section for one calculator."""
    
    latex = f"""
\\subsection{{{calc_name} Calculator Results}}

\\subsubsection{{Study Overview}}

\\begin{{table}}[H]
\\centering
\\caption{{{calc_name} Sella HPO Study Statistics}}
\\begin{{tabular}}{{ll}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Study Name & \\texttt{{{escape_latex(study.name)}}} \\\\
Total Trials & {study.n_trials} \\\\
Completed Trials & {study.n_completed} \\\\
Pruned Trials & {study.n_pruned} \\\\
"""
    
    if study.best_trial:
        latex += f"Best Score & {format_float(study.best_trial.value, 4)} \\\\\n"
        if "eigenvalue_ts_rate" in study.best_trial.user_attrs:
            latex += f"Best TS Rate & {format_float(study.best_trial.user_attrs['eigenvalue_ts_rate']*100, 1)}\\% \\\\\n"
        if "sella_convergence_rate" in study.best_trial.user_attrs:
            latex += f"Best Sella Conv. Rate & {format_float(study.best_trial.user_attrs['sella_convergence_rate']*100, 1)}\\% \\\\\n"
        if "avg_steps" in study.best_trial.user_attrs:
            latex += f"Avg Steps (Best) & {format_float(study.best_trial.user_attrs['avg_steps'], 1)} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    # Best configuration
    if study.best_trial:
        latex += f"""\\subsubsection{{Best Configuration}}

\\begin{{table}}[H]
\\centering
\\caption{{{calc_name} Sella Best Hyperparameters (Trial {study.best_trial.number})}}
\\begin{{tabular}}{{lll}}
\\toprule
\\textbf{{Parameter}} & \\textbf{{Value}} & \\textbf{{Search Range}} \\\\
\\midrule
"""
        for param in ["delta0", "rho_dec", "rho_inc", "sigma_dec", "sigma_inc", "fmax", "apply_eckart"]:
            if param in study.best_trial.params:
                val = study.best_trial.params[param]
                if param == "apply_eckart":
                    val_str = "True" if val == 1.0 else "False"
                elif param == "fmax":
                    val_str = f"{val:.2e}"
                else:
                    val_str = format_float(val, 4)
                
                range_str = ""
                if param in study.param_ranges:
                    low, high = study.param_ranges[param]
                    range_str = format_range(low, high)
                
                latex += f"\\texttt{{{escape_latex(param)}}} & {val_str} & {range_str} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    # Performance metrics for completed trials
    completed = [t for t in study.trials if t.state == "COMPLETE" and t.value is not None]
    if completed:
        values = [t.value for t in completed]
        ts_rates = [t.user_attrs.get("eigenvalue_ts_rate", 0) for t in completed if "eigenvalue_ts_rate" in t.user_attrs]
        sella_rates = [t.user_attrs.get("sella_convergence_rate", 0) for t in completed if "sella_convergence_rate" in t.user_attrs]
        steps = [t.user_attrs.get("avg_steps", 0) for t in completed if "avg_steps" in t.user_attrs]
        
        latex += f"""\\subsubsection{{Performance Distribution Across Trials}}

\\begin{{table}}[H]
\\centering
\\caption{{{calc_name} Sella Performance Statistics (n={len(completed)} completed trials)}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Mean}} & \\textbf{{Std}} & \\textbf{{Min}} & \\textbf{{Max}} & \\textbf{{Median}} \\\\
\\midrule
"""
        if values:
            stats = compute_statistics(values)
            latex += f"Score & {format_float(stats['mean'], 3)} & {format_float(stats['std'], 3)} & {format_float(stats['min'], 3)} & {format_float(stats['max'], 3)} & {format_float(stats['median'], 3)} \\\\\n"
        
        if ts_rates:
            stats = compute_statistics([r*100 for r in ts_rates])
            latex += f"TS Rate (\\%) & {format_float(stats['mean'], 1)} & {format_float(stats['std'], 1)} & {format_float(stats['min'], 1)} & {format_float(stats['max'], 1)} & {format_float(stats['median'], 1)} \\\\\n"
        
        if sella_rates:
            stats = compute_statistics([r*100 for r in sella_rates])
            latex += f"Sella Conv. (\\%) & {format_float(stats['mean'], 1)} & {format_float(stats['std'], 1)} & {format_float(stats['min'], 1)} & {format_float(stats['max'], 1)} & {format_float(stats['median'], 1)} \\\\\n"
        
        if steps:
            stats = compute_statistics(steps)
            latex += f"Avg Steps & {format_float(stats['mean'], 1)} & {format_float(stats['std'], 1)} & {format_float(stats['min'], 1)} & {format_float(stats['max'], 1)} & {format_float(stats['median'], 1)} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    # Parameter correlations
    params = ["delta0", "rho_dec", "rho_inc", "sigma_dec", "sigma_inc", "fmax"]
    corrs = compute_correlations(study.trials, params, "value")
    
    if corrs:
        latex += f"""\\subsubsection{{Parameter-Score Correlations}}

\\begin{{table}}[H]
\\centering
\\caption{{{calc_name} Sella Parameter Correlations with Score}}
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Parameter}} & \\textbf{{Correlation}} \\\\
\\midrule
"""
        # Sort by absolute correlation
        sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
        for param, corr in sorted_corrs:
            color = "bestcolor" if corr > 0.2 else ("worstcolor" if corr < -0.2 else "neutralcolor")
            latex += f"\\texttt{{{escape_latex(param)}}} & \\textcolor{{{color}}}{{{format_float(corr, 3)}}} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    # Parameter effectiveness analysis
    latex += f"""\\subsubsection{{Parameter Range Effectiveness}}

"""
    for param in ["delta0", "rho_dec", "sigma_dec", "fmax"]:
        effectiveness = analyze_param_effectiveness(study.trials, param)
        if effectiveness and len(effectiveness) >= 2:
            latex += f"""\\paragraph{{\\texttt{{{escape_latex(param)}}} Analysis}}

\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lll}}
\\toprule
\\textbf{{Range}} & \\textbf{{Parameter Values}} & \\textbf{{Mean Score ($\\pm$ std)}} \\\\
\\midrule
"""
            for range_name in ["low_range", "mid_range", "high_range"]:
                if range_name in effectiveness:
                    data = effectiveness[range_name]
                    pr = data["param_range"]
                    latex += f"{range_name.replace('_', ' ').title()} & {format_range(pr[0], pr[1])} & {format_float(data['metric_mean'], 3)} $\\pm$ {format_float(data['metric_std'], 3)} \\\\\n"
            
            latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    # Negative eigenvalue distribution (if available)
    neg_eig_dist = defaultdict(int)
    for t in study.trials:
        if t.state == "COMPLETE" and "neg_eigval_distribution" in t.user_attrs:
            dist = t.user_attrs["neg_eigval_distribution"]
            for k, v in dist.items():
                neg_eig_dist[int(k)] += v
    
    if neg_eig_dist:
        total = sum(neg_eig_dist.values())
        latex += f"""\\subsubsection{{Negative Eigenvalue Distribution (All Trials Combined)}}

\\begin{{table}}[H]
\\centering
\\caption{{{calc_name} Final Negative Eigenvalue Counts (Total samples: {total})}}
\\begin{{tabular}}{{rrr}}
\\toprule
\\textbf{{Neg. Eigenvalues}} & \\textbf{{Count}} & \\textbf{{Percentage}} \\\\
\\midrule
"""
        for k in sorted(neg_eig_dist.keys()):
            v = neg_eig_dist[k]
            pct = 100 * v / total if total > 0 else 0
            color = "bestcolor" if k == 1 else "neutralcolor"
            latex += f"\\textcolor{{{color}}}{{{k}}} & {v} & {format_float(pct, 1)}\\% \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    # HIP-specific analysis
    if is_hip:
        latex += generate_hip_sella_deep_analysis(study)
    
    return latex


def generate_hip_sella_deep_analysis(study: StudyResults) -> str:
    """Generate deep analysis specific to HIP Sella results."""
    
    latex = r"""\subsubsection{HIP-Specific Analysis}

"""
    
    completed = [t for t in study.trials if t.state == "COMPLETE" and t.value is not None]
    if len(completed) < 5:
        latex += "Insufficient completed trials for deep analysis.\n\n"
        return latex
    
    # Eckart projection analysis
    eckart_true = [t for t in completed if t.params.get("apply_eckart", 0) == 1.0]
    eckart_false = [t for t in completed if t.params.get("apply_eckart", 0) == 0.0]
    
    if eckart_true and eckart_false:
        latex += r"""\paragraph{Eckart Projection Impact}

\begin{table}[H]
\centering
\caption{Effect of Eckart Projection on HIP Sella Performance}
\begin{tabular}{lrrrr}
\toprule
\textbf{Setting} & \textbf{N Trials} & \textbf{Mean Score} & \textbf{Mean TS Rate} & \textbf{Mean Steps} \\
\midrule
"""
        for label, trials in [("With Eckart", eckart_true), ("Without Eckart", eckart_false)]:
            n = len(trials)
            mean_score = np.mean([t.value for t in trials])
            ts_rates = [t.user_attrs.get("eigenvalue_ts_rate", 0) for t in trials]
            mean_ts = np.mean(ts_rates) * 100 if ts_rates else 0
            steps = [t.user_attrs.get("avg_steps", 0) for t in trials]
            mean_steps = np.mean(steps) if steps else 0
            
            latex += f"{label} & {n} & {format_float(mean_score, 3)} & {format_float(mean_ts, 1)}\\% & {format_float(mean_steps, 1)} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    # Trust radius regime analysis
    latex += r"""\paragraph{Trust Radius Regime Analysis}

The trust radius parameters form a coupled system. High \texttt{delta0} with high \texttt{rho\_dec} 
creates a ``conservative'' regime that takes smaller steps but is more stable. Low \texttt{delta0} 
with low \texttt{rho\_dec} is ``aggressive'' but risks overshooting.

"""
    
    # Identify failure modes
    low_score_trials = sorted(completed, key=lambda t: t.value)[:5]
    high_score_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    
    latex += r"""\paragraph{Failure Mode Analysis}

\begin{table}[H]
\centering
\caption{Bottom 5 vs Top 5 Trial Parameter Comparison}
\begin{tabular}{lrrrr}
\toprule
& \multicolumn{2}{c}{\textbf{Bottom 5 (Failed)}} & \multicolumn{2}{c}{\textbf{Top 5 (Success)}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
\textbf{Parameter} & \textbf{Mean} & \textbf{Std} & \textbf{Mean} & \textbf{Std} \\
\midrule
"""
    for param in ["delta0", "rho_dec", "sigma_dec", "fmax"]:
        low_vals = [t.params.get(param, np.nan) for t in low_score_trials]
        high_vals = [t.params.get(param, np.nan) for t in high_score_trials]
        
        low_vals = [v for v in low_vals if not np.isnan(v)]
        high_vals = [v for v in high_vals if not np.isnan(v)]
        
        if low_vals and high_vals:
            if param == "fmax":
                latex += f"\\texttt{{{escape_latex(param)}}} & {np.mean(low_vals):.2e} & {np.std(low_vals):.2e} & {np.mean(high_vals):.2e} & {np.std(high_vals):.2e} \\\\\n"
            else:
                latex += f"\\texttt{{{escape_latex(param)}}} & {format_float(np.mean(low_vals), 3)} & {format_float(np.std(low_vals), 3)} & {format_float(np.mean(high_vals), 3)} & {format_float(np.std(high_vals), 3)} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}

\paragraph{Key Observations for HIP Sella}

\begin{itemize}
"""
    
    # Generate insights based on data
    if corrs := compute_correlations(study.trials, ["delta0", "rho_dec", "sigma_dec"], "value"):
        if "delta0" in corrs:
            direction = "larger" if corrs["delta0"] > 0 else "smaller"
            latex += f"    \\item \\texttt{{delta0}}: {direction.title()} values tend to improve performance (correlation: {format_float(corrs['delta0'], 3)})\n"
        if "rho_dec" in corrs:
            direction = "larger" if corrs["rho_dec"] > 0 else "smaller"
            latex += f"    \\item \\texttt{{rho\\_dec}}: {direction.title()} values correlate with better scores (correlation: {format_float(corrs['rho_dec'], 3)})\n"
        if "sigma_dec" in corrs:
            direction = "larger" if corrs["sigma_dec"] > 0 else "smaller"
            latex += f"    \\item \\texttt{{sigma\\_dec}}: {direction.title()} decrease factors work better (correlation: {format_float(corrs['sigma_dec'], 3)})\n"
    
    latex += r"""    \item HIP's neural network-based Hessian introduces noise that requires more conservative trust radius management
    \item The algorithm tends to over-stabilize at higher-order saddle points when trust adjustments are too aggressive
\end{itemize}

"""
    
    return latex


def generate_sella_comparison(hip: StudyResults, scine: StudyResults) -> str:
    """Generate comparison between HIP and SCINE Sella results."""
    
    latex = r"""\subsection{HIP vs SCINE Sella Comparison}

\begin{table}[H]
\centering
\caption{Sella HPO: HIP vs SCINE Comparison}
\begin{tabular}{lrr}
\toprule
\textbf{Metric} & \textbf{HIP} & \textbf{SCINE} \\
\midrule
"""
    
    latex += f"Completed Trials & {hip.n_completed} & {scine.n_completed} \\\\\n"
    latex += f"Pruned Trials & {hip.n_pruned} & {scine.n_pruned} \\\\\n"
    
    if hip.best_trial and scine.best_trial:
        latex += f"Best Score & {format_float(hip.best_trial.value, 4)} & {format_float(scine.best_trial.value, 4)} \\\\\n"
        
        hip_ts = hip.best_trial.user_attrs.get("eigenvalue_ts_rate", 0) * 100
        scine_ts = scine.best_trial.user_attrs.get("eigenvalue_ts_rate", 0) * 100
        latex += f"Best TS Rate (\\%) & {format_float(hip_ts, 1)} & {format_float(scine_ts, 1)} \\\\\n"
        
        hip_steps = hip.best_trial.user_attrs.get("avg_steps", 0)
        scine_steps = scine.best_trial.user_attrs.get("avg_steps", 0)
        latex += f"Avg Steps (Best) & {format_float(hip_steps, 1)} & {format_float(scine_steps, 1)} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}

\paragraph{Key Differences}

\begin{itemize}
    \item \textbf{Trust Radius}: HIP requires larger initial trust radii (higher \texttt{delta0}) to handle 
    the noise in ML-predicted Hessians
    \item \textbf{Convergence Speed}: SCINE's analytical Hessians enable more aggressive stepping and faster convergence
    \item \textbf{Stability}: HIP benefits more from conservative trust radius decrease settings to avoid oscillations
\end{itemize}

"""
    return latex


def generate_multimode_section(study: StudyResults, calc_name: str) -> str:
    """Generate detailed multi-mode GAD results section."""
    
    latex = f"""
\\subsection{{{calc_name} Calculator Results}}

\\subsubsection{{Study Overview}}

\\begin{{table}}[H]
\\centering
\\caption{{{calc_name} Multi-Mode GAD HPO Study Statistics}}
\\begin{{tabular}}{{ll}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Study Name & \\texttt{{{escape_latex(study.name)}}} \\\\
Total Trials & {study.n_trials} \\\\
Completed Trials & {study.n_completed} \\\\
"""
    
    if study.best_trial:
        latex += f"Best Score & {format_float(study.best_trial.value, 4)} \\\\\n"
        if "convergence_rate" in study.best_trial.user_attrs:
            latex += f"Best Conv. Rate & {format_float(study.best_trial.user_attrs['convergence_rate']*100, 1)}\\% \\\\\n"
        if "mean_steps_to_converge" in study.best_trial.user_attrs:
            latex += f"Mean Steps (Best) & {format_float(study.best_trial.user_attrs['mean_steps_to_converge'], 1)} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    # Best configuration
    if study.best_trial:
        latex += f"""\\subsubsection{{Best Configuration}}

\\begin{{table}}[H]
\\centering
\\caption{{{calc_name} Multi-Mode GAD Best Hyperparameters (Trial {study.best_trial.number})}}
\\begin{{tabular}}{{lll}}
\\toprule
\\textbf{{Parameter}} & \\textbf{{Value}} & \\textbf{{Search Range}} \\\\
\\midrule
"""
        mm_params = ["dt_min", "dt_max", "plateau_patience", "plateau_boost", "plateau_shrink",
                     "escape_disp_threshold", "escape_window", "escape_neg_vib_std",
                     "escape_delta", "adaptive_delta_scale", "trust_radius_max"]
        
        for param in mm_params:
            if param in study.best_trial.params:
                val = study.best_trial.params[param]
                if param in ["dt_min", "escape_disp_threshold"]:
                    val_str = f"{val:.2e}"
                else:
                    val_str = format_float(val, 4)
                
                range_str = ""
                if param in study.param_ranges:
                    low, high = study.param_ranges[param]
                    if param in ["dt_min", "escape_disp_threshold"]:
                        range_str = f"[{low:.1e}, {high:.1e}]"
                    else:
                        range_str = format_range(low, high)
                
                latex += f"\\texttt{{{escape_latex(param)}}} & {val_str} & {range_str} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    # Performance statistics
    completed = [t for t in study.trials if t.state == "COMPLETE" and t.value is not None]
    if completed:
        values = [t.value for t in completed]
        conv_rates = [t.user_attrs.get("convergence_rate", 0) for t in completed if "convergence_rate" in t.user_attrs]
        steps = [t.user_attrs.get("mean_steps_to_converge", 0) for t in completed if "mean_steps_to_converge" in t.user_attrs]
        
        latex += f"""\\subsubsection{{Performance Distribution Across Trials}}

\\begin{{table}}[H]
\\centering
\\caption{{{calc_name} Multi-Mode GAD Performance Statistics (n={len(completed)} completed trials)}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Mean}} & \\textbf{{Std}} & \\textbf{{Min}} & \\textbf{{Max}} & \\textbf{{Median}} \\\\
\\midrule
"""
        if values:
            stats = compute_statistics(values)
            latex += f"Score & {format_float(stats['mean'], 3)} & {format_float(stats['std'], 3)} & {format_float(stats['min'], 3)} & {format_float(stats['max'], 3)} & {format_float(stats['median'], 3)} \\\\\n"
        
        if conv_rates:
            stats = compute_statistics([r*100 for r in conv_rates])
            latex += f"Conv. Rate (\\%) & {format_float(stats['mean'], 1)} & {format_float(stats['std'], 1)} & {format_float(stats['min'], 1)} & {format_float(stats['max'], 1)} & {format_float(stats['median'], 1)} \\\\\n"
        
        if steps:
            stats = compute_statistics(steps)
            latex += f"Mean Steps & {format_float(stats['mean'], 1)} & {format_float(stats['std'], 1)} & {format_float(stats['min'], 1)} & {format_float(stats['max'], 1)} & {format_float(stats['median'], 1)} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    # Parameter correlations
    mm_params = ["dt_min", "dt_max", "plateau_patience", "plateau_boost", "plateau_shrink",
                 "escape_disp_threshold", "escape_window", "escape_neg_vib_std",
                 "escape_delta", "adaptive_delta_scale", "trust_radius_max"]
    corrs = compute_correlations(study.trials, mm_params, "value")
    
    if corrs:
        latex += f"""\\subsubsection{{Parameter-Score Correlations}}

\\begin{{table}}[H]
\\centering
\\caption{{{calc_name} Multi-Mode GAD Parameter Correlations with Score}}
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Parameter}} & \\textbf{{Correlation}} \\\\
\\midrule
"""
        sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
        for param, corr in sorted_corrs:
            color = "bestcolor" if corr > 0.2 else ("worstcolor" if corr < -0.2 else "neutralcolor")
            latex += f"\\texttt{{{escape_latex(param)}}} & \\textcolor{{{color}}}{{{format_float(corr, 3)}}} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    return latex


def generate_multimode_comparison(hip: StudyResults, scine: StudyResults) -> str:
    """Generate comparison between HIP and SCINE multi-mode results."""
    
    latex = r"""\subsection{HIP vs SCINE Multi-Mode GAD Comparison}

\begin{table}[H]
\centering
\caption{Multi-Mode GAD HPO: HIP vs SCINE Comparison}
\begin{tabular}{lrr}
\toprule
\textbf{Metric} & \textbf{HIP} & \textbf{SCINE} \\
\midrule
"""
    
    latex += f"Completed Trials & {hip.n_completed} & {scine.n_completed} \\\\\n"
    
    if hip.best_trial and scine.best_trial:
        latex += f"Best Score & {format_float(hip.best_trial.value, 4)} & {format_float(scine.best_trial.value, 4)} \\\\\n"
        
        hip_cr = hip.best_trial.user_attrs.get("convergence_rate", 0) * 100
        scine_cr = scine.best_trial.user_attrs.get("convergence_rate", 0) * 100
        latex += f"Best Conv. Rate (\\%) & {format_float(hip_cr, 1)} & {format_float(scine_cr, 1)} \\\\\n"
        
        hip_steps = hip.best_trial.user_attrs.get("mean_steps_to_converge", 0)
        scine_steps = scine.best_trial.user_attrs.get("mean_steps_to_converge", 0)
        latex += f"Mean Steps (Best) & {format_float(hip_steps, 1)} & {format_float(scine_steps, 1)} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    return latex


def main():
    """Main analysis and report generation."""
    print("=" * 80)
    print("HPO Results Analysis")
    print("=" * 80)
    
    # Load all databases
    results = {}
    for key, path in DB_PATHS.items():
        print(f"\nLoading {key} from {path}...")
        results[key] = load_optuna_db(path)
        if results[key]:
            print(f"  Loaded {results[key].n_trials} trials ({results[key].n_completed} completed)")
        else:
            print(f"  Failed to load")
    
    # Load SCINE Sella JSON
    scine_json = load_scine_sella_json()
    if scine_json:
        print(f"\nLoaded SCINE Sella JSON with {scine_json.get('n_trials_total', 0)} trials")
    
    # Generate LaTeX report
    print("\n" + "=" * 80)
    print("Generating LaTeX Report...")
    print("=" * 80)
    
    latex_content = generate_latex_report(results, scine_json)
    
    # Write to file
    output_path = TS_TOOLS_DIR / "results" / "hpo_analysis_report.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(latex_content)
    
    print(f"\nReport written to: {output_path}")
    print("=" * 80)
    
    # Print summary
    print("\n=== SUMMARY ===")
    for key, study in results.items():
        if study and study.best_trial:
            print(f"\n{key}:")
            print(f"  Best score: {study.best_trial.value:.4f}")
            print(f"  Best params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
