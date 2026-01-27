"""Extended logging infrastructure for GAD failure diagnosis.

This module provides comprehensive metrics computation and logging for:
1. Eigenvalue spectrum analysis (first 6 vibrational modes)
2. Eigenvalue gaps (critical for singularity detection where λ₁ ≈ λ₂)
3. Morse index tracking (count of negative eigenvalues)
4. GAD direction quality metrics
5. Escape event logging with pre/post diagnostics
6. TR mode verification (near-zero eigenvalue diagnostics)
7. Visualization utilities for dt_eff trajectories and diagnostics
"""

from .metrics import ExtendedMetrics, compute_extended_metrics
from .escape_logger import EscapeEvent, create_escape_event
from .trajectory_logger import TrajectoryLogger
from .visualization import (
    collect_trajectories_from_dir,
    plot_dt_eff_trajectories,
    plot_tr_mode_diagnostics,
    plot_morse_index_evolution,
    create_summary_report,
    generate_all_plots,
)

__all__ = [
    "ExtendedMetrics",
    "compute_extended_metrics",
    "EscapeEvent",
    "create_escape_event",
    "TrajectoryLogger",
    "collect_trajectories_from_dir",
    "plot_dt_eff_trajectories",
    "plot_tr_mode_diagnostics",
    "plot_morse_index_evolution",
    "create_summary_report",
    "generate_all_plots",
]
