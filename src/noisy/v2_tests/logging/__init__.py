"""Extended logging infrastructure for GAD failure diagnosis.

This module provides comprehensive metrics computation and logging for:
1. Eigenvalue spectrum analysis (first 6 vibrational modes)
2. Eigenvalue gaps (critical for singularity detection where λ₁ ≈ λ₂)
3. Morse index tracking (count of negative eigenvalues)
4. GAD direction quality metrics
5. Escape event logging with pre/post diagnostics
"""

from .metrics import ExtendedMetrics, compute_extended_metrics
from .escape_logger import EscapeEvent, create_escape_event
from .trajectory_logger import TrajectoryLogger

__all__ = [
    "ExtendedMetrics",
    "compute_extended_metrics",
    "EscapeEvent",
    "create_escape_event",
    "TrajectoryLogger",
]
