"""V2 Tests: Comprehensive logging and baseline experiments for understanding GAD failures.

This module provides:
1. Extended logging infrastructure to diagnose WHY GAD gets stuck
2. Baseline algorithm implementations (plain GAD, HiSD, adaptive k-HiSD, iHiSD)
3. Kick experiment implementations (gradient descent, random, adaptive k-reflect)
4. Analysis tools for understanding eigenvalue gaps, Morse indices, and escape behavior

Key hypothesis from Levitt-Ortner and iHiSD papers:
- GAD gets stuck at singularity sets S = {x : λ₁(x) = λ₂(x)}
- GAD gets stuck at high-index saddles (index > 1)
- v₂ kicking works by escaping these via implicit higher-index HiSD steps
"""

from .logging import (
    ExtendedMetrics,
    EscapeEvent,
    TrajectoryLogger,
    compute_extended_metrics,
)

__all__ = [
    "ExtendedMetrics",
    "EscapeEvent",
    "TrajectoryLogger",
    "compute_extended_metrics",
]
