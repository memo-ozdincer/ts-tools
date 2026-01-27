"""Experiment runners for systematic comparison.

Implemented:
1. run_with_diagnostics - GAD + v₂ kicking with comprehensive logging

Runners in baselines/:
2. run_adaptive_k_hisd - Adaptive k-HiSD runner

Usage:
  - For parallel execution, use the scripts in src/noisy/:
    - scine_multi_mode_eckartmw_parallel_diagnostics.py (GAD + v₂ kick)
    - scine_adaptive_k_hisd_parallel.py (Adaptive k-HiSD)
"""

from .run_with_diagnostics import run_multi_mode_with_diagnostics

__all__ = [
    "run_multi_mode_with_diagnostics",
]
