"""Baseline algorithm implementations for comparison.

Implemented:
1. k-HiSD (k_hisd.py) - High-index Saddle Dynamics with k reflections
2. Adaptive k-HiSD (run_adaptive_k_hisd.py) - Automatically adjusts k = Morse index

To implement:
3. Plain GAD (no mode tracking, no kicking)
4. iHiSD (gradient flow → HiSD crossover with α dynamics)

Key insight from diagnostic analysis:
- v₂ kicking doesn't reduce Morse index (stays at ~5)
- v₂ kicking works by "unsticking" GAD (preventing dt→0)
- Adaptive k-HiSD should be more principled: k-reflection for index-k saddles

All baselines share the same logging infrastructure for fair comparison.
"""

from .k_hisd import (
    compute_hisd_direction,
    compute_reflection_matrix,
    adaptive_k_hisd_step,
    compute_adaptive_k,
)

__all__ = [
    "compute_hisd_direction",
    "compute_reflection_matrix",
    "adaptive_k_hisd_step",
    "compute_adaptive_k",
]
