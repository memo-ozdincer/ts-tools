"""Baseline algorithm implementations for comparison.

This module will contain:
1. Plain GAD (no mode tracking, no kicking)
2. GAD + Mode Tracking (no kicking)
3. HiSD k=1
4. Adaptive k-HiSD
5. iHiSD (gradient flow → HiSD crossover)

All baselines share the same logging infrastructure for fair comparison.
"""
