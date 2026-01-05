"""Sella transition state optimizer integration.

This module provides Sella-based TS refinement using the RS-P-RFO algorithm
with internal coordinates. It wraps HIP and SCINE calculators as ASE calculators
to interface with Sella's optimizer.

Key features:
- RS-P-RFO (Restricted-Step Partitioned Rational Function Optimization)
- Internal coordinates for chemically robust optimization
- Trust radius management to prevent oscillation/divergence
- Post-optimization eigenvalue validation using existing Hessian pipeline

Usage:
    # HIP (GPU)
    python -m src.experiments.Sella.hip_sella --start-from midpoint_rt_noise1.0A

    # SCINE (CPU)
    python -m src.experiments.Sella.scine_sella --start-from midpoint_rt_noise1.0A
"""

from .sella_ts import run_sella_ts, validate_ts_eigenvalues
from .ase_calculators import HipASECalculator, ScineASECalculator, create_ase_calculator

__all__ = [
    "run_sella_ts",
    "validate_ts_eigenvalues",
    "HipASECalculator",
    "ScineASECalculator",
    "create_ase_calculator",
]
