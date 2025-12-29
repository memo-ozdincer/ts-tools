"""Noisy geometry runners for GAD-based transition state search.

This package contains production-ready scripts for TS search starting from
noisy (perturbed) geometries. These are the go-to scripts for handling
realistic, imperfect input structures.

The key algorithm is the multi-mode escape mechanism with Eckart-projected,
mass-weighted Hessians. This stabilizes eigenvector computations by:
1. Removing translation/rotation null modes via Eckart projection
2. Mass-weighting ensures physically meaningful vibrational modes

Modules:
    multi_mode_eckartmw: Core algorithm implementation
    hip_multi_mode_eckartmw: HIP calculator entrypoint
    scine_multi_mode_eckartmw: SCINE calculator entrypoint

Usage:
    python -m src.noisy.hip_multi_mode_eckartmw --h5-path ... --wandb
    python -m src.noisy.scine_multi_mode_eckartmw --h5-path ... --wandb
"""

from .multi_mode_eckartmw import (
    run_multi_mode_escape,
    perform_escape_perturbation,
    get_projected_hessian,
    gad_euler_step_projected,
)

__all__ = [
    "run_multi_mode_escape",
    "perform_escape_perturbation",
    "get_projected_hessian",
    "gad_euler_step_projected",
]
