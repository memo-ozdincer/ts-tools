"""Core (minimal) algorithms for TS search.

These modules intentionally contain only algorithmic logic and depend on an
external `predict_fn` callable for energies/forces/Hessians.

The goal is to keep these files small, testable, and independent of HIP/SCINE,
W&B, dataset loading, or experiment CLI concerns.
"""

from .types import PredictFn

__all__ = ["PredictFn"]
