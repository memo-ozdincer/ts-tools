from __future__ import annotations

from typing import Any, Tuple

from ..dependencies.calculators import make_hip_predict_fn, make_scine_predict_fn


def make_predict_fn_from_calculator(calculator, calculator_type: str):
    """Return a `PredictFn` for the given calculator."""

    calc = (calculator_type or "hip").lower()
    if calc == "scine":
        return make_scine_predict_fn(calculator)
    return make_hip_predict_fn(calculator)
