"""HIP-only entrypoint for Sella TS refinement experiment.

This wraps the shared implementation in `sella_experiment.py` but
forces `--calculator=hip` so HIP experiments are cleanly separated from
SCINE runs.

Sella uses RS-P-RFO (Restricted-Step Partitioned Rational Function Optimization)
with internal coordinates for robust saddle point optimization.

Key differences from GAD-based methods:
- Uses trust-radius controlled steps (prevents oscillation/divergence)
- Works in internal coordinates (bonds/angles/dihedrals) for chemical robustness
- No need for eigenvector tracking - Sella handles this internally
"""
from __future__ import annotations

from .sella_experiment import main as _main


def main(argv: list[str] | None = None) -> None:
    """Run Sella TS experiment with HIP calculator."""
    _main(
        argv=argv,
        default_calculator="hip",
        enforce_calculator=True,
        script_name_prefix="exp-hip-sella",
    )


if __name__ == "__main__":
    main()
