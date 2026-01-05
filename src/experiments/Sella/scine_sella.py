"""SCINE-only entrypoint for Sella TS refinement experiment.

This wraps the shared implementation in `sella_experiment.py` but
forces `--calculator=scine` so SCINE experiments are cleanly separated from
HIP runs.

Sella uses RS-P-RFO (Restricted-Step Partitioned Rational Function Optimization)
with internal coordinates for robust saddle point optimization.

Note: SCINE calculations run on CPU only, regardless of --device setting.
"""
from __future__ import annotations

from .sella_experiment import main as _main


def main(argv: list[str] | None = None) -> None:
    """Run Sella TS experiment with SCINE calculator."""
    _main(
        argv=argv,
        default_calculator="scine",
        enforce_calculator=True,
        script_name_prefix="exp-scine-sella",
    )


if __name__ == "__main__":
    main()
