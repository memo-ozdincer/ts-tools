from __future__ import annotations

"""SCINE-only entrypoint for the Eckart-MW multi-mode escape experiment.

This wraps the shared implementation in `src.experiments.multi_mode_eckartmw` but
forces `--calculator=scine` so SCINE experiments are cleanly separated from
HIP runs.

Key difference from scine_multi_mode.py:
- Uses Eckart-projected, mass-weighted Hessian for ALL eigenvector computations
- GAD direction uses lowest vibrational eigenvector (skips TR modes)
- Escape v2 direction uses second vibrational eigenvector (skips TR modes)
"""

from .multi_mode_eckartmw import main as _main


def main(argv: list[str] | None = None) -> None:
    _main(
        argv=argv,
        default_calculator="scine",
        enforce_calculator=True,
        script_name_prefix="exp-scine-multi-mode-eckartmw",
    )


if __name__ == "__main__":
    main()
