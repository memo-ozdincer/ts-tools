from __future__ import annotations

"""SCINE-only entrypoint for the Eckart-MW multi-mode noisy geometry runner.

This wraps the shared implementation in `src.noisy.multi_mode_eckartmw` but
forces `--calculator=scine` so SCINE runs are cleanly separated from HIP runs.

This is the go-to script for noisy/perturbed starting geometries with the SCINE
(Sparrow/DFTB) analytical calculator.

Key features:
- Uses Eckart-projected, mass-weighted Hessian for ALL eigenvector computations
- GAD direction uses lowest vibrational eigenvector (skips TR modes)
- Escape v2 direction uses second vibrational eigenvector (skips TR modes)
- Displacement-based plateau detection for robust convergence
"""

from .multi_mode_eckartmw import main as _main


def main(argv: list[str] | None = None) -> None:
    _main(
        argv=argv,
        default_calculator="scine",
        enforce_calculator=True,
        script_name_prefix="noisy-scine-multi-mode-eckartmw",
    )


if __name__ == "__main__":
    main()
