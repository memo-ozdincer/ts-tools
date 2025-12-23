from __future__ import annotations

"""Compatibility wrapper.

Historically, SLURM scripts invoked `python -m src.lbfgs_energy_minimizer`.
The maintained implementation now lives in `src.experiments.lbfgs_energy_minimizer`.
"""

from .experiments.lbfgs_energy_minimizer import main


if __name__ == "__main__":
    main()
