from __future__ import annotations

"""HIP-only entrypoint for the GAD plateau dt-control experiment.

This wraps the shared implementation in `src.experiments.scine_gad_plateau` but
forces `--calculator=hip` so noisy HIP experiments are cleanly separated from
SCINE runs (different Hessian/diagnostic characteristics).
"""

from .scine_gad_plateau import main as _main


def main(argv: list[str] | None = None) -> None:
    _main(
        argv=argv,
        default_calculator="hip",
        enforce_calculator=True,
        script_name_prefix="exp-hip-gad-plateau",
    )


if __name__ == "__main__":
    main()
