from __future__ import annotations

"""HIP-only entrypoint for the multi-mode escape experiment.

This wraps the shared implementation in `src.experiments.multi_mode` but
forces `--calculator=hip` so HIP experiments are cleanly separated from
SCINE runs.
"""

from .multi_mode import main as _main


def main(argv: list[str] | None = None) -> None:
    _main(
        argv=argv,
        default_calculator="hip",
        enforce_calculator=True,
        script_name_prefix="exp-hip-multi-mode",
    )


if __name__ == "__main__":
    main()
