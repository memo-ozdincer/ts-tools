from __future__ import annotations

"""SCINE-only entrypoint for the multi-mode escape experiment.

This wraps the shared implementation in `src.experiments.multi_mode` but
forces `--calculator=scine` so SCINE experiments are cleanly separated from
HIP runs.
"""

from .multi_mode import main as _main


def main(argv: list[str] | None = None) -> None:
    _main(
        argv=argv,
        default_calculator="scine",
        enforce_calculator=True,
        script_name_prefix="exp-scine-multi-mode",
    )


if __name__ == "__main__":
    main()
