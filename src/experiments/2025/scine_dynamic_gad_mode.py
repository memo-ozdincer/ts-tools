from __future__ import annotations

"""SCINE-only entrypoint for the Dynamic GAD Mode Switching experiment.

This wraps the shared implementation in `src.experiments.dynamic_gad_mode` but
forces `--calculator=scine` so SCINE experiments are cleanly separated from
HIP runs.

Key difference from multi_mode approach:
- Instead of kicks, dynamically switches which eigenvector GAD follows
- Escalate (v1 -> v2 -> v3) when stuck (tiny displacement)
- De-escalate (v3 -> v2 -> v1) when progress resumes
"""

from .dynamic_gad_mode import main as _main


def main(argv: list[str] | None = None) -> None:
    _main(
        argv=argv,
        default_calculator="scine",
        enforce_calculator=True,
        script_name_prefix="exp-scine-dynamic-gad-mode",
    )


if __name__ == "__main__":
    main()
