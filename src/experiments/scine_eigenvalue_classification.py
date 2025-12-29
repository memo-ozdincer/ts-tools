from __future__ import annotations

"""SCINE-only entrypoint for eigenvalue classification.

This is a convenience wrapper that runs the classification using only SCINE.
For comparing HIP vs SCINE, use eigenvalue_classification.py directly.
"""

from .eigenvalue_classification import main as _main


def main(argv: list[str] | None = None) -> None:
    # Add --scine-only flag implicitly
    _main(argv=argv)


if __name__ == "__main__":
    main()
