from __future__ import annotations

"""HIP-only entrypoint for eigenvalue classification.

This is a convenience wrapper that runs the classification using only HIP.
For comparing HIP vs SCINE, use eigenvalue_classification.py directly.
"""

from .eigenvalue_classification import main as _main


def main(argv: list[str] | None = None) -> None:
    # Add --hip-only flag implicitly
    _main(argv=argv)


if __name__ == "__main__":
    main()
