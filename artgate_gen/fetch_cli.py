"""Command-line entrypoint for :mod:`fetch`, called by ``run.sh``.

A separate file rather than a ``python -c`` block in the shell script: inline
program logic in another language is invisible to this repo's linters, so it
lives here where ruff can see it. Mirrors :mod:`repair_cli`.

Only ``fetch_thrifty`` is exposed -- it is the strategy ``run.sh`` has always
used (formerly the deleted ``fetch_model.py``'s only entry point), and it is
idempotent, so a plain re-run is always safe.
"""

from __future__ import annotations

import argparse

from kuhy_credit import credit_line

from artgate_gen.fetch import fetch_thrifty


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    desc = "Populate the local Hugging Face cache for the models this package uses."
    return argparse.ArgumentParser(description=desc, epilog=credit_line())


def main(argv: list[str] | None = None) -> int:
    """Populate the local Hugging Face cache for the models this package uses.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv[1:]``.

    Returns:
        ``0`` always; :func:`~artgate_gen.fetch.fetch_thrifty` reports its own
        status and only raises on a genuine download failure.

    """
    build_parser().parse_args(argv)
    fetch_thrifty()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
