"""Command-line entrypoint for :mod:`fetch`, called by ``run.sh``.

A separate file rather than a ``python -c`` block in the shell script: inline
program logic in another language is invisible to this repo's linters, so it
lives here where ruff can see it. Mirrors :mod:`repair_cli`.

Only ``fetch_thrifty`` is exposed -- it is the strategy ``run.sh`` has always
used (formerly the deleted ``fetch_model.py``'s only entry point), and it is
idempotent, so a plain re-run is always safe.
"""

from __future__ import annotations

from artgate_gen.fetch import fetch_thrifty


def main() -> int:
    """Populate the local Hugging Face cache for the models this package uses.

    Returns:
        ``0`` always; :func:`~artgate_gen.fetch.fetch_thrifty` reports its own
        status and only raises on a genuine download failure.

    """
    fetch_thrifty()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
