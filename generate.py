"""Compatibility shim: the CLI now lives in :mod:`artgate_gen.cli`.

Kept at the repository root because `run.sh` and the `item-icons` skill invoke
this path directly. New callers should use the `artgate-gen` console script.
"""

from __future__ import annotations

from artgate_gen.cli import main

if __name__ == "__main__":
    main()
