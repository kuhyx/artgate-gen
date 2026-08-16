"""Terminal output for the command-line entry points.

Ruff's ``T201`` forbids bare ``print`` so that stray debug output cannot reach
a library path unnoticed. These CLIs genuinely do write to the terminal, so
they route through a named function instead of suppressing the rule: the intent
is explicit at every call site, and a stray ``print`` elsewhere still fails the
lint.
"""

from __future__ import annotations

import sys


def emit(message: str = "") -> None:
    """Write one line to stdout, flushed.

    Flushing matters because generation interleaves this with a progress
    display and with subprocess output; an unflushed line arrives out of order.

    Args:
        message: The line to write.

    """
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def emit_error(message: str) -> None:
    """Write one line to stderr, flushed.

    Args:
        message: The line to write.

    """
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()
