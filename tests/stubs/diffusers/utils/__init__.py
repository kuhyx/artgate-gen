"""Stand-in for :mod:`diffusers.utils`.

A real submodule rather than an attribute on the parent, because the pipeline
loader does ``from diffusers.utils import logging``, which requires an
importable module path.
"""

from __future__ import annotations


class _Logging:
    """Stands in for ``diffusers.utils.logging``."""

    def __init__(self) -> None:
        """Start with progress bars enabled."""
        self.progress_disabled = False

    def disable_progress_bar(self) -> None:
        """Record that progress bars were silenced."""
        self.progress_disabled = True


logging = _Logging()
