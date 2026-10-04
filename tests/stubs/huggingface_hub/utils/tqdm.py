"""Stand-in for :mod:`huggingface_hub.utils.tqdm`.

``fetch.py`` imports ``disable_progress_bars`` from here directly (the
defining submodule) rather than from ``huggingface_hub.utils``, so this file
has to exist as its own importable module.
"""

from __future__ import annotations


def disable_progress_bars() -> None:
    """Silence progress bars; a no-op here."""
