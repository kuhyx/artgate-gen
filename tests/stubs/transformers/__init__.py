"""A minimal stand-in for :mod:`transformers`.

Only the progress-bar control is touched: the pipeline loader silences the
per-component "Loading weights" bars, which come from transformers rather than
from diffusers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.stubs._inert import fallback

if TYPE_CHECKING:
    from collections.abc import Callable


class _Logging:
    """Stands in for ``transformers.logging``."""

    def disable_progress_bar(self) -> None:
        """Silence progress bars; a no-op here."""


logging = _Logging()


def __getattr__(name: str) -> Callable[..., None]:
    """Return an inert placeholder for any unexercised attribute.

    Args:
        name: The attribute being looked up.

    Returns:
        A no-op callable.

    """
    return fallback(name)
