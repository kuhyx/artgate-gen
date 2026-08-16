"""Shared fallback used by the library stubs.

Each stub declares the surface this project actually touches. Anything else a
library might expose falls through to here and becomes a no-op, so an
unexercised attribute produces an inert call rather than an AttributeError
that would look like a real failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def inert(*args: object, **kwargs: object) -> None:
    """Accept any arguments and do nothing.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    """
    del args, kwargs


def fallback(name: str) -> Callable[..., None]:
    """Return an inert callable for an unexercised attribute.

    Args:
        name: The attribute being looked up.

    Returns:
        A no-op callable.

    Raises:
        AttributeError: For dunder lookups, so introspection behaves normally.

    """
    if name.startswith("__"):
        raise AttributeError(name)
    return inert
