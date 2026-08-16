"""Assertion helpers used in place of the bare ``assert`` statement.

This repository lints under ``select = ["ALL"]`` with no per-file ignores, so
``S101`` (bare assert) applies to tests too. Rather than carve out an
exemption, tests call these helpers: they raise ``AssertionError`` exactly as
``assert`` would, so pytest reports failures normally, and they force every
expectation to carry a written explanation.

The messages are built into a local variable before raising because ``EM102``
forbids f-strings passed straight to an exception constructor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Container, Sized


def check(*, condition: bool, message: str) -> None:
    """Raise an assertion error when a condition does not hold.

    Args:
        condition: The condition that must be true.
        message: Explanation shown when the condition fails.

    Raises:
        AssertionError: If ``condition`` is false.

    """
    if not condition:
        raise AssertionError(message)


def check_equal[T](*, actual: T, expected: T, message: str) -> None:
    """Raise an assertion error when two values differ.

    Args:
        actual: The observed value.
        expected: The value that was required.
        message: Explanation shown when they differ.

    Raises:
        AssertionError: If the values are not equal.

    """
    if actual != expected:
        detail = f"{message}: expected {expected!r}, got {actual!r}"
        raise AssertionError(detail)


def check_in[T](*, member: T, container: Container[T], message: str) -> None:
    """Raise an assertion error when a member is absent from a container.

    Args:
        member: The value that must be present.
        container: The container to search.
        message: Explanation shown when the member is missing.

    Raises:
        AssertionError: If ``member`` is not in ``container``.

    """
    if member not in container:
        detail = f"{message}: {member!r} not found in {container!r}"
        raise AssertionError(detail)


def check_length(*, sized: Sized, expected: int, message: str) -> None:
    """Raise an assertion error when a sized object has the wrong length.

    Args:
        sized: The object whose length is checked.
        expected: The required length.
        message: Explanation shown when the length differs.

    Raises:
        AssertionError: If the length does not match.

    """
    actual = len(sized)
    if actual != expected:
        detail = f"{message}: expected {expected} items, got {actual}"
        raise AssertionError(detail)
