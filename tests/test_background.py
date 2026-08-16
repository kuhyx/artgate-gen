"""Tests for border-seeded background removal.

The interesting case is the one the border-ring approach exists for: an
interior region that happens to match the background colour must survive,
because nothing connects it to the edge. A corner flood would eat it.
"""

from __future__ import annotations

import numpy as np

from artgate_gen.background import (
    DEFAULT_TOLERANCE,
    OPAQUE,
    TRANSPARENT,
    strip_background,
)
from tests.assertions import check, check_equal

EDGE = 8
BACKGROUND = (0, 0, 0)
SUBJECT = (200, 180, 40)


def _canvas(colour: tuple[int, int, int] = BACKGROUND) -> np.ndarray:
    """Build an opaque square filled with one colour.

    Args:
        colour: The RGB fill.

    Returns:
        An ``(EDGE, EDGE, 4)`` uint8 array.

    """
    arr = np.zeros((EDGE, EDGE, 4), dtype=np.uint8)
    arr[:, :, :3] = colour
    arr[:, :, 3] = OPAQUE
    return arr


def test_uniform_background_is_fully_cleared() -> None:
    """An image that is entirely background ends up fully transparent."""
    out = strip_background(_canvas())
    check_equal(
        actual=int(out[:, :, 3].max()),
        expected=TRANSPARENT,
        message="every pixel should be cleared",
    )


def test_centred_subject_is_kept() -> None:
    """A subject that differs from the border survives."""
    arr = _canvas()
    arr[3:5, 3:5, :3] = SUBJECT
    out = strip_background(arr)
    check_equal(
        actual=int(out[3, 3, 3]),
        expected=OPAQUE,
        message="subject pixel should stay opaque",
    )
    check_equal(
        actual=int(out[0, 0, 3]),
        expected=TRANSPARENT,
        message="border pixel should be cleared",
    )


def test_enclosed_background_coloured_region_survives() -> None:
    """A background-coloured hole inside the subject is not eaten.

    This is the whole reason the flood seeds the border rather than the
    corners: the enclosed region matches the background exactly, and only
    connectivity distinguishes it.
    """
    arr = _canvas()
    arr[2:6, 2:6, :3] = SUBJECT
    arr[3:5, 3:5, :3] = BACKGROUND
    out = strip_background(arr)
    check_equal(
        actual=int(out[3, 3, 3]),
        expected=OPAQUE,
        message="enclosed background-coloured pixels must survive",
    )


def test_subject_touching_the_border_is_not_cleared() -> None:
    """A subject reaching the edge keeps the pixels that differ."""
    arr = _canvas()
    arr[:, 0, :3] = SUBJECT
    out = strip_background(arr)
    check_equal(
        actual=int(out[0, 0, 3]),
        expected=OPAQUE,
        message="differing border pixel is not background",
    )


def test_tolerance_controls_what_counts_as_background() -> None:
    """A near-background colour is cleared only under a wide tolerance."""
    arr = _canvas()
    arr[3:5, 3:5, :3] = (10, 10, 10)
    strict = strip_background(arr, tol=5)
    loose = strip_background(arr, tol=DEFAULT_TOLERANCE)
    check_equal(
        actual=int(strict[3, 3, 3]),
        expected=OPAQUE,
        message="a tight tolerance keeps the near-background subject",
    )
    check(
        condition=int(loose[3, 3, 3]) == TRANSPARENT,
        message="a wide tolerance treats it as background",
    )


def test_non_square_canvas_seeds_each_border_pixel_once() -> None:
    """Corners are queued once even though both seeding loops reach them.

    The two loops walk the top/bottom rows and then the left/right columns, so
    all four corners are visited twice; the ``seen`` guard is what stops them
    being enqueued twice.
    """
    arr = np.zeros((4, 9, 4), dtype=np.uint8)
    arr[:, :, :3] = BACKGROUND
    arr[:, :, 3] = OPAQUE
    arr[1:3, 4:6, :3] = SUBJECT
    out = strip_background(arr)
    check_equal(
        actual=int(out[1, 4, 3]),
        expected=OPAQUE,
        message="subject survives on a non-square canvas",
    )
    check_equal(
        actual=int(out[0, 0, 3]),
        expected=TRANSPARENT,
        message="corner is cleared exactly once",
    )


def test_single_row_canvas_does_not_double_seed() -> None:
    """On a one-pixel-tall image the top and bottom rows are the same row.

    The first seeding loop then visits every pixel twice, so the ``seen``
    guard has to reject the second visit.
    """
    arr = np.zeros((1, 5, 4), dtype=np.uint8)
    arr[:, :, :3] = BACKGROUND
    arr[:, :, 3] = OPAQUE
    arr[0, 2, :3] = SUBJECT
    out = strip_background(arr)
    check_equal(
        actual=int(out[0, 2, 3]),
        expected=OPAQUE,
        message="the differing pixel survives",
    )
    check_equal(
        actual=int(out[0, 0, 3]),
        expected=TRANSPARENT,
        message="background pixels are still cleared",
    )
