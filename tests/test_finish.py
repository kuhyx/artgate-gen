"""Tests for the finishing stage.

Order is the thing under test: the background must go before quantisation, or
the arbitrary RGB carried by transparent pixels lands in the palette.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from artgate_gen.finish import MAX_COLOURS, finish
from artgate_gen.gate.pixels import opaque_rgb
from tests.assertions import check, check_equal

SOURCE_EDGE = 64
TARGET_EDGE = 16


def _render() -> Image.Image:
    """Build a synthetic render: a bright square on a black field.

    Returns:
        An RGB image shaped like a generator's output.

    """
    arr = np.zeros((SOURCE_EDGE, SOURCE_EDGE, 3), dtype=np.uint8)
    arr[16:48, 16:48] = (220, 60, 40)
    return Image.fromarray(arr, "RGB")


def test_output_is_square_at_the_requested_size() -> None:
    """The icon comes back at the requested edge length."""
    out = finish(_render(), TARGET_EDGE)
    check_equal(
        actual=out.size,
        expected=(TARGET_EDGE, TARGET_EDGE),
        message="finished icon size",
    )


def test_output_is_rgba_with_a_cleared_background() -> None:
    """The black field becomes transparent."""
    out = finish(_render(), TARGET_EDGE)
    arr = np.asarray(out, dtype=np.uint8)
    check_equal(actual=out.mode, expected="RGBA", message="mode")
    check_equal(actual=int(arr[0, 0, 3]), expected=0, message="corner is cleared")


def test_palette_is_capped_at_the_default() -> None:
    """The opaque palette respects the calibrated cap."""
    arr = np.random.default_rng(0).integers(
        0,
        255,
        (SOURCE_EDGE, SOURCE_EDGE, 3),
        dtype=np.uint8,
    )
    arr[0, :] = 0
    arr[-1, :] = 0
    arr[:, 0] = 0
    arr[:, -1] = 0
    out = finish(Image.fromarray(arr, "RGB"), TARGET_EDGE)
    distinct = np.unique(opaque_rgb(np.asarray(out, dtype=np.uint8)), axis=0)
    check(
        condition=distinct.shape[0] <= MAX_COLOURS,
        message="palette is capped at the default",
    )


def test_explicit_colour_cap_is_honoured() -> None:
    """An explicit cap overrides the default."""
    cap = 4
    arr = np.random.default_rng(1).integers(
        0,
        255,
        (SOURCE_EDGE, SOURCE_EDGE, 3),
        dtype=np.uint8,
    )
    arr[0, :] = 0
    arr[-1, :] = 0
    arr[:, 0] = 0
    arr[:, -1] = 0
    out = finish(Image.fromarray(arr, "RGB"), TARGET_EDGE, cap)
    distinct = np.unique(opaque_rgb(np.asarray(out, dtype=np.uint8)), axis=0)
    check(condition=distinct.shape[0] <= cap, message="explicit cap is honoured")
