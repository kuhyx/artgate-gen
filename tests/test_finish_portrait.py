"""Tests for ``finish_portrait``.

Split out of ``test_finish.py`` to stay under the 250-line cap. A portrait is
full-bleed like a backdrop (no ``strip_background``), but never tiles (no
wrap-blend) and defaults to unquantised, matching the accepted portraits in
``art/MANIFEST.json``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from artgate_gen.finish import MAX_COLOURS, finish_portrait
from tests.assertions import check, check_equal

SOURCE_EDGE = 64
TARGET_EDGE = 16
# Alpha value of a fully opaque pixel.
OPAQUE = 255


def _render() -> Image.Image:
    """Build a synthetic render: a bright square on a black field.

    Returns:
        An RGB image shaped like a generator's output.
    """
    arr = np.zeros((SOURCE_EDGE, SOURCE_EDGE, 3), dtype=np.uint8)
    arr[16:48, 16:48] = (220, 60, 40)
    return Image.fromarray(arr, "RGB")


def test_portrait_is_fully_opaque() -> None:
    """A portrait has no transparency, same reasoning as a backdrop."""
    out = finish_portrait(_render(), (TARGET_EDGE, TARGET_EDGE))
    alpha = np.asarray(out, dtype=np.uint8)[:, :, 3] == OPAQUE
    check(condition=bool(alpha.all()), message="every pixel is opaque")


def test_portrait_is_finished_at_its_canvas() -> None:
    """The portrait comes back at the requested canvas."""
    out = finish_portrait(_render(), (TARGET_EDGE, TARGET_EDGE))
    check_equal(
        actual=out.size,
        expected=(TARGET_EDGE, TARGET_EDGE),
        message="finished portrait size",
    )


def test_portrait_default_skips_quantisation() -> None:
    """``colours=0`` (the default) leaves the palette untouched.

    The accepted portraits in art/MANIFEST.json shipped at `max_colors: 0`.
    A synthetic render with more distinct colours than any small cap would
    allow must come back with all of them if quantisation was truly skipped.
    """
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 255, (SOURCE_EDGE, SOURCE_EDGE, 3), dtype=np.uint8)
    out = finish_portrait(Image.fromarray(arr, "RGB"), (SOURCE_EDGE, SOURCE_EDGE))
    distinct = np.unique(
        np.asarray(out, dtype=np.uint8)[:, :, :3].reshape(-1, 3), axis=0
    )
    check(
        condition=distinct.shape[0] > MAX_COLOURS,
        message="an unquantised portrait keeps more colours than the icon cap",
    )


def test_portrait_honours_an_explicit_colour_cap() -> None:
    """A non-zero cap does quantise, for when that decision is made later."""
    cap = 4
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 255, (SOURCE_EDGE, SOURCE_EDGE, 3), dtype=np.uint8)
    out = finish_portrait(Image.fromarray(arr, "RGB"), (SOURCE_EDGE, SOURCE_EDGE), cap)
    distinct = np.unique(
        np.asarray(out, dtype=np.uint8)[:, :, :3].reshape(-1, 3), axis=0
    )
    check(condition=distinct.shape[0] <= cap, message="explicit cap is honoured")


def test_portrait_keeps_the_whole_image() -> None:
    """No inward flood: a full-bleed portrait survives finishing intact."""
    out = finish_portrait(_render(), (TARGET_EDGE, TARGET_EDGE))
    alpha = np.asarray(out, dtype=np.uint8)[:, :, 3] == OPAQUE
    check_equal(
        actual=float(alpha.mean()),
        expected=1.0,
        message="the opaque fraction of a finished portrait",
    )
