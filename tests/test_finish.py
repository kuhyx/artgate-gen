"""Tests for the finishing stage.

Order is the thing under test: the background must go before quantisation, or
the arbitrary RGB carried by transparent pixels lands in the palette.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from artgate_gen.finish import (
    MAX_COLOURS,
    WRAP_OVERLAP,
    finish,
    finish_backdrop,
    wrap_blend,
)
from artgate_gen.gate.pixels import opaque_rgb
from tests.assertions import check, check_equal

SOURCE_EDGE = 64
TARGET_EDGE = 16

# A miniature of the real 2304x512 -> 2048x512 geometry. Nothing here allocates
# the real size: the maths is identical at 1/16th of it.
STRIP_CANVAS = (32, 8)
STRIP_OVERLAP = 16
STRIP_RENDER = (STRIP_CANVAS[0] * 4 + STRIP_OVERLAP, STRIP_CANVAS[1] * 4)
# The source metric's calibration: ~1.6 for a seamless wrap, ~31 for a hard
# split. Anything under 4.0 is statistically invisible.
SEAM_THRESHOLD = 4.0
# Alpha value of a fully opaque pixel.
OPAQUE = 255


def _seam_ratio(rgb: np.ndarray) -> float:
    """Return the horizontal wrap-seam ratio of an image.

    Args:
        rgb: An ``(h, w, 3)`` float array.

    Returns:
        Seam energy relative to the mean interior column difference.
    """
    seam = float(np.abs(rgb[:, -1, :] - rgb[:, 0, :]).mean())
    inner = float(np.abs(np.diff(rgb, axis=1)).mean())
    return seam / inner if inner else 0.0


def _strip(seed: int = 0) -> Image.Image:
    """Build a synthetic non-tiling render: a hard ramp plus noise.

    Returns:
        An oversized RGB strip that does NOT wrap continuously.
    """
    rng = np.random.default_rng(seed)
    width, height = STRIP_RENDER
    arr = np.zeros((height, width, 3), dtype=np.int64)
    for x in range(width):
        arr[:, x, :] = int(255 * x / (width - 1))
    noise = rng.integers(-20, 20, (height, width, 3))
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8), "RGB")


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


def test_wrap_blend_crops_by_the_overlap() -> None:
    """The blended result is exactly ``overlap`` narrower than its source."""
    arr = np.asarray(_strip().convert("RGBA"), dtype=np.uint8)
    out = wrap_blend(arr, STRIP_OVERLAP)
    check_equal(
        actual=(out.shape[1], out.shape[0]),
        expected=(STRIP_RENDER[0] - STRIP_OVERLAP, STRIP_RENDER[1]),
        message="the blended strip size",
    )


def test_wrap_blend_makes_a_seamed_render_tile() -> None:
    """The whole point: a non-tiling render becomes statistically seamless.

    Both arms are asserted. Checking only the blended figure would pass even if
    ``wrap_blend`` did nothing, because it would not establish that the input
    was seamed in the first place.
    """
    arr = np.asarray(_strip().convert("RGBA"), dtype=np.uint8)
    before = _seam_ratio(arr[:, : STRIP_RENDER[0] - STRIP_OVERLAP, :3].astype(float))
    after = _seam_ratio(wrap_blend(arr, STRIP_OVERLAP)[:, :, :3].astype(float))
    check(
        condition=before > SEAM_THRESHOLD,
        message=f"the raw crop is seamed ({before:.2f} > {SEAM_THRESHOLD})",
    )
    check(
        condition=after < SEAM_THRESHOLD,
        message=f"the blended strip is seamless ({after:.2f} < {SEAM_THRESHOLD})",
    )


def test_wrap_blend_defaults_to_the_module_overlap() -> None:
    """Omitting the overlap uses ``WRAP_OVERLAP``."""
    width = WRAP_OVERLAP * 2
    arr = np.zeros((4, width, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    check_equal(
        actual=wrap_blend(arr).shape[1],
        expected=width - WRAP_OVERLAP,
        message="the default overlap",
    )


def test_backdrop_is_fully_opaque() -> None:
    """A strip has no transparency: every finished pixel is opaque.

    This is what makes skipping ``strip_background`` verifiable rather than
    merely trusted.
    """
    out = finish_backdrop(_strip(), STRIP_CANVAS)
    alpha = np.asarray(out, dtype=np.uint8)[:, :, 3] == OPAQUE
    check(condition=bool(alpha.all()), message="every pixel is opaque")


def test_backdrop_is_finished_at_its_canvas() -> None:
    """The strip comes back non-square, at the requested canvas."""
    out = finish_backdrop(_strip(), STRIP_CANVAS)
    check_equal(actual=out.size, expected=STRIP_CANVAS, message="finished strip size")


def test_backdrop_honours_an_explicit_colour_cap() -> None:
    """The palette cap applies to strips too."""
    cap = 4
    out = finish_backdrop(_strip(1), STRIP_CANVAS, cap)
    distinct = np.unique(opaque_rgb(np.asarray(out, dtype=np.uint8)), axis=0)
    check(condition=distinct.shape[0] <= cap, message="explicit cap is honoured")


def test_backdrop_rejects_a_render_with_no_overlap() -> None:
    """A render with nothing to blend fails loudly instead of tiling badly.

    Silently accepting it would produce a strip whose wrap was never blended --
    exactly the seamed output the whole render path exists to avoid, but
    indistinguishable from a good one without measuring it.
    """
    exact = Image.new("RGB", (STRIP_CANVAS[0] * 4, STRIP_CANVAS[1] * 4), (10, 20, 30))
    try:
        finish_backdrop(exact, STRIP_CANVAS)
    except ValueError as error:
        check(
            condition="no wrap overlap" in str(error),
            message="the error names the missing overlap",
        )
    else:
        check(condition=False, message="a render with no overlap must raise")


def test_backdrop_keeps_the_whole_image() -> None:
    """No inward flood: a full-bleed strip survives finishing intact.

    ``strip_background`` measured 1.000 -> 0.531 opaque on a synthetic sky.
    Routing a strip through the icon finisher instead of this one would
    reintroduce exactly that.
    """
    out = finish_backdrop(_strip(), STRIP_CANVAS)
    alpha = np.asarray(out, dtype=np.uint8)[:, :, 3] == OPAQUE
    check_equal(
        actual=float(alpha.mean()),
        expected=1.0,
        message="the opaque fraction of a finished strip",
    )
