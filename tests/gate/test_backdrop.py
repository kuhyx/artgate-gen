"""Backdrop-gate tests, with a calibration control pair per gate.

Each threshold is justified by a provably-failing and a provably-passing
input, per the ``TestCalibrationControls`` precedent in ``test_contrast.py``:
a gate whose threshold has no controls is a bug waiting to reject good art.
SEAM_ENERGY's tests live in ``test_backdrop_seam.py`` to keep both files
under the 250-line cap; they share this module's fixtures.
"""

from __future__ import annotations

import numpy as np

from artgate_gen.gate.backdrop import (
    EDGE_BAND,
    MIN_ROW_LUMA_RANGE,
    check_fully_opaque,
    check_letterbox,
    check_vertical_structure,
    row_luma_range,
)
from artgate_gen.gate.codes import Code
from tests.assertions import check, check_equal

WIDTH = 256
HEIGHT = 64


def _solid(value: int, alpha: int = 255) -> np.ndarray:
    """Build a uniform 256x64 RGBA strip.

    Args:
        value: Grey level for all channels.
        alpha: Alpha for every pixel.

    Returns:
        A ``(HEIGHT, WIDTH, 4)`` uint8 array.
    """
    arr = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    arr[:, :, :3] = value
    arr[:, :, 3] = alpha
    return arr


def _gradient(top: int, bottom: int, alpha: int = 255) -> np.ndarray:
    """Build a smooth vertical grey gradient, opaque throughout.

    Args:
        top: Grey level at row 0.
        bottom: Grey level at the last row.
        alpha: Alpha for every pixel.

    Returns:
        A ``(HEIGHT, WIDTH, 4)`` uint8 array.
    """
    arr = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    ramp = np.linspace(top, bottom, HEIGHT).reshape(HEIGHT, 1)
    arr[:, :, 0] = ramp
    arr[:, :, 1] = ramp
    arr[:, :, 2] = ramp
    arr[:, :, 3] = alpha
    return arr


class TestFullyOpaque:
    """A backdrop is full-bleed by construction; no pixel may be transparent."""

    def test_any_transparent_pixel_fails(self) -> None:
        """A single non-opaque pixel is enough to fail."""
        arr = _solid(128)
        arr[0, 0, 3] = 254
        check_equal(
            actual=check_fully_opaque(arr),
            expected=[Code.FULLY_OPAQUE_REQUIRED],
            message="check_fully_opaque(arr) == [Code.FULLY_OPAQUE_REQUIRED]",
        )

    def test_fully_opaque_passes(self) -> None:
        """A strip with alpha 255 throughout passes."""
        check_equal(
            actual=check_fully_opaque(_solid(128)),
            expected=[],
            message="check_fully_opaque(_solid(128)) == []",
        )


class TestLetterbox:
    """Both a near-black edge band AND a bright interior are required.

    Regression target: a legitimately dark night-sky render must not
    false-positive just because it is dark at the edges -- it is dark
    everywhere, which is a different thing from a composed scene padded with
    black bars.
    """

    def test_black_bars_over_bright_interior_fails(self) -> None:
        """A near-black top/bottom band over a bright interior is a bar."""
        arr = _solid(200)
        arr[:EDGE_BAND, :, :3] = 5
        arr[-EDGE_BAND:, :, :3] = 5
        check_equal(
            actual=check_letterbox(arr),
            expected=[Code.LETTERBOX_BARS],
            message="check_letterbox(arr) == [Code.LETTERBOX_BARS]",
        )

    def test_uniformly_bright_strip_passes(self) -> None:
        """A strip with no dark edge band at all passes."""
        check_equal(
            actual=check_letterbox(_solid(150)),
            expected=[],
            message="check_letterbox(_solid(150)) == []",
        )

    def test_uniformly_dark_strip_passes(self) -> None:
        """A dark-throughout night scene is not a letterbox bar.

        The edge bands are near-black, same as a real bar, but the interior
        is equally dark -- the AND condition is what tells them apart.
        """
        arr = _solid(15)
        check(
            condition=check_letterbox(arr) == [],
            message="a uniformly dark strip (no bright interior) must pass",
        )

    def test_dark_edge_needs_bright_interior_to_fail(self) -> None:
        """A dark edge band alone, without a bright interior, is not a bar."""
        arr = _solid(40)
        arr[:EDGE_BAND, :, :3] = 5
        check(
            condition=check_letterbox(arr) == [],
            message="a dark edge over a dim (not bright) interior must pass",
        )


class TestVerticalStructure:
    """A flat wash has no sky-to-ground composition and must fail."""

    def test_uniform_colour_fails(self) -> None:
        """A perfectly flat strip has zero row-luma range."""
        check_equal(
            actual=row_luma_range(_solid(128)),
            expected=0.0,
            message="row_luma_range(_solid(128)) == 0.0",
        )
        check_equal(
            actual=check_vertical_structure(_solid(128)),
            expected=[Code.NO_VERTICAL_STRUCTURE],
            message="check_vertical_structure(_solid(128)) == "
            "[Code.NO_VERTICAL_STRUCTURE]",
        )

    def test_sky_to_ground_gradient_passes(self) -> None:
        """A real-scale sky-to-ground gradient clears the threshold."""
        arr = _gradient(top=180, bottom=80)
        check(
            condition=row_luma_range(arr) > MIN_ROW_LUMA_RANGE,
            message="row_luma_range(arr) > MIN_ROW_LUMA_RANGE",
        )
        check_equal(
            actual=check_vertical_structure(arr),
            expected=[],
            message="check_vertical_structure(arr) == []",
        )

    def test_range_is_robust_to_which_end_is_brighter(self) -> None:
        """Range does not depend on the sign of top-minus-bottom.

        A top/bottom-only comparison would read near zero here even though
        real vertical structure exists between the ends.
        """
        arr = _gradient(top=100, bottom=98)
        arr[HEIGHT // 2, :, :3] = 200
        check(
            condition=row_luma_range(arr) > MIN_ROW_LUMA_RANGE,
            message="a bright interior band must register in the range even "
            "when the endpoints are nearly equal",
        )
