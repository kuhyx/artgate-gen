"""Silhouette-contrast tests, with a calibration control pair.

The threshold is justified by controls rather than taste: a near-white subject
scores 5.0 against white and must fail; a mid-grey subject scores ~128 against
both and must pass. A gate whose threshold has no controls is a bug waiting to
reject good art.
"""

from __future__ import annotations

import numpy as np
import pytest

from artgate_gen.gate.codes import Code
from artgate_gen.gate.contrast import (
    DEFAULT_MIN_DELTA,
    check_margin,
    check_silhouette,
    silhouette_delta,
)
from tests.assertions import check, check_equal


def _solid(value: int, alpha: int = 255, size: int = 8) -> np.ndarray:
    """Build a uniform RGBA square.

    Args:
        value: Grey level for all channels.
        alpha: Alpha for every pixel.
        size: Square edge length.

    Returns:
        An ``(size, size, 4)`` uint8 array.
    """
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, :, :3] = value
    arr[:, :, 3] = alpha
    return arr


class TestCalibrationControls:
    """Provably-failing and provably-passing inputs bracket the threshold."""

    def test_near_white_fails_against_light(self) -> None:
        """A near-white subject is invisible on white and must fail."""
        arr = _solid(250)
        check(
            condition=silhouette_delta(arr, 255) < DEFAULT_MIN_DELTA,
            message="silhouette_delta(arr, 255) < DEFAULT_MIN_DELTA",
        )
        check_equal(
            actual=check_silhouette(arr),
            expected=[Code.SILHOUETTE_LOW_CONTRAST],
            message="check_silhouette(arr) == [Code.SILHOUETTE_LOW_CONTRAST]",
        )

    def test_near_black_fails_against_dark(self) -> None:
        """A near-black subject is invisible on black and must fail."""
        arr = _solid(4)
        check(
            condition=silhouette_delta(arr, 0) < DEFAULT_MIN_DELTA,
            message="silhouette_delta(arr, 0) < DEFAULT_MIN_DELTA",
        )
        check_equal(
            actual=check_silhouette(arr),
            expected=[Code.SILHOUETTE_LOW_CONTRAST],
            message="check_silhouette(arr) == [Code.SILHOUETTE_LOW_CONTRAST]",
        )

    def test_mid_grey_passes_both(self) -> None:
        """A mid-grey subject reads on both backgrounds and must pass."""
        arr = _solid(128)
        check_equal(
            actual=check_silhouette(arr),
            expected=[],
            message="check_silhouette(arr) == []",
        )

    def test_both_backgrounds_are_required(self) -> None:
        """Passing one background is not sufficient.

        Near-white separates hugely from black (250) yet still fails, which is
        exactly the case a single-background check would wave through.
        """
        arr = _solid(250)
        check_equal(
            actual=silhouette_delta(arr, 0),
            expected=250.0,
            message="silhouette_delta(arr, 0) == 250.0",
        )
        check_equal(
            actual=check_silhouette(arr),
            expected=[Code.SILHOUETTE_LOW_CONTRAST],
            message="check_silhouette(arr) == [Code.SILHOUETTE_LOW_CONTRAST]",
        )


class TestMargin:
    """A sprite needs a clear border; a square tile is not an item icon.

    Regression: twelve SDXL icons measured 100% opaque coverage and passed
    every other gate, because nothing checked whether the asset had a
    silhouette at all. Gate-pass measured hygiene, not usability.
    """

    def test_full_bleed_is_rejected(self) -> None:
        """A fully-opaque canvas has no silhouette and must fail."""
        check_equal(
            actual=check_margin(_solid(128), 1),
            expected=[Code.MARGIN_TOO_SMALL],
            message="check_margin(_solid(128), 1) == [Code.MARGIN_TOO_SMALL]",
        )

    def test_clear_border_passes(self) -> None:
        """Content inset from every edge passes."""
        arr = _solid(128, alpha=0)
        arr[2:6, 2:6, 3] = 255
        check_equal(
            actual=check_margin(arr, 1),
            expected=[],
            message="check_margin(arr, 1) == []",
        )

    @pytest.mark.parametrize("edge", ["top", "bottom", "left", "right"])
    def test_any_touched_edge_is_rejected(self, edge: str) -> None:
        """Touching a single edge is enough to fail."""
        arr = _solid(128, alpha=0)
        arr[2:6, 2:6, 3] = 255
        index = {
            "top": (0, 3),
            "bottom": (-1, 3),
            "left": (3, 0),
            "right": (3, -1),
        }[edge]
        arr[index[0], index[1], 3] = 255
        check_equal(
            actual=check_margin(arr, 1),
            expected=[Code.MARGIN_TOO_SMALL],
            message="check_margin(arr, 1) == [Code.MARGIN_TOO_SMALL]",
        )

    def test_zero_margin_disables_the_gate(self) -> None:
        """A style that permits full bleed opts out explicitly."""
        check_equal(
            actual=check_margin(_solid(128), 0),
            expected=[],
            message="check_margin(_solid(128), 0) == []",
        )

    def test_wider_margin_is_stricter(self) -> None:
        """A 2px requirement rejects art that only clears 1px."""
        arr = _solid(128, alpha=0)
        arr[1:7, 1:7, 3] = 255
        check_equal(
            actual=check_margin(arr, 1),
            expected=[],
            message="check_margin(arr, 1) == []",
        )
        check_equal(
            actual=check_margin(arr, 2),
            expected=[Code.MARGIN_TOO_SMALL],
            message="check_margin(arr, 2) == [Code.MARGIN_TOO_SMALL]",
        )


class TestEdgeCases:
    """Degenerate inputs fail closed rather than dividing by zero."""

    def test_fully_transparent_is_empty_canvas(self) -> None:
        """An image that paints nothing reports EMPTY_CANVAS."""
        check_equal(
            actual=check_silhouette(_solid(128, alpha=0)),
            expected=[Code.EMPTY_CANVAS],
            message="check_silhouette(_solid(128, alpha=0)) == [Code.EMPTY_CAN...",
        )

    def test_delta_is_zero_when_nothing_is_painted(self) -> None:
        """The alpha-weighted mean is defined for a blank image."""
        check_equal(
            actual=silhouette_delta(_solid(128, alpha=0), 0),
            expected=0.0,
            message="silhouette_delta(_solid(128, alpha=0), 0) == 0.0",
        )

    def test_alpha_weighting_ignores_transparent_margin(self) -> None:
        """A transparent margin does not drag the subject's luma."""
        arr = _solid(250, size=8)
        arr[4:, :, 3] = 0
        check_equal(
            actual=silhouette_delta(arr, 255),
            expected=5.0,
            message="silhouette_delta(arr, 255) == 5.0",
        )

    def test_threshold_is_configurable(self) -> None:
        """A style may demand more separation than the default."""
        arr = _solid(128)
        check_equal(
            actual=check_silhouette(arr, min_delta=200.0),
            expected=[Code.SILHOUETTE_LOW_CONTRAST],
            message="check_silhouette(arr, min_delta=200.0) == [Code.SILHOUETT...",
        )
