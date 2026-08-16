"""Gate tests, including calibration controls and two regression bugs.

Every threshold here is justified by a control pair -- one provably-passing
input and one provably-failing input -- because a guessed threshold is the
single most expensive mistake available: it rejects good art and sends you
looking for a fault in the artwork that is actually in the gate.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageFilter

from artgate_gen.gate.codes import Code
from artgate_gen.gate.pixels import (
    block_impurity,
    check_alpha_binary,
    check_color_count,
    check_palette,
    check_scale_invariance,
    detect_scale,
    opaque_rgb,
)
from tests.assertions import check_equal


def _rgba(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to an RGBA uint8 array.

    Args:
        img: Any PIL image.

    Returns:
        An ``(h, w, 4)`` uint8 array.
    """
    return np.asarray(img.convert("RGBA"), dtype=np.uint8)


@pytest.fixture(name="base_art")
def fixture_base_art() -> Image.Image:
    """Return a deterministic 16x16 opaque pixel-art stand-in.

    Returns:
        A 16x16 RGBA image with fully-opaque alpha.
    """
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 255, (16, 16, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    return Image.fromarray(arr, "RGBA")


class TestScaleInvarianceControls:
    """Control pair for the scale gate: it must separate honest from fake."""

    def test_nearest_upscale_is_pure(self, base_art: Image.Image) -> None:
        """Nearest-neighbour upscaling is provably blocky."""
        up = base_art.resize((64, 64), Image.Resampling.NEAREST)
        check_equal(
            actual=block_impurity(_rgba(up), 4),
            expected=0.0,
            message="block_impurity(_rgba(up), 4) == 0.0",
        )

    def test_bilinear_upscale_is_impure(self, base_art: Image.Image) -> None:
        """Interpolated upscaling is provably not blocky."""
        up = base_art.resize((64, 64), Image.Resampling.BILINEAR)
        check_equal(
            actual=block_impurity(_rgba(up), 4),
            expected=1.0,
            message="block_impurity(_rgba(up), 4) == 1.0",
        )

    def test_gate_accepts_honest_and_rejects_fake(self, base_art: Image.Image) -> None:
        """The gate turns those two controls into the right verdicts."""
        honest = _rgba(base_art.resize((64, 64), Image.Resampling.NEAREST))
        fake = _rgba(base_art.resize((64, 64), Image.Resampling.BILINEAR))
        check_equal(
            actual=check_scale_invariance(honest, 4),
            expected=[],
            message="check_scale_invariance(honest, 4) == []",
        )
        check_equal(
            actual=check_scale_invariance(fake, 4),
            expected=[Code.SCALE_NOT_INVARIANT],
            message="check_scale_invariance(fake, 4) == [Code.SCALE_NOT_INVARI...",
        )

    def test_detects_scale_without_declaration(self, base_art: Image.Image) -> None:
        """Auto-detection recovers the upscale factor."""
        up = _rgba(base_art.resize((64, 64), Image.Resampling.NEAREST))
        check_equal(
            actual=detect_scale(up), expected=4, message="detect_scale(up) == 4"
        )

    def test_native_resolution_passes_trivially(self, base_art: Image.Image) -> None:
        """Documented limitation: at scale 1 this gate cannot discriminate.

        This is why alpha and colour gates carry the load on native output.
        """
        native = _rgba(base_art)
        check_equal(
            actual=detect_scale(native), expected=1, message="detect_scale(native) == 1"
        )
        check_equal(
            actual=check_scale_invariance(native, None),
            expected=[],
            message="check_scale_invariance(native, None) == []",
        )


class TestAlphaBeforeColor:
    """Regression: gate ordering, and the empty-opaque-set hole it exposed."""

    def test_soft_alpha_is_rejected(self) -> None:
        """A blurred, soft-alpha image fails the alpha gate."""
        rng = np.random.default_rng(7)
        arr = rng.integers(0, 255, (32, 32, 4), dtype=np.uint8)
        soft = Image.fromarray(arr, "RGBA").filter(ImageFilter.GaussianBlur(4))
        check_equal(
            actual=check_alpha_binary(_rgba(soft)),
            expected=[Code.ALPHA_NOT_BINARY],
            message="check_alpha_binary(_rgba(soft)) == [Code.ALPHA_NOT_BINARY]",
        )

    def test_color_gate_fails_closed_with_no_opaque_pixels(self) -> None:
        """Regression: an opaque-only count returns 0 and must not pass.

        Measured during design: a 512px smooth image downscaled to 32x32 had
        99 distinct alpha values and therefore zero fully-opaque pixels, so a
        naive opaque-only colour count reported 0 and missed it entirely.
        """
        arr = np.full((8, 8, 4), 128, dtype=np.uint8)
        arr[:, :, 3] = 128
        check_equal(
            actual=opaque_rgb(arr).size, expected=0, message="opaque_rgb(arr).size == 0"
        )
        check_equal(
            actual=check_color_count(arr, 16),
            expected=[Code.NO_OPAQUE_PIXELS],
            message="check_color_count(arr, 16) == [Code.NO_OPAQUE_PIXELS]",
        )

    def test_palette_gate_fails_closed_with_no_opaque_pixels(self) -> None:
        """The palette gate shares the fail-closed behaviour."""
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        check_equal(
            actual=check_palette(arr, frozenset({(0, 0, 0)})),
            expected=[Code.NO_OPAQUE_PIXELS],
            message="check_palette(arr, frozenset({(0, 0, 0)})) == [Code.NO_OP...",
        )


class TestTransparentRgbInflation:
    """Regression: transparent pixels retaining RGB inflated the colour count."""

    def test_transparent_rgb_does_not_inflate_count(self) -> None:
        """Two colours plus junk-RGB transparent pixels still counts as two."""
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[0, :, :3] = (10, 20, 30)
        arr[0, :, 3] = 255
        arr[1, :, :3] = (40, 50, 60)
        arr[1, :, 3] = 255
        arr[2:, :, :3] = (99, 88, 77)
        arr[2:, :, 3] = 0
        check_equal(
            actual=check_color_count(arr, 2),
            expected=[],
            message="check_color_count(arr, 2) == []",
        )

    def test_exceeding_the_cap_is_reported(self) -> None:
        """Three opaque colours against a cap of two fails."""
        arr = np.zeros((3, 1, 4), dtype=np.uint8)
        arr[0, 0] = (1, 1, 1, 255)
        arr[1, 0] = (2, 2, 2, 255)
        arr[2, 0] = (3, 3, 3, 255)
        check_equal(
            actual=check_color_count(arr, 2),
            expected=[Code.TOO_MANY_COLORS],
            message="check_color_count(arr, 2) == [Code.TOO_MANY_COLORS]",
        )


class TestPalette:
    """Palette conformance over opaque pixels."""

    def test_conforming_art_passes(self) -> None:
        """Art drawn only from the palette passes."""
        arr = np.zeros((2, 2, 4), dtype=np.uint8)
        arr[:, :, :3] = (10, 20, 30)
        arr[:, :, 3] = 255
        check_equal(
            actual=check_palette(arr, frozenset({(10, 20, 30)})),
            expected=[],
            message="check_palette(arr, frozenset({(10, 20, 30)})) == []",
        )

    def test_off_palette_pixel_is_reported(self) -> None:
        """A single stray colour fails the gate."""
        arr = np.zeros((1, 2, 4), dtype=np.uint8)
        arr[0, 0] = (10, 20, 30, 255)
        arr[0, 1] = (99, 99, 99, 255)
        check_equal(
            actual=check_palette(arr, frozenset({(10, 20, 30)})),
            expected=[Code.OFF_PALETTE],
            message="check_palette(arr, frozenset({(10, 20, 30)})) == [Code.OF...",
        )


class TestAlphaBinary:
    """Alpha binarity accepts only fully-transparent and fully-opaque."""

    def test_binary_alpha_passes(self) -> None:
        """A mask of only 0 and 255 passes."""
        arr = np.zeros((2, 2, 4), dtype=np.uint8)
        arr[0, :, 3] = 255
        check_equal(
            actual=check_alpha_binary(arr),
            expected=[],
            message="check_alpha_binary(arr) == []",
        )
