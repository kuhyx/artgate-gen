"""SEAM_ENERGY tests, split from ``test_backdrop.py``.

Split to respect the 250-line file cap; shares that module's
``_solid``/``WIDTH``/``HEIGHT`` fixtures.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from artgate_gen.gate.backdrop import (
    SEAM_ENERGY_MAX,
    check_seam_energy,
    wrap_discontinuity,
)
from artgate_gen.gate.codes import Code
from tests.assertions import check, check_equal
from tests.gate.test_backdrop import HEIGHT, WIDTH, _solid

# Tolerance for the seamless-control regression test against tiletool.py's
# documented ~1.6 reference value.
SEAMLESS_CONTROL_TOLERANCE = 0.5

# A hard split must score far above SEAM_ENERGY_MAX; this is not the
# threshold itself, just a floor loose enough to be robust to minor
# arithmetic differences between this port and the reference implementation.
SEAMED_CONTROL_FLOOR = 100.0

# Luma jump added at a wrap edge to build a hard-step control. Not derived
# from SEAM_ENERGY_MAX -- just large enough that the resulting ratio clears
# threshold with obvious headroom.
BOUNDARY_STEP_LUMA = 158


def _sinusoid(size: int = 64) -> np.ndarray:
    """Build tiletool.py's seamless-control sinusoid, opaque throughout.

    A sinusoid with an integer number of periods wraps exactly on both axes,
    which is what makes it a provably-seamless control.

    Args:
        size: Edge length of the square test image.

    Returns:
        A ``(size, size, 4)`` uint8 array.
    """
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            v = int(
                128
                + 100
                * math.sin(2 * math.pi * x / size)
                * math.cos(2 * math.pi * y / size)
            )
            arr[y, x, :3] = v
    arr[:, :, 3] = 255
    return arr


class TestSeamEnergy:
    """Ported from tiletool.py: a wrap-edge jump measured against interior."""

    def test_reproduces_reference_seamless_control(self) -> None:
        """A wrapping sinusoid measures near the documented ~1.6."""
        horizontal, vertical = wrap_discontinuity(_sinusoid())
        check(
            condition=abs(horizontal - 1.6) < SEAMLESS_CONTROL_TOLERANCE,
            message="seamless control horizontal ratio is near 1.6",
        )
        check(
            condition=vertical < SEAM_ENERGY_MAX,
            message="seamless control vertical ratio passes threshold",
        )

    def test_reproduces_reference_seamed_control(self) -> None:
        """A hard half/half split measures far above threshold."""
        arr = _solid(20)
        arr[:, WIDTH // 2 :, :3] = 230
        horizontal, _vertical = wrap_discontinuity(arr)
        check(
            condition=horizontal > SEAMED_CONTROL_FLOOR,
            message="a hard horizontal split must score far above threshold",
        )

    def test_hard_horizontal_split_fails(self) -> None:
        """A strip with a hard left/right split fails SEAM_ENERGY."""
        arr = _solid(20)
        arr[:, WIDTH // 2 :, :3] = 230
        check_equal(
            actual=check_seam_energy(arr),
            expected=[Code.SEAM_ENERGY],
            message="check_seam_energy(arr) == [Code.SEAM_ENERGY]",
        )

    def test_hard_vertical_split_fails(self) -> None:
        """A strip with a hard top/bottom split fails SEAM_ENERGY."""
        arr = _solid(20)
        arr[HEIGHT // 2 :, :, :3] = 230
        check_equal(
            actual=check_seam_energy(arr),
            expected=[Code.SEAM_ENERGY],
            message="check_seam_energy(arr) == [Code.SEAM_ENERGY]",
        )

    def test_smooth_gradient_passes_both_axes(self) -> None:
        """A strip that is smooth on both axes clears SEAM_ENERGY."""
        check_equal(
            actual=check_seam_energy(_sinusoid()),
            expected=[],
            message="check_seam_energy(_sinusoid()) == []",
        )

    def test_uniform_image_does_not_divide_by_zero(self) -> None:
        """A perfectly flat interior yields a defined 0.0 ratio, not NaN.

        Two individually-correct gates composing into a hole is the pattern
        `pixels.py` already warns about; the equivalent hole here would be a
        uniform image passing SEAM_ENERGY only because 0/0 raised instead of
        returning a value the gate loop could reason about.
        """
        horizontal, vertical = wrap_discontinuity(_solid(128))
        check_equal(
            actual=horizontal,
            expected=0.0,
            message="wrap_discontinuity(_solid(128))[0] == 0.0",
        )
        check_equal(
            actual=vertical,
            expected=0.0,
            message="wrap_discontinuity(_solid(128))[1] == 0.0",
        )
        check_equal(
            actual=check_seam_energy(_solid(128)),
            expected=[],
            message="check_seam_energy(_solid(128)) == []",
        )

    @pytest.mark.parametrize("axis", ["horizontal", "vertical"])
    def test_hard_step_at_either_wrap_edge_fails(self, axis: str) -> None:
        """A hard step confined to one wrap edge fails SEAM_ENERGY."""
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[:, :, :3] = 100
        if axis == "horizontal":
            arr[:, 0, :3] = BOUNDARY_STEP_LUMA
        else:
            arr[0, :, :3] = BOUNDARY_STEP_LUMA
        arr[:, :, 3] = 255
        check_equal(
            actual=check_seam_energy(arr),
            expected=[Code.SEAM_ENERGY],
            message=f"a hard step on the {axis} wrap edge must fail",
        )
