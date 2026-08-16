"""Tests for the wrap-seam metric.

The controls are the point. This metric is a port, and the only thing that
makes its 4.0 threshold meaningful is that the source implementation measured
~1.6 on a seamless wrapping sinusoid and ~31 on a hard split. Both are asserted
here, so a change that quietly rescales the ratio fails rather than silently
moving every future verdict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from tests.assertions import check, check_equal
from tools.seam_probe import THRESHOLD, main, measure, wrap_discontinuity

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

WIDTH = 256
HEIGHT = 64
# The source implementation's calibration point for a seamless wrap.
SEAMLESS_MAX = 2.0


def _sinusoid() -> np.ndarray:
    """Build a horizontally wrapping sinusoid: the seamless control.

    Returns:
        An ``(h, w, 3)`` float array that tiles exactly.
    """
    x = np.arange(WIDTH)
    row = 127 + 120 * np.sin(2 * np.pi * x / WIDTH)
    return row[None, :, None].repeat(HEIGHT, 0).repeat(3, 2)


def _hard_split() -> np.ndarray:
    """Build a half-black/half-white image: the seamed control.

    Returns:
        An ``(h, w, 3)`` float array with a hard wrap discontinuity.
    """
    rng = np.random.default_rng(0)
    arr = np.zeros((HEIGHT, WIDTH, 3))
    arr[:, WIDTH // 2 :] = 255
    return np.clip(arr + rng.integers(-8, 8, (HEIGHT, WIDTH, 3)), 0, 255).astype(float)


def test_a_wrapping_sinusoid_reads_as_seamless() -> None:
    """The seamless control lands near 1.0, well under the threshold."""
    horizontal, _ = wrap_discontinuity(_sinusoid())
    check(
        condition=horizontal < SEAMLESS_MAX,
        message=f"the sinusoid is seamless ({horizontal:.2f} < {SEAMLESS_MAX})",
    )


def test_a_hard_split_reads_as_seamed() -> None:
    """The seamed control lands far above the threshold."""
    horizontal, _ = wrap_discontinuity(_hard_split())
    check(
        condition=horizontal > THRESHOLD,
        message=f"the hard split is seamed ({horizontal:.2f} > {THRESHOLD})",
    )


def test_a_uniform_image_does_not_divide_by_zero() -> None:
    """A flat image has no interior variation, so the ratio is defined as 0."""
    flat = np.full((HEIGHT, WIDTH, 3), 128.0)
    check_equal(
        actual=wrap_discontinuity(flat),
        expected=(0.0, 0.0),
        message="the uniform-image guard",
    )


def test_vertical_seams_are_measured_independently() -> None:
    """A vertical-only discontinuity shows on the vertical axis, not the horizontal.

    A parallax strip does not tile vertically, so the two axes must not be
    conflated: a vertical seam is information, not a failure.
    """
    arr = np.zeros((HEIGHT, WIDTH, 3))
    arr[HEIGHT // 2 :, :] = 255
    horizontal, vertical = wrap_discontinuity(arr)
    check_equal(actual=horizontal, expected=0.0, message="no horizontal seam")
    check(condition=vertical > THRESHOLD, message="a vertical seam is detected")


def test_measure_reads_a_png(tmp_path: Path) -> None:
    """Measuring from disk matches measuring the array."""
    path = tmp_path / "strip.png"
    arr = _sinusoid().astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)
    check_equal(
        actual=measure(path),
        expected=wrap_discontinuity(arr.astype(float)),
        message="the on-disk measurement",
    )


def test_main_passes_a_seamless_strip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seamless strip exits 0 and the report is written where asked."""
    path = tmp_path / "seamless.png"
    Image.fromarray(_sinusoid().astype(np.uint8), "RGB").save(path)
    out = tmp_path / "ratios.txt"
    monkeypatch.setattr("sys.argv", ["seam_probe", str(path), "--out", str(out)])
    check_equal(actual=main(), expected=0, message="the exit status")
    check(condition="PASS" in out.read_text(encoding="utf-8"), message="PASS reported")


def test_main_fails_a_seamed_strip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seamed strip exits non-zero, so the probe can gate a script.

    Fail-closed: the whole point of measuring is that a bad wrap is invisible
    on the strip itself, so the exit status has to carry the verdict.
    """
    path = tmp_path / "seamed.png"
    Image.fromarray(_hard_split().astype(np.uint8), "RGB").save(path)
    monkeypatch.setattr("sys.argv", ["seam_probe", str(path)])
    check_equal(actual=main(), expected=1, message="the exit status")
