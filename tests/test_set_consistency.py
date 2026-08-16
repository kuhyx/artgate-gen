"""Tests for the set-consistency metric.

The controls are the point, as they are for the seam probe. Two synthetic sets
are built: one where a garment region holds a single colour across every frame,
and one where it drifts. The metric must separate them, and it must keep
separating them if anyone rescales it.

The empty-selection case has its own test because it is the failure that
matters most in practice: a region spec that matches nothing returns a
confident ``(0, 0, 0)``, and a caller who does not look at the debug sheet
would read that as a perfectly consistent black garment.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from artgate_gen.set_consistency import (
    RegionSpec,
    channel_spread,
    debug_sheet,
    load,
    mean_colour,
    measure_set,
    saturation,
    select,
    subject_width,
)
from artgate_gen.set_consistency_cli import main
from tests.assertions import check, check_equal

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

SIZE = 64
# The synthetic drift built below moves the garment's blue by exactly 60 units,
# so the metric must report close to that rather than merely "more than zero".
DRIFT = 60.0
DRIFT_TOLERANCE = 5.0
# Saturation is a 0..1 ratio; grey sits at the floor and a pure channel at the
# ceiling, so these bracket the helper's full range.
NEAR_ZERO = 0.01
NEAR_ONE = 0.99
# A blue-over-red window with no brightness ceiling, matching the garment in
# the synthetic frames below.
GARMENT = RegionSpec(
    name="garment",
    channel_lead=(2, 0),
    lead_margin=12.0,
    min_saturation=0.18,
    max_saturation=1.0,
    min_value=0.0,
    max_value=255.0,
    top=0.5,
    bottom=1.0,
)


def _frame(blue: int) -> np.ndarray:
    """Build a frame: grey background, a blue garment across the lower half.

    Args:
        blue: The garment's blue channel.

    Returns:
        An ``(h, w, 3)`` float array.
    """
    arr = np.full((SIZE, SIZE, 3), 150.0)
    arr[SIZE // 2 :, :] = (40.0, 60.0, float(blue))
    return arr


def _write(tmp_path: Path, name: str, arr: np.ndarray) -> Path:
    """Write a float array as a PNG.

    Args:
        tmp_path: Directory to write into.
        name: File stem.
        arr: The image data.

    Returns:
        The written path.
    """
    path = tmp_path / f"{name}.png"
    Image.fromarray(arr.astype("uint8"), "RGB").save(path)
    return path


def test_a_consistent_set_measures_near_zero_spread(tmp_path: Path) -> None:
    """Identical garment colour across frames reads as no drift."""
    paths = [_write(tmp_path, f"f{i}", _frame(200)) for i in range(4)]
    colours, _ = measure_set(paths, GARMENT)
    check(
        condition=max(channel_spread(colours)) < 1.0,
        message=f"identical frames should not drift, got {channel_spread(colours)}",
    )


def test_a_drifting_set_is_separated_from_a_consistent_one(tmp_path: Path) -> None:
    """A garment whose blue moves 60 units reports roughly that spread."""
    paths = [
        _write(tmp_path, f"f{i}", _frame(blue)) for i, blue in enumerate((180, 240))
    ]
    colours, _ = measure_set(paths, GARMENT)
    spread = channel_spread(colours)
    check(
        condition=abs(spread[2] - DRIFT) < DRIFT_TOLERANCE,
        message=f"a {DRIFT:.0f}-unit drift should measure near it, got {spread[2]:.1f}",
    )


def test_an_empty_selection_reports_zeros_rather_than_raising() -> None:
    """A region matching nothing returns zeros instead of dividing by empty."""
    arr = np.full((SIZE, SIZE, 3), 150.0)
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    check_equal(
        actual=mean_colour(arr, mask),
        expected=(0.0, 0.0, 0.0),
        message="an empty mask has no mean colour to report",
    )


def test_selection_honours_every_bound() -> None:
    """Each bound in a region spec excludes what it claims to."""
    arr = _frame(200)
    wide = select(arr, GARMENT)
    check(condition=bool(wide.any()), message="the garment should be selected")
    check(
        condition=not wide[: SIZE // 2].any(),
        message="the band bound should exclude the upper half",
    )

    dark_only = select(arr, replace(GARMENT, max_value=50.0))
    check(
        condition=not dark_only.any(),
        message="a brightness ceiling below the garment excludes it",
    )
    flat = select(arr, replace(GARMENT, min_saturation=NEAR_ONE))
    check(
        condition=not flat.any(),
        message="a saturation floor above the garment excludes it",
    )
    capped = select(arr, replace(GARMENT, max_saturation=NEAR_ZERO))
    check(
        condition=not capped.any(),
        message="a saturation ceiling below the garment excludes it",
    )
    warm = select(arr, replace(GARMENT, channel_lead=(0, 2)))
    check(
        condition=not warm.any(),
        message="a warm-lead window excludes a cool garment",
    )


def test_saturation_is_zero_for_grey_and_one_for_a_pure_channel() -> None:
    """The saturation helper spans its full range on the obvious controls."""
    grey = np.full((2, 2, 3), 128.0)
    check(
        condition=float(saturation(grey).max()) < NEAR_ZERO,
        message="grey has no saturation",
    )
    pure = np.zeros((2, 2, 3))
    pure[:, :, 2] = 255.0
    check(
        condition=float(saturation(pure).min()) > NEAR_ONE,
        message="a pure channel is fully saturated",
    )


def test_saturation_of_black_does_not_divide_by_zero() -> None:
    """Black pixels clamp the denominator rather than producing NaN."""
    black = np.zeros((2, 2, 3))
    check(
        condition=bool(np.isfinite(saturation(black)).all()),
        message="black must not produce NaN saturation",
    )


def test_subject_width_grows_with_the_subject() -> None:
    """A wider subject occupies a larger fraction of columns."""
    narrow = np.full((SIZE, SIZE, 3), 150.0)
    narrow[:, 28:36] = 10.0
    wide = np.full((SIZE, SIZE, 3), 150.0)
    wide[:, 8:56] = 10.0
    check(
        condition=subject_width(wide) > subject_width(narrow),
        message="a wider subject should measure wider",
    )


def test_load_reads_a_png_from_disk(tmp_path: Path) -> None:
    """Loading returns the written pixels as floats."""
    path = _write(tmp_path, "one", _frame(200))
    check_equal(
        actual=load(path).shape,
        expected=(SIZE, SIZE, 3),
        message="a loaded frame keeps its shape",
    )


def test_debug_sheet_writes_one_cell_per_render(tmp_path: Path) -> None:
    """The debug sheet is as wide as the set and tints the selection."""
    paths = [_write(tmp_path, f"f{i}", _frame(200)) for i in range(3)]
    out = debug_sheet(paths, GARMENT, tmp_path / "nested" / "sheet.png")
    with Image.open(out) as sheet:
        check_equal(
            actual=sheet.size,
            expected=(SIZE * 3, SIZE),
            message="the sheet holds one cell per render",
        )


def test_main_passes_a_consistent_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI exits 0 and says PASS when the set holds together."""
    paths = [_write(tmp_path, f"f{i}", _frame(200)) for i in range(3)]
    code = main([str(p) for p in paths])
    check_equal(actual=code, expected=0, message="a consistent set passes")
    check(
        condition="PASS" in capsys.readouterr().out,
        message="the verdict is reported",
    )


def test_main_fails_a_drifting_set_and_can_write_a_debug_sheet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI exits 1 on drift and writes the sheet when asked."""
    paths = [
        _write(tmp_path, f"f{i}", _frame(blue)) for i, blue in enumerate((180, 250))
    ]
    sheet = tmp_path / "debug.png"
    code = main(
        [*[str(p) for p in paths], "--debug-sheet", str(sheet), "--max-value", "255"]
    )
    check_equal(actual=code, expected=1, message="a drifting set fails")
    check(condition=sheet.exists(), message="the debug sheet is written")
    check(
        condition="FAIL" in capsys.readouterr().out,
        message="the failing verdict is reported",
    )
