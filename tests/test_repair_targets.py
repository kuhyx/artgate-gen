"""Tests for repair's non-square canvas handling (session 3).

Split out of `test_repair.py` to stay under the repo's line cap. Repair
branches on square vs non-square canvas, not on `TARGETS` membership: it
operates on an already-finished image, never an oversized render, so
`finish_backdrop`'s wrap-blend (which demands overlap width beyond the
canvas) is the wrong tool and used to crash on a correctly-sized strip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from artgate_gen.repair import repair
from tests.assertions import check, check_equal

if TYPE_CHECKING:
    from pathlib import Path

SUBJECT_COLOUR = (200, 40, 40, 255)

# Backdrop-shaped, never declares MARGIN_TOO_SMALL -- like the real
# backdrop_strip.toml -- so CANVAS_SIZE repair must not inset.
SPEC_BACKDROP = """
name = "t"
canvas = [256, 64]
max_colors = 16
gates = ["CANVAS_SIZE", "TOO_MANY_COLORS"]
"""


def _write_spec(tmp_path: Path, body: str = SPEC_BACKDROP) -> Path:
    """Write a spec file.

    Args:
        tmp_path: Pytest temporary directory.
        body: TOML contents.

    Returns:
        The spec path.

    """
    path = tmp_path / "spec.toml"
    path.write_text(body, encoding="utf-8")
    return path


def _write_icon(tmp_path: Path, arr: np.ndarray) -> Path:
    """Write an RGBA array as a PNG.

    Args:
        tmp_path: Pytest temporary directory.
        arr: The image data.

    Returns:
        The icon path.

    """
    path = tmp_path / "icon.png"
    Image.fromarray(arr, "RGBA").save(path)
    return path


def test_repair_does_not_inset_a_backdrop_on_canvas_size(tmp_path: Path) -> None:
    """A backdrop failing CANVAS_SIZE is resized, never inset to a square.

    Regression guard: CANVAS_SIZE is in REPAIRABLE, so a backdrop failing it
    used to enter inset_subject and come back as a (256, 256) square.
    """
    arr = np.full((72, 288, 4), SUBJECT_COLOUR, dtype=np.uint8)
    path = _write_icon(tmp_path, arr)
    check(
        condition=repair(path, _write_spec(tmp_path), ["CANVAS_SIZE"]),
        message="a backdrop canvas failure is repairable",
    )
    with Image.open(path) as image:
        size = image.size
    check_equal(
        actual=size,
        expected=(256, 64),
        message="the backdrop keeps its own aspect ratio, not a square",
    )


def test_repair_does_not_crash_on_an_already_correctly_sized_backdrop(
    tmp_path: Path,
) -> None:
    """Repairing a palette failure on a canvas-correct strip must not crash.

    Regression guard: routing repair through `finish_backdrop` made this
    raise -- its wrap-blend requires the render to be wider than the canvas
    by a real overlap, which an already-finished, correctly-sized strip never
    has. A strip never declares SEAM_ENERGY as repairable, so there is no
    seam to redo here; only the palette needs fixing.
    """
    arr = np.full((64, 256, 4), SUBJECT_COLOUR, dtype=np.uint8)
    path = _write_icon(tmp_path, arr)
    check(
        condition=repair(path, _write_spec(tmp_path), ["TOO_MANY_COLORS"]),
        message="a palette failure on a correctly-sized strip is repairable",
    )
    with Image.open(path) as image:
        size = image.size
    check_equal(
        actual=size,
        expected=(256, 64),
        message="the strip's canvas size is unchanged",
    )
