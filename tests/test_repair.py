"""Tests for icon repair.

Repair never asserts its own success -- the caller re-gates. What is tested
here is that it attempts a repair exactly when the failure is one re-encoding
can actually fix, and leaves everything else alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from artgate_gen.repair import REPAIRABLE, repair
from tests.assertions import check, check_equal

if TYPE_CHECKING:
    from pathlib import Path

EDGE = 32
SUBJECT_COLOUR = (200, 40, 40, 255)

SPEC_32 = """
name = "t"
canvas = [32, 32]
max_colors = 16
alpha_binary = true
min_margin = 2
gates = ["CANVAS_SIZE", "TOO_MANY_COLORS", "ALPHA_NOT_BINARY", "MARGIN_TOO_SMALL"]
"""

SPEC_NO_CANVAS = """
name = "t"
max_colors = 16
gates = ["TOO_MANY_COLORS"]
"""


def _write_spec(tmp_path: Path, body: str = SPEC_32) -> Path:
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


def _subject_canvas(edge: int = EDGE) -> np.ndarray:
    """Build a canvas with an off-centre opaque block.

    Args:
        edge: Canvas edge length.

    Returns:
        An ``(edge, edge, 4)`` uint8 array.

    """
    arr = np.zeros((edge, edge, 4), dtype=np.uint8)
    arr[0:8, 0:8] = SUBJECT_COLOUR
    return arr


def test_repair_declines_when_there_are_no_codes(tmp_path: Path) -> None:
    """Nothing to fix means nothing is written."""
    path = _write_icon(tmp_path, _subject_canvas())
    check(
        condition=not repair(path, _write_spec(tmp_path), []),
        message="no codes, no repair",
    )


def test_repair_declines_unrepairable_codes(tmp_path: Path) -> None:
    """A contrast failure is a fact about the art, not an encoding problem."""
    path = _write_icon(tmp_path, _subject_canvas())
    check(
        condition=not repair(
            path,
            _write_spec(tmp_path),
            ["SILHOUETTE_LOW_CONTRAST"],
        ),
        message="unrepairable code is declined",
    )


def test_repair_declines_a_mixed_code_set(tmp_path: Path) -> None:
    """One unrepairable code disqualifies the whole set."""
    path = _write_icon(tmp_path, _subject_canvas())
    check(
        condition=not repair(
            path,
            _write_spec(tmp_path),
            ["TOO_MANY_COLORS", "SILHOUETTE_LOW_CONTRAST"],
        ),
        message="a mixed set is declined",
    )


def test_repair_rewrites_for_a_palette_failure(tmp_path: Path) -> None:
    """A palette failure is repaired by re-running the finishing chain."""
    path = _write_icon(tmp_path, _subject_canvas())
    check(
        condition=repair(path, _write_spec(tmp_path), ["TOO_MANY_COLORS"]),
        message="a palette failure is repairable",
    )


def test_repair_insets_for_a_margin_failure(tmp_path: Path) -> None:
    """A margin failure moves the subject inward."""
    path = _write_icon(tmp_path, _subject_canvas())
    check(
        condition=repair(path, _write_spec(tmp_path), ["MARGIN_TOO_SMALL"]),
        message="a margin failure is repairable",
    )
    with Image.open(path) as image:
        out = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    check_equal(
        actual=int(out[0:2, :, 3].max()),
        expected=0,
        message="the margin band is now clear",
    )


def test_repair_insets_for_a_canvas_failure(tmp_path: Path) -> None:
    """A canvas-size failure also routes through the inset path."""
    path = _write_icon(tmp_path, _subject_canvas(16))
    check(
        condition=repair(path, _write_spec(tmp_path), ["CANVAS_SIZE"]),
        message="a canvas failure is repairable",
    )
    with Image.open(path) as image:
        size = image.size
    check_equal(
        actual=size,
        expected=(EDGE, EDGE),
        message="the icon was resized to the spec canvas",
    )


def test_repair_falls_back_to_the_image_width_without_a_spec_canvas(
    tmp_path: Path,
) -> None:
    """With no canvas in the spec, the icon's own width is the target."""
    path = _write_icon(tmp_path, _subject_canvas())
    check(
        condition=repair(
            path,
            _write_spec(tmp_path, SPEC_NO_CANVAS),
            ["TOO_MANY_COLORS"],
        ),
        message="repair runs without a spec canvas",
    )
    with Image.open(path) as image:
        size = image.size
    check_equal(
        actual=size,
        expected=(EDGE, EDGE),
        message="the original width is preserved",
    )


def test_repairable_set_excludes_art_facts() -> None:
    """Contrast and emptiness are not encoding problems."""
    check(
        condition="SILHOUETTE_LOW_CONTRAST" not in REPAIRABLE,
        message="contrast is a fact about the art",
    )
