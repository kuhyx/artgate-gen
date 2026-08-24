"""Tests for repair's pure geometry helpers: `opaque_bbox` and `inset_subject`.

Split out of `test_repair.py` to stay under the repo's line cap. `repair()`
itself -- the orchestration that decides *whether* to inset and how to
re-finish -- is tested there; this file only covers the two building blocks.
"""

from __future__ import annotations

import numpy as np

from artgate_gen.repair import inset_subject, opaque_bbox
from tests.assertions import check, check_equal

EDGE = 32
SUBJECT_COLOUR = (200, 40, 40, 255)


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


def test_bbox_is_none_for_a_fully_transparent_canvas() -> None:
    """Nothing opaque means no bounds."""
    arr = np.zeros((EDGE, EDGE, 4), dtype=np.uint8)
    check(condition=opaque_bbox(arr) is None, message="no opaque pixels")


def test_bbox_tracks_the_opaque_region() -> None:
    """Bounds are inclusive-exclusive around the opaque block."""
    check_equal(
        actual=opaque_bbox(_subject_canvas()),
        expected=(0, 0, 8, 8),
        message="bbox of the top-left block",
    )


def test_inset_returns_input_when_nothing_is_opaque() -> None:
    """An empty canvas cannot be re-centred, so it comes back unchanged."""
    arr = np.zeros((EDGE, EDGE, 4), dtype=np.uint8)
    out = inset_subject(arr, EDGE, 2)
    check(condition=out is arr, message="the input is returned unchanged")


def test_inset_returns_input_when_the_margin_leaves_no_room() -> None:
    """A margin wider than the canvas leaves no inner box to scale into."""
    arr = _subject_canvas()
    out = inset_subject(arr, 4, 10)
    check(condition=out is arr, message="no inner box means no repair")


def test_inset_centres_the_subject_within_the_margin() -> None:
    """The subject is re-centred and the border band is left clear."""
    out = inset_subject(_subject_canvas(), EDGE, 2)
    check_equal(actual=out.shape, expected=(EDGE, EDGE, 4), message="canvas shape")
    check_equal(
        actual=int(out[0:2, :, 3].max()),
        expected=0,
        message="the top margin band is clear",
    )
    bounds = opaque_bbox(out)
    check(condition=bounds is not None, message="the subject survived")
