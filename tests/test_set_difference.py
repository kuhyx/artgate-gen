"""Tests for the expression-difference metric.

The controls carry the weight here, because this metric's failure mode is
subtle: it must respond to features MOVING inside a face and stay flat when the
same face is merely rendered larger, smaller or off-centre. A version that
confuses those two was written first, on real portraits, and its numbers ranked
a set of identical faces above a set of visibly different ones.

So the synthetic frames below vary one thing at a time: same face at two
scales, same face at two offsets, and different features at one scale. The
metric must call the first two the same and the third different.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from artgate_gen.set_consistency import RegionSpec
from artgate_gen.set_difference import (
    MIN_LANDMARK_PX,
    NORMALISED_PX,
    LandmarkNotFoundError,
    band_difference,
    landmark_anchor,
    normalised_crop,
)
from artgate_gen.set_difference_report import (
    difference_sheet,
    landmark_pixels,
    pairwise,
    weakest_pair,
)
from tests.assertions import check, check_equal

if TYPE_CHECKING:
    from pathlib import Path

SIZE = 128
# Two crops of the same face at different scales should differ by less than
# this; two faces with a moved mouth by more. The gap between the two bounds is
# what makes the metric usable rather than merely non-constant.
SAME_FACE_TOLERANCE = 12.0
# The synthetic mouth is a two-pixel line, so moving it changes a far smaller
# share of the crop than a real expression does -- a real one reshades the
# whole lower face. The floor is set against what this deliberately sparse
# stimulus can produce, and the ordering assertions below carry the real
# weight: a moved feature must beat pure scale and offset noise, and it must
# land in the band it belongs to.
MOVED_FEATURE_FLOOR = 3.0
# A green-over-red window, matching the synthetic eyes below.
EYES = RegionSpec(
    name="eyes",
    channel_lead=(1, 0),
    lead_margin=25.0,
    min_saturation=0.35,
    max_saturation=1.0,
    min_value=60.0,
    max_value=255.0,
    top=0.0,
    bottom=0.7,
)


def _face(*, scale: int = 1, dx: int = 0, dy: int = 0, mouth_y: int = 18) -> np.ndarray:
    """Build a frame carrying a schematic face: two green eyes and a mouth.

    The proportions follow what was measured on the real portraits this metric
    runs against -- the irises span roughly the width of the face and the mouth
    sits about eighteen rows below the eyeline at scale 1 -- so a crop tuned on
    real output frames the synthetic face the same way.

    Args:
        scale: Multiplier on every feature's size and separation.
        dx: Horizontal offset in pixels.
        dy: Vertical offset in pixels.
        mouth_y: Mouth rows below the eyeline, so a caller can move one feature.

    Returns:
        An ``(h, w, 3)`` float array.
    """
    arr = np.full((SIZE, SIZE, 3), 150.0)
    cx, cy = SIZE // 2 + dx, SIZE // 2 + dy
    eye = 3 * scale
    gap = 10 * scale
    for side in (-gap, gap):
        top, left = cy - eye, cx + side - eye
        arr[top : top + 2 * eye, left : left + 2 * eye] = (20.0, 200.0, 40.0)
    mouth_top = cy + mouth_y * scale
    arr[mouth_top : mouth_top + 2 * scale, cx - gap : cx + gap] = (90.0, 40.0, 40.0)
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


def test_the_same_face_at_two_scales_reads_as_unchanged() -> None:
    """Rendering a face larger must not register as a changed expression.

    This is the defect that made the first version of this metric useless: it
    rewarded whatever filled more of the frame.
    """
    small = normalised_crop(_face(scale=1), EYES)
    large = normalised_crop(_face(scale=2), EYES)
    whole, _, _ = band_difference(small, large)
    check(
        condition=whole < SAME_FACE_TOLERANCE,
        message=f"same face at two scales should match, got {whole:.2f}",
    )


def test_the_same_face_at_two_offsets_reads_as_unchanged() -> None:
    """Moving a face within the frame must not register as a changed expression."""
    centred = normalised_crop(_face(), EYES)
    shifted = normalised_crop(_face(dx=12, dy=-9), EYES)
    whole, _, _ = band_difference(centred, shifted)
    check(
        condition=whole < SAME_FACE_TOLERANCE,
        message=f"same face at two offsets should match, got {whole:.2f}",
    )


def test_a_moved_mouth_is_separated_from_an_unchanged_face() -> None:
    """A feature that actually moves must outscore scale and offset noise.

    The ordering is what matters: a moved mouth must register more than the
    same face merely drawn at another scale, or the metric cannot tell an
    expression change from a framing change.
    """
    base = normalised_crop(_face(), EYES)
    moved = normalised_crop(_face(mouth_y=8), EYES)
    rescaled = normalised_crop(_face(scale=2), EYES)
    whole, upper, lower = band_difference(base, moved)
    scale_noise, _, _ = band_difference(base, rescaled)
    check(
        condition=whole > MOVED_FEATURE_FLOOR,
        message=f"a moved mouth should register, got {whole:.2f}",
    )
    check(
        condition=whole > scale_noise,
        message=f"a moved mouth ({whole:.2f}) must beat rescaling ({scale_noise:.2f})",
    )
    check(
        condition=lower > upper,
        message=f"a mouth moves the lower band, got {lower:.2f} vs {upper:.2f}",
    )


def test_a_landmark_below_the_floor_raises_rather_than_guessing() -> None:
    """Too few landmark pixels must fail loudly, not anchor a crop on noise."""
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[0, : MIN_LANDMARK_PX - 1] = True
    with pytest.raises(LandmarkNotFoundError):
        landmark_anchor(mask)


def test_a_landmark_at_the_floor_is_accepted() -> None:
    """The boundary is inclusive: exactly the minimum is enough to anchor."""
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[0, :MIN_LANDMARK_PX] = True
    _, _, width = landmark_anchor(mask)
    check_equal(
        actual=width,
        expected=float(MIN_LANDMARK_PX),
        message="a landmark at the floor should report its true width",
    )


def test_a_face_against_the_frame_edge_is_still_centred() -> None:
    """A crop running past the edge pads rather than sliding the landmark.

    Clamping the box inwards would move the face within the crop, which is the
    misalignment the anchoring exists to remove. The offset stays inside the
    region's own vertical band: pushed past it one eye is cut off, and the
    anchor is then measuring a different landmark rather than a moved one.
    """
    centred = normalised_crop(_face(), EYES)
    cornered = normalised_crop(_face(dx=-58, dy=-24), EYES)
    whole, _, _ = band_difference(centred, cornered)
    check(
        condition=whole < SAME_FACE_TOLERANCE,
        message=f"an edge face should still align, got {whole:.2f}",
    )


def test_every_pair_is_reported_once(tmp_path: Path) -> None:
    """Four renders yield the six unordered pairs, not sixteen."""
    paths = [_write(tmp_path, f"f{i}", _face(mouth_y=8 + i * 4)) for i in range(4)]
    check_equal(
        actual=len(pairwise(paths, EYES)),
        expected=6,
        message="four renders make six unordered pairs",
    )


def test_the_weakest_pair_is_the_least_different(tmp_path: Path) -> None:
    """The gate is written against the closest two, so they must be found."""
    paths = [_write(tmp_path, "a", _face()), _write(tmp_path, "b", _face())]
    paths.append(_write(tmp_path, "c", _face(mouth_y=8)))
    weakest = weakest_pair(pairwise(paths, EYES))
    check_equal(
        actual={weakest.left, weakest.right},
        expected={"a", "b"},
        message="the two identical frames are the weakest pair",
    )


def test_weakest_pair_rejects_an_empty_set() -> None:
    """No pairs means no gate, which must raise rather than return a number."""
    with pytest.raises(ValueError, match="no pairs"):
        weakest_pair([])


def test_landmark_pixels_counts_each_render(tmp_path: Path) -> None:
    """A count per render is what makes a failed detection visible."""
    paths = [_write(tmp_path, "big", _face(scale=2))]
    paths.append(_write(tmp_path, "small", _face(scale=1)))
    counts = dict(landmark_pixels(paths, EYES))
    check(
        condition=counts["big"] > counts["small"],
        message=f"a larger face carries more landmark pixels, got {counts}",
    )


def test_the_difference_sheet_is_one_tile_per_render(tmp_path: Path) -> None:
    """The strip is reviewed by eye, so its geometry must be predictable."""
    paths = [_write(tmp_path, f"f{i}", _face()) for i in range(3)]
    out = difference_sheet(paths, EYES, tmp_path / "nested" / "strip.png")
    with Image.open(out) as sheet:
        check_equal(
            actual=sheet.size,
            expected=(NORMALISED_PX * 3, NORMALISED_PX),
            message="the strip is one normalised tile per render",
        )
