"""Measure whether a set of renders shows several expressions or only one.

The mirror image of `set_consistency`. That module asks "is this one
character?" and improves as the set varies less. Optimising it alone has a
measured failure mode: a set of seven identical faces is its perfect score. An
expression set built that way passes every consistency gate and is useless,
because the thing the set exists for -- telling the expressions apart -- is
exactly what got optimised away.

This measures the difference that must be present. Given a set of same-sized
renders and the region carrying the expression, it reports how far each pair
differs inside that region.

WHY THE REGION IS NORMALISED BY A LANDMARK. Differencing a fixed crop was
considered first and rejected before any number was generated, because it
cannot separate the two things it would be used to compare. Framing distance
varies with prompt length -- describing a garment pulls the camera back and
shrinks the face -- so a fixed upper-centre crop is mostly hair and background
on a wide render and nearly all face on a close one. Its mean difference would
then rise with face SIZE regardless of whether any brow or mouth moved, and it
would rank prompt variants by how much of the frame the face fills. That is the
same defect as a consistency score improving as the subject becomes less
visible, pointed the other way.

Cropping to the bounding box of a broad skin region was tried next and also
rejected, on real output: the strip written by `difference_sheet` showed the
eyeline drifting vertically from cell to cell. A skin window catches the neck
and upper chest, and how much of those is visible varies per render, so the box
extends a different distance below the chin each time. Two further confounds
sat on top of it -- a non-square box resampled to a square adds aspect
distortion that differs per image, and a selection that finds nothing silently
degrades to the whole frame.

So the crop is anchored on a LANDMARK region instead: its centroid sets the
centre and its horizontal extent sets the scale, with a fixed aspect ratio
throughout. Horizontal extent is the robust choice because the expressions
being measured deform the landmark vertically -- narrowed, downcast and dead
eyes all squash its height while leaving pupil separation alone. Translation,
scale and aspect all cancel; only the arrangement of features survives.

BANDS. The crop is reported whole and split into an upper band (brows, eyes)
and a lower band (mouth). An expression that changes only the mouth is diluted
by an unchanged forehead when the crop is scored as one block, and the mouth is
half of what distinguishes these sets.

This metric measures DIFFERENCE, not legibility. Seven faces differing by noise
score well and read as one face to a person. It exists to stop a set that is
provably identical, not to certify that a set is good. A human looking at the
output remains the gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from PIL import Image

from artgate_gen.set_consistency import select

if TYPE_CHECKING:  # pragma: no cover
    from artgate_gen.set_consistency import RegionSpec

# Every crop is resampled to this square before differencing, so face size
# cancels out of the comparison. Large enough to keep brow and mouth detail at
# the 256px canvas these portraits ship at, small enough that resampling noise
# stays well under a real expression change.
NORMALISED_PX = 96

# Where the upper band (brows, eyes) ends and the lower band (mouth) begins, as
# a fraction of crop height. The crop is anchored so the landmark sits at
# LANDMARK_Y, and everything below it is nose, mouth and chin.
BAND_SPLIT = 0.5

# How wide the crop is as a multiple of the landmark's horizontal extent, and
# where the landmark sits vertically within it. Both are MEASURED off real
# renders, not assumed: on this character the outer edges of the two irises
# span 1.0-1.4x the width of the whole face, and the eyeline sits at 0.27-0.30
# of face height. An earlier guess of 3.0 was built on the intuition that eyes
# span a third of a face; it produced crops two thirds background and scored a
# moved mouth at 1.18, because the mouth fell outside the crop entirely.
#
# 1.6 spans brow to chin with a little margin at that measured ratio, and 0.34
# places the eyeline high enough to keep the mouth inside the lower band.
CROP_SCALE = 1.6
LANDMARK_Y = 0.34

# Below this many selected pixels a landmark is treated as not found: the crop
# would be anchored on noise, and a number derived from it must not be quietly
# returned alongside good ones.
MIN_LANDMARK_PX = 12


@dataclass(frozen=True, slots=True)
class PairDifference:
    """How far two renders differ inside the expression region.

    Attributes:
        left: Name of the first render.
        right: Name of the second render.
        whole: Mean absolute difference over the whole normalised crop.
        upper: Mean absolute difference over the brow and eye band.
        lower: Mean absolute difference over the mouth band.
    """

    left: str
    right: str
    whole: float
    upper: float
    lower: float


class LandmarkNotFoundError(ValueError):
    """Raised when a landmark region selects too few pixels to anchor a crop."""


def landmark_anchor(mask: npt.NDArray[np.bool_]) -> tuple[float, float, float]:
    """Return a landmark's centre and horizontal extent.

    Args:
        mask: Boolean ``(h, w)`` selection of the landmark.

    Returns:
        ``(centre_y, centre_x, width)`` in pixels.

    Raises:
        LandmarkNotFoundError: If fewer than `MIN_LANDMARK_PX` pixels are selected.
    """
    rows, cols = np.nonzero(mask)
    if rows.size < MIN_LANDMARK_PX:
        msg = f"landmark selected {rows.size} px, need {MIN_LANDMARK_PX}"
        raise LandmarkNotFoundError(msg)
    width = float(cols.max() - cols.min() + 1)
    return (float(rows.mean()), float(cols.mean()), width)


def normalised_crop(
    rgb: npt.NDArray[np.float64], region: RegionSpec
) -> npt.NDArray[np.float64]:
    """Crop a render around its landmark at a fixed scale and aspect.

    The landmark's centroid sets the centre and its horizontal extent sets the
    scale, so two renders at different framing distances yield crops that put
    the same features in the same places. The crop is always square in source
    pixels before resampling, so no aspect distortion is introduced.

    Args:
        rgb: An ``(h, w, 3)`` float array.
        region: The colour window locating the landmark.

    Returns:
        A ``(NORMALISED_PX, NORMALISED_PX, 3)`` float array.

    Raises:
        LandmarkNotFoundError: If the landmark is not present.
    """
    centre_y, centre_x, width = landmark_anchor(select(rgb, region))
    side = width * CROP_SCALE
    top = centre_y - side * LANDMARK_Y
    left = centre_x - side / 2.0
    # A crop this size often runs past the frame edge on a close render, and
    # PIL rejects a negative box offset. Pad by a whole side length first and
    # shift the box to match, so the landmark stays where it was asked to be
    # instead of being clamped inwards -- clamping would slide the face within
    # the crop and reintroduce exactly the misalignment this anchoring removes.
    pad = int(np.ceil(side))
    padded = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    image = Image.fromarray(padded.astype(np.uint8), mode="RGB")
    box = (left + pad, top + pad, left + pad + side, top + pad + side)
    resized = image.resize(
        (NORMALISED_PX, NORMALISED_PX), Image.Resampling.BILINEAR, box=box
    )
    return np.asarray(resized, dtype=np.float64)


def band_difference(
    left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]
) -> tuple[float, float, float]:
    """Return whole, upper and lower mean absolute differences of two crops.

    Args:
        left: First normalised crop.
        right: Second normalised crop.

    Returns:
        ``(whole, upper, lower)`` mean absolute differences in 0-255 units.
    """
    delta = np.abs(left - right)
    split = int(NORMALISED_PX * BAND_SPLIT)
    return (
        float(delta.mean()),
        float(delta[:split].mean()),
        float(delta[split:].mean()),
    )
