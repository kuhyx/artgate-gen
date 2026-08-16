"""Measure whether a set of renders shows one subject or several.

A fixed seed does not hold a character together. It fixes the initial noise;
changing any part of the prompt -- an expression clause, a pose -- re-conditions
the whole generation, so every attribute the prompt leaves unpinned is free to
be re-invented per render. An expression set built that way passes every
per-image gate and still shows seven different people.

This measures the drift. Given a set of same-sized renders and a colour region
of interest, it reports how far the region's mean colour moves across the set,
plus how much the subject's width varies.

WHY A COLOUR REGION AND NOT A PIXEL BOX. Sampling a fixed rectangle was tried
first, on a real seven-portrait set, and it failed: framing distance varies
between renders, so a box over the torso on one image lands on the background
on another. It returned the backdrop's grey as the garment colour on two of
seven images. Selecting by colour finds the garment wherever it sits.

The caller supplies the region because this module knows nothing about any
particular character. `RegionSpec` is a hue/saturation/brightness window;
`debug_sheet` renders the selection so it is checked by eye before any number
derived from it is believed. That check is not optional: a selection that has
silently wandered onto the background produces confident, meaningless numbers.

This metric measures AGREEMENT, not correctness. A set that is uniformly the
wrong colour scores perfectly. Only a human looking at the output catches a
mistake the whole set shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

RGB = tuple[float, float, float]

# How far a column's mean colour must sit from the border reference before it
# counts as subject rather than background, in 0-255 units. Low enough to catch
# a garment close to the backdrop in tone, high enough to ignore the gentle
# vertical gradient these backdrops carry.
SUBJECT_DEVIATION = 8.0


@dataclass(frozen=True, slots=True)
class RegionSpec:
    """A colour window selecting one material across a set of renders.

    Attributes:
        name: What this region is, for reporting.
        channel_lead: ``(high, low)`` channel indices; the high channel must
            exceed the low one by ``lead_margin``. ``(2, 0)`` selects
            blue-over-red, ``(0, 2)`` warm-over-cool.
        lead_margin: How far, in 0-255 units, the lead channel must win by.
        min_saturation: Lower saturation bound, ``0.0..1.0``.
        max_saturation: Upper saturation bound, ``0.0..1.0``.
        min_value: Lower brightness bound, in 0-255 units.
        max_value: Upper brightness bound, in 0-255 units.
        top: Fraction of the frame to skip from the top, ``0.0..1.0``.
        bottom: Fraction of the frame to stop at, ``0.0..1.0``.
    """

    name: str
    channel_lead: tuple[int, int]
    lead_margin: float
    min_saturation: float
    max_saturation: float
    min_value: float
    max_value: float
    top: float
    bottom: float


def saturation(rgb: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Per-pixel saturation in ``0.0..1.0``.

    Args:
        rgb: An ``(h, w, 3)`` float array.

    Returns:
        An ``(h, w)`` float array.
    """
    high = rgb.max(axis=2)
    ratio: npt.NDArray[np.float64] = (high - rgb.min(axis=2)) / np.maximum(high, 1.0)
    return ratio


def select(rgb: npt.NDArray[np.float64], region: RegionSpec) -> npt.NDArray[np.bool_]:
    """Select the pixels of one render matching a region spec.

    Args:
        rgb: An ``(h, w, 3)`` float array.
        region: The colour window to apply.

    Returns:
        A boolean ``(h, w)`` mask.
    """
    height = rgb.shape[0]
    high, low = region.channel_lead
    lead = rgb[:, :, high] - rgb[:, :, low] > region.lead_margin
    sat = saturation(rgb)
    val = rgb.mean(axis=2)

    band = np.zeros(rgb.shape[:2], dtype=bool)
    band[int(height * region.top) : int(height * region.bottom), :] = True

    return (
        lead
        & band
        & (sat > region.min_saturation)
        & (sat <= region.max_saturation)
        & (val > region.min_value)
        & (val <= region.max_value)
    )


def mean_colour(rgb: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_]) -> RGB:
    """Mean colour over a mask, or zeros when nothing is selected.

    Args:
        rgb: An ``(h, w, 3)`` float array.
        mask: A boolean ``(h, w)`` selection.

    Returns:
        The mean ``(r, g, b)``.
    """
    if not mask.any():
        return (0.0, 0.0, 0.0)
    picked = rgb[mask]
    return (
        float(picked[:, 0].mean()),
        float(picked[:, 1].mean()),
        float(picked[:, 2].mean()),
    )


def subject_width(rgb: npt.NDArray[np.float64]) -> float:
    """Estimate the fraction of columns the subject occupies.

    The edge columns of a render are background by construction -- a portrait
    is framed, not cropped to the silhouette -- so the leftmost and rightmost
    columns are taken as the background reference. A column that differs from
    that reference holds the subject.

    An earlier version tested each column's *vertical* variation instead, on
    the theory that flat columns are background. That degenerates: a subject
    which is itself a flat vertical band has no vertical variation either, and
    the metric returned 0.0 for a subject occupying three quarters of the
    frame. Comparing against the border is robust to that.

    Args:
        rgb: An ``(h, w, 3)`` float array.

    Returns:
        A fraction in ``0.0..1.0``.
    """
    border = np.concatenate([rgb[:, :1, :], rgb[:, -1:, :]], axis=1)
    reference = border.reshape(-1, 3).mean(axis=0)
    deviation = np.abs(rgb - reference).mean(axis=2).mean(axis=0)
    return float((deviation > SUBJECT_DEVIATION).mean())


def load(path: Path) -> npt.NDArray[np.float64]:
    """Load one render as a float RGB array.

    Args:
        path: Image to load.

    Returns:
        An ``(h, w, 3)`` float array.
    """
    with Image.open(path) as image:
        rgb: npt.NDArray[np.float64] = np.asarray(
            image.convert("RGB"), dtype=np.float64
        )
        return rgb


def channel_spread(colours: list[RGB]) -> RGB:
    """Per-channel max-minus-min across a set of mean colours.

    Args:
        colours: One mean colour per render.

    Returns:
        The ``(r, g, b)`` spread.
    """
    array = np.asarray(colours, dtype=np.float64)
    ranges = array.max(axis=0) - array.min(axis=0)
    return (float(ranges[0]), float(ranges[1]), float(ranges[2]))


def measure_set(paths: list[Path], region: RegionSpec) -> tuple[list[RGB], list[float]]:
    """Measure a whole set.

    Args:
        paths: The renders forming one set.
        region: The colour window to track across them.

    Returns:
        ``(mean colour per render, subject width per render)``.
    """
    colours: list[RGB] = []
    widths: list[float] = []
    for path in paths:
        rgb = load(path)
        colours.append(mean_colour(rgb, select(rgb, region)))
        widths.append(subject_width(rgb))
    return colours, widths


def debug_sheet(paths: list[Path], region: RegionSpec, out: Path) -> Path:
    """Write a sheet showing which pixels the region selected on each render.

    Args:
        paths: The renders to visualise.
        region: The colour window to apply.
        out: Where to write the sheet.

    Returns:
        The written path.
    """
    tinted: list[Image.Image] = []
    for path in paths:
        rgb = load(path)
        shown = rgb.copy()
        shown[select(rgb, region)] = (255.0, 0.0, 255.0)
        tinted.append(Image.fromarray(shown.astype("uint8"), "RGB"))

    width, height = tinted[0].size
    sheet = Image.new("RGB", (width * len(tinted), height), (18, 18, 22))
    for index, image in enumerate(tinted):
        sheet.paste(image, (index * width, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
