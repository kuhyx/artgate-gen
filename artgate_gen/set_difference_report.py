"""Set-level reporting over `set_difference`'s per-image crop maths.

Split from `set_difference` to hold both files under the 250-line cap, the same
way `set_consistency_cli` is split from `set_consistency`. The maths of
locating and normalising one crop lives there; everything that walks a whole
set and produces something a person reads lives here.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from artgate_gen.set_consistency import load, select
from artgate_gen.set_difference import (
    NORMALISED_PX,
    PairDifference,
    band_difference,
    normalised_crop,
)

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from artgate_gen.set_consistency import RegionSpec


def landmark_pixels(paths: list[Path], region: RegionSpec) -> list[tuple[str, int]]:
    """Count landmark pixels per render, so a failed detection is visible.

    A silently mislocated crop produces numbers indistinguishable from good
    ones. Reporting the count is what makes that failure showable -- and on
    these portraits the count is a finding in its own right, because a face
    rendered too small to carry an expression has too few landmark pixels to
    anchor on either.

    Args:
        paths: The renders forming one set.
        region: The colour window locating the landmark.

    Returns:
        ``(stem, pixel count)`` per render, in input order.
    """
    return [(p.stem, int(select(load(p), region).sum())) for p in paths]


def pairwise(paths: list[Path], region: RegionSpec) -> list[PairDifference]:
    """Difference every pair of renders inside the expression region.

    Args:
        paths: The renders forming one set.
        region: The colour window locating the landmark.

    Returns:
        One `PairDifference` per unordered pair, in input order.
    """
    crops = [(p.stem, normalised_crop(load(p), region)) for p in paths]
    out: list[PairDifference] = []
    for (left_name, left), (right_name, right) in combinations(crops, 2):
        whole, upper, lower = band_difference(left, right)
        out.append(PairDifference(left_name, right_name, whole, upper, lower))
    return out


def weakest_pair(differences: list[PairDifference]) -> PairDifference:
    """Return the pair that differs least -- the set's binding constraint.

    A set is only as legible as its two most similar members, so this is the
    number a gate is written against, not the mean.

    Args:
        differences: Pairwise differences over one set.

    Returns:
        The `PairDifference` with the smallest whole-crop difference.

    Raises:
        ValueError: If no pairs were supplied.
    """
    if not differences:
        msg = "no pairs to compare"
        raise ValueError(msg)
    return min(differences, key=lambda d: d.whole)


def difference_sheet(paths: list[Path], region: RegionSpec, out: Path) -> Path:
    """Write the normalised crops side by side for review by eye.

    The crops are what the numbers are computed from. If the landmark has
    wandered on one render, its number is meaningless and only looking at the
    strip will show it. This caught two bad crop schemes before either was
    trusted, which is why it exists.

    Args:
        paths: The renders forming one set.
        region: The colour window locating the landmark.
        out: Destination PNG path.

    Returns:
        The written path.
    """
    crops = [normalised_crop(load(p), region) for p in paths]
    strip = Image.new("RGB", (NORMALISED_PX * len(crops), NORMALISED_PX))
    for index, crop in enumerate(crops):
        tile = Image.fromarray(crop.astype(np.uint8), mode="RGB")
        strip.paste(tile, (index * NORMALISED_PX, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out)
    return out
