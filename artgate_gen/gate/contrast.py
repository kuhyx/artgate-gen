"""Silhouette contrast: does the subject read against light AND dark?

One of only three gates valid for every style, because it makes no assumption
about palette, resolution or alpha. An asset that vanishes into one background
is unusable regardless of how it was produced.

Both backgrounds are required. Checking a single background is how art that
looks fine on a dark UI mock disappears entirely on a light one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from artgate_gen.gate.codes import Code

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

# Rec. 709 luma weights, matching how perceived brightness is normally modelled.
_LUMA = (0.2126, 0.7152, 0.0722)

# Minimum mean luma separation (0-255) between subject and background.
# Calibrated against controls: a mid-grey blob on white scores ~1 and must
# fail; a normal opaque icon scores far above this against at least one side.
DEFAULT_MIN_DELTA = 12.0


def _luma(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert RGB to Rec. 709 luma.

    Args:
        rgb: An ``(n, 3)`` float array.

    Returns:
        An ``(n,)`` array of luma values.
    """
    return rgb[:, 0] * _LUMA[0] + rgb[:, 1] * _LUMA[1] + rgb[:, 2] * _LUMA[2]


def silhouette_delta(rgba: NDArray[np.uint8], background: int) -> float:
    """Measure mean luma separation between the subject and a background.

    Alpha-weighted, so a mostly-transparent asset is judged on the pixels it
    actually paints rather than on its empty margin.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.
        background: Grey level of the backdrop, 0-255.

    Returns:
        The absolute luma difference; 0.0 if nothing is painted.
    """
    flat = rgba.reshape(-1, 4).astype(np.float64)
    weight = flat[:, 3] / 255.0
    total = float(weight.sum())
    if total == 0.0:
        return 0.0
    subject = float((_luma(flat[:, :3]) * weight).sum() / total)
    return abs(subject - float(background))


def check_margin(rgba: NDArray[np.uint8], min_margin: int) -> list[Code]:
    """Verify a transparent border surrounds the subject.

    An item icon must be a *sprite*, not a square tile: without a transparent
    margin it cannot sit on an inventory slot or a world background. This gate
    exists because it was missed -- twelve diffusion icons scored 100% opaque
    coverage and passed every other gate, because nothing checked whether the
    asset had a silhouette at all.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.
        min_margin: Required clear border width in pixels.

    Returns:
        ``[MARGIN_TOO_SMALL]`` if any edge band contains opaque pixels.
    """
    if min_margin <= 0:
        return []
    alpha = rgba[:, :, 3]
    band = min_margin
    edges = (
        alpha[:band, :],
        alpha[-band:, :],
        alpha[:, :band],
        alpha[:, -band:],
    )
    if any(int(edge.max(initial=0)) > 0 for edge in edges):
        return [Code.MARGIN_TOO_SMALL]
    return []


def check_silhouette(
    rgba: NDArray[np.uint8],
    min_delta: float = DEFAULT_MIN_DELTA,
) -> list[Code]:
    """Verify the subject separates from both a light and a dark background.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.
        min_delta: Minimum required luma separation from each background.

    Returns:
        ``[SILHOUETTE_LOW_CONTRAST]`` if either background is too close, or
        ``[EMPTY_CANVAS]`` if the image paints nothing at all.
    """
    flat = rgba.reshape(-1, 4)
    if float(flat[:, 3].sum()) == 0.0:
        return [Code.EMPTY_CANVAS]
    on_dark = silhouette_delta(rgba, 0)
    on_light = silhouette_delta(rgba, 255)
    if min(on_dark, on_light) < min_delta:
        return [Code.SILHOUETTE_LOW_CONTRAST]
    return []
