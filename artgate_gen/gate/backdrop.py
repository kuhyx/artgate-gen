"""Backdrop-only gates: opacity, letterboxing, flat washes, and seams.

A parallax backdrop is *by design* a 100%-opaque full-bleed rectangle, which
inverts the icon-era assumptions baked into ``contrast.py`` and ``pixels.py``
(``MARGIN_TOO_SMALL`` and ``SILHOUETTE_LOW_CONTRAST`` would manufacture false
failures on exactly the art this module is meant to accept). Disabling those
gates without adding replacements would repeat this repo's third fail-open --
twelve fully-opaque SDXL icons passed every check until something finally
asked whether the asset had a silhouette at all. These four gates are that
replacement for backdrop art specifically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from artgate_gen.gate.codes import Code
from artgate_gen.gate.pixels import OPAQUE

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

# Rec. 709 luma weights, matching contrast.py's silhouette measurement.
_LUMA = (0.2126, 0.7152, 0.0722)

# Rows/columns sampled from each edge when checking for a letterbox bar.
# Measured: an 8px band lets a real 6px bar dilute below the near-black cutoff
# and pass. 4px keeps a genuine bar concentrated enough to be caught.
EDGE_BAND = 4

# Mean luma an edge band must fall under to count as a "bar" rather than
# legitimate dark art (a night sky, a shadowed foreground). Calibrated against
# the 9 real accepted strips: measured edge-band luma ranges 47.3-161.0, so
# 30.0 leaves 1.6x headroom above the darkest real edge without approaching
# true black (0-10).
LETTERBOX_DARK_MAX = 30.0

# Minimum mean luma the interior (excluding both edge bands) must reach for an
# edge band to be judged a bar rather than a uniformly dark image. Calibrated
# against the 9 real strips: measured interior luma ranges 83.1-136.8, so 60.0
# leaves 1.4x headroom below the darkest real interior. A genuinely dark scene
# whose interior also reads below 60 is deliberately treated as "dark
# throughout", not letterboxed -- the AND condition, not an oversight.
LETTERBOX_INTERIOR_MIN = 60.0

# Minimum spread across all rows' mean luma required to prove a backdrop
# has sky-to-ground structure rather than being a flat wash.
# Calibrated against the 9 real strips: measured range 45.1-138.5, so 20.0
# leaves over 2x headroom above the lowest real strip while still failing a
# uniform-colour control (range 0.0).
MIN_ROW_LUMA_RANGE = 20.0

# Maximum seam/interior discontinuity ratio, on both axes, before a strip
# fails SEAM_ENERGY. Ported from tiletool.py's calibration: a wrapping
# sinusoid (seamless control) measures ~1.6, a hard half/half split (seamed
# control) measures 255. Verified during planning: this port reproduces
# seam-ratios.txt row-for-row on all 9 real strips.
SEAM_ENERGY_MAX = 4.0


def _luma(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert an RGB array's last axis to Rec. 709 luma.

    Args:
        rgb: An array whose last axis is length-3 RGB.

    Returns:
        An array one dimension smaller, of luma values.
    """
    return rgb[..., 0] * _LUMA[0] + rgb[..., 1] * _LUMA[1] + rgb[..., 2] * _LUMA[2]


def check_fully_opaque(rgba: NDArray[np.uint8]) -> list[Code]:
    """Verify every pixel is fully opaque.

    A backdrop has no subject and no background to separate from one -- it is
    full-bleed by construction. This makes ``finish_backdrop``'s bypass of
    ``strip_background`` (which would otherwise flood 47% of a synthetic sky
    away) verifiable rather than trusted.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.

    Returns:
        ``[FULLY_OPAQUE_REQUIRED]`` if any pixel's alpha is below 255.
    """
    if bool((rgba[:, :, 3] < OPAQUE).any()):
        return [Code.FULLY_OPAQUE_REQUIRED]
    return []


def check_letterbox(rgba: NDArray[np.uint8]) -> list[Code]:
    """Verify the strip has no black bars padding a smaller composed scene.

    Requires BOTH a near-black top-or-bottom edge band AND a bright interior,
    so a legitimately dark night-sky render -- dark at the edges because it is
    dark everywhere -- does not false-positive.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.

    Returns:
        ``[LETTERBOX_BARS]`` if either edge is a bar against a bright
        interior.
    """
    luma = _luma(rgba[:, :, :3].astype(np.float64))
    top = float(luma[:EDGE_BAND, :].mean())
    bottom = float(luma[-EDGE_BAND:, :].mean())
    interior = float(luma[EDGE_BAND:-EDGE_BAND, :].mean())
    barred = (top < LETTERBOX_DARK_MAX or bottom < LETTERBOX_DARK_MAX) and (
        interior > LETTERBOX_INTERIOR_MIN
    )
    if barred:
        return [Code.LETTERBOX_BARS]
    return []


def row_luma_range(rgba: NDArray[np.uint8]) -> float:
    """Measure the spread of per-row mean luma across a strip's height.

    Deliberately the max-minus-min across every row, not top-row-minus-
    bottom-row: a top/bottom comparison is sign-sensitive and noisy on real
    art, where the brightest row is not always row 0 (a sunset sky can be
    darker at the top than a lit foreground is at the bottom). Range is
    robust to which end is brighter and only reads 0 on a genuine flat wash.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.

    Returns:
        The difference between the brightest and dimmest row's mean luma.
    """
    luma = _luma(rgba[:, :, :3].astype(np.float64))
    row_means = luma.mean(axis=1)
    return float(row_means.max() - row_means.min())


def check_vertical_structure(rgba: NDArray[np.uint8]) -> list[Code]:
    """Verify the strip has sky-to-ground structure rather than a flat wash.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.

    Returns:
        ``[NO_VERTICAL_STRUCTURE]`` if no row differs enough from another to
        prove any vertical composition exists.
    """
    if row_luma_range(rgba) < MIN_ROW_LUMA_RANGE:
        return [Code.NO_VERTICAL_STRUCTURE]
    return []


def wrap_discontinuity(rgba: NDArray[np.uint8]) -> tuple[float, float]:
    """Measure how visible a strip's wrap seam would be, on both axes.

    Ported from ``tiletool.py``'s ``wrap_discontinuity``: the wrap-edge colour
    difference is compared against the mean difference between adjacent
    interior pixels, so a ratio near 1.0 reads as ordinary texture variation
    and a ratio far above it reads as a hard seam. Vectorised for arbitrary
    ``(h, w)`` rather than the original's fixed 32x32 tile.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.

    Returns:
        ``(horizontal, vertical)`` seam ratios. 0.0 on an axis whose interior
        has zero variation (a uniform image), rather than dividing by zero.
    """
    rgb = rgba[:, :, :3].astype(np.float64)

    def diff(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.abs(a - b).sum(axis=-1) / 3.0

    seam_h = float(diff(rgb[:, -1, :], rgb[:, 0, :]).mean())
    inner_h = float(diff(rgb[:, :-1, :], rgb[:, 1:, :]).mean())
    seam_v = float(diff(rgb[-1, :, :], rgb[0, :, :]).mean())
    inner_v = float(diff(rgb[:-1, :, :], rgb[1:, :, :]).mean())

    horizontal = seam_h / inner_h if inner_h else 0.0
    vertical = seam_v / inner_v if inner_v else 0.0
    return horizontal, vertical


def check_seam_energy(rgba: NDArray[np.uint8]) -> list[Code]:
    """Verify a strip tiles without a visible seam on either axis.

    A parallax strip tiles horizontally on screen (three instances are live
    at once), so a discontinuous wrap shows as a seam on every frame. The
    vertical axis is checked too even though the strip never tiles
    vertically: a large vertical discontinuity is still evaluated against the
    same threshold, by design.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.

    Returns:
        ``[SEAM_ENERGY]`` if either axis's ratio exceeds ``SEAM_ENERGY_MAX``.
    """
    horizontal, vertical = wrap_discontinuity(rgba)
    if horizontal > SEAM_ENERGY_MAX or vertical > SEAM_ENERGY_MAX:
        return [Code.SEAM_ENERGY]
    return []
