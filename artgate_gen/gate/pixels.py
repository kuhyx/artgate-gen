"""Pixel-level gates: alpha binarity, palette conformance, colour count, scale.

Ordering is load-bearing and was established empirically, not by taste:
``ALPHA_NOT_BINARY`` must be evaluated before ``TOO_MANY_COLORS``. A soft-alpha
image has *no* fully-opaque pixels, so an opaque-only colour count returns 0
and silently passes -- two individually correct gates composing into a hole.
``COLOR_COUNT`` therefore also fails closed on an empty opaque set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from artgate_gen.gate.codes import Code

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

# Alpha value denoting a fully opaque pixel.
OPAQUE = 255

# Alpha value denoting a fully transparent pixel.
TRANSPARENT = 0


def opaque_rgb(rgba: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Select the RGB values of fully-opaque pixels.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.

    Returns:
        An ``(n, 3)`` array of RGB triples; may be empty.
    """
    flat = rgba.reshape(-1, 4)
    # numpy's boolean indexing is typed as returning Any, so the element type
    # has to be restated for mypy; the runtime dtype is unchanged.
    return cast("NDArray[np.uint8]", flat[flat[:, 3] == OPAQUE][:, :3])


def check_alpha_binary(rgba: NDArray[np.uint8]) -> list[Code]:
    """Verify alpha is exactly ``{0, 255}``.

    Semi-transparent edges are the clearest signature of a generator that has
    not been through a pixel-art finishing stage.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.

    Returns:
        ``[ALPHA_NOT_BINARY]`` if any other alpha value occurs, else ``[]``.
    """
    values = set(np.unique(rgba[:, :, 3]).tolist())
    if values <= {TRANSPARENT, OPAQUE}:
        return []
    return [Code.ALPHA_NOT_BINARY]


def check_color_count(rgba: NDArray[np.uint8], max_colors: int) -> list[Code]:
    """Verify the opaque palette size, failing closed when nothing is opaque.

    Counting opaque RGB only is required because fully-transparent pixels
    retain arbitrary RGB values, which inflate a naive RGBA count above the
    cap and reject correct art.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.
        max_colors: Maximum permitted distinct opaque RGB values.

    Returns:
        The failing codes, empty if the image conforms.
    """
    opaque = opaque_rgb(rgba)
    if opaque.size == 0:
        return [Code.NO_OPAQUE_PIXELS]
    distinct = np.unique(opaque, axis=0).shape[0]
    if distinct > max_colors:
        return [Code.TOO_MANY_COLORS]
    return []


def check_palette(
    rgba: NDArray[np.uint8],
    palette: frozenset[tuple[int, int, int]],
) -> list[Code]:
    """Verify every opaque pixel uses an allowed colour.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.
        palette: The allowed opaque RGB triples.

    Returns:
        ``[OFF_PALETTE]`` if any opaque pixel is outside the palette.
    """
    opaque = opaque_rgb(rgba)
    if opaque.size == 0:
        return [Code.NO_OPAQUE_PIXELS]
    used = {tuple(int(c) for c in row) for row in np.unique(opaque, axis=0)}
    if used <= palette:
        return []
    return [Code.OFF_PALETTE]


def block_impurity(rgba: NDArray[np.uint8], scale: int) -> float:
    """Measure the fraction of ``scale``-sized blocks that are not uniform.

    True pixel art delivered at an integer upscale has perfectly uniform
    blocks; anything resampled with interpolation does not.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.
        scale: The block size to test.

    Returns:
        0.0 for perfectly blocky input, up to 1.0 for fully smooth input.
    """
    height, width = rgba.shape[:2]
    height -= height % scale
    width -= width % scale
    cropped = rgba[:height, :width]
    blocks = cropped.reshape(
        height // scale, scale, width // scale, scale, 4
    ).transpose(0, 2, 1, 3, 4)
    flat = blocks.reshape(-1, scale * scale, 4)
    impure = (flat != flat[:, :1, :]).any(axis=(1, 2))
    return float(impure.mean())


def detect_scale(rgba: NDArray[np.uint8]) -> int:
    """Find the largest integer upscale factor the image is consistent with.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.

    Returns:
        The largest factor dividing both dimensions with uniform blocks; 1 if
        the image is native-resolution.
    """
    height, width = rgba.shape[:2]
    smallest = min(height, width)
    for candidate in range(smallest, 1, -1):
        divides = height % candidate == 0 and width % candidate == 0
        if divides and block_impurity(rgba, candidate) == 0.0:
            return candidate
    return 1


def check_scale_invariance(rgba: NDArray[np.uint8], scale: int | None) -> list[Code]:
    """Verify the image is genuinely blocky at its declared scale.

    Note:
        This gate passes *trivially* at scale 1, so it is only meaningful for
        assets delivered at an integer upscale of 2 or more. On native
        small-resolution output the discriminating gates are alpha binarity
        and colour count, not this one.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.
        scale: The declared factor, or None to auto-detect.

    Returns:
        ``[SCALE_NOT_INVARIANT]`` if blocks are not uniform at ``scale``.
    """
    factor = detect_scale(rgba) if scale is None else scale
    if factor <= 1:
        return []
    if block_impurity(rgba, factor) > 0.0:
        return [Code.SCALE_NOT_INVARIANT]
    return []
