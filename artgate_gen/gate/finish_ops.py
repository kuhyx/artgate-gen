"""The finishing stage: turn continuous-tone output into gate-passing pixel art.

Routes that render continuously -- diffusion (b), SVG rasterisation and 3D
renders (d) -- cannot satisfy a pixel-art spec natively. Their output has
anti-aliased alpha and thousands of colours. This module is the shared bridge,
and its existence is itself a bake-off finding: for pixel work those routes are
*generators feeding a finisher*, not pixel-art generators.

Order matters and mirrors the gate order. Alpha is thresholded FIRST, so the
colour quantiser only ever sees pixels that will survive; quantising first
would blend transparent pixels' arbitrary RGB into the palette.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

# Alpha at or above this becomes fully opaque; below it becomes transparent.
ALPHA_CUTOFF = 128


def binarize_alpha(rgba: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Threshold alpha to exactly ``{0, 255}`` and canonicalise clear pixels.

    Fully-transparent pixels are set to ``(0, 0, 0, 0)``. Without this their
    leftover RGB inflates a distinct-colour count and rejects correct art --
    the exact bug measured during design at 18 colours against a cap of 16.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.

    Returns:
        A new array with binary alpha.
    """
    out = rgba.copy()
    opaque = out[:, :, 3] >= ALPHA_CUTOFF
    out[:, :, 3] = np.where(opaque, 255, 0).astype(np.uint8)
    out[~opaque] = 0
    return out


def downscale(rgba: NDArray[np.uint8], target: int) -> NDArray[np.uint8]:
    """Box-downscale to a square target, preserving straight alpha.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.
        target: Target edge length in pixels.

    Returns:
        A ``(target, target, 4)`` uint8 array.
    """
    img = Image.fromarray(rgba, "RGBA").resize((target, target), Image.Resampling.BOX)
    return np.asarray(img, dtype=np.uint8)


def quantize(rgba: NDArray[np.uint8], max_colors: int) -> NDArray[np.uint8]:
    """Reduce the opaque palette to at most ``max_colors`` entries.

    Quantisation runs over RGB only; the caller is expected to have binarised
    alpha already, so transparent pixels contribute nothing to the palette.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array with binary alpha.
        max_colors: Maximum distinct opaque RGB values.

    Returns:
        A new array whose opaque pixels use at most ``max_colors`` colours.
    """
    rgb = Image.fromarray(rgba[:, :, :3], "RGB").quantize(
        colors=max_colors, method=Image.Quantize.MEDIANCUT
    )
    flat = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
    alpha = rgba[:, :, 3]
    out = np.dstack([flat, alpha])
    out[alpha == 0] = 0
    return out.astype(np.uint8)


def finish(
    rgba: NDArray[np.uint8],
    target: int,
    max_colors: int,
) -> NDArray[np.uint8]:
    """Run the full finishing pipeline.

    Args:
        rgba: Continuous-tone source, any size.
        target: Target square edge length.
        max_colors: Maximum distinct opaque colours.

    Returns:
        Pixel art satisfying the alpha, colour and canvas gates.
    """
    return quantize(binarize_alpha(downscale(rgba, target)), max_colors)
