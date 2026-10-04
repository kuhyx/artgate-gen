"""Reduce a continuous-tone render to gate-passing pixel art.

Two finishers, because two kinds of art need opposite treatment.

``finish_icon`` is the original: the background goes first, then the downscale,
then alpha is binarised, and only then is the palette quantised. Quantising
earlier would blend the arbitrary RGB carried by transparent pixels into the
palette and inflate the colour count -- the exact bug that once produced 18
colours against a cap of 16.

``finish_backdrop`` deliberately skips two of those steps. A parallax strip is
full-bleed: every pixel is subject, so there is no background to strip and no
transparency to binarise. Running ``strip_background`` on one destroys it --
measured, opaque fraction 1.000 -> 0.531 on a synthetic gradient sky, because
the flood starts from the border ring and the sky is within tolerance of the
border median. The strip instead gets a wrap-blend, which is what lets three
instances tile horizontally without a seam.

``finish_portrait`` is the third: also full-bleed (a head-and-shoulders
portrait fills the frame, sometimes with a background element like a brick
wall), so it shares the backdrop's ``strip_background`` bypass, but there is
no horizontal tiling requirement, so no wrap-blend either -- a straight
downscale. It also skips quantisation by default: the accepted portraits in
``art/MANIFEST.json`` were shipped at ``max_colors: 0`` (unquantised), and
introducing a cap now would change pixels in art that already survived five
sessions of identity/consistency work to get right.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from artgate_gen.background import strip_background
from artgate_gen.gate.finish_ops import binarize_alpha, downscale_to, quantize

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

# Both icon specs cap the palette at 32, and the number is calibrated against
# real art rather than chosen: 250 genuine 32x32 icons top out at 30 colours and
# 300 genuine 48x48 icons top out at 14. Hand-made art does not spend more
# colours on a bigger canvas, so this cap does NOT scale with `size`. Deriving
# it from the canvas instead produced 64-colour icons that failed
# TOO_MANY_COLORS on all 18 subjects.
MAX_COLOURS = 32

# Width of the horizontal overlap a backdrop is rendered with and blended over.
# At the 2304x512 render size this is exactly the 256px excess over the 2048x512
# finished width, so the blend consumes the surplus and nothing is cropped away
# un-blended.
WRAP_OVERLAP = 256


def wrap_blend(
    rgba: NDArray[np.uint8], overlap: int = WRAP_OVERLAP
) -> NDArray[np.uint8]:
    """Fold an oversized render's right edge back over its left, then crop.

    Renders arrive ``overlap`` pixels wider than they finish. The surplus strip
    on the right is cross-faded onto the left edge with a linear ramp, so the
    first and last columns of the result are near-identical and the image tiles
    horizontally. Raw diffusion output does not: measured seam ratios of
    realistic non-tiling renders are 5.94 and 255.00 against a threshold of 4.0,
    and the same renders blended are 1.03 and 0.58.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array, wider than its finished width by
            ``overlap``.
        overlap: Width of the blended region in pixels.

    Returns:
        An ``(h, w - overlap, 4)`` uint8 array that wraps continuously.
    """
    keep = rgba.shape[1] - overlap
    out = rgba[:, :keep].astype(np.float64).copy()
    tail = rgba[:, keep:].astype(np.float64)
    # Ramp from 0 at the left edge to 1 just past the overlap: the tail fully
    # replaces column 0 and fades out, so there is no step at either end of the
    # blended region.
    ramp = np.linspace(0.0, 1.0, overlap, endpoint=False).reshape(1, overlap, 1)
    out[:, :overlap] = tail * (1.0 - ramp) + out[:, :overlap] * ramp
    return np.clip(out, 0, 255).astype(np.uint8)


def finish_icon(
    raw: Image.Image, canvas: tuple[int, int], colours: int = MAX_COLOURS
) -> Image.Image:
    """Strip the background and reduce a render to an icon.

    Args:
        raw: The full-resolution render.
        canvas: Target ``(width, height)`` in pixels.
        colours: Palette cap.

    Returns:
        The finished icon, ready for the gate.
    """
    arr: NDArray[np.uint8] = np.asarray(raw.convert("RGBA"), dtype=np.uint8)
    stripped = strip_background(arr)
    reduced = quantize(binarize_alpha(downscale_to(stripped, canvas)), colours)
    return Image.fromarray(reduced, "RGBA")


def finish_backdrop(
    raw: Image.Image, canvas: tuple[int, int], colours: int = MAX_COLOURS
) -> Image.Image:
    """Wrap-blend a strip and reduce it, keeping every pixel opaque.

    Neither ``strip_background`` nor ``binarize_alpha`` runs here; see the module
    docstring for what the former does to a full-bleed image. Alpha is forced to
    255 rather than merely left alone, so the result is opaque by construction
    and the bypass is verifiable instead of trusted.

    The overlap is derived from the render's own geometry rather than assumed to
    be ``WRAP_OVERLAP``. The *height* fixes the reduction factor unambiguously
    (nothing is cropped vertically), so the blended width must be that factor
    times the canvas width, and the surplus is the overlap. Deriving it from the
    width instead would fail on the production case: 2304 is itself an exact
    multiple of 256, so a width-based rule computes an overlap of zero.

    Args:
        raw: The full-resolution render, oversized horizontally.
        canvas: Target ``(width, height)`` in pixels.
        colours: Palette cap.

    Returns:
        The finished strip, tiling horizontally.

    Raises:
        ValueError: If the render leaves no horizontal overlap to blend.
    """
    arr: NDArray[np.uint8] = np.asarray(raw.convert("RGBA"), dtype=np.uint8)
    factor = arr.shape[0] // canvas[1]
    overlap = arr.shape[1] - factor * canvas[0]
    if overlap <= 0:
        msg = (
            f"a {arr.shape[1]}x{arr.shape[0]} render leaves no wrap overlap at "
            f"canvas {canvas[0]}x{canvas[1]}; it must be wider than "
            f"{factor * canvas[0]}px"
        )
        raise ValueError(msg)
    reduced = quantize(downscale_to(wrap_blend(arr, overlap), canvas), colours)
    opaque = reduced.copy()
    opaque[:, :, 3] = 255
    return Image.fromarray(opaque, "RGBA")


def finish_portrait(
    raw: Image.Image, canvas: tuple[int, int], colours: int = 0
) -> Image.Image:
    """Downscale a full-bleed portrait render, keeping every pixel opaque.

    Neither ``strip_background`` nor ``binarize_alpha`` runs here, for the same
    reason as ``finish_backdrop``: a head-and-shoulders portrait is full-bleed,
    and the flood measured destroying 47% of a synthetic sky would do the same
    to a portrait's background. Unlike a backdrop there is no wrap-blend step --
    a portrait never tiles.

    Args:
        raw: The full-resolution render.
        canvas: Target ``(width, height)`` in pixels.
        colours: Palette cap. ``0`` (the default) skips quantisation entirely,
            matching the accepted portraits' ``max_colors: 0`` -- they were
            never quantised, and capping them now would move pixels in art
            that already passed manual review.

    Returns:
        The finished portrait, opaque by construction.
    """
    arr: NDArray[np.uint8] = np.asarray(raw.convert("RGBA"), dtype=np.uint8)
    reduced = downscale_to(arr, canvas)
    if colours:
        reduced = quantize(reduced, colours)
    opaque = reduced.copy()
    opaque[:, :, 3] = 255
    return Image.fromarray(opaque, "RGBA")


def finish(raw: Image.Image, size: int, colours: int | None = None) -> Image.Image:
    """Reduce a render to a square icon.

    Retained as the square-only entry point used by the CLI and the MCP tool.

    Args:
        raw: The full-resolution render.
        size: Target edge length in pixels.
        colours: Palette cap; defaults to ``MAX_COLOURS``.

    Returns:
        The finished icon, ready for the gate.
    """
    return finish_icon(raw, (size, size), colours or MAX_COLOURS)
