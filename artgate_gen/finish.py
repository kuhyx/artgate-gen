"""Reduce a continuous-tone render to a gate-passing pixel icon.

Order is load-bearing and mirrors the gate's own order: the background goes
first, then the downscale, then alpha is binarised, and only then is the
palette quantised. Quantising earlier would blend the arbitrary RGB carried by
transparent pixels into the palette and inflate the colour count -- the exact
bug that once produced 18 colours against a cap of 16.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from artgate_gen.background import strip_background
from artgate_gen.gate.finish_ops import binarize_alpha, downscale, quantize

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Both icon specs cap the palette at 32, and the number is calibrated against
# real art rather than chosen: 250 genuine 32x32 icons top out at 30 colours and
# 300 genuine 48x48 icons top out at 14. Hand-made art does not spend more
# colours on a bigger canvas, so this cap does NOT scale with `size`. Deriving
# it from the canvas instead produced 64-colour icons that failed
# TOO_MANY_COLORS on all 18 subjects.
MAX_COLOURS = 32


def finish(raw: Image.Image, size: int, colours: int | None = None) -> Image.Image:
    """Strip the background and reduce a render to a square icon.

    Args:
        raw: The full-resolution render.
        size: Target edge length in pixels.
        colours: Palette cap; defaults to ``MAX_COLOURS``.

    Returns:
        The finished icon, ready for the gate.
    """
    arr: NDArray[np.uint8] = np.asarray(raw.convert("RGBA"), dtype=np.uint8)
    stripped = strip_background(arr)
    reduced = quantize(
        binarize_alpha(downscale(stripped, size)),
        colours or MAX_COLOURS,
    )
    return Image.fromarray(reduced, "RGBA")
