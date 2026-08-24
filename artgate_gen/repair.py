"""Best-effort repair of an icon that failed the artgate gates.

Scope, deliberately narrow: this fixes *mechanical* violations -- too many
colours, soft alpha, a subject touching the canvas edge -- because those are
properties of the encoding, not of the art. It cannot fix
``SILHOUETTE_LOW_CONTRAST`` (the subject genuinely does not separate from its
background) and it must not try; an icon that fails that gate needs a different
render, not a different quantiser.

The repair is a single re-finish, never a per-code patch. Patching one code in
isolation breaks others: insetting the subject to clear the margin band resamples
the image, which re-introduces continuous alpha and new colours, so an icon
"repaired" for MARGIN_TOO_SMALL would come back failing ALPHA_NOT_BINARY and
TOO_MANY_COLORS. Applying the geometric change first and then re-running the
whole finisher chain -- the same one generate.py uses -- keeps every gate
satisfied simultaneously.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from artgate_gen.gate.codes import Code
from artgate_gen.gate.finish_ops import downscale_to, quantize
from artgate_gen.gate.finish_ops import finish as finish_square
from artgate_gen.gate.spec import load_spec

if TYPE_CHECKING:
    from pathlib import Path

# Repairable failures. Anything else -- notably SILHOUETTE_LOW_CONTRAST and
# EMPTY_CANVAS -- is a fact about the art that re-encoding cannot change.
REPAIRABLE = frozenset(
    {"TOO_MANY_COLORS", "ALPHA_NOT_BINARY", "MARGIN_TOO_SMALL", "CANVAS_SIZE"}
)


def opaque_bbox(rgba: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the ``(top, left, bottom, right)`` bounds of the opaque subject.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.

    Returns:
        Inclusive-exclusive bounds, or ``None`` when nothing is opaque.
    """
    rows = np.any(rgba[:, :, 3] > 0, axis=1)
    cols = np.any(rgba[:, :, 3] > 0, axis=0)
    if not rows.any() or not cols.any():
        return None
    top, bottom = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    left, right = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    return top, left, bottom, right


def inset_subject(rgba: np.ndarray, size: int, margin: int) -> np.ndarray:
    """Rescale the subject so a clear border of ``margin`` px surrounds it.

    The subject is cropped to its opaque bounds, scaled to fit the box left over
    once the margin is reserved on all four sides, and re-centred. Aspect ratio
    is preserved -- squashing an icon to fit is a worse outcome than the failure.

    Args:
        rgba: An ``(h, w, 4)`` uint8 source.
        size: Target square canvas edge.
        margin: Required clear border width in pixels.

    Returns:
        A ``(size, size, 4)`` uint8 array, or the input unchanged when the
        canvas has no opaque pixels to move.
    """
    bounds = opaque_bbox(rgba)
    if bounds is None:
        return rgba
    top, left, bottom, right = bounds

    # One extra pixel of inset beyond the required margin. The gate tests the
    # band *after* downscaling and alpha-binarisation, and a subject placed
    # exactly on the boundary can bleed back into it when those run.
    inner = size - 2 * (margin + 1)
    if inner < 1:
        return rgba

    subject = Image.fromarray(rgba[top:bottom, left:right], "RGBA")
    scale = min(inner / subject.width, inner / subject.height)
    scaled = subject.resize(
        (max(1, round(subject.width * scale)), max(1, round(subject.height * scale))),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(
        scaled,
        ((size - scaled.width) // 2, (size - scaled.height) // 2),
    )
    return np.asarray(canvas, dtype=np.uint8)


def repair(path: Path, spec_path: Path, codes: list[str]) -> bool:
    """Rewrite ``path`` in place so it satisfies ``spec_path``.

    Args:
        path: The failing icon; overwritten only when a repair is attempted.
        spec_path: The artgate spec the icon is gated against.
        codes: The gate codes the icon failed.

    Returns:
        ``True`` when a repair was written. The caller must re-gate to learn
        whether it actually worked -- this function never asserts success.
    """
    if not codes or not set(codes) <= REPAIRABLE:
        return False

    spec = load_spec(spec_path)
    # Opened once, under a context manager: reading `.width` from a second
    # unclosed handle leaked a file descriptor per repair.
    with Image.open(path) as image:
        source = image.convert("RGBA")
        canvas = spec.canvas or (image.width, image.width)
    rgba = np.asarray(source, dtype=np.uint8)
    width, _height = canvas

    # Only when the spec DECLARES MARGIN_TOO_SMALL, not merely when the icon
    # failed it: a backdrop never declares this gate (a full-bleed strip has
    # no margin to clear), so this guard is what stops a backdrop entering
    # CANVAS_SIZE repair from being rebuilt as a square by inset_subject --
    # width != height, so inset_subject's square canvas would be wrong anyway.
    if ("MARGIN_TOO_SMALL" in codes or "CANVAS_SIZE" in codes) and spec.runs(
        Code.MARGIN_TOO_SMALL
    ):
        rgba = inset_subject(rgba, width, spec.min_margin)

    # Square vs non-square, not TARGETS membership: repair operates on an
    # already-finished image, never an oversized render, so finish_backdrop's
    # wrap-blend is the wrong tool here -- it demands overlap width beyond the
    # canvas, which a correctly-sized strip never has, and crashes. A strip
    # never declares SEAM_ENERGY as repairable, so there is no seam to redo;
    # only quantise, resize, and force full opacity, matching what
    # finish_backdrop guarantees without re-blending.
    if canvas[0] == canvas[1]:
        fixed = finish_square(rgba, width, spec.max_colors or 32)
    else:
        reduced = quantize(downscale_to(rgba, canvas), spec.max_colors or 32)
        reduced[:, :, 3] = 255
        fixed = reduced
    Image.fromarray(fixed, "RGBA").save(path)
    return True
