"""What each output size *is*: resolution, prompt, palette and finisher.

This table exists so that adding an output kind is a data change, not an
if-chain. The gate layer already treats gate applicability as data -- a spec
declares its gates -- and this is the same idea one stage earlier: the
difference between an item icon and a parallax backdrop is a row here, not a
branch inside ``generate`` or ``finish``.

The finisher is a **callable per target**, deliberately not a ``strip_bg: bool``
flag. A backdrop must skip the background strip because ``strip_background``
floods inward from every border and a full-bleed sky is within tolerance of the
border median -- measured, it destroyed 47% of a synthetic backdrop (opaque
1.000 -> 0.531) while correctly isolating a centred icon (1.000 -> 0.141). One
boolean would express that, but the second such difference would produce an
if-chain inside the finisher; a callable does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from artgate_gen.finish import finish_backdrop, finish_icon, finish_portrait

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from PIL import Image

# Verbatim from the bake-off run that produced the accepted icons. This string
# is asserted character-for-character by the test suite: it is the only reason
# the 31 icons in `art/` are reproducible, and an "improvement" to it silently
# invalidates every accepted result.
ICON_STYLE = "pixel art, game item icon, centered, plain black background, crisp"
ICON_NEGATIVE = (
    "photo, realistic, blurry, soft focus, gradient background, text, "
    "watermark, multiple objects, hands, drop shadow"
)

# The backdrop stem drops "game item icon", "centered" and "plain black
# background" -- all three describe a single object on a void, which is the
# opposite of a full-bleed landscape.
BACKDROP_STYLE = "pixel art, side-scrolling parallax background, wide landscape, crisp"
# Two deliberate differences from ICON_NEGATIVE:
#   - "multiple objects" is DROPPED. It is actively wrong for a landscape, which
#     is nothing but multiple objects, and duplicated motifs are the known SDXL
#     failure mode at extreme aspect ratios -- suppressing the phrase there
#     fights the wrong battle.
#   - "black bars, letterbox, border, frame" is ADDED. A 4.5:1 request pushes
#     SDXL toward composing a normal scene and padding it, so letterboxing is
#     attacked at the prompt level as well as by the gate.
BACKDROP_NEGATIVE = (
    "photo, realistic, blurry, soft focus, text, watermark, hands, "
    "drop shadow, black bars, letterbox, border, frame"
)

# Verbatim from `art/MANIFEST.json`'s accepted `category: "portraits"` entries
# (e.g. `neutral_s11`) in roadside-assistance -- the recipe that survived
# session 1b/1c/1d/1e's identity-consistency work. Only the reusable stem and
# negative live here; the subject-specific expression/pose clause is per-call,
# same as ICON_STYLE and BACKDROP_STYLE.
PORTRAIT_STYLE = (
    "pixel art, close-up character portrait, head and shoulders, "
    "three-quarter view, crisp"
)
PORTRAIT_NEGATIVE = (
    "photo, realistic, blurry, text, watermark, multiple heads, deformed "
    "face, shoulder strap, harness, backpack, hood, scarf, green jacket, "
    "olive, khaki, army green, camouflage, teal, white jacket, white coat, "
    "lab coat, full body, wide shot, standing, hands, distant"
)


@dataclass(frozen=True, slots=True)
class Target:
    """One output kind: what to render, how big, and how to finish it.

    Attributes:
        spec: Style spec filename under ``artgate_gen/styles``.
        render: ``(width, height)`` handed to the diffusion pipeline.
        canvas: ``(width, height)`` of the finished art.
        style: Prompt stem prepended to the subject.
        negative: Negative prompt.
        max_colors: Palette cap for the finisher.
        finisher: Turns one render into finished art at ``canvas``.
    """

    spec: str
    render: tuple[int, int]
    canvas: tuple[int, int]
    style: str
    negative: str
    max_colors: int
    finisher: Callable[[Image.Image, tuple[int, int], int], Image.Image]


# Keyed by finished canvas, so a non-square target is expressible at all -- the
# int-keyed `SPECS` in gating.py cannot represent (256, 64).
#
# Backdrops render at 2560x512 rather than the 2048x512 they finish to: the
# extra 512px is the wrap overlap that `wrap_blend` folds back in, which is what
# makes horizontal tiling possible by construction rather than by luck.
# 2048/256 = 512/64 = 8 makes the reduction a uniform integer factor on both
# axes.
#
# THE OVERLAP IS 512 BECAUSE 256 WAS MEASURED AND FAILED. Over nine real
# renders (three prompts x three seeds) a 256px overlap left worst-case seam
# energy at 6.01 against a threshold of 4.0 -- one strip in nine, and the
# failure was not visible by eye on the strip itself. Widening to 512 took the
# same nine to a worst case of 1.42 and improved eight of them:
#
#     strip              ov=256   ov=512
#     desert-dusk_s1       6.01     0.46   <- the failure
#     desert-storm_s1      2.52     0.62
#     desert-dusk_s2       2.64     0.79
#     WORST                6.01     1.42
#
# The residual at 256 was a local step at the seam, not a global left-right
# brightness gap: the outer 64px columns of the failing render averaged
# (181,107,86) and (189,110,86), close enough that no amount of ramp width was
# fighting a real luminance difference. A narrow ramp simply cannot absorb a
# sharp feature that lands near the wrap.
TARGETS: Final[dict[tuple[int, int], Target]] = {
    (32, 32): Target(
        spec="pixel_icon.toml",
        render=(1024, 1024),
        canvas=(32, 32),
        style=ICON_STYLE,
        negative=ICON_NEGATIVE,
        max_colors=32,
        finisher=finish_icon,
    ),
    (64, 64): Target(
        spec="pixel_icon_64.toml",
        render=(1024, 1024),
        canvas=(64, 64),
        style=ICON_STYLE,
        negative=ICON_NEGATIVE,
        max_colors=32,
        finisher=finish_icon,
    ),
    (256, 64): Target(
        spec="backdrop_strip.toml",
        render=(2560, 512),
        canvas=(256, 64),
        style=BACKDROP_STYLE,
        negative=BACKDROP_NEGATIVE,
        max_colors=32,
        finisher=finish_backdrop,
    ),
    # max_colors=0 means "skip quantisation", matching the accepted portraits'
    # own `max_colors: 0` in art/MANIFEST.json -- see finish_portrait and
    # portrait.toml for why introducing a cap now is a decision this project
    # explicitly defers, not an oversight.
    (256, 256): Target(
        spec="portrait.toml",
        render=(1024, 1024),
        canvas=(256, 256),
        style=PORTRAIT_STYLE,
        negative=PORTRAIT_NEGATIVE,
        max_colors=0,
        finisher=finish_portrait,
    ),
}
