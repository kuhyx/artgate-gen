"""Render the session-1 probes: nine backdrop strips and nine portraits.

This is a one-off driver for the milestone-0 go/no-go, not part of the package.
It answers two questions that only real output can settle:

1. Does a wrap-blended SDXL strip land under the 4.0 seam threshold? Raw output
   does not -- measured 5.94 and 255.00 on realistic non-tiling renders.
2. Can this pipeline resolve a face at all? ``art/README.md`` records that a
   test cat lost its eyes and tongue at 64px, so the portrait is shown at 64,
   128 and 256 to find where expression survives.

The portrait probe deliberately runs through NEITHER finisher: it box-downscales
the raw render and stops. ``finish_icon`` would strip the background and
binarise alpha, ``finish_backdrop`` would force opacity, and quantising to 32
colours would itself eat facial detail -- any of which would make the verdict
about the finisher rather than about resolution.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from artgate_gen import pipeline
from artgate_gen.console import emit
from artgate_gen.gate.finish_ops import downscale_to
from artgate_gen.targets import TARGETS

if TYPE_CHECKING:
    from artgate_gen._types import Pipeline

ROOT = Path(__file__).resolve().parent.parent
RAW_BACKDROP = ROOT / "art" / "raw-backdrop"
FINISHED = ROOT / "art" / "backdrops-256x64"
PORTRAITS = ROOT / "art" / "portrait-probe"

SEEDS = (0, 1, 2)

# One prompt family, three phrasings. Varying the seed answers "is wrap-blend
# robust to generator variance"; varying the phrasing answers "is it robust to
# content". Both matter, because a seam that is content-sensitive cannot be
# declared satisfiable by construction.
BACKDROPS = {
    "desert-night": (
        "empty desert highway at night, distant mesas, starry sky, telephone poles"
    ),
    "desert-dusk": (
        "desert highway at dusk, orange sky, saguaro cacti, distant mountains"
    ),
    "desert-storm": (
        "desert highway under storm clouds, lightning, dark rocky buttes, rain"
    ),
}

# The portrait is the game's selling point, so it gets the same 3x3 grid. The
# expression clause is the only thing that varies within a seed -- that is the
# identity question: a fixed seed fixes initial noise, not a character.
PORTRAIT_STEM = "anime girl, portrait, short white hair, headset, blue jacket"
PORTRAIT_EXPRESSIONS = {
    "neutral": "neutral expression, looking at viewer",
    "smiling": "smiling warmly, happy expression",
    "angry": "angry, frowning, furrowed brow",
}
PORTRAIT_SIZES = (64, 128, 256)


def _render_backdrops(pipe: Pipeline) -> None:
    """Render, wrap-blend and finish nine strips.

    Args:
        pipe: The loaded diffusion pipeline.
    """
    target = TARGETS[256, 64]
    for name, prompt in BACKDROPS.items():
        for seed in SEEDS:
            stem = f"{name}_s{seed}"
            image = pipeline.generate(pipe, prompt, seed, target)
            image.save(RAW_BACKDROP / f"{stem}.png")
            target.finisher(image, target.canvas, target.max_colors).save(
                FINISHED / f"{stem}.png"
            )
            emit(f"backdrop {stem} {image.size} -> {target.canvas}")


def _render_portraits(pipe: Pipeline) -> None:
    """Render nine portraits, each saved at three sizes, unquantised.

    Args:
        pipe: The loaded diffusion pipeline.
    """
    # A square 1024 render with the icon stem replaced: this is a probe of what
    # the generator resolves, so it must not inherit "game item icon, centered,
    # plain black background", which describes an object rather than a face.
    base = TARGETS[64, 64]
    target = replace(
        base,
        style="pixel art, character portrait, front facing, crisp",
        negative=(
            "photo, realistic, blurry, soft focus, text, watermark, "
            "multiple heads, deformed face, extra limbs"
        ),
    )
    for name, clause in PORTRAIT_EXPRESSIONS.items():
        for seed in SEEDS:
            stem = f"{name}_s{seed}"
            image = pipeline.generate(pipe, f"{PORTRAIT_STEM}, {clause}", seed, target)
            image.save(PORTRAITS / f"{stem}_1024.png")
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            for size in PORTRAIT_SIZES:
                small = downscale_to(rgba, (size, size))
                Image.fromarray(small, "RGBA").save(PORTRAITS / f"{stem}_{size}.png")
            emit(f"portrait {stem} -> {PORTRAIT_SIZES}")


def main() -> int:
    """Render every probe.

    Returns:
        0 on success.
    """
    for directory in (RAW_BACKDROP, FINISHED, PORTRAITS):
        directory.mkdir(parents=True, exist_ok=True)

    pipe = pipeline.load_pipeline()
    _render_backdrops(pipe)
    _render_portraits(pipe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
