"""The one MCP tool that loads the diffusion model.

Kept apart from the model-free tools so the server module stays small and the
seam between "needs a GPU" and "does not" is visible in the file layout.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from artgate_gen.gating import gate
from artgate_gen.targets import TARGETS
from artgate_gen.unload import PipelineState

# Module-level rather than private: the tools share one pipeline, and the
# tests need to inspect and reset residency without reaching into internals.
state = PipelineState()


def pipeline() -> object:
    """Return the loaded pipeline, loading it on first use.

    Returns:
        The diffusion pipeline.

    """
    with state.lock:
        if state.pipe is None:
            module = importlib.import_module("artgate_gen.pipeline")
            state.pipe = module.load_pipeline()
        state.last_used = time.time()
        return state.pipe


def generate_icons(
    out_dir: Annotated[
        str,
        Field(description="Directory to write icons into; created if absent."),
    ],
    subjects: Annotated[
        dict[str, str],
        Field(
            description=(
                "Map of icon name to a description of the object, e.g. "
                '{"lantern": "a brass lantern with a lit wick"}. Batch every '
                "subject into one call: the model is not kept warm, so a "
                "second call may pay a ~40s reload."
            ),
        ),
    ],
    size: Annotated[
        int | list[int],
        Field(
            description=(
                "Icon edge in pixels (64, recommended, or 32), or [width, "
                "height] for a non-square target, e.g. [256, 64] for a "
                "parallax backdrop strip."
            ),
        ),
    ] = 64,
    candidates: Annotated[
        int,
        Field(
            description=(
                "Renders per subject, each with its own seed. The hit rate is "
                "not 1.0 -- ask for 3 when quality matters and pick."
            ),
        ),
    ] = 1,
    seed: Annotated[
        int,
        Field(description="Base seed; candidate n uses seed + n."),
    ] = 11,
    *,
    keep_raw: Annotated[
        bool,
        Field(
            description=(
                "Also save the 1024px render, which is often better than any "
                "downscale of it."
            ),
        ),
    ] = False,
) -> dict[str, Any]:
    """Generate item icons with SDXL plus the pixel-art LoRA, then gate them.

    Args:
        out_dir: Directory to write icons into; created if absent.
        subjects: Map of icon name to a description of the object, e.g.
            ``{"lantern": "a brass lantern with a lit wick"}``.
        size: Icon edge in pixels. 64 (default, recommended) or 32, or
            ``[width, height]`` for a non-square target such as ``[256, 64]``.
        candidates: Renders per subject, each with its own seed. The hit rate
            is not 1.0 -- ask for 3 when quality matters and pick.
        seed: Base seed; candidate *n* uses ``seed + n``.
        keep_raw: Also save the full-resolution render, which is often better
            than any downscale of it.

    Returns:
        Per-icon paths plus each icon's deterministic gate result.

    """
    module = importlib.import_module("artgate_gen.pipeline")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "raw"
    if keep_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    canvas = (size, size) if isinstance(size, int) else (size[0], size[1])

    # Derived from `size` rather than taken as a parameter: `generate_icons`
    # is already at the repo's six-argument limit, and those six are the MCP
    # tool's published schema. An unsupported size returns an error rather than
    # raising KeyError at the MCP boundary.
    target = TARGETS.get(canvas)
    if target is None:
        return {
            "icons": [],
            "summary": f"no target for size {canvas}; supported sizes are "
            f"{sorted(TARGETS)}",
        }

    pipe = pipeline()
    results = []
    for name, prompt in subjects.items():
        for candidate in range(candidates):
            image = module.generate(pipe, prompt, seed + candidate, target)
            suffix = "" if candidates == 1 else f"_{candidate}"
            if keep_raw:
                image.save(raw_dir / f"{name}{suffix}.png")
            path = out / f"{name}{suffix}.png"
            target.finisher(image, canvas, target.max_colors).save(path)
            results.append(
                {
                    "name": f"{name}{suffix}",
                    "path": str(path),
                    "seed": seed + candidate,
                    **gate(path, canvas),
                },
            )

    passed = sum(1 for result in results if result.get("passed"))
    return {
        "icons": results,
        "summary": f"{passed}/{len(results)} passed the {canvas} gate",
        "note": "Gates check canvas/palette/alpha/margin/contrast, NOT whether "
        "the art looks good. Look at the images before reporting done.",
    }
