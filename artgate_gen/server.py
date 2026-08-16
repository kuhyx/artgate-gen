"""MCP server exposing the artgate diffusion route as tools.

Icon generation becomes a tool call rather than a shell command, and the gate
verdict comes back with the art in one round trip.

**VRAM policy.** The model is not kept warm. SDXL holds roughly 7 GB and the
same GPU runs local LLM work, so the pipeline loads on first use and is dropped
after an idle window. The first call after an unload pays the load again, which
is why ``generate_icons`` takes a batch of subjects -- send them in one call
rather than one call per icon.

**Adjudication stays deterministic.** The gate runs in-process here, but it is
the same vendored implementation the CLI uses and it never judges whether art
looks good: it measures canvas, palette, alpha, scale, contrast and margin.
Looking at the images is still the caller's job.

Importing this module has no side effects. The idle watcher starts in
:func:`main`, so tests and introspection tools can import it freely.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP
from PIL import Image

from artgate_gen.finish import finish
from artgate_gen.gate.cli import evaluate
from artgate_gen.gate.spec import load_spec
from artgate_gen.unload import (
    IDLE_UNLOAD_SECONDS,
    PipelineState,
    start_unloader,
)

mcp = FastMCP("artgate")

# Icon edge length to the style spec that governs it.
SPECS = {
    32: "pixel_icon.toml",
    64: "pixel_icon_64.toml",
}

STYLES_DIR = Path(__file__).resolve().parent / "styles"

# Alpha at or above this counts a pixel as opaque when describing an icon.
OPAQUE_CUTOFF = 127

_state = PipelineState()


def _pipeline() -> object:
    """Return the loaded pipeline, loading it on first use.

    Returns:
        The diffusion pipeline.

    """
    with _state.lock:
        if _state.pipe is None:
            pipeline = importlib.import_module("artgate_gen.pipeline")
            _state.pipe = pipeline.load_pipeline()
        _state.last_used = time.time()
        return _state.pipe


def _gate(path: Path, size: int) -> dict[str, Any]:
    """Run the deterministic gate against one icon.

    Args:
        path: Path to the PNG to check.
        size: Icon edge length, selecting the spec.

    Returns:
        Whether the gate ran, whether it passed, and any failure codes.

    """
    spec_name = SPECS.get(size)
    if spec_name is None:
        return {"gated": False, "reason": f"no spec for size {size}"}
    spec = load_spec(STYLES_DIR / spec_name)
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    codes = evaluate(rgba, spec)
    return {
        "gated": True,
        "passed": not codes,
        "codes": [str(code) for code in codes],
    }


@mcp.tool()
def generate_icons(
    out_dir: str,
    subjects: dict[str, str],
    size: int = 64,
    candidates: int = 1,
    seed: int = 11,
    *,
    keep_raw: bool = False,
) -> dict[str, Any]:
    """Generate item icons with SDXL plus the pixel-art LoRA, then gate them.

    Args:
        out_dir: Directory to write icons into; created if absent.
        subjects: Map of icon name to a description of the object, e.g.
            ``{"lantern": "a brass lantern with a lit wick"}``.
        size: Icon edge in pixels. 64 (default, recommended) or 32.
        candidates: Renders per subject, each with its own seed. The hit rate
            is not 1.0 -- ask for 3 when quality matters and pick.
        seed: Base seed; candidate *n* uses ``seed + n``.
        keep_raw: Also save the full-resolution render, which is often better
            than any downscale of it.

    Returns:
        Per-icon paths plus each icon's deterministic gate result.

    """
    pipeline = importlib.import_module("artgate_gen.pipeline")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "raw"
    if keep_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    pipe = _pipeline()
    results = []
    for name, prompt in subjects.items():
        for candidate in range(candidates):
            image = pipeline.generate(pipe, prompt, seed + candidate)
            suffix = "" if candidates == 1 else f"_{candidate}"
            if keep_raw:
                image.save(raw_dir / f"{name}{suffix}.png")
            path = out / f"{name}{suffix}.png"
            finish(image, size).save(path)
            results.append(
                {
                    "name": f"{name}{suffix}",
                    "path": str(path),
                    "seed": seed + candidate,
                    **_gate(path, size),
                },
            )

    passed = sum(1 for result in results if result.get("passed"))
    return {
        "icons": results,
        "summary": f"{passed}/{len(results)} passed the {size}px gate",
        "note": "Gates check canvas/palette/alpha/margin/contrast, NOT whether "
        "the art looks good. Look at the images before reporting done.",
    }


@mcp.tool()
def gate_icon(path: str, size: int = 64) -> dict[str, Any]:
    """Run the deterministic artgate gates against an existing icon.

    Needs no model and no GPU, so it works in a base install.

    Args:
        path: Path to a PNG.
        size: Which spec to check against -- 64 or 32.

    Returns:
        Whether it passed, and any failure codes.

    """
    return _gate(Path(path), size)


@mcp.tool()
def describe_icon(path: str) -> dict[str, Any]:
    """Report an icon's measurable properties without loading the model.

    Useful for checking what a file actually is before regenerating it.

    Args:
        path: Path to a PNG.

    Returns:
        Size, opaque coverage and distinct opaque colour count.

    """
    image = Image.open(path).convert("RGBA")
    pixels = list(image.getdata())
    opaque = [p for p in pixels if p[3] > OPAQUE_CUTOFF]
    return {
        "size": list(image.size),
        "opaque_fraction": round(len(opaque) / len(pixels), 4),
        "distinct_colours": len({p[:3] for p in opaque}),
    }


@mcp.tool()
def model_status() -> dict[str, Any]:
    """Report whether the SDXL pipeline is currently resident in VRAM.

    Returns:
        Residency, idle time, and the configured unload window.

    """
    with _state.lock:
        loaded = _state.pipe is not None
        idle = round(time.time() - _state.last_used) if _state.last_used else None
    return {
        "loaded": loaded,
        "idle_seconds": idle,
        "unloads_after_seconds": IDLE_UNLOAD_SECONDS,
        "note": "Not kept warm on purpose -- SDXL holds ~7GB and the GPU is "
        "shared with local LLM work. First call after an unload pays "
        "a ~40s load, so batch subjects into one generate_icons call.",
    }


def main() -> None:
    """Start the idle watcher and serve over stdio."""
    start_unloader(_state)
    mcp.run()
