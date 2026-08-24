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

import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from mcp.server.fastmcp import FastMCP
from PIL import Image
from pydantic import Field

from artgate_gen.gating import gate
from artgate_gen.generation import generate_icons as _generate_icons
from artgate_gen.generation import state
from artgate_gen.unload import IDLE_UNLOAD_SECONDS, start_unloader

mcp = FastMCP("artgate")

# Alpha at or above this counts a pixel as opaque when describing an icon.
OPAQUE_CUTOFF = 127

# Registered here rather than decorated at its definition, so the module that
# owns the pipeline does not have to import the server.
generate_icons = mcp.tool()(_generate_icons)


@mcp.tool()
def gate_icon(
    path: Annotated[str, Field(description="Path to a PNG to check.")],
    size: Annotated[
        int | list[int],
        Field(
            description=(
                "Which spec to check against: 64 or 32, or [width, height] "
                "for a non-square target such as [256, 64]. Any other value "
                "returns gated=false rather than a verdict."
            ),
        ),
    ] = 64,
) -> dict[str, Any]:
    """Run the deterministic artgate gates against an existing icon.

    Needs no model and no GPU, so it works in a base install.

    Args:
        path: Path to a PNG.
        size: Which spec to check against -- 64 or 32, or ``[width, height]``
            for a non-square target such as ``[256, 64]``.

    Returns:
        Whether it passed, and any failure codes.

    """
    canvas = size if isinstance(size, int) else (size[0], size[1])
    return gate(Path(path), canvas)


@mcp.tool()
def describe_icon(
    path: Annotated[str, Field(description="Path to a PNG to measure.")],
) -> dict[str, Any]:
    """Report an icon's measurable properties without loading the model.

    Useful for checking what a file actually is before regenerating it.

    Args:
        path: Path to a PNG.

    Returns:
        Size, opaque coverage and distinct opaque colour count.

    """
    # numpy rather than `Image.getdata()`, which Pillow deprecated for removal
    # in 14, and which materialises a Python tuple per pixel besides. Opened
    # under a context manager so the file handle is not left dangling.
    with Image.open(path) as handle:
        rgba = np.asarray(handle.convert("RGBA"), dtype=np.uint8)
        size = list(handle.size)

    flat = rgba.reshape(-1, 4)
    opaque = flat[flat[:, 3] > OPAQUE_CUTOFF]
    distinct = np.unique(opaque[:, :3], axis=0) if opaque.size else opaque
    return {
        "size": size,
        "opaque_fraction": round(len(opaque) / len(flat), 4),
        "distinct_colours": int(distinct.shape[0]),
    }


@mcp.tool()
def model_status() -> dict[str, Any]:
    """Report whether the SDXL pipeline is currently resident in VRAM.

    Returns:
        Residency, idle time, and the configured unload window.

    """
    with state.lock:
        loaded = state.pipe is not None
        idle = round(time.time() - state.last_used) if state.last_used else None
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
    start_unloader(state)
    mcp.run()
