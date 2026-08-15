"""MCP server exposing the artgate diffusion route as tools.

Wraps `generate.py` so icon generation is a tool call instead of a shell
command, and so the gate result comes back with the art in one round trip.

**VRAM policy.** Unlike the Blender host, this does NOT keep its model warm.
SDXL holds ~7 GB, and the same GPU runs local LLM work; a permanently resident
pipeline would squat on that. The pipeline is loaded on first use and dropped
after `IDLE_UNLOAD_SECONDS` of inactivity. First call after an unload pays the
~40 s load again, which is why `generate_icons` takes a list of subjects --
batch them into one call rather than calling once per icon.

Adjudication stays deterministic: this server shells out to the artgate CLI and
reports its exit code. It never judges art itself.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

import generate
from mcp.server.fastmcp import FastMCP
from PIL import Image

mcp = FastMCP("artgate")

ARTGATE_REPO = Path("/home/kuhy/testsAndMisc")
SPECS = {
    32: "python_pkg/artgate/styles/pixel_icon.toml",
    64: "python_pkg/artgate/styles/pixel_icon_64.toml",
}
IDLE_UNLOAD_SECONDS = 600

_pipe: Any = None
_last_used = 0.0
_lock = threading.Lock()


def _unload_if_idle() -> None:
    """Drop the pipeline once it has been unused for the idle window."""
    global _pipe
    while True:
        time.sleep(60)
        with _lock:
            if _pipe is not None and time.time() - _last_used > IDLE_UNLOAD_SECONDS:
                _pipe = None
                import torch

                torch.cuda.empty_cache()


threading.Thread(target=_unload_if_idle, daemon=True).start()


def _pipeline() -> Any:
    """Return the loaded pipeline, loading it if necessary."""
    global _pipe, _last_used
    with _lock:
        if _pipe is None:
            _pipe = generate.load_pipeline()
        _last_used = time.time()
        return _pipe


def _gate(path: Path, size: int) -> dict[str, Any]:
    """Run the deterministic artgate CLI against one icon."""
    spec = SPECS.get(size)
    if spec is None:
        return {"gated": False, "reason": f"no spec for size {size}"}
    result = subprocess.run(
        ["python", "-m", "python_pkg.artgate.cli", str(path), "--spec", spec, "--json"],
        cwd=ARTGATE_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"gated": False, "reason": result.stderr.strip()[:200]}
    return {
        "gated": True,
        "passed": payload.get("exit") == 0,
        "codes": payload.get("codes", []),
    }


@mcp.tool()
def generate_icons(
    out_dir: str,
    subjects: dict[str, str],
    size: int = 64,
    candidates: int = 1,
    seed: int = 11,
    keep_raw: bool = False,
) -> dict[str, Any]:
    """Generate item icons with SDXL + the pixel-art LoRA, then gate them.

    Args:
        out_dir: Directory to write icons into; created if absent.
        subjects: Map of icon name to a description of the object, e.g.
            ``{"lantern": "a brass lantern with a lit wick"}``.
        size: Icon edge in pixels. 64 (default, recommended) or 32.
        candidates: Renders per subject, each with its own seed. The hit rate
            is not 1.0 -- ask for 3 when quality matters and pick.
        seed: Base seed; candidate *n* uses ``seed + n``.
        keep_raw: Also save the 1024px render, which is often better than any
            downscale of it.

    Returns:
        Per-icon paths plus each icon's deterministic gate result.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "raw"
    if keep_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    pipe = _pipeline()
    results = []
    for name, prompt in subjects.items():
        for c in range(candidates):
            image = generate.generate(pipe, prompt, seed + c)
            suffix = "" if candidates == 1 else f"_{c}"
            if keep_raw:
                image.save(raw_dir / f"{name}{suffix}.png")
            path = out / f"{name}{suffix}.png"
            generate.finish(image, size).save(path)
            results.append(
                {
                    "name": f"{name}{suffix}",
                    "path": str(path),
                    "seed": seed + c,
                    **_gate(path, size),
                },
            )

    passed = sum(1 for r in results if r.get("passed"))
    return {
        "icons": results,
        "summary": f"{passed}/{len(results)} passed the {size}px gate",
        "note": "Gates check canvas/palette/alpha/margin/contrast, NOT whether "
        "the art looks good. Look at the images before reporting done.",
    }


@mcp.tool()
def gate_icon(path: str, size: int = 64) -> dict[str, Any]:
    """Run the deterministic artgate gates against an existing icon.

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
        Size, mode, opaque coverage and distinct opaque colour count.
    """
    image = Image.open(path).convert("RGBA")
    pixels = list(image.getdata())
    opaque = [p for p in pixels if p[3] > 127]
    return {
        "size": list(image.size),
        "opaque_fraction": round(len(opaque) / len(pixels), 4),
        "distinct_colours": len({p[:3] for p in opaque}),
    }


@mcp.tool()
def model_status() -> dict[str, Any]:
    """Report whether the SDXL pipeline is currently resident in VRAM."""
    with _lock:
        loaded = _pipe is not None
        idle = round(time.time() - _last_used) if _last_used else None
    return {
        "loaded": loaded,
        "idle_seconds": idle,
        "unloads_after_seconds": IDLE_UNLOAD_SECONDS,
        "note": "Not kept warm on purpose -- SDXL holds ~7GB and the GPU is "
        "shared with local LLM work. First call after an unload pays "
        "a ~40s load, so batch subjects into one generate_icons call.",
    }


if __name__ == "__main__":
    mcp.run()
