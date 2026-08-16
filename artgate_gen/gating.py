"""The in-process gate used by both the MCP server and the CLI shim.

Split from the server module so that the model-free half -- checking and
describing icons -- stays independent of anything that touches the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from artgate_gen.gate.cli import evaluate
from artgate_gen.gate.spec import load_spec

# Icon edge length to the style spec that governs it.
SPECS = {
    32: "pixel_icon.toml",
    64: "pixel_icon_64.toml",
}

STYLES_DIR = Path(__file__).resolve().parent / "styles"


def gate(path: Path, size: int) -> dict[str, Any]:
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
