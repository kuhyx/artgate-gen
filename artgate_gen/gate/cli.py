"""Command-line entry point: image + spec in, exit code and JSON out.

Adjudication is by exit code, never by a model judging a model. Three values,
because conflating a harness fault with an art rejection would convert our own
bugs into art verdicts and poison the bake-off metrics:

* 0 -- every applicable gate passed
* 1 -- the asset is well-formed but failed at least one gate
* 2 -- the spec or the image could not be read
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, UnidentifiedImageError

from artgate_gen.gate.backdrop import (
    check_fully_opaque,
    check_letterbox,
    check_seam_energy,
    check_vertical_structure,
)
from artgate_gen.gate.codes import (
    EXIT_GATE_FAILURE,
    EXIT_HARNESS_ERROR,
    EXIT_PASS,
    Code,
)
from artgate_gen.gate.contrast import check_margin, check_silhouette
from artgate_gen.gate.pixels import (
    check_alpha_binary,
    check_color_count,
    check_palette,
    check_scale_invariance,
)
from artgate_gen.gate.spec import SpecError, TargetSpec, load_spec

if TYPE_CHECKING:
    from collections.abc import Callable


def _check_canvas(rgba: np.ndarray, spec: TargetSpec) -> list[Code]:
    """Verify the image matches the spec's exact canvas size.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.
        spec: The target specification.

    Returns:
        ``[CANVAS_SIZE]`` on a mismatch, else empty.
    """
    if spec.canvas is None:
        return []
    height, width = rgba.shape[:2]
    if (width, height) != spec.canvas:
        return [Code.CANVAS_SIZE]
    return []


def evaluate(rgba: np.ndarray, spec: TargetSpec) -> list[Code]:
    """Run every gate the spec declares applicable.

    Gate order matters: alpha binarity is evaluated before the colour count,
    because a soft-alpha image has no fully-opaque pixels and would otherwise
    yield a colour count of zero and pass.

    Args:
        rgba: An ``(h, w, 4)`` uint8 array.
        spec: The target specification.

    Returns:
        The failing reason codes, empty if the asset passes.
    """
    # A table rather than an if-chain: adding a gate is one entry, and the
    # ORDER is explicit and load-bearing -- alpha must precede the colour
    # count, or a soft-alpha image yields zero opaque pixels and passes.
    checks: list[tuple[Code, Callable[[], list[Code]]]] = [
        (Code.CANVAS_SIZE, lambda: _check_canvas(rgba, spec)),
        (Code.ALPHA_NOT_BINARY, lambda: check_alpha_binary(rgba)),
        (
            Code.TOO_MANY_COLORS,
            lambda: check_color_count(rgba, spec.max_colors or 0),
        ),
        (Code.OFF_PALETTE, lambda: check_palette(rgba, spec.palette)),
        (
            Code.SCALE_NOT_INVARIANT,
            lambda: check_scale_invariance(rgba, spec.scale),
        ),
        (Code.SILHOUETTE_LOW_CONTRAST, lambda: check_silhouette(rgba)),
        (Code.MARGIN_TOO_SMALL, lambda: check_margin(rgba, spec.min_margin)),
        # Backdrop-only gates. FULLY_OPAQUE_REQUIRED comes first of this
        # block: the luma-based checks below it are meaningless on an image
        # with no opaque pixels, the same reason ALPHA_NOT_BINARY precedes
        # TOO_MANY_COLORS above.
        (Code.FULLY_OPAQUE_REQUIRED, lambda: check_fully_opaque(rgba)),
        (Code.LETTERBOX_BARS, lambda: check_letterbox(rgba)),
        (Code.NO_VERTICAL_STRUCTURE, lambda: check_vertical_structure(rgba)),
        (Code.SEAM_ENERGY, lambda: check_seam_energy(rgba, spec.seam_axes)),
    ]

    failures: list[Code] = []
    for code, run in checks:
        if spec.runs(code):
            failures.extend(run())

    deduped: list[Code] = []
    for code in failures:
        if code not in deduped:
            deduped.append(code)
    return deduped


def _load_image(path: Path) -> np.ndarray:
    """Read an image as an RGBA uint8 array.

    Args:
        path: Path to the image.

    Returns:
        An ``(h, w, 4)`` uint8 array.

    Raises:
        SpecError: If the file is missing or not a readable image.
    """
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGBA"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        msg = f"cannot read image {path}: {exc}"
        raise SpecError(msg) from exc


def main(argv: list[str] | None = None) -> int:
    """Gate one image against one spec.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="artgate",
        description="Gate generated game art deterministically.",
    )
    parser.add_argument("image", type=Path, help="PNG to check")
    parser.add_argument("--spec", type=Path, required=True, help="TOML spec")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable verdict on stdout",
    )
    args = parser.parse_args(argv)

    try:
        spec = load_spec(args.spec)
        rgba = _load_image(args.image)
    except SpecError as exc:
        sys.stderr.write(f"artgate: {exc}\n")
        return EXIT_HARNESS_ERROR

    codes = evaluate(rgba, spec)
    status = EXIT_PASS if not codes else EXIT_GATE_FAILURE

    if args.json:
        verdict = {
            "image": str(args.image),
            "style": spec.name,
            "exit": status,
            "codes": [str(c) for c in codes],
        }
        sys.stdout.write(f"{json.dumps(verdict, sort_keys=True)}\n")
    else:
        label = "PASS" if status == EXIT_PASS else "FAIL"
        detail = "" if not codes else " " + ",".join(str(c) for c in codes)
        sys.stdout.write(f"{label} {args.image}{detail}\n")

    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
