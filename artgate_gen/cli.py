"""Command-line entry point: render every requested subject, then finish it.

Kept separate from the pipeline so that argument parsing can be tested without
importing torch.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from artgate_gen.console import emit
from artgate_gen.profiling import RunProfile
from artgate_gen.targets import TARGETS

# Shown in `--size`'s error message on a malformed value.
SIZE_METAVAR = "SIZE|WxH"


def _parse_size(raw: str) -> tuple[int, int]:
    """Parse ``--size``: a bare edge length or an explicit ``WxH``.

    Args:
        raw: ``"64"`` or ``"256x64"``.

    Returns:
        ``(width, height)``.

    Raises:
        ValueError: If ``raw`` matches neither form. argparse turns this into
            a usage error rather than a traceback.
    """
    if "x" in raw:
        width_str, _, height_str = raw.partition("x")
        return (int(width_str), int(height_str))
    edge = int(raw)
    return (edge, edge)


# Shown by `--help`. Kept as its own constant rather than reusing the module
# docstring: the docstring explains the module to a reader of the source, while
# this explains the tool to someone running it.
DESCRIPTION = """Generate game art with SDXL + the pixel-art LoRA, then finish
it to icons.

This is the winning route from the art bake-off (row B / "b_pixel"): SDXL base
1.0 with `nerijs/pixel-art-xl` at 1024px, background-stripped, then downscaled
and quantised by the artgate finisher. It beat procedural pixel drawing, SVG
vector, and Blender 3D on the same twelve subjects.

Two findings from the bake-off are baked into the defaults:

* **64px, not 32px.** SDXL renders at 1024, so a 32px icon is a 32x downscale
  and fine detail (a coin's emblem, a shield's radial pattern) mushes. The same
  source survives at 64px.
* **Border-ring background removal.** Flooding from the four corners with a
  fixed tolerance cannot distinguish a dark object from a dark field. Seeding
  every border pixel and requiring connectivity to the border means an interior
  region that merely matches the background colour is never eaten.
"""


def _parse_subjects(
    parser: argparse.ArgumentParser,
    items: list[str],
) -> list[tuple[str, str]]:
    """Split ``name=prompt`` arguments into pairs.

    Args:
        parser: Used to report a usage error on a malformed pair.
        items: The raw ``name=prompt`` strings.

    Returns:
        The parsed pairs, in order.
    """
    pairs = []
    for item in items:
        if "=" not in item:
            parser.error(f"expected name=prompt, got {item!r}")
        name, _, prompt = item.partition("=")
        pairs.append((name, prompt))
    return pairs


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Split out so the flag contract can be tested without running a generation.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("out", type=Path, help="output directory")
    parser.add_argument(
        "subjects",
        nargs="+",
        help="'name=prompt' pairs, e.g. lantern='a brass lantern with a lit wick'",
    )
    parser.add_argument(
        "--size",
        type=_parse_size,
        default=(64, 64),
        metavar=SIZE_METAVAR,
        help="icon size: an edge length (default 64) or WxH, e.g. 256x64",
    )
    parser.add_argument("--colours", type=int, default=None, help="palette cap")
    parser.add_argument(
        "--candidates",
        type=int,
        default=1,
        help="renders per subject, each with its own seed",
    )
    parser.add_argument("--seed", type=int, default=11, help="base seed")
    parser.add_argument("--keep-raw", action="store_true", help="also save 1024px")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="report where the time went, plus peak GPU/RAM/load",
    )
    parser.add_argument(
        "--profile-json",
        type=Path,
        default=None,
        help="also write the full timing trace to this file",
    )
    return parser


def main() -> None:
    """Generate every requested subject at the requested size."""
    # Resolved here, not imported at module scope: `artgate_gen.pipeline`
    # pulls in torch and diffusers, and deferring it means `--help`, argument
    # parsing and the tests all work without the ML stack installed.
    pipeline = importlib.import_module("artgate_gen.pipeline")

    parser = build_parser()
    args = parser.parse_args()

    pairs = _parse_subjects(parser, args.subjects)

    args.out.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out / "raw"
    if args.keep_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    # An unknown size fails here with a usage error rather than a KeyError,
    # and rather than rendering art no spec can gate.
    target = TARGETS.get(args.size)
    if target is None:
        parser.error(f"no target for size {args.size}")

    with RunProfile() as profile:
        pipe = pipeline.load_pipeline(profile)
        for name, prompt in pairs:
            for c in range(args.candidates):
                seed = args.seed + c
                suffix = "" if args.candidates == 1 else f"_{c}"
                # Timed separately: denoising is GPU-bound and the finisher is a
                # pure-Python flood fill over a megapixel, so lumping them
                # together hides which one actually costs.
                with profile.phase(f"render:{name}{suffix}", seed=seed):
                    image = pipeline.generate(pipe, prompt, seed, target)
                if args.keep_raw:
                    image.save(raw_dir / f"{name}{suffix}.png")
                with profile.phase(f"finish:{name}{suffix}"):
                    colours = args.colours or target.max_colors
                    finished = target.finisher(image, args.size, colours)
                out_path = (args.out / f"{name}{suffix}.png").resolve()
                finished.save(out_path)
                # Absolute file:// URL so the terminal makes it clickable and
                # the icon can be opened without hunting for the output dir.
                emit(
                    f"generated {name}{suffix} (seed {seed}) -> {out_path.as_uri()}",
                )

    if args.profile:
        emit(profile.report())
    if args.profile_json:
        profile.write_json(args.profile_json)
