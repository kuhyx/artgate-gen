"""Command line for the set-consistency metric.

Split from ``set_consistency.py`` to keep both files under the repository's
250-line cap, and along the seam the repository already uses elsewhere
(``repair.py`` / ``repair_cli.py``): the metric is importable without dragging
in argument parsing, and the CLI holds nothing but presentation.

The defaults describe a dark, saturated, blue-dominant garment in the lower
half of a portrait, because that is the case the metric was built against. They
are defaults, not assumptions -- every bound is a flag, and `--debug-sheet`
exists so the selection is confirmed by eye before its numbers are believed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from artgate_gen.set_consistency import (
    RegionSpec,
    channel_spread,
    debug_sheet,
    measure_set,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--lead", type=int, nargs=2, default=[2, 0])
    parser.add_argument("--lead-margin", type=float, default=12.0)
    parser.add_argument("--min-saturation", type=float, default=0.18)
    parser.add_argument("--max-saturation", type=float, default=1.0)
    parser.add_argument("--min-value", type=float, default=0.0)
    parser.add_argument("--max-value", type=float, default=140.0)
    parser.add_argument("--top", type=float, default=0.5)
    parser.add_argument("--bottom", type=float, default=1.0)
    parser.add_argument("--limit", type=float, default=10.0)
    parser.add_argument("--name", default="region")
    parser.add_argument("--debug-sheet", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Measure a set from the command line.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        0 when the spread is within ``--limit``, 1 otherwise.
    """
    args = build_parser().parse_args(argv)
    region = RegionSpec(
        name=args.name,
        channel_lead=(args.lead[0], args.lead[1]),
        lead_margin=args.lead_margin,
        min_saturation=args.min_saturation,
        max_saturation=args.max_saturation,
        min_value=args.min_value,
        max_value=args.max_value,
        top=args.top,
        bottom=args.bottom,
    )

    paths = sorted(args.paths)
    colours, widths = measure_set(paths, region)
    lines = [f"region {region.name!r} across {len(paths)} renders", ""]
    for path, colour in zip(paths, colours, strict=True):
        r, g, b = colour
        lines.append(f"  ({r:6.1f},{g:6.1f},{b:6.1f})  {path.name}")

    spread = channel_spread(colours)
    worst = max(spread)
    verdict = "PASS" if worst <= args.limit else "FAIL"
    lines.extend(
        [
            "",
            f"spread r={spread[0]:.1f} g={spread[1]:.1f} b={spread[2]:.1f}",
            f"width spread {max(widths) - min(widths):.3f}",
            f"{verdict}  worst {worst:.1f} against limit {args.limit}",
        ]
    )
    sys.stdout.write("\n".join(lines) + "\n")

    if args.debug_sheet is not None:
        written = debug_sheet(paths, region, args.debug_sheet)
        sys.stdout.write(f"debug sheet -> {written}\n")

    return 0 if worst <= args.limit else 1


if __name__ == "__main__":
    raise SystemExit(main())
