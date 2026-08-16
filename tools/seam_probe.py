"""Measure how visible the wrap seam is on a horizontally tiling strip.

Ported from ``~/.claude/skills/seamless-tilesets/references/tiletool.py``
(``wrap_discontinuity``, lines 29-55) rather than reinvented, because that
implementation ships calibrated controls: a wrapping sinusoid measures ~1.6 and
a hard half/half split ~31, which is what makes the 4.0 threshold meaningful
instead of arbitrary. The only change is generalising a square ``SIZE``
constant to independent width and height, since a 256x64 strip is 4:1.

The ratio compares the colour difference across the wrap boundary against the
mean difference between adjacent interior columns. Near 1.0 means the seam is
statistically invisible -- indistinguishable from any other pair of neighbouring
columns. Much above 1.0 means a hard edge that will be visible on every frame,
since three instances of a parallax strip are on screen at once.

Vertical is reported too, but a parallax strip does not tile vertically, so only
the horizontal figure gates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Calibrated on the source implementation's controls: ~1.6 for a seamless
# wrapping sinusoid, ~31 for a hard split. 4.0 sits well clear of both.
THRESHOLD = 4.0


def wrap_discontinuity(rgb: np.ndarray) -> tuple[float, float]:
    """Return ``(horizontal, vertical)`` seam energy for one image.

    Args:
        rgb: An ``(h, w, 3)`` float array.

    Returns:
        Horizontal and vertical seam ratios. A ratio is 0.0 when the
        corresponding interior difference is zero, which happens only for a
        uniform image and would otherwise divide by zero.
    """
    seam_h = float(np.abs(rgb[:, -1, :] - rgb[:, 0, :]).mean())
    inner_h = float(np.abs(np.diff(rgb, axis=1)).mean())
    seam_v = float(np.abs(rgb[-1, :, :] - rgb[0, :, :]).mean())
    inner_v = float(np.abs(np.diff(rgb, axis=0)).mean())
    return (
        seam_h / inner_h if inner_h else 0.0,
        seam_v / inner_v if inner_v else 0.0,
    )


def measure(path: Path) -> tuple[float, float]:
    """Measure one PNG on disk.

    Args:
        path: Image to measure.

    Returns:
        Horizontal and vertical seam ratios.
    """
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return wrap_discontinuity(rgb)


def main() -> int:
    """Measure every given strip and report against the threshold.

    Returns:
        0 if every strip is under the threshold, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strips", type=Path, nargs="+", help="PNGs to measure")
    parser.add_argument(
        "--out", type=Path, default=None, help="also write results here"
    )
    args = parser.parse_args()

    lines = [f"threshold {THRESHOLD}", ""]
    worst = 0.0
    for path in sorted(args.strips):
        horizontal, vertical = measure(path)
        worst = max(worst, horizontal)
        verdict = "PASS" if horizontal < THRESHOLD else "FAIL"
        lines.append(f"{verdict}  h={horizontal:6.2f}  v={vertical:6.2f}  {path.name}")

    lines.extend(["", f"worst horizontal {worst:.2f}"])
    report = "\n".join(lines)
    sys.stdout.write(report + "\n")
    if args.out is not None:
        args.out.write_text(report + "\n", encoding="utf-8")
    return 0 if worst < THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
