"""Print the spec TOML filename for a ``--size`` value, or nothing on a miss.

Used by ``run.sh`` so the shell script does not duplicate ``TARGETS``.
"""

from __future__ import annotations

import sys

from artgate_gen.console import emit
from artgate_gen.targets import TARGETS


def _parse_size(raw: str) -> tuple[int, int] | None:
    """Parse a ``run.sh --size`` value: a bare edge length or ``WxH``.

    Args:
        raw: ``"64"`` or ``"256x64"``.

    Returns:
        ``(width, height)``, or ``None`` if ``raw`` is not a valid size.
    """
    try:
        if "x" in raw:
            width_str, _, height_str = raw.partition("x")
            parsed = (int(width_str), int(height_str))
        else:
            edge = int(raw)
            parsed = (edge, edge)
    except ValueError:
        return None
    else:
        return parsed


def main() -> None:
    """Print the resolved spec filename, or nothing if the size is unknown."""
    canvas = _parse_size(sys.argv[1])
    target = TARGETS.get(canvas) if canvas is not None else None
    if target is not None:
        emit(target.spec)


if __name__ == "__main__":
    main()
