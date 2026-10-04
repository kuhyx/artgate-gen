"""Per-field TOML parsers for `TargetSpec`, split out of `spec.py`.

Split to respect the 250-line file cap; `spec.py` grew past it once
`seam_axes` parsing was added. Each function here validates and converts one
raw TOML value into the type `TargetSpec` expects, raising `SpecError` (never
constructing a permissive default) on anything malformed.
"""

from __future__ import annotations

from artgate_gen.gate.codes import DECLARABLE, RESULT_ONLY, Code
from artgate_gen.gate.spec_error import SpecError

# A canvas is declared as exactly [width, height].
_CANVAS_DIMS = 2

# Length of a '#rrggbb' colour literal.
_HEX_RGB_LEN = 7

_VALID_SEAM_AXES = frozenset({"horizontal", "vertical"})


def parse_palette(raw: object) -> frozenset[tuple[int, int, int]]:
    """Parse a list of ``#rrggbb`` strings into RGB triples.

    Args:
        raw: The ``palette`` value from a spec file.

    Returns:
        The parsed colours, empty if unset.

    Raises:
        SpecError: If an entry is not a ``#rrggbb`` hex string.
    """
    if raw is None:
        return frozenset()
    if not isinstance(raw, list):
        msg = "palette must be a list of '#rrggbb' strings"
        raise SpecError(msg)
    out: set[tuple[int, int, int]] = set()
    for entry in raw:
        malformed = (
            not isinstance(entry, str)
            or not entry.startswith("#")
            or len(entry) != _HEX_RGB_LEN
        )
        if malformed:
            msg = f"bad palette entry {entry!r}: expected '#rrggbb'"
            raise SpecError(msg)
        try:
            value = int(entry[1:], 16)
        except ValueError as exc:
            msg = f"bad palette entry {entry!r}: not hexadecimal"
            raise SpecError(msg) from exc
        out.add(((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF))
    return frozenset(out)


def parse_gates(raw: object) -> frozenset[Code]:
    """Parse the list of applicable gate names.

    Args:
        raw: The ``gates`` value from a spec file.

    Returns:
        The applicable gate codes.

    Raises:
        SpecError: If ``gates`` is missing, empty, or names an unknown gate.
    """
    if not isinstance(raw, list) or not raw:
        msg = "spec must list at least one gate under 'gates'"
        raise SpecError(msg)
    known = {c.value for c in Code}
    unknown = [g for g in raw if g not in known]
    if unknown:
        msg = f"unknown gate(s): {sorted(map(str, unknown))}"
        raise SpecError(msg)
    codes = frozenset(Code(g) for g in raw)

    result_only = codes & RESULT_ONLY
    if result_only:
        names = sorted(str(c) for c in result_only)
        msg = f"{names} are emitted by gates, not declarable in 'gates'"
        raise SpecError(msg)

    unimplemented = codes - DECLARABLE
    if unimplemented:
        names = sorted(str(c) for c in unimplemented)
        msg = f"gate(s) {names} are declared but not implemented"
        raise SpecError(msg)
    return codes


def parse_canvas(raw: object) -> tuple[int, int] | None:
    """Parse the required canvas size.

    Args:
        raw: The ``canvas`` value from a spec file.

    Returns:
        The ``(width, height)`` pair, or None if unset.

    Raises:
        SpecError: If present but not a pair of positive integers.
    """
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != _CANVAS_DIMS:
        msg = "canvas must be [width, height] positive integers"
        raise SpecError(msg)
    width, height = raw
    valid = all(isinstance(v, int) and v > 0 for v in (width, height))
    if not valid:
        msg = "canvas must be [width, height] positive integers"
        raise SpecError(msg)
    return (int(width), int(height))


def parse_seam_axes(raw: object) -> tuple[str, ...]:
    """Parse which axes SEAM_ENERGY should check.

    Args:
        raw: The ``seam_axes`` value from a spec file.

    Returns:
        The declared axes, defaulting to both when unset.

    Raises:
        SpecError: If present but not a list of ``"horizontal"``/``"vertical"``.
    """
    if raw is None:
        return ("horizontal", "vertical")
    if not isinstance(raw, list) or not raw or not set(raw) <= _VALID_SEAM_AXES:
        msg = "seam_axes must be a non-empty list of 'horizontal'/'vertical'"
        raise SpecError(msg)
    return tuple(raw)
