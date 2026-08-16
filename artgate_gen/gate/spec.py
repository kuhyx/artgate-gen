"""Target specifications: which gates apply, and with what thresholds.

Gate applicability is *data*, not an if-chain. Palette conformance and
scale-invariance are meaningless for hand-painted work, so running them there
would manufacture false failures; a per-style spec makes that a one-line
declaration instead of branching logic scattered through the gates.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from artgate_gen.gate.codes import DECLARABLE, RESULT_ONLY, Code

if TYPE_CHECKING:
    from pathlib import Path

# A canvas is declared as exactly [width, height].
_CANVAS_DIMS = 2

# Length of a '#rrggbb' colour literal.
_HEX_RGB_LEN = 7


class SpecError(ValueError):
    """Raised when a spec file is missing, malformed, or self-contradictory.

    Distinct from a gate failure: this maps to the harness-error exit code,
    never to an art verdict.
    """


@dataclass(frozen=True)
class TargetSpec:
    """A frozen description of what a passing asset looks like.

    Attributes:
        name: Human-readable style name, used in reports and contact sheets.
        canvas: Required exact ``(width, height)``, or ``None`` to skip.
        palette: Allowed opaque RGB triples; empty means the gate is skipped.
        max_colors: Cap on distinct opaque RGB values, or ``None`` to skip.
        alpha_binary: Whether alpha must be exactly ``{0, 255}``.
        scale: Declared integer upscale factor, or ``None`` to auto-detect.
        min_margin: Required transparent border in pixels.
        gates: The applicable gate codes; anything absent is not run.
    """

    name: str
    canvas: tuple[int, int] | None = None
    palette: frozenset[tuple[int, int, int]] = frozenset()
    max_colors: int | None = None
    alpha_binary: bool = False
    scale: int | None = None
    min_margin: int = 0
    gates: frozenset[Code] = field(default_factory=frozenset)

    def runs(self, code: Code) -> bool:
        """Report whether a gate applies to this style.

        Args:
            code: The gate's reason code.

        Returns:
            True if the gate should be evaluated for this spec.
        """
        return code in self.gates


def _parse_palette(raw: object) -> frozenset[tuple[int, int, int]]:
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


def _parse_gates(raw: object) -> frozenset[Code]:
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


def _parse_canvas(raw: object) -> tuple[int, int] | None:
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


def load_spec(path: Path) -> TargetSpec:
    """Load and validate a TOML target spec.

    Args:
        path: Path to the spec file.

    Returns:
        The parsed spec.

    Raises:
        SpecError: If the file is missing or malformed.
    """
    try:
        data: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"cannot read spec {path}: {exc}"
        raise SpecError(msg) from exc
    except tomllib.TOMLDecodeError as exc:
        msg = f"malformed spec {path}: {exc}"
        raise SpecError(msg) from exc

    name = data.get("name")
    if not isinstance(name, str) or not name:
        msg = "spec must define a non-empty 'name'"
        raise SpecError(msg)

    spec = TargetSpec(
        name=name,
        canvas=_parse_canvas(data.get("canvas")),
        palette=_parse_palette(data.get("palette")),
        max_colors=data.get("max_colors"),
        alpha_binary=bool(data.get("alpha_binary", False)),
        scale=data.get("scale"),
        min_margin=int(data.get("min_margin", 0)),
        gates=_parse_gates(data.get("gates")),
    )
    _require_config(spec)
    return spec


def _require_config(spec: TargetSpec) -> None:
    """Reject a spec that declares a gate without the config it needs.

    Declaring ``TOO_MANY_COLORS`` without ``max_colors`` would silently skip
    the gate at evaluation time and report a pass, which is indistinguishable
    from art that genuinely passed. Failing here maps to the harness-error
    exit code instead -- a spec fault, never an art verdict.

    Args:
        spec: The freshly parsed spec.

    Raises:
        SpecError: If a declared gate has no usable configuration.
    """
    missing: list[str] = []
    if spec.runs(Code.CANVAS_SIZE) and spec.canvas is None:
        missing.append("CANVAS_SIZE needs 'canvas'")
    if spec.runs(Code.TOO_MANY_COLORS) and spec.max_colors is None:
        missing.append("TOO_MANY_COLORS needs 'max_colors'")
    if spec.runs(Code.OFF_PALETTE) and not spec.palette:
        missing.append("OFF_PALETTE needs a non-empty 'palette'")
    if spec.runs(Code.ALPHA_NOT_BINARY) and not spec.alpha_binary:
        missing.append("ALPHA_NOT_BINARY needs 'alpha_binary = true'")
    if spec.runs(Code.MARGIN_TOO_SMALL) and spec.min_margin <= 0:
        missing.append("MARGIN_TOO_SMALL needs a positive 'min_margin'")
    if missing:
        msg = "; ".join(missing)
        raise SpecError(msg)
