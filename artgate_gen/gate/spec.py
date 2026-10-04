"""Target specifications: which gates apply, and with what thresholds.

Gate applicability is *data*, not an if-chain. Palette conformance and
scale-invariance are meaningless for hand-painted work, so running them there
would manufacture false failures; a per-style spec makes that a one-line
declaration instead of branching logic scattered through the gates.

Per-field TOML parsing lives in `spec_fields.py`, split out to keep this file
under the 250-line cap.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from artgate_gen.gate.codes import Code
from artgate_gen.gate.spec_error import SpecError
from artgate_gen.gate.spec_fields import (
    parse_canvas,
    parse_gates,
    parse_palette,
    parse_seam_axes,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["SpecError", "TargetSpec", "load_spec"]


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
        seam_axes: Which axes SEAM_ENERGY checks, when declared. Defaults to
            both -- a spec for anything that tiles on only one axis (e.g.
            `backdrop_strip`, which never tiles top-to-bottom) opts out of
            the other explicitly rather than silently failing on it.
    """

    name: str
    canvas: tuple[int, int] | None = None
    palette: frozenset[tuple[int, int, int]] = frozenset()
    max_colors: int | None = None
    alpha_binary: bool = False
    scale: int | None = None
    min_margin: int = 0
    gates: frozenset[Code] = field(default_factory=frozenset)
    seam_axes: tuple[str, ...] = ("horizontal", "vertical")

    def runs(self, code: Code) -> bool:
        """Report whether a gate applies to this style.

        Args:
            code: The gate's reason code.

        Returns:
            True if the gate should be evaluated for this spec.
        """
        return code in self.gates


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
        canvas=parse_canvas(data.get("canvas")),
        palette=parse_palette(data.get("palette")),
        max_colors=data.get("max_colors"),
        alpha_binary=bool(data.get("alpha_binary", False)),
        scale=data.get("scale"),
        min_margin=int(data.get("min_margin", 0)),
        gates=parse_gates(data.get("gates")),
        seam_axes=parse_seam_axes(data.get("seam_axes")),
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
