"""Spec-loading tests.

A malformed spec must raise ``SpecError`` (harness error, exit 2) rather than
producing a permissive spec that silently skips gates -- a spec typo that
turned gates off would be indistinguishable from art that passed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import artgate_gen
from artgate_gen.gate.codes import Code
from artgate_gen.gate.spec import SpecError, TargetSpec, load_spec
from tests.assertions import check, check_equal

VALID = """
name = "pixel_icon"
canvas = [32, 32]
palette = ["#1b1d21", "#ffffff"]
max_colors = 16
alpha_binary = true
scale = 1
min_margin = 1
gates = ["ALPHA_NOT_BINARY", "OFF_PALETTE", "TOO_MANY_COLORS"]
"""


def _write(tmp_path: Path, text: str) -> Path:
    """Write a spec file and return its path.

    Args:
        tmp_path: Pytest temporary directory.
        text: TOML contents.

    Returns:
        The written path.
    """
    path = tmp_path / "spec.toml"
    path.write_text(text, encoding="utf-8")
    return path


class TestValidSpec:
    """A well-formed spec parses into the expected frozen dataclass."""

    def test_fields_round_trip(self, tmp_path: Path) -> None:
        """Every declared field is parsed."""
        spec = load_spec(_write(tmp_path, VALID))
        check_equal(
            actual=spec.name, expected="pixel_icon", message="spec.name == 'pixel_icon'"
        )
        check_equal(
            actual=spec.canvas, expected=(32, 32), message="spec.canvas == (32, 32)"
        )
        check_equal(
            actual=spec.max_colors, expected=16, message="spec.max_colors == 16"
        )
        check(condition=spec.alpha_binary is True, message="spec.alpha_binary is True")
        check_equal(actual=spec.min_margin, expected=1, message="spec.min_margin == 1")
        check(
            condition=(27, 29, 33) in spec.palette,
            message="(27, 29, 33) in spec.palette",
        )

    def test_gate_applicability(self, tmp_path: Path) -> None:
        """``runs`` reports only the declared gates."""
        spec = load_spec(_write(tmp_path, VALID))
        check(
            condition=spec.runs(Code.OFF_PALETTE), message="spec.runs(Code.OFF_PALETTE)"
        )
        check(
            condition=not spec.runs(Code.QUADRANT_WEIGHT),
            message="not spec.runs(Code.QUADRANT_WEIGHT)",
        )

    def test_defaults_when_optional_fields_absent(self, tmp_path: Path) -> None:
        """Optional fields default, for a gate needing no extra config.

        ``SILHOUETTE_LOW_CONTRAST`` is used deliberately: it is the only
        declarable gate with a self-contained threshold, so it exercises the
        defaults without tripping the require-config check.
        """
        body = 'name = "m"\ngates = ["SILHOUETTE_LOW_CONTRAST"]\n'
        spec = load_spec(_write(tmp_path, body))
        check(condition=spec.canvas is None, message="spec.canvas is None")
        check_equal(
            actual=spec.palette,
            expected=frozenset(),
            message="spec.palette == frozenset()",
        )
        check(condition=spec.max_colors is None, message="spec.max_colors is None")
        check(condition=spec.scale is None, message="spec.scale is None")
        check_equal(actual=spec.min_margin, expected=0, message="spec.min_margin == 0")
        check(
            condition=spec.alpha_binary is False, message="spec.alpha_binary is False"
        )

    def test_default_construction_has_no_gates(self) -> None:
        """A bare TargetSpec runs nothing."""
        check(
            condition=not TargetSpec(name="bare").runs(Code.OFF_PALETTE),
            message="not TargetSpec(name='bare').runs(Code.OFF_PALETTE)",
        )

    def test_seam_axes_defaults_to_both(self, tmp_path: Path) -> None:
        """A spec that never mentions seam_axes checks both, unchanged."""
        spec = load_spec(_write(tmp_path, VALID))
        check_equal(
            actual=spec.seam_axes,
            expected=("horizontal", "vertical"),
            message="spec.seam_axes == ('horizontal', 'vertical')",
        )

    def test_seam_axes_can_be_scoped(self, tmp_path: Path) -> None:
        """A spec may declare only the axis it actually tiles on."""
        body = 'name = "m"\ngates = ["SEAM_ENERGY"]\nseam_axes = ["horizontal"]\n'
        spec = load_spec(_write(tmp_path, body))
        check_equal(
            actual=spec.seam_axes,
            expected=("horizontal",),
            message="spec.seam_axes == ('horizontal',)",
        )


class TestMalformedSpec:
    """Every malformed input raises SpecError, never a permissive spec."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """A missing spec file is a harness error."""
        with pytest.raises(SpecError, match="cannot read spec"):
            load_spec(tmp_path / "absent.toml")

    def test_invalid_toml(self, tmp_path: Path) -> None:
        """Unparsable TOML is a harness error."""
        with pytest.raises(SpecError, match="malformed spec"):
            load_spec(_write(tmp_path, "name = [unclosed"))

    @pytest.mark.parametrize("body", ["gates = []", 'name = ""\ngates = []'])
    def test_missing_name(self, tmp_path: Path, body: str) -> None:
        """A blank or absent name is rejected."""
        with pytest.raises(SpecError, match="non-empty 'name'"):
            load_spec(_write(tmp_path, body))

    @pytest.mark.parametrize("gates", ["", "gates = []", 'gates = "nope"'])
    def test_missing_gates(self, tmp_path: Path, gates: str) -> None:
        """A spec with no gates would pass everything, so it is rejected."""
        with pytest.raises(SpecError, match="at least one gate"):
            load_spec(_write(tmp_path, f'name = "m"\n{gates}\n'))

    def test_unknown_gate(self, tmp_path: Path) -> None:
        """A typo in a gate name must not silently disable it."""
        with pytest.raises(SpecError, match="unknown gate"):
            load_spec(_write(tmp_path, 'name = "m"\ngates = ["NOT_A_GATE"]\n'))

    @pytest.mark.parametrize(
        "palette",
        ['palette = "#fff"', 'palette = ["fff"]', 'palette = ["#gggggg"]'],
    )
    def test_bad_palette(self, tmp_path: Path, palette: str) -> None:
        """Palette entries must be '#rrggbb'."""
        body = f'name = "m"\ngates = ["OFF_PALETTE"]\n{palette}\n'
        with pytest.raises(SpecError, match="palette"):
            load_spec(_write(tmp_path, body))

    @pytest.mark.parametrize(
        "canvas",
        ["canvas = 32", "canvas = [32]", "canvas = [0, 32]", 'canvas = ["a","b"]'],
    )
    def test_bad_canvas(self, tmp_path: Path, canvas: str) -> None:
        """Canvas must be a pair of positive integers."""
        body = f'name = "m"\ngates = ["CANVAS_SIZE"]\n{canvas}\n'
        with pytest.raises(SpecError, match="canvas must be"):
            load_spec(_write(tmp_path, body))

    @pytest.mark.parametrize(
        "seam_axes",
        [
            "seam_axes = []",
            'seam_axes = "horizontal"',
            'seam_axes = ["diagonal"]',
        ],
    )
    def test_bad_seam_axes(self, tmp_path: Path, seam_axes: str) -> None:
        """seam_axes must be a non-empty list of horizontal/vertical."""
        body = f'name = "m"\ngates = ["SEAM_ENERGY"]\n{seam_axes}\n'
        with pytest.raises(SpecError, match="seam_axes"):
            load_spec(_write(tmp_path, body))


class TestFailClosed:
    """A declared gate that cannot actually run is a spec fault, not a pass.

    Regression: ``hand_painted.toml`` declared ``SILHOUETTE_LOW_CONTRAST``
    while no implementation existed. The spec parsed, the gate never ran, and
    a featureless image scored exit 0 -- indistinguishable from real art
    passing. All three mechanisms below are now rejected at load time.
    """

    def test_unimplemented_gate_is_rejected(self, tmp_path: Path) -> None:
        """A gate with no implementation cannot be declared."""
        body = 'name = "m"\ngates = ["QUADRANT_WEIGHT"]\n'
        with pytest.raises(SpecError, match="not implemented"):
            load_spec(_write(tmp_path, body))

    def test_result_only_code_is_rejected(self, tmp_path: Path) -> None:
        """A result code is emitted by a gate, never requested by a spec."""
        body = 'name = "m"\ncanvas = [8, 8]\ngates = ["NO_OPAQUE_PIXELS"]\n'
        with pytest.raises(SpecError, match="emitted by gates"):
            load_spec(_write(tmp_path, body))

    @pytest.mark.parametrize(
        ("body", "match"),
        [
            ('gates = ["CANVAS_SIZE"]', "CANVAS_SIZE needs"),
            ('gates = ["TOO_MANY_COLORS"]', "TOO_MANY_COLORS needs"),
            ('gates = ["OFF_PALETTE"]', "OFF_PALETTE needs"),
            ('gates = ["ALPHA_NOT_BINARY"]', "ALPHA_NOT_BINARY needs"),
            ('gates = ["MARGIN_TOO_SMALL"]', "MARGIN_TOO_SMALL needs"),
        ],
    )
    def test_gate_without_config_is_rejected(
        self, tmp_path: Path, body: str, match: str
    ) -> None:
        """Declaring a gate without its config would silently skip it."""
        with pytest.raises(SpecError, match=match):
            load_spec(_write(tmp_path, f'name = "m"\n{body}\n'))

    def test_shipped_styles_load(self) -> None:
        """Every style shipped in the package must itself be valid."""
        styles = Path(artgate_gen.__file__).resolve().parent / "styles"
        loaded = [load_spec(p) for p in sorted(styles.glob("*.toml"))]
        check(condition=len(loaded) > 0, message="at least one style ships")
        check(
            condition=all(spec.gates for spec in loaded),
            message="all(spec.gates for spec in loaded)",
        )


def test_code_str_is_bare_value() -> None:
    """Codes format without an enum prefix, so JSONL stays greppable."""
    check_equal(
        actual=f"{Code.OFF_PALETTE}",
        expected="OFF_PALETTE",
        message="f'{Code.OFF_PALETTE}' == 'OFF_PALETTE'",
    )
