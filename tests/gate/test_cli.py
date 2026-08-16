"""CLI tests, covering all three exit codes.

The three-valued exit code is the contract the whole bake-off rests on, so
each value is asserted explicitly: a harness fault must never be recorded as
an art rejection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from artgate_gen.gate.cli import evaluate, main
from artgate_gen.gate.codes import (
    EXIT_GATE_FAILURE,
    EXIT_HARNESS_ERROR,
    EXIT_PASS,
    Code,
)
from artgate_gen.gate.spec import TargetSpec
from tests.assertions import check, check_equal

if TYPE_CHECKING:
    from pathlib import Path

SPEC = """
name = "t"
canvas = [4, 4]
max_colors = 4
alpha_binary = true
palette = ["#0a141e", "#28323c"]
gates = [
  "ALPHA_NOT_BINARY", "TOO_MANY_COLORS", "OFF_PALETTE",
  "CANVAS_SIZE", "SCALE_NOT_INVARIANT",
]
"""


def _write_png(path: Path, rgb: tuple[int, int, int], size: int = 4) -> Path:
    """Write a uniform opaque PNG.

    Args:
        path: Destination file.
        rgb: The fill colour.
        size: Square edge length.

    Returns:
        The written path.
    """
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, :, :3] = rgb
    arr[:, :, 3] = 255
    Image.fromarray(arr, "RGBA").save(path)
    return path


@pytest.fixture(name="spec_file")
def fixture_spec_file(tmp_path: Path) -> Path:
    """Write the shared test spec.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to the spec.
    """
    path = tmp_path / "t.toml"
    path.write_text(SPEC, encoding="utf-8")
    return path


class TestExitCodes:
    """Each exit code is reachable and means exactly one thing."""

    def test_pass(
        self, tmp_path: Path, spec_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Conforming art exits 0."""
        img = _write_png(tmp_path / "ok.png", (10, 20, 30))
        check_equal(
            actual=main([str(img), "--spec", str(spec_file)]),
            expected=EXIT_PASS,
            message="main([str(img), '--spec', str(spec_file)]) == EXIT_PASS",
        )
        check(
            condition="PASS" in capsys.readouterr().out,
            message="'PASS' in capsys.readouterr().out",
        )

    def test_gate_failure(
        self, tmp_path: Path, spec_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Off-palette art exits 1 and names the code."""
        img = _write_png(tmp_path / "bad.png", (200, 10, 10))
        check_equal(
            actual=main([str(img), "--spec", str(spec_file)]),
            expected=EXIT_GATE_FAILURE,
            message="main([str(img), '--spec', str(spec_file)]) == EXIT_GATE_F...",
        )
        check(
            condition="OFF_PALETTE" in capsys.readouterr().out,
            message="'OFF_PALETTE' in capsys.readouterr().out",
        )

    def test_missing_spec_is_harness_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A missing spec exits 2, not 1."""
        img = _write_png(tmp_path / "x.png", (10, 20, 30))
        code = main([str(img), "--spec", str(tmp_path / "absent.toml")])
        check_equal(
            actual=code,
            expected=EXIT_HARNESS_ERROR,
            message="code == EXIT_HARNESS_ERROR",
        )
        check(
            condition="artgate:" in capsys.readouterr().err,
            message="'artgate:' in capsys.readouterr().err",
        )

    def test_unreadable_image_is_harness_error(
        self, tmp_path: Path, spec_file: Path
    ) -> None:
        """A non-image file exits 2, not 1."""
        junk = tmp_path / "junk.png"
        junk.write_text("not an image", encoding="utf-8")
        code = main([str(junk), "--spec", str(spec_file)])
        check_equal(
            actual=code,
            expected=EXIT_HARNESS_ERROR,
            message="code == EXIT_HARNESS_ERROR",
        )

    def test_json_output(
        self, tmp_path: Path, spec_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """JSON mode emits the machine-readable verdict."""
        img = _write_png(tmp_path / "ok.png", (10, 20, 30))
        main([str(img), "--spec", str(spec_file), "--json"])
        out = capsys.readouterr().out
        check(condition='"exit": 0' in out, message="''exit': 0' in out")
        check(condition='"style": "t"' in out, message="''style': 't'' in out")


class TestEvaluate:
    """Gate selection is driven by the spec, not by hardcoded branching."""

    def test_canvas_mismatch_reported(self) -> None:
        """A wrong-sized canvas is caught."""
        arr = np.zeros((8, 8, 4), dtype=np.uint8)
        arr[:, :, 3] = 255
        spec = TargetSpec(name="s", canvas=(4, 4), gates=frozenset({Code.CANVAS_SIZE}))
        check_equal(
            actual=evaluate(arr, spec),
            expected=[Code.CANVAS_SIZE],
            message="evaluate(arr, spec) == [Code.CANVAS_SIZE]",
        )

    def test_gates_not_declared_are_not_run(self) -> None:
        """An off-palette image passes when the palette gate is not declared."""
        arr = np.zeros((2, 2, 4), dtype=np.uint8)
        arr[:, :, :3] = (99, 99, 99)
        arr[:, :, 3] = 255
        spec = TargetSpec(
            name="s",
            palette=frozenset({(1, 2, 3)}),
            gates=frozenset({Code.CANVAS_SIZE}),
        )
        check_equal(
            actual=evaluate(arr, spec), expected=[], message="evaluate(arr, spec) == []"
        )

    def test_codes_are_deduplicated(self) -> None:
        """A code raised by two gates is reported once."""
        arr = np.zeros((2, 2, 4), dtype=np.uint8)
        spec = TargetSpec(
            name="s",
            max_colors=4,
            palette=frozenset({(1, 2, 3)}),
            gates=frozenset({Code.TOO_MANY_COLORS, Code.OFF_PALETTE}),
        )
        check_equal(
            actual=evaluate(arr, spec),
            expected=[Code.NO_OPAQUE_PIXELS],
            message="evaluate(arr, spec) == [Code.NO_OPAQUE_PIXELS]",
        )

    def test_margin_gate_runs_when_declared(self) -> None:
        """A full-bleed asset fails once the margin gate is declared.

        Regression: twelve diffusion icons were 100% opaque -- square tiles,
        not sprites -- and passed every other gate, because no gate asked
        whether the asset had a silhouette at all.
        """
        arr = np.zeros((8, 8, 4), dtype=np.uint8)
        arr[:, :, 3] = 255
        spec = TargetSpec(
            name="s", min_margin=1, gates=frozenset({Code.MARGIN_TOO_SMALL})
        )
        check_equal(
            actual=evaluate(arr, spec),
            expected=[Code.MARGIN_TOO_SMALL],
            message="evaluate(arr, spec) == [Code.MARGIN_TOO_SMALL]",
        )

    def test_silhouette_gate_runs_when_declared(self) -> None:
        """A near-white asset fails the silhouette gate.

        Regression: ``SILHOUETTE_LOW_CONTRAST`` was declarable in a spec but
        had no implementation, so it parsed cleanly, ran nothing and reported
        a pass. ``load_spec`` now rejects undeclarable gates, and this asserts
        the gate genuinely executes once declared.
        """
        arr = np.zeros((8, 8, 4), dtype=np.uint8)
        arr[:, :, :3] = 250
        arr[:, :, 3] = 255
        spec = TargetSpec(name="s", gates=frozenset({Code.SILHOUETTE_LOW_CONTRAST}))
        check_equal(
            actual=evaluate(arr, spec),
            expected=[Code.SILHOUETTE_LOW_CONTRAST],
            message="evaluate(arr, spec) == [Code.SILHOUETTE_LOW_CONTRAST]",
        )
