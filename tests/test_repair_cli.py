"""Tests for the repair command-line entry point.

The verdict parser has to survive the gate's stdout carrying warning lines
around the JSON, which is why it scans rather than parsing the whole stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from artgate_gen.repair_cli import main, parse_codes
from tests.assertions import check_equal, check_length

if TYPE_CHECKING:
    from pathlib import Path

EDGE = 32
EXIT_OK = 0
EXIT_NO_REPAIR = 1

SPEC = """
name = "t"
canvas = [32, 32]
max_colors = 16
alpha_binary = true
min_margin = 2
gates = ["CANVAS_SIZE", "TOO_MANY_COLORS", "ALPHA_NOT_BINARY", "MARGIN_TOO_SMALL"]
"""


def test_parse_codes_reads_the_json_line() -> None:
    """A clean verdict yields its codes."""
    check_equal(
        actual=parse_codes('{"codes": ["TOO_MANY_COLORS"], "exit": 1}'),
        expected=["TOO_MANY_COLORS"],
        message="codes from a clean verdict",
    )


def test_parse_codes_skips_surrounding_noise() -> None:
    """Warning lines around the JSON are ignored."""
    raw = 'warning: something\n{"codes": ["CANVAS_SIZE"], "exit": 1}\ntrailing\n'
    check_equal(
        actual=parse_codes(raw),
        expected=["CANVAS_SIZE"],
        message="codes despite surrounding noise",
    )


def test_parse_codes_skips_an_unparseable_json_line() -> None:
    """A line that starts like JSON but is malformed does not abort the scan."""
    raw = '{not json\n{"codes": ["ALPHA_NOT_BINARY"], "exit": 1}\n'
    check_equal(
        actual=parse_codes(raw),
        expected=["ALPHA_NOT_BINARY"],
        message="scanning continues past a malformed line",
    )


def test_parse_codes_returns_empty_when_there_is_no_verdict() -> None:
    """Output with no JSON at all yields no codes."""
    check_length(sized=parse_codes("no verdict here\n"), expected=0, message="no codes")


def test_main_repairs_and_reports(tmp_path: Path) -> None:
    """A repairable verdict drives a repair and reports success."""
    spec = tmp_path / "spec.toml"
    spec.write_text(SPEC, encoding="utf-8")
    icon = tmp_path / "icon.png"
    arr = np.zeros((EDGE, EDGE, 4), dtype=np.uint8)
    arr[0:8, 0:8] = (200, 40, 40, 255)
    Image.fromarray(arr, "RGBA").save(icon)

    code = main(
        [
            str(icon),
            "--spec",
            str(spec),
            "--codes",
            '{"codes": ["MARGIN_TOO_SMALL"], "exit": 1}',
        ],
    )
    check_equal(actual=code, expected=EXIT_OK, message="repair reports success")


def test_main_declines_an_unrepairable_verdict(tmp_path: Path) -> None:
    """An unrepairable verdict still exits cleanly, having changed nothing."""
    spec = tmp_path / "spec.toml"
    spec.write_text(SPEC, encoding="utf-8")
    icon = tmp_path / "icon.png"
    arr = np.zeros((EDGE, EDGE, 4), dtype=np.uint8)
    arr[0:8, 0:8] = (200, 40, 40, 255)
    Image.fromarray(arr, "RGBA").save(icon)

    code = main(
        [
            str(icon),
            "--spec",
            str(spec),
            "--codes",
            '{"codes": ["SILHOUETTE_LOW_CONTRAST"], "exit": 1}',
        ],
    )
    check_equal(
        actual=code,
        expected=EXIT_NO_REPAIR,
        message="no repair was written",
    )
