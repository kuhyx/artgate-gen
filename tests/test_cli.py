"""Tests for the generation command-line entry point.

The flag contract is the part that matters most: `run.sh` and the `item-icons`
skill both invoke this by path, so a renamed or dropped flag breaks callers
outside this repository.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from artgate_gen.cli import _parse_size, build_parser, main
from tests.assertions import check, check_equal

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_FLAGS = {
    "backend",
    "out",
    "subjects",
    "size",
    "colours",
    "candidates",
    "seed",
    "keep_raw",
    "profile",
    "profile_json",
    "help",
}

DEFAULT_SIZE = (64, 64)
DEFAULT_SEED = 11


def test_parser_exposes_the_published_flags() -> None:
    """Every flag callers depend on still parses.

    Asserted through the public parse path rather than by inspecting the
    parser's internals, so this tests the contract a caller actually sees.
    """
    args = build_parser().parse_args(
        [
            "out",
            "axe=an axe",
            "--size",
            "32",
            "--colours",
            "8",
            "--candidates",
            "2",
            "--seed",
            "5",
            "--keep-raw",
            "--profile",
            "--profile-json",
            "trace.json",
        ],
    )
    parsed = {key for key in vars(args) if not key.startswith("_")}
    check_equal(
        actual=parsed,
        expected=EXPECTED_FLAGS - {"help"},
        message="the published flag set",
    )


def test_parser_defaults_match_the_documented_ones() -> None:
    """Defaults are part of the contract, not an implementation detail."""
    args = build_parser().parse_args(["out", "axe=an axe"])
    check_equal(actual=args.size, expected=DEFAULT_SIZE, message="default size")
    check_equal(actual=args.seed, expected=DEFAULT_SEED, message="default seed")
    check_equal(actual=args.candidates, expected=1, message="default candidates")
    check(condition=args.colours is None, message="no palette cap by default")
    check(condition=not args.keep_raw, message="raw renders are not kept by default")


def test_malformed_subject_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subject without '=' is a usage error, not a silent skip."""
    monkeypatch.setattr(sys, "argv", ["generate.py", str(tmp_path), "nopair"])
    with pytest.raises(SystemExit):
        main()


def test_parse_size_accepts_a_bare_edge_length() -> None:
    """A plain integer is a square target."""
    check_equal(actual=_parse_size("64"), expected=(64, 64), message="bare edge length")


def test_parse_size_accepts_an_explicit_wxh() -> None:
    """`WxH` selects a non-square target."""
    check_equal(actual=_parse_size("256x64"), expected=(256, 64), message="WxH form")


def test_generation_at_a_non_square_size_uses_the_backdrop_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--size 256x64` resolves the real (256, 64) backdrop target."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate.py", str(tmp_path), "dunes=a desert dune line", "--size", "256x64"],
    )
    main()
    with Image.open(tmp_path / "dunes.png") as icon:
        check_equal(
            actual=icon.size,
            expected=(256, 64),
            message="the finished backdrop's canvas size",
        )


def test_unsupported_size_is_a_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A size with no target fails cleanly instead of raising KeyError.

    Before the target table this silently produced art at a size no spec could
    gate, which is worse than refusing: it looks like a successful run.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate.py", str(tmp_path), "axe=a stone axe", "--size", "48"],
    )
    with pytest.raises(SystemExit):
        main()
