"""Tests for the generation command-line entry point.

The flag contract is the part that matters most: `run.sh` and the `item-icons`
skill both invoke this by path, so a renamed or dropped flag breaks callers
outside this repository.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from artgate_gen.cli import build_parser, main
from tests.assertions import check, check_equal, check_length

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_FLAGS = {
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

DEFAULT_SIZE = 64
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


def test_generation_writes_and_gates_an_icon(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full run renders, finishes and reports a clickable path."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate.py", str(tmp_path), "axe=a stone axe", "--size", "32"],
    )
    main()
    out = capsys.readouterr().out
    check(condition="generated axe" in out, message="reports the generated icon")
    check(condition="file://" in out, message="reports a clickable URL")
    check(condition=(tmp_path / "axe.png").exists(), message="the icon was written")


def test_multiple_candidates_are_suffixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking for several candidates writes one file per seed."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            str(tmp_path),
            "axe=a stone axe",
            "--size",
            "32",
            "--candidates",
            "2",
        ],
    )
    main()
    written = sorted(p.name for p in tmp_path.glob("axe*.png"))
    check_equal(
        actual=written,
        expected=["axe_0.png", "axe_1.png"],
        message="candidates are suffixed",
    )


def test_keep_raw_saves_the_full_resolution_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The raw render is kept alongside the finished icon when asked."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            str(tmp_path),
            "axe=a stone axe",
            "--size",
            "32",
            "--keep-raw",
        ],
    )
    main()
    check(
        condition=(tmp_path / "raw" / "axe.png").exists(),
        message="the raw render was kept",
    )


def test_profile_flag_reports_timings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The profile flag adds a timing report to stdout."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            str(tmp_path),
            "axe=a stone axe",
            "--size",
            "32",
            "--profile",
        ],
    )
    main()
    check(condition="TOTAL" in capsys.readouterr().out, message="timings are reported")


def test_profile_json_writes_a_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trace file is written when a path is given."""
    trace = tmp_path / "trace.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            str(tmp_path),
            "axe=a stone axe",
            "--size",
            "32",
            "--profile-json",
            str(trace),
        ],
    )
    main()
    check(condition=trace.exists(), message="the trace file was written")


def test_subject_parsing_splits_on_the_first_equals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prompt containing '=' keeps everything after the first separator."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate.py", str(tmp_path), "sign=a sign reading x=y", "--size", "32"],
    )
    main()
    check_length(
        sized=list(tmp_path.glob("sign.png")),
        expected=1,
        message="the subject name is everything before the first '='",
    )
