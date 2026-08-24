"""Tests for the `run.sh` size-to-spec resolver.

Exists so `run.sh` never re-derives `TARGETS` in bash: this is the one place
that mapping is duplicated as a lookup, and it must agree with `targets.py`
by construction, not by two authors keeping two files in sync by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.assertions import check_equal
from tools.spec_for_size import _parse_size, main

if TYPE_CHECKING:
    import pytest


def test_parse_size_accepts_a_bare_edge_length() -> None:
    """A plain integer is a square target."""
    check_equal(actual=_parse_size("64"), expected=(64, 64), message="bare edge length")


def test_parse_size_accepts_an_explicit_wxh() -> None:
    """`WxH` selects a non-square target."""
    check_equal(actual=_parse_size("256x64"), expected=(256, 64), message="WxH form")


def test_parse_size_returns_none_for_a_malformed_value() -> None:
    """A value that is neither an int nor `WxH` is not a crash."""
    check_equal(actual=_parse_size("banana"), expected=None, message="malformed size")


def test_main_prints_the_spec_for_a_known_size(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A size with a TARGETS entry prints its spec filename."""
    monkeypatch.setattr("sys.argv", ["spec_for_size", "64"])
    main()
    check_equal(
        actual=capsys.readouterr().out.strip(),
        expected="pixel_icon_64.toml",
        message="the resolved spec filename",
    )


def test_main_prints_the_spec_for_a_non_square_size(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`WxH` also resolves, for the backdrop target."""
    monkeypatch.setattr("sys.argv", ["spec_for_size", "256x64"])
    main()
    check_equal(
        actual=capsys.readouterr().out.strip(),
        expected="backdrop_strip.toml",
        message="the resolved backdrop spec filename",
    )


def test_main_prints_nothing_for_an_unknown_size(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A size with no target prints nothing, so `run.sh`'s `[[ -z ]]` fires."""
    monkeypatch.setattr("sys.argv", ["spec_for_size", "48"])
    main()
    check_equal(actual=capsys.readouterr().out, expected="", message="no output")


def test_main_prints_nothing_for_a_malformed_size(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed value is also a clean no-output miss, not a traceback."""
    monkeypatch.setattr("sys.argv", ["spec_for_size", "banana"])
    main()
    check_equal(actual=capsys.readouterr().out, expected="", message="no output")
