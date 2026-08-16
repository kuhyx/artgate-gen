"""Tests for the terminal output helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from artgate_gen.console import emit, emit_error
from tests.assertions import check_equal

if TYPE_CHECKING:
    import pytest


def test_emit_writes_a_line_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """A message reaches stdout with a trailing newline."""
    emit("hello")
    check_equal(actual=capsys.readouterr().out, expected="hello\n", message="stdout")


def test_emit_with_no_argument_writes_a_blank_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Calling with no message still terminates the line."""
    emit()
    check_equal(actual=capsys.readouterr().out, expected="\n", message="blank line")


def test_emit_error_writes_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    """Errors go to stderr, not stdout, so they survive a stdout redirect."""
    emit_error("bad")
    captured = capsys.readouterr()
    check_equal(actual=captured.err, expected="bad\n", message="stderr")
    check_equal(actual=captured.out, expected="", message="stdout stays clean")
