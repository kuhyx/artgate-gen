"""Tests for the fetch command-line entry point.

``main`` has no branches of its own -- it delegates entirely to
:func:`~artgate_gen.fetch.fetch_thrifty`, which :mod:`test_fetch` already
exercises against both branches of the cache probe. This just proves the
entry point wires up and reports success.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from artgate_gen.fetch_cli import main
from tests.assertions import check, check_equal

if TYPE_CHECKING:
    import pytest

EXIT_OK = 0


def test_main_fetches_and_reports_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A cached run reports the cache status and exits cleanly."""
    code = main()
    check_equal(actual=code, expected=EXIT_OK, message="fetch reports success")
    out = capsys.readouterr().out
    check(condition="cached" in out, message="reports the cache status")
