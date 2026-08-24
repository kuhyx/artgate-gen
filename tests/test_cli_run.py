"""End-to-end tests for the CLI's generation run.

Split out of `test_cli.py` to stay under the repo's line cap. That file keeps
the flag/parser contract; this one drives `main()` through a full render (via
the stub pipeline) and checks what lands on disk.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from artgate_gen.cli import main
from tests.assertions import check, check_equal, check_length

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


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
