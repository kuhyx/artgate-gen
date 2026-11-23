"""Tests for the memory-capped user scope every render runs in."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from artgate_gen import pipeline, sd_cli
from artgate_gen.targets import TARGETS
from tests.assertions import check, check_equal

if TYPE_CHECKING:
    from pathlib import Path

ARGV = ["sd-cli", "-p", "a lantern"]


def _user_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Fake a user session: a bus socket path and a recording ``systemd-run``.

    The fake strips its own options up to ``--`` and execs the rest, as
    ``systemd-run --scope`` does, so a render through it still completes.

    Returns:
        Where the fake writes the argv it was given.
    """
    runtime = tmp_path / "run"
    runtime.mkdir()
    (runtime / "bus").touch()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    record = tmp_path / "systemd-run.argv"
    fake = bindir / "systemd-run"
    fake.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > "{record}"\n'
        'while [ "$1" != "--" ]; do shift; done\nshift\nexec "$@"\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.delenv(sd_cli.SCOPE_ENV, raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("PATH", f"{bindir}:/usr/bin:/bin")
    return record


def test_a_render_runs_in_a_memory_capped_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The weight read must not push other processes' pages to swap."""
    record = _user_session(monkeypatch, tmp_path)
    target = replace(TARGETS[64, 64], render=(64, 64))
    pipeline.generate(pipeline.load_pipeline(), "a lantern", 7, target)
    argv = record.read_text(encoding="utf-8").splitlines()
    for flag in ("--user", "--scope", "--quiet", "--collect"):
        check(condition=flag in argv, message=f"{flag} is passed")
    check(condition="MemoryHigh=2560M" in argv, message="page cache is capped")
    check(condition="MemorySwapMax=0" in argv, message="sd-cli never swaps")
    command = argv[argv.index("--") + 1]
    check(condition=command.endswith("sd-cli"), message="sd-cli follows --")


def test_the_scope_wraps_the_argv_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Everything after ``--`` is the render's own command line."""
    _user_session(monkeypatch, tmp_path)
    wrapped = sd_cli.memory_capped(ARGV)
    check_equal(actual=wrapped[-len(ARGV) :], expected=ARGV, message="tail")
    check_equal(actual=wrapped[-len(ARGV) - 1], expected="--", message="separator")


@pytest.mark.parametrize("missing", ["opt-out", "bus", "systemd-run"])
def test_without_a_user_session_sd_cli_runs_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: str
) -> None:
    """CI, containers and the opt-out run the bare command."""
    _user_session(monkeypatch, tmp_path)
    if missing == "opt-out":
        monkeypatch.setenv(sd_cli.SCOPE_ENV, "0")
    elif missing == "bus":
        (tmp_path / "run" / "bus").unlink()
    else:
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    check_equal(actual=sd_cli.memory_capped(ARGV), expected=ARGV, message=missing)


def test_no_runtime_dir_means_no_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without ``XDG_RUNTIME_DIR`` there is no user bus to ask."""
    _user_session(monkeypatch, tmp_path)
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    check_equal(actual=sd_cli.memory_capped(ARGV), expected=ARGV, message="argv")
