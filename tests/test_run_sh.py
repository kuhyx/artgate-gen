"""Tests for ``run.sh``'s setup steps, run against a stub venv in a temp copy.

The real setup installs uv, sd-cli and Python packages and fetches gigabytes,
so each test copies ``run.sh`` next to stubs that only log their arguments.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.assertions import check

REPO = Path(__file__).resolve().parent.parent
STUB = '#!/bin/sh\nprintf "%s\\n" "$*" >> "{log}"\n'


@pytest.fixture
def stub_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Copy ``run.sh`` beside a logging stub python, sd-cli installer and uv.

    Args:
        tmp_path: Where the copy goes.

    Returns:
        The copied ``run.sh`` and the stub call log.
    """
    log = tmp_path / "calls.log"
    for rel in ("venv/bin/python", "scripts/install_sd_cpp.sh", "bin/uv"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(STUB.format(log=log), encoding="utf-8")
        path.chmod(0o755)
    shutil.copy(REPO / "run.sh", tmp_path / "run.sh")
    return tmp_path / "run.sh", log


def _setup(run_sh: Path, *args: str) -> None:
    """Run ``run.sh --setup-only`` with the stub ``uv`` first on PATH."""
    path = f"{run_sh.parent / 'bin'}{os.pathsep}{os.environ['PATH']}"
    subprocess.run(
        [str(run_sh), "--setup-only", *args],
        check=True,
        capture_output=True,
        env={**os.environ, "PATH": path},
    )


def test_sdxl_setup_fetches_the_sdxl_models(stub_repo: tuple[Path, Path]) -> None:
    """The default backend still pre-fetches its checkpoint and LoRA."""
    run_sh, log = stub_repo
    _setup(run_sh)
    calls = log.read_text(encoding="utf-8")
    check(condition="artgate_gen.fetch_cli" in calls, message=calls)


def test_sd35_setup_downloads_no_sdxl_models(stub_repo: tuple[Path, Path]) -> None:
    """``--backend sd35`` never pulls SDXL: an empty cache must stay empty."""
    run_sh, log = stub_repo
    _setup(run_sh, "--backend", "sd35")
    calls = log.read_text(encoding="utf-8")
    check(condition="artgate_gen.fetch_cli" not in calls, message=calls)
