"""Every user-facing entry point shows the shared kuhy credit.

The credit comes from ``kuhy_credit`` (kuhyx/utils), never a copy: each test
compares against ``credit_line()`` so a change in utils flows through.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from kuhy_credit import URL, credit_line

from artgate_gen import (
    cli,
    fetch_cli,
    repair_cli,
    sd35_fetch,
    server,
    set_consistency_cli,
)
from artgate_gen.gate import cli as gate_cli
from tests.assertions import check, check_equal

if TYPE_CHECKING:
    from collections.abc import Callable

REPO = Path(__file__).resolve().parent.parent
CREDIT = credit_line()
PIN = r"kuhy-credit @ git\+https://github\.com/kuhyx/utils@(kuhy-credit-v[\d.]+)#"


@pytest.mark.parametrize(
    "main",
    [
        lambda: cli.build_parser().parse_args(["--help"]),
        lambda: fetch_cli.main(["--help"]),
        lambda: gate_cli.main(["--help"]),
        lambda: repair_cli.main(["--help"]),
        lambda: sd35_fetch.main(["--help"]),
        lambda: set_consistency_cli.main(["--help"]),
    ],
    ids=[
        "artgate-gen",
        "fetch",
        "artgate-gate",
        "repair",
        "sd35_fetch",
        "set_consistency",
    ],
)
def test_python_help_ends_with_credit(
    main: Callable[[], object], capsys: pytest.CaptureFixture[str]
) -> None:
    """``--help`` of each Python CLI ends on the credit line."""
    with pytest.raises(SystemExit):
        main()
    out = capsys.readouterr().out.rstrip()
    check(
        condition=out.endswith(CREDIT), message=f"help ends with credit: {out[-80:]!r}"
    )


def test_run_sh_help_ends_with_credit() -> None:
    """``./run.sh --help`` ends on the credit line."""
    done = subprocess.run(
        [str(REPO / "run.sh"), "--help"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    out = done.stdout.rstrip()
    check(
        condition=out.endswith(CREDIT),
        message=f"run.sh help ends with credit: {out[-80:]!r}",
    )


def test_mcp_server_advertises_site() -> None:
    """MCP clients see the site as the server's website URL."""
    check_equal(actual=server.mcp.website_url, expected=URL, message="website_url")


def _pin(text: str) -> str:
    """Return the kuhy-credit tag a manifest line pins."""
    found = re.findall(PIN, text)
    check_equal(actual=len(found), expected=1, message="manifest pins kuhy-credit once")
    return str(found[0])


def test_requirements_and_pyproject_pin_same_credit_tag() -> None:
    """``run.sh`` installs from requirements.txt, so it must carry the same pin."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    deps = "\n".join(project["project"]["dependencies"])
    reqs = (REPO / "requirements.txt").read_text(encoding="utf-8")
    check_equal(actual=_pin(reqs), expected=_pin(deps), message="same tag in both")


def test_run_sh_help_without_venv_uses_path(tmp_path: Path) -> None:
    """With no ./venv (CI, a fresh clone), the ``kuhy-credit`` on PATH prints it."""
    shutil.copy(REPO / "run.sh", tmp_path / "run.sh")
    path = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    done = subprocess.run(
        [str(tmp_path / "run.sh"), "--help"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PATH": path},
    )
    out = done.stdout.rstrip()
    check(
        condition=out.endswith(CREDIT),
        message=f"run.sh help ends with credit: {out[-80:]!r}",
    )
