"""The ``sd-cli`` binary: finding it, and running one render through it.

Shared by every backend. They differ only in argv; exit-status handling, log
capture and the PNG read-back live here once.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable

# The binary is found on PATH by this name unless the env var names a path.
# The override is the test seam: the suite points it at a fake binary.
SD_CLI = "sd-cli"
SD_CLI_ENV = "ARTGATE_SD_CLI"

# How much of sd-cli's log to quote when a render fails.
LOG_TAIL_LINES = 15


class SdCliMissingError(FileNotFoundError):
    """The ``sd-cli`` binary is not installed or not where it was pointed."""


class SdCliError(RuntimeError):
    """``sd-cli`` ran but did not produce the requested image."""


def find_sd_cli() -> str:
    """Return the path of the ``sd-cli`` binary.

    Returns:
        An executable path.

    Raises:
        SdCliMissingError: When it cannot be found, naming the setup command.
    """
    wanted = os.environ.get(SD_CLI_ENV) or SD_CLI
    found = shutil.which(wanted)
    if found is None:
        message = (
            f"stable-diffusion.cpp binary {wanted!r} not found; "
            "install it with ./run.sh --setup-only"
        )
        raise SdCliMissingError(message)
    return found


def run_sd_cli(
    argv_for: Callable[[Path], list[str]], size: tuple[int, int]
) -> Image.Image:
    """Run one ``sd-cli`` render and read back the PNG it wrote.

    Shared by every backend: they differ only in argv. sd-cli's log is
    captured, never inherited: the MCP server speaks over stdio, and a render's
    log lines on stdout would corrupt the protocol.

    Args:
        argv_for: Builds the command line, given the PNG path to write.
        size: The width and height the render must have.

    Returns:
        The render, as RGB.

    Raises:
        SdCliError: When sd-cli fails, writes nothing, or writes the wrong size.
    """
    with tempfile.TemporaryDirectory(prefix="artgate-sd-") as scratch:
        out = Path(scratch) / "render.png"
        done = subprocess.run(
            argv_for(out),
            capture_output=True,
            text=True,
            check=False,
        )
        log = "\n".join((done.stdout + done.stderr).splitlines()[-LOG_TAIL_LINES:])
        if done.returncode != 0:
            message = f"sd-cli exited {done.returncode}:\n{log}"
            raise SdCliError(message)
        if not out.is_file():
            message = f"sd-cli exited 0 but wrote no image:\n{log}"
            raise SdCliError(message)
        with Image.open(out) as image:
            image.load()
            render = image.convert("RGB")
    if render.size != size:
        message = f"sd-cli rendered {render.size}, expected {size}"
        raise SdCliError(message)
    return render
