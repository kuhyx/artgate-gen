"""The ``sd-cli`` binary: finding it, and running one render through it.

Shared by every backend. They differ only in argv; exit-status handling, log
capture and the PNG read-back live here once.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
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

# sd-cli's log line once every PNG is on disk.
SAVED_MARK = "images saved"
# About 1 run in 4, sd-cli aborts inside exit() after saving: a static map in
# libstable-diffusion.so frees into a heap that is already corrupted (glibc
# "free(): chunks in smallbin corrupted"). MangoHud off does not stop it. The
# PNG is complete by then, so that one abort is not a failed render. -6 is the
# direct child; 134 is the same abort reported through a wrapper shell.
EXIT_ABORT_CODES = frozenset({-signal.SIGABRT, 128 + signal.SIGABRT})

# Each render reads gigabytes of weights through the page cache. Uncapped, that
# growth made the kernel swap other processes' idle pages out (and back in,
# minutes later) on a box with swappiness 100. In its own cgroup the read is
# reclaimed from sd-cli's own cache instead. 2560M sits above sd-cli's ~1.8 GiB
# peak RSS (SD3.5); MemorySwapMax=0 keeps its own pages out of swap.
MEMORY_HIGH = "2560M"
# "0" runs sd-cli bare even where a user session exists (the test suite does).
SCOPE_ENV = "ARTGATE_SD_SCOPE"


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


def sd_cli_env() -> dict[str, str]:
    """Return the environment for an ``sd-cli`` child: ours, MangoHud off.

    A session-wide ``MANGOHUD=1`` (set in environment.d for games) makes the
    Vulkan loader inject MangoHud's implicit overlay layer into every Vulkan
    process. Inside sd-cli it corrupts the heap: about 1 run in 4 aborted at
    exit with ``free(): corrupted unsorted chunks`` after saving the image.
    ``DISABLE_MANGOHUD=1`` is the layer manifest's own off switch.

    Returns:
        A copy of ``os.environ`` with ``DISABLE_MANGOHUD=1``.
    """
    return {**os.environ, "DISABLE_MANGOHUD": "1"}


def memory_capped(argv: list[str]) -> list[str]:
    """Wrap ``argv`` in a memory-capped systemd user scope, when there is one.

    ``systemd-run --scope`` execs the command in place, so stdout, stderr, the
    exit status and the environment pass through unchanged. CI and containers
    have no user bus; there the command runs bare.

    Args:
        argv: The sd-cli command line.

    Returns:
        The command line to run.
    """
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if (
        os.environ.get(SCOPE_ENV) == "0"
        or not runtime
        or not (Path(runtime) / "bus").exists()
        or shutil.which("systemd-run") is None
    ):
        return argv
    return [
        *("systemd-run", "--user", "--scope", "--quiet", "--collect"),
        *("-p", f"MemoryHigh={MEMORY_HIGH}", "-p", "MemorySwapMax=0", "--"),
        *argv,
    ]


def _aborted_after_save(returncode: int, log: str, out: Path) -> bool:
    """Tell sd-cli's harmless exit-time abort apart from a failed render.

    Args:
        returncode: sd-cli's exit status.
        log: Its full stdout and stderr.
        out: The PNG path it was told to write.

    Returns:
        True only for an abort after sd-cli logged the image saved and the
        PNG exists. Size and the 64 gate are still checked afterwards.
    """
    return returncode in EXIT_ABORT_CODES and SAVED_MARK in log and out.is_file()


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
            memory_capped(argv_for(out)),
            env=sd_cli_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        full_log = done.stdout + done.stderr
        log = "\n".join(full_log.splitlines()[-LOG_TAIL_LINES:])
        if _aborted_after_save(done.returncode, full_log, out):
            sys.stderr.write(
                f"sd-cli aborted after saving (exit {done.returncode}); render kept\n"
            )
        elif done.returncode != 0:
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
