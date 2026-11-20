"""A stand-in for the ``sd-cli`` binary, run as a real subprocess by the tests.

The pipeline talks to stable-diffusion.cpp only through argv, exit status and
the PNG it writes, so the fake honours exactly that contract: it parses the
same flags, renders a flat image at ``-W`` x ``-H`` to ``-o``, and records its
argv for assertions. ``FAKE_SD_MODE`` steers the failure paths the pipeline
must report visibly instead of trusting exit code 0.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PIL import Image

# Flat colour, matching what the old diffusers stub rendered, so the finisher
# and the gate downstream see the same input they always did.
FILL = (40, 60, 90)


def _flag(argv: list[str], name: str) -> str:
    """Return the value following ``name`` in ``argv``.

    Args:
        argv: The command-line arguments.
        name: The flag to look up.

    Returns:
        The flag's value.
    """
    return argv[argv.index(name) + 1]


def main(argv: list[str]) -> int:
    """Pretend to render, or fail in the way ``FAKE_SD_MODE`` asks for.

    Args:
        argv: The command-line arguments, without the program name.

    Returns:
        The process exit status.
    """
    record = os.environ.get("FAKE_SD_ARGV")
    if record:
        Path(record).write_text(json.dumps(argv), encoding="utf-8")
    mode = os.environ.get("FAKE_SD_MODE", "ok")
    if mode == "fail":
        sys.stderr.write("[E] fake: failed to allocate Vulkan buffer\n")
        return 1
    if mode == "no-output":
        return 0
    width, height = int(_flag(argv, "-W")), int(_flag(argv, "-H"))
    if mode == "wrong-size":
        width //= 2
    Image.new("RGB", (width, height), FILL).save(_flag(argv, "-o"))
    sys.stdout.write("[I] fake: generate_image completed in 0.01s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
