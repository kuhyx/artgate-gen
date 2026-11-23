"""Report, and only on ``--download`` fetch, the SD3.5-Large subset the loader reads.

``--download`` then writes the q8_0 GGUF copies renders load (see ``sd35``).
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import subprocess
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from kuhy_credit import credit_line

from artgate_gen.console import emit
from artgate_gen.sd35 import (
    ALLOW_PATTERNS,
    LORA,
    LORA_WEIGHTS,
    MODEL,
    QUANT,
    QUANTISED,
    quant_path,
)
from artgate_gen.sd_cli import LOG_TAIL_LINES, SdCliError, find_sd_cli

GB = 1e9


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The configured parser.
    """
    return argparse.ArgumentParser(description=__doc__, epilog=credit_line())


def _is_cached(filename: str) -> bool:
    """Return whether the model file ``filename`` is in the local HF cache."""
    # huggingface_hub's "not cached" errors vary by version; see sd35._cached.
    with contextlib.suppress(Exception):
        return bool(hf_hub_download(MODEL, filename, local_files_only=True))
    return False


def planned_bytes(*, missing_only: bool = False) -> int:
    """Sum the sizes of the model files matched by ``ALLOW_PATTERNS``.

    Args:
        missing_only: Count only the files not already cached.

    Returns:
        The total size in bytes.
    """
    files = HfApi().list_repo_tree(MODEL, recursive=True)
    return sum(
        int(getattr(f, "size", 0))
        for f in files
        if any(fnmatch(f.path, pattern) for pattern in ALLOW_PATTERNS)
        and not (missing_only and _is_cached(f.path))
    )


def convert(source: Path, target: Path) -> None:
    """Write ``target``, the ``QUANT`` GGUF copy of ``source``, unless it exists.

    sd-cli writes to a temporary name first, so a killed conversion never
    leaves a truncated copy that a render would then load.

    Args:
        source: The fp16 safetensors file.
        target: Where the copy goes.

    Raises:
        SdCliError: When sd-cli fails, with the tail of its log.
    """
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".partial-{target.name}")
    done = subprocess.run(
        [
            *(find_sd_cli(), "-M", "convert", "-m", str(source)),
            *("--type", QUANT, "-o", str(partial)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        partial.unlink(missing_ok=True)
        log = "\n".join((done.stdout + done.stderr).splitlines()[-LOG_TAIL_LINES:])
        message = f"sd-cli convert exited {done.returncode}:\n{log}"
        raise SdCliError(message)
    partial.replace(target)


def main(argv: list[str] | None = None) -> int:
    """Print download size and free disk; download only with ``--download``."""
    parser = build_parser()
    parser.add_argument("--download", action="store_true", help="fetch the weights")
    args = parser.parse_args(argv)
    if args.download:
        snapshot_download(MODEL, allow_patterns=list(ALLOW_PATTERNS))
        snapshot_download(LORA, allow_patterns=[LORA_WEIGHTS])
        for filename in QUANTISED:
            source = hf_hub_download(MODEL, filename, local_files_only=True)
            convert(Path(source), quant_path(filename))
        return 0
    free = shutil.disk_usage(Path.home()).free
    todo = planned_bytes(missing_only=True)
    status = f"{todo / GB:.1f} GB to fetch" if todo else "cached"
    emit(f"{MODEL}: {status}, {free / GB:.0f} GB free")
    missing = [quant_path(f).name for f in QUANTISED if not quant_path(f).is_file()]
    if missing:
        emit(f"{QUANT} copies missing: {', '.join(missing)}; --download makes them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
