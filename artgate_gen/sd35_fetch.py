"""Report, and only on ``--download`` fetch, the SD3.5-Large subset the loader reads."""

from __future__ import annotations

import argparse
import shutil
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from artgate_gen.console import emit
from artgate_gen.sd35 import ALLOW_PATTERNS, LORA, LORA_WEIGHTS, MODEL

GB = 1e9


def planned_bytes() -> int:
    """Sum the sizes of the model files matched by ``ALLOW_PATTERNS``."""
    files = HfApi().list_repo_tree(MODEL, recursive=True)
    return sum(
        int(getattr(f, "size", 0))
        for f in files
        if any(fnmatch(f.path, pattern) for pattern in ALLOW_PATTERNS)
    )


def main(argv: list[str] | None = None) -> int:
    """Print download size and free disk; download only with ``--download``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="fetch the weights")
    args = parser.parse_args(argv)
    if args.download:
        snapshot_download(MODEL, allow_patterns=list(ALLOW_PATTERNS))
        snapshot_download(LORA, allow_patterns=[LORA_WEIGHTS])
        return 0
    free = shutil.disk_usage(Path.home()).free
    emit(f"{MODEL}: {planned_bytes() / GB:.1f} GB to fetch, {free / GB:.0f} GB free")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
