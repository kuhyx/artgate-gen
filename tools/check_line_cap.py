"""Fail when any tracked text file exceeds the repository's line cap.

The cap is a hard rule with no exemption mechanism, and that absence is
deliberate: an allow-list is exactly how a hard cap decays into a soft one.
A file that grows past the limit gets split, not excused.

Binary files are skipped by extension. Because an extension deny-list rots
silently as new formats arrive, a file that *claims* to be text and then fails
to decode is reported as an error rather than skipped -- that way the deny-list
announces its own gaps instead of hiding them.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

LINE_CAP = 250

# Resolved at import rather than hardcoded: an absolute path satisfies ruff's
# S607 without a suppression, but a literal "/usr/bin/git" would break on
# systems that put it elsewhere (macOS, NixOS).
GIT = shutil.which("git") or "git"

BINARY_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".bmp",
        ".safetensors",
        ".pt",
        ".pth",
        ".bin",
        ".onnx",
        ".zip",
        ".gz",
        ".xz",
        ".zst",
        ".tar",
        ".pdf",
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".mp3",
        ".mp4",
        ".wav",
        ".so",
        ".dylib",
        ".dll",
    },
)


def tracked_files(root: Path) -> list[Path]:
    """List every file git tracks, relative to ``root``.

    Using git's index rather than a filesystem walk keeps untracked scratch
    output and the virtualenv structurally out of scope.

    Args:
        root: Repository root to query.

    Returns:
        Tracked paths, as absolute paths.

    """
    result = subprocess.run(
        [GIT, "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [root / name for name in result.stdout.split("\0") if name]


def count_lines(path: Path) -> int | None:
    """Count the lines in a text file.

    A trailing newline terminates the final line rather than starting a new
    one, so a file ending in a single newline counts its last line once.

    Args:
        path: File to measure.

    Returns:
        The line count, or ``None`` if the file is not decodable as UTF-8.

    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    return len(text.splitlines())


def check_paths(paths: list[Path], root: Path) -> list[str]:
    """Collect a violation message for every offending file.

    Args:
        paths: Files to inspect.
        root: Repository root, used to render readable relative paths.

    Returns:
        Human-readable violation lines; empty when everything passes.

    """
    violations: list[str] = []
    for path in sorted(paths):
        if path.suffix.lower() in BINARY_SUFFIXES or not path.is_file():
            continue
        display = path.relative_to(root) if path.is_relative_to(root) else path
        lines = count_lines(path)
        if lines is None:
            violations.append(
                f"{display}: not valid UTF-8; add its suffix to BINARY_SUFFIXES",
            )
        elif lines > LINE_CAP:
            violations.append(f"{display}: {lines} lines exceeds cap of {LINE_CAP}")
    return violations


def main(argv: list[str] | None = None) -> int:
    """Check the given files, or every tracked file when none are given.

    Args:
        argv: Optional explicit paths, as passed by pre-commit.

    Returns:
        ``0`` when every file is within the cap, ``1`` otherwise.

    """
    args = sys.argv[1:] if argv is None else argv
    root = Path(__file__).resolve().parent.parent
    paths = [Path(a).resolve() for a in args] if args else tracked_files(root)

    violations = check_paths(paths, root)
    for violation in violations:
        sys.stdout.write(f"{violation}\n")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
