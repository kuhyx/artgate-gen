"""Tests for the 250-line cap enforcer.

The boundary cases matter more than the happy path here: a checker that is off
by one at exactly the cap either blocks compliant files or lets a violation
through, and both failures are silent.
"""

from __future__ import annotations

from pathlib import Path

from tests.assertions import check, check_equal, check_length
from tools.check_line_cap import (
    LINE_CAP,
    check_paths,
    count_lines,
    main,
    tracked_files,
)

EXIT_OK = 0
EXIT_VIOLATION = 1
OVER_CAP = LINE_CAP + 1


def write_lines(path: Path, count: int) -> None:
    """Write a file containing exactly ``count`` lines.

    Args:
        path: Destination path.
        count: Number of lines to write.

    """
    path.write_text("x\n" * count, encoding="utf-8")


def test_count_lines_ignores_trailing_newline(tmp_path: Path) -> None:
    """A file ending in one newline counts its last line once."""
    target = tmp_path / "a.txt"
    target.write_text("one\ntwo\n", encoding="utf-8")
    check_equal(actual=count_lines(target), expected=2, message="trailing newline")


def test_count_lines_returns_none_for_undecodable(tmp_path: Path) -> None:
    """A file that is not valid UTF-8 reports ``None`` rather than raising."""
    target = tmp_path / "b.txt"
    target.write_bytes(b"\xff\xfe\x00binary")
    check(condition=count_lines(target) is None, message="undecodable returns None")


def test_file_at_exactly_the_cap_passes(tmp_path: Path) -> None:
    """A file of exactly ``LINE_CAP`` lines is compliant."""
    root = tmp_path
    target = root / "at_cap.py"
    write_lines(target, LINE_CAP)
    check_length(sized=check_paths([target], root), expected=0, message="at cap")


def test_file_one_over_the_cap_fails(tmp_path: Path) -> None:
    """A file one line over the cap is reported."""
    root = tmp_path
    target = root / "over.py"
    write_lines(target, OVER_CAP)
    violations = check_paths([target], root)
    check_length(sized=violations, expected=1, message="one over cap")
    check(condition="over.py" in violations[0], message="names the file")


def test_binary_suffix_is_skipped(tmp_path: Path) -> None:
    """A long file with a binary extension is not measured."""
    root = tmp_path
    target = root / "big.png"
    write_lines(target, OVER_CAP)
    check_length(sized=check_paths([target], root), expected=0, message="png skipped")


def test_undecodable_text_extension_is_reported(tmp_path: Path) -> None:
    """A binary file wearing a text extension is an error, not a silent skip."""
    root = tmp_path
    target = root / "sneaky.md"
    target.write_bytes(b"\xff\xfe\x00")
    violations = check_paths([target], root)
    check_length(sized=violations, expected=1, message="undecodable reported")
    check(condition="BINARY_SUFFIXES" in violations[0], message="suggests the fix")


def test_directories_are_ignored(tmp_path: Path) -> None:
    """A directory passed in by mistake is skipped rather than crashing."""
    root = tmp_path
    (root / "subdir").mkdir()
    check_length(
        sized=check_paths([root / "subdir"], root),
        expected=0,
        message="directory skipped",
    )


def test_path_outside_root_is_displayed_absolute(tmp_path: Path) -> None:
    """A file outside the root still reports, using its absolute path."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    write_lines(outside, OVER_CAP)
    violations = check_paths([outside], root)
    check_length(sized=violations, expected=1, message="outside root reported")


def test_main_with_explicit_paths_returns_violation_code(tmp_path: Path) -> None:
    """``main`` returns 1 and prints when given an oversized file."""
    target = tmp_path / "big.py"
    write_lines(target, OVER_CAP)
    check_equal(actual=main([str(target)]), expected=EXIT_VIOLATION, message="exit 1")


def test_main_with_compliant_path_returns_zero(tmp_path: Path) -> None:
    """``main`` returns 0 when every given file is within the cap."""
    target = tmp_path / "small.py"
    write_lines(target, 10)
    check_equal(actual=main([str(target)]), expected=EXIT_OK, message="exit 0")


def test_main_without_args_scans_tracked_files() -> None:
    """With no arguments ``main`` falls back to the git index."""
    result = main([])
    check(condition=result in {EXIT_OK, EXIT_VIOLATION}, message="returns a valid code")


def test_tracked_files_returns_repository_contents() -> None:
    """``tracked_files`` lists real files from the git index."""
    root = Path(__file__).resolve().parent.parent
    files = tracked_files(root)
    check(condition=len(files) > 0, message="git index is not empty")
    check(
        condition=any(f.name == "pyproject.toml" for f in files),
        message="pyproject.toml is tracked",
    )
