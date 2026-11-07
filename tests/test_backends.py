"""Tests for backbone selection: SDXL stays the default, SD3.5 is opt-in."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from artgate_gen import backends, pipeline, sd35
from artgate_gen.cli import build_parser, main
from tests.assertions import check, check_equal

if TYPE_CHECKING:
    from pathlib import Path

    from artgate_gen.profiling import RunProfile


def test_sdxl_is_the_default_backend() -> None:
    """Omitting ``--backend`` keeps the SDXL route every caller relies on."""
    args = build_parser().parse_args(["out", "axe=a stone axe"])
    check_equal(actual=args.backend, expected="sdxl", message="the default backend")
    check_equal(actual=backends.DEFAULT, expected="sdxl", message="the registry")


def test_each_name_resolves_to_its_module() -> None:
    """The registry maps a name onto the module that renders with it."""
    check(condition=backends.load("sdxl") is pipeline, message="sdxl is pipeline")
    check(condition=backends.load("sd35") is sd35, message="sd35 is its own module")


def test_an_unknown_backend_is_a_usage_error() -> None:
    """A typo fails at parse time rather than after a model load."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["out", "axe=a stone axe", "--backend", "flux"])


def test_the_cli_renders_with_the_chosen_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--backend sd35`` routes the whole run through the SD3.5 module."""
    used: list[str] = []
    original = sd35.load_pipeline

    def spy(profile: RunProfile | None = None) -> object:
        used.append("sd35")
        return original(profile)

    monkeypatch.setattr(sd35, "load_pipeline", spy)
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate.py", str(tmp_path), "axe=a stone axe", "--backend", "sd35"],
    )
    main()
    check_equal(actual=used, expected=["sd35"], message="the SD3.5 loader ran")
    check(condition=(tmp_path / "axe.png").exists(), message="the icon was written")
