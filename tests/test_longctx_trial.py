"""Tests for the long-context trial driver, run end to end on the stubs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from artgate_gen import sd35
from tests.assertions import check, check_equal
from tools import longctx_trial

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image

    from artgate_gen.targets import Target

PORTRAIT_STYLE = "pixel art, close-up character portrait"
UNRELATED = {"name": "axe", "category": "items", "file": "items/axe.png"}


def _entry(name: str, category: str, canvas: list[int]) -> dict[str, object]:
    """Build one SDXL manifest entry, shaped like roadside-assistance's.

    Args:
        name: Entry name.
        category: Entry category.
        canvas: Final canvas size.

    Returns:
        The entry.
    """
    return {
        "name": name,
        "category": category,
        "file": f"{category}/{name}.png",
        "prompt": f"anime girl face, {name}",
        "seed": 11,
        "style": PORTRAIT_STYLE,
        "negative": "photo",
        "render": [canvas[0] * 3, canvas[1] * 2],
        "canvas": canvas,
        "max_colors": 0,
    }


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    """Write a small art tree with two portraits, one backdrop and a bystander.

    Args:
        tmp_path: Pytest's per-test directory.

    Returns:
        The manifest path.
    """
    art = tmp_path / "art"
    art.mkdir()
    entries = [
        _entry("neutral_s11", "portraits", [256, 256]),
        _entry("sad_s11", "portraits", [256, 256]),
        _entry("coast_s0", "backdrops", [256, 64]),
        UNRELATED,
    ]
    path = art / "MANIFEST.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _run(manifest: Path, *extra: str) -> int:
    """Run the driver against ``manifest`` with the default trial folder.

    Args:
        manifest: The manifest fixture.
        *extra: Further arguments.

    Returns:
        The driver's exit code.
    """
    out = manifest.parent / "sd35-trial"
    argv = ["--manifest", str(manifest), "--out", str(out), "--backdrops", "coast_s0"]
    return longctx_trial.main([*argv, "--long-subject", "neutral_s11", *extra])


def _trial(manifest: Path) -> dict[str, dict[str, Any]]:
    """Return the trial's entries keyed by name.

    Args:
        manifest: The manifest fixture.

    Returns:
        The ``longctx-sd35`` entries.
    """
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    return {e["name"]: e for e in entries if e["category"] == "longctx-sd35"}


def test_every_subject_is_rendered_and_recorded(manifest: Path) -> None:
    """Short arms for all subjects plus one long arm, each with full metadata."""
    check_equal(actual=_run(manifest), expected=0, message="a clean run")
    trial = _trial(manifest)
    check_equal(
        actual=set(trial),
        expected={"neutral_s11", "sad_s11", "coast_s0", "neutral_s11_long"},
        message="the rendered arms",
    )
    for entry in trial.values():
        check_equal(actual=entry["status"], expected="ok", message=entry["name"])
        check_equal(actual=entry["model"], expected=sd35.MODEL, message="model")
        check_equal(actual=entry["lora_scale"], expected=1.0, message="strength")
        check(condition=entry["prompt"].startswith("pixel art"), message="trigger")
        check(condition="peak_vram_mb" in entry, message="peak VRAM recorded")
        check(condition=entry["seconds"] >= 0, message="seconds recorded")
        check(condition=(manifest.parent / entry["file"]).exists(), message="file")


def test_bystanders_are_untouched_and_reruns_do_not_duplicate(manifest: Path) -> None:
    """Only the trial's own category is replaced; others keep their order."""
    original = json.loads(manifest.read_text(encoding="utf-8"))
    _run(manifest)
    _run(manifest)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    check_equal(actual=entries[:4], expected=original, message="prefix unchanged")
    check(condition=UNRELATED in entries, message="the bystander survives")
    check_equal(actual=len(entries), expected=8, message="4 sources + 4 trial")


def test_the_long_arm_is_longer_with_the_same_seed(manifest: Path) -> None:
    """The long twin costs more CLIP tokens and keeps the short arm's seed."""
    _run(manifest)
    trial = _trial(manifest)
    short, long_arm = trial["neutral_s11"], trial["neutral_s11_long"]
    check(
        condition=long_arm["clip_tokens"] > max(short["clip_tokens"], 77),
        message=f"{long_arm['clip_tokens']} vs {short['clip_tokens']}",
    )
    check_equal(actual=long_arm["seed"], expected=short["seed"], message="seed")
    check_equal(actual=long_arm["t5_max_len"], expected=sd35.T5_MAX_LEN, message="t5")


def test_a_failed_render_is_recorded_not_dropped(
    manifest: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash becomes an ``error`` entry and a non-zero exit."""
    original = sd35.generate

    def flaky(
        pipe: sd35.Sd35Pipeline, prompt: str, seed: int, target: Target
    ) -> Image.Image:
        if "sad_s11" in prompt:
            message = "CUDA out of memory"
            raise RuntimeError(message)
        return original(pipe, prompt, seed, target)

    monkeypatch.setattr(sd35, "generate", flaky)
    check_equal(actual=_run(manifest), expected=1, message="failure is reported")
    failed = _trial(manifest)["sad_s11"]
    check_equal(actual=failed["status"], expected="error", message="status")
    check(condition="out of memory" in failed["error"], message="the error text")


@pytest.mark.parametrize(
    ("out", "extra"),
    [
        ("../elsewhere-trial", ()),
        ("portraits", ()),
        ("sd35-trial", ("--long-subject", "nobody")),
        ("sd35-trial", ("--backdrops", "moon_s0")),
    ],
)
def test_bad_arguments_are_usage_errors(
    manifest: Path,
    out: str,
    extra: tuple[str, ...],
) -> None:
    """Outside the art tree, a non-trial folder, or an unknown subject all fail."""
    argv = ["--manifest", str(manifest), "--out", str(manifest.parent / out)]
    argv += ["--backdrops", "coast_s0", "--long-subject", "neutral_s11"]
    with pytest.raises(SystemExit):
        longctx_trial.main([*argv, *extra])
