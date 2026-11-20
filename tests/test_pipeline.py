"""Tests for the stable-diffusion.cpp pipeline module.

Generation is a subprocess call to ``sd-cli``. Every test here runs it for real
against ``tests/fake_sd_cli.py`` (wired in by ``conftest``), so the argv, the
exit-status handling and the PNG read-back are all exercised end to end; only
the model itself is fake.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from artgate_gen import pipeline, prompt_budget
from artgate_gen.profiling import RunProfile
from artgate_gen.targets import TARGETS
from tests.assertions import check, check_equal, check_length

if TYPE_CHECKING:
    from pathlib import Path

    from artgate_gen._types import Pipeline

RENDER_EDGE = 64
# Small on purpose: the real backdrop target renders at 2560x512, and no test
# allocates a megapixel to prove that width and height are threaded separately.
NON_SQUARE = (128, 32)
SEED = 7


def _argv_of(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Ask the fake binary to record its argv, and return where it lands."""
    record = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_SD_ARGV", str(record))
    return record


def _icon(pipe: Pipeline) -> tuple[int, int]:
    """Render the small icon target and return the image size."""
    target = replace(TARGETS[64, 64], render=(RENDER_EDGE, RENDER_EDGE))
    return pipeline.generate(pipe, "a brass lantern", SEED, target).size


def test_load_pipeline_uses_the_binary_named_by_the_env_seam(
    fake_sd_cli: Path,
) -> None:
    """The env var wins over PATH, which is how tests reach the fake."""
    pipe = pipeline.load_pipeline()
    check_equal(actual=pipe.binary, expected=str(fake_sd_cli), message="binary")


def test_load_pipeline_falls_back_to_path(
    monkeypatch: pytest.MonkeyPatch,
    fake_sd_cli: Path,
) -> None:
    """With no override the binary is looked up on PATH by its own name."""
    monkeypatch.delenv(pipeline.SD_CLI_ENV)
    monkeypatch.setenv("PATH", str(fake_sd_cli.parent))
    pipe = pipeline.load_pipeline()
    check_equal(actual=pipe.binary, expected=str(fake_sd_cli), message="PATH hit")


def test_a_missing_binary_fails_visibly_with_the_setup_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No sd-cli means an error naming the fix, not a crash mid-render."""
    monkeypatch.setenv(pipeline.SD_CLI_ENV, str(tmp_path / "absent"))
    with pytest.raises(pipeline.SdCliMissingError, match=r"run\.sh --setup-only"):
        pipeline.load_pipeline()


def test_load_pipeline_resolves_every_weight_file() -> None:
    """The checkpoint, VAE, LoRA and tokenizer all come from the HF cache."""
    pipe = pipeline.load_pipeline()
    check_equal(
        actual=pipe.model.name, expected=pipeline.CHECKPOINT, message="checkpoint"
    )
    check_equal(actual=pipe.vae.name, expected=pipeline.VAE_WEIGHTS, message="vae")
    check(
        condition=pipeline.LORA in str(pipe.lora_dir),
        message="the LoRA directory is the LoRA repo's snapshot",
    )
    check_equal(actual=pipe.tokenizer_dir.name, expected="tokenizer", message="tok")


def test_resolution_falls_back_to_the_network_when_not_cached() -> None:
    """A cache miss retries online instead of failing the run."""
    hub = sys.modules["huggingface_hub"]
    hub.missing.add(pipeline.LORA)
    pipe = pipeline.load_pipeline()
    check(condition=pipeline.LORA in str(pipe.lora_dir), message="fetched online")
    check_equal(
        actual=hub.calls.count(pipeline.LORA),
        expected=2,
        message="one local probe, then one network fetch",
    )


def test_load_pipeline_records_a_phase_when_profiled() -> None:
    """Passing a profile times the resolution step."""
    with RunProfile(interval=0.01) as profile:
        pipeline.load_pipeline(profile)
    names = [phase.name for phase in profile.phases]
    check(condition="load:resolve" in names, message="resolution is timed")


def test_load_pipeline_without_a_profile_times_nothing() -> None:
    """The null span keeps the unprofiled path free of bookkeeping."""
    with RunProfile(interval=0.01) as profile:
        pipeline.load_pipeline()
    check_length(sized=profile.phases, expected=0, message="nothing was timed")


def test_generate_returns_one_image() -> None:
    """Rendering yields a single image read back from the binary's PNG."""
    check_equal(
        actual=_icon(pipeline.load_pipeline()),
        expected=(RENDER_EDGE, RENDER_EDGE),
        message="the rendered image size",
    )


def test_generate_honours_a_non_square_target() -> None:
    """A non-square target renders at its own width and height."""
    target = replace(TARGETS[256, 64], render=NON_SQUARE)
    image = pipeline.generate(pipeline.load_pipeline(), "a desert", SEED, target)
    check_equal(actual=image.size, expected=NON_SQUARE, message="non-square size")


def test_generate_passes_the_benchmarked_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Steps, guidance, sampler, seed, VAE and LoRA all reach the binary."""
    record = _argv_of(monkeypatch, tmp_path)
    pipe = pipeline.load_pipeline()
    _icon(pipe)
    argv = json.loads(record.read_text(encoding="utf-8"))
    for flag, value in {
        "--steps": "25",
        "--cfg-scale": "7.0",
        "--sampling-method": "euler",
        "-s": str(SEED),
        "-m": str(pipe.model),
        "--vae": str(pipe.vae),
        "--lora-model-dir": str(pipe.lora_dir),
        "-n": pipeline.NEGATIVE,
    }.items():
        check(condition=flag in argv, message=f"{flag} is passed")
        check_equal(actual=argv[argv.index(flag) + 1], expected=value, message=flag)
    prompt = argv[argv.index("-p") + 1]
    check(
        condition=prompt.startswith(f"{pipeline.STYLE}, a brass lantern"),
        message="the style stem precedes the subject",
    )
    check(
        condition=prompt.endswith("<lora:pixel-art-xl:1>"),
        message="the pixel-art LoRA is applied at full strength",
    )


def test_prompt_weighting_reaches_the_binary_verbatim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``(word:1.2)`` is the syntax sd.cpp and the old lpw pipeline share."""
    record = _argv_of(monkeypatch, tmp_path)
    target = replace(TARGETS[64, 64], render=(RENDER_EDGE, RENDER_EDGE))
    pipeline.generate(pipeline.load_pipeline(), "(red:1.3) potion", SEED, target)
    argv = json.loads(record.read_text(encoding="utf-8"))
    check(
        condition="(red:1.3) potion" in argv[argv.index("-p") + 1],
        message="the weight survives untouched",
    )


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("fail", "failed to allocate Vulkan buffer"),
        ("no-output", "no image"),
        ("wrong-size", "expected"),
    ],
)
def test_a_bad_render_raises_with_the_reason(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    match: str,
) -> None:
    """Exit 0 is not trusted: a missing or wrong-sized PNG is an error too."""
    monkeypatch.setenv("FAKE_SD_MODE", mode)
    with pytest.raises(pipeline.SdCliError, match=match):
        _icon(pipeline.load_pipeline())


def test_token_count_grows_with_the_prompt() -> None:
    """Longer prompts cost more tokens than shorter ones.

    sd.cpp splits a long prompt into 77-token windows rather than truncating
    it, so this is the advisory signal that a prompt has grown past the first
    attention window, not a limit check.
    """
    pipe = pipeline.load_pipeline()
    short = prompt_budget.token_count(pipe, "a lantern")
    long_prompt = prompt_budget.token_count(pipe, "a lantern " * 40)
    check(
        condition=long_prompt > short,
        message=f"a longer prompt should cost more, got {long_prompt} vs {short}",
    )
    transformers = sys.modules["transformers"]
    check_equal(
        actual=transformers.loads[-1],
        expected=str(pipe.tokenizer_dir),
        message="the tokenizer comes from the SDXL snapshot",
    )
