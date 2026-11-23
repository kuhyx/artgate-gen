"""Tests for the SD3.5-Large backbone, rendered through the fake ``sd-cli``.

What is under test is this module's own contract: it never downloads (the
weights sit behind a licence only kuhy can accept), it hands ``sd-cli`` the
q8_0 copies of the model and T5 plus the two CLIP encoders, all of which fit
a 24 GB card with no offload, and the LoRA trigger leads the prompt exactly
once.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from artgate_gen import sd35, sd_cli
from artgate_gen.profiling import RunProfile
from artgate_gen.targets import TARGETS
from tests.assertions import check, check_equal

if TYPE_CHECKING:
    from artgate_gen.targets import Target

SEED = 5


def _small(size: tuple[int, int]) -> Target:
    """Shrink a real target's render so the fake image stays tiny.

    Args:
        size: The target's canvas key.

    Returns:
        The target with a 32px render.
    """
    return replace(TARGETS[size], render=(32, 32))


def _flag(argv: list[str], name: str) -> str:
    """Return the value after ``name`` in a recorded argv.

    Args:
        argv: The recorded command line.
        name: The flag.

    Returns:
        Its value.
    """
    return argv[argv.index(name) + 1]


def _render_argv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Render once and return the argv the fake ``sd-cli`` received.

    Args:
        monkeypatch: Sets the fake's argv-record path.
        tmp_path: Where the record lands.

    Returns:
        The argv, without the program name.
    """
    record = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_SD_ARGV", str(record))
    sd35.generate(sd35.load_pipeline(), "a fox", SEED, _small((256, 256)))
    argv: list[str] = json.loads(record.read_text(encoding="utf-8"))
    return argv


def test_argv_names_the_model_and_all_three_text_encoders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The model (VAE inside) and T5 are q8_0 copies; CLIP stays as fetched."""
    argv = _render_argv(monkeypatch, tmp_path)
    for flag, name in (("--clip_l", sd35.CLIP_L), ("--clip_g", sd35.CLIP_G)):
        check(condition=_flag(argv, flag).endswith(name), message=flag)
    quantised = (("-m", "sd3.5_large_q8_0.gguf"), ("--t5xxl", "t5xxl_fp16_q8_0.gguf"))
    for flag, name in quantised:
        check_equal(
            actual=_flag(argv, flag),
            expected=str(tmp_path / "xdg-cache" / "artgate-gen" / "sd35" / name),
            message=flag,
        )
    check(condition="--vae" not in argv, message="the VAE is inside the model")
    # Offload moved 26 GB through a 10 GiB-capped box and swapped for an hour.
    check(condition="--offload-to-cpu" not in argv, message="all in VRAM")
    check_equal(
        actual=_flag(argv, "--flash-attn"),
        expected="off,diffusion=on",
        message="flash attention on the diffusion model only",
    )


def test_argv_applies_the_lora_with_the_card_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The LoRA tag, steps, CFG, sampler and seed reach the binary."""
    argv = _render_argv(monkeypatch, tmp_path)
    prompt = _flag(argv, "-p")
    check(condition=prompt.startswith("pixel art, "), message=prompt)
    check(condition=prompt.endswith(sd35.LORA_TAG), message=prompt)
    check_equal(
        actual=sd35.LORA_TAG,
        expected="<lora:pixel-art-3.5L-v2_000000300:1>",
        message="the tag names the LoRA file's stem",
    )
    check(
        condition=_flag(argv, "--lora-model-dir").endswith(sd35.LORA),
        message="the LoRA dir",
    )
    check_equal(actual=_flag(argv, "--steps"), expected="12", message="steps")
    check_equal(actual=_flag(argv, "--cfg-scale"), expected="3.5", message="cfg")
    check_equal(
        actual=_flag(argv, "--sampling-method"), expected="euler", message="sampler"
    )
    check_equal(actual=_flag(argv, "-s"), expected=str(SEED), message="seed")


def test_generate_returns_one_render_at_the_target_size() -> None:
    """The PNG ``sd-cli`` writes comes back as one RGB image."""
    image = sd35.generate(sd35.load_pipeline(), "a fox", SEED, _small((64, 64)))
    check_equal(actual=image.size, expected=(32, 32), message="one render")


def test_a_failed_render_raises_through_the_shared_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sd-cli's failure surfaces exactly as it does for SDXL."""
    monkeypatch.setenv("FAKE_SD_MODE", "fail")
    with pytest.raises(sd_cli.SdCliError, match="Vulkan buffer"):
        sd35.generate(sd35.load_pipeline(), "a fox", SEED, _small((64, 64)))


@pytest.fixture
def local_only(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record the ``local_files_only`` flag of every cache lookup ``sd35`` makes.

    Args:
        monkeypatch: Wraps the module's ``hf_hub_download``.

    Returns:
        The recorded flags, in call order.
    """
    flags: list[object] = []
    hub = sys.modules["huggingface_hub"]

    def spy(repo: str, filename: str, **kwargs: object) -> str:
        flags.append(kwargs.get("local_files_only"))
        return str(hub.hf_hub_download(repo, filename, **kwargs))

    monkeypatch.setattr(sd35, "hf_hub_download", spy)
    return flags


def test_load_never_reaches_the_network(local_only: list[object]) -> None:
    """Every lookup is local-only: a 27 GB fetch is never a side effect."""
    sd35.load_pipeline()
    check(condition=bool(local_only), message="the cache was consulted")
    check_equal(
        actual=set(local_only), expected={True}, message="never a network lookup"
    )


def test_load_refuses_uncached_weights_and_names_the_fetch(
    local_only: list[object],
) -> None:
    """A missing file is a clear error naming it and the fetch command."""
    sys.modules["huggingface_hub"].missing.add(sd35.MODEL)
    with pytest.raises(sd35.NotCachedError, match="sd35_fetch") as caught:
        sd35.load_pipeline()
    check(condition=sd35.CLIP_L in str(caught.value), message=str(caught.value))
    check_equal(actual=set(local_only), expected={True}, message="no download started")


def test_load_refuses_missing_quantised_copies_and_names_the_fetch() -> None:
    """Cached safetensors without their q8_0 copies still stop the render."""
    sd35.quant_path(sd35.T5XXL).unlink()
    with pytest.raises(sd35.NotCachedError, match="sd35_fetch --download") as caught:
        sd35.load_pipeline()
    message = str(caught.value)
    check(condition="t5xxl_fp16_q8_0.gguf" in message, message=message)
    check(condition="sd3.5_large_q8_0" not in message, message=message)


def test_quant_dir_defaults_under_home_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without XDG_CACHE_HOME the copies sit in ~/.cache, not the HF cache."""
    monkeypatch.delenv("XDG_CACHE_HOME")
    check_equal(
        actual=sd35.quant_dir(),
        expected=Path.home() / ".cache" / "artgate-gen" / "sd35",
        message="default cache dir",
    )


def test_load_times_the_resolution_when_profiled() -> None:
    """A profile sees the resolution step."""
    with RunProfile(interval=0.01) as profile:
        sd35.load_pipeline(profile)
    names = {phase.name for phase in profile.phases}
    check_equal(actual=names, expected={"load:resolve"}, message="timed steps")


def test_the_trigger_leads_the_prompt_exactly_once() -> None:
    """A style stem already opening with the trigger is not doubled."""
    prompt = sd35.full_prompt("a fox", _small((256, 256)))
    check(condition=prompt.startswith("pixel art, "), message=prompt)
    check_equal(actual=prompt.count("pixel art"), expected=1, message=prompt)


def test_the_trigger_is_added_when_the_stem_lacks_it() -> None:
    """A stem without the trigger gets it prepended."""
    target = replace(_small((64, 64)), style="crisp")
    check_equal(
        actual=sd35.full_prompt("a fox", target),
        expected="pixel art, crisp, a fox",
        message="the trigger is prepended",
    )


def test_token_count_uses_the_sd35_tokenizer() -> None:
    """CLIP tokens are counted with no torch, from the SD3.5 snapshot."""
    pipe = sd35.load_pipeline()
    short = sd35.token_count(pipe, "a fox")
    longer = sd35.token_count(pipe, "a fox with a red scarf and a lantern")
    check(condition=longer > short > 0, message=f"{short} -> {longer}")
    check(
        condition=sd35.MODEL in str(pipe.tokenizer_dir),
        message="the SD3.5 tokenizer",
    )
