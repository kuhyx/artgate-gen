"""Tests for the SD3.5-Large backbone, run against the library stubs.

What is under test is this module's own contract: it never downloads, it
offloads instead of moving the whole stack onto a 24 GB card, the LoRA trigger
leads the prompt exactly once, and the long T5 channel is actually requested.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import diffusers
import pytest

from artgate_gen import sd35
from artgate_gen.profiling import RunProfile
from artgate_gen.targets import TARGETS
from tests.assertions import check, check_equal

if TYPE_CHECKING:
    from artgate_gen.targets import Target

SEED = 5


def _small(size: tuple[int, int]) -> Target:
    """Shrink a real target's render so the stub image stays tiny.

    Args:
        size: The target's canvas key.

    Returns:
        The target with a 32px render.
    """
    return replace(TARGETS[size], render=(32, 32))


def _last_call(pipe: object) -> dict[str, Any]:
    """Return the keyword arguments of the stub pipeline's latest render.

    Args:
        pipe: A pipeline built by the stub loader.

    Returns:
        The recorded keyword arguments.
    """
    calls = cast("Any", pipe).calls
    return cast("dict[str, Any]", calls[-1])


def test_load_offloads_and_applies_the_lora() -> None:
    """The stack is offloaded, not moved whole onto the GPU."""
    pipe = cast("Any", sd35.load_pipeline())
    check(condition=pipe.offloaded, message="model CPU offload is enabled")
    check(condition=pipe.lora_loaded, message="the pixel-art LoRA is applied")
    check(condition=pipe.progress_disabled, message="the inference bar is silenced")
    check_equal(actual=pipe.device, expected="cpu", message="no blanket .to(cuda)")


def test_load_never_reaches_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every load is local-only: a 27 GB fetch is never a side effect."""
    seen: dict[str, object] = {}
    original = diffusers.DiffusionPipeline.from_pretrained

    def spy(*args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(diffusers.DiffusionPipeline, "from_pretrained", spy)
    sd35.load_pipeline()
    check_equal(
        actual=seen.get("local_files_only"),
        expected=True,
        message="from_pretrained is told not to download",
    )


def test_load_refuses_an_uncached_model() -> None:
    """A missing snapshot is a clear error naming the fetch command."""
    sys.modules["huggingface_hub"].missing.add(sd35.MODEL)
    with pytest.raises(sd35.NotCachedError, match="sd35_fetch"):
        sd35.load_pipeline()


def test_load_times_each_step_when_profiled() -> None:
    """A profile sees the pipeline, LoRA and offload steps separately."""
    with RunProfile(interval=0.01) as profile:
        sd35.load_pipeline(profile)
    names = {phase.name for phase in profile.phases}
    check_equal(
        actual=names,
        expected={"load:pipeline", "load:lora", "load:offload"},
        message="the timed load steps",
    )


def test_the_trigger_leads_the_prompt_exactly_once() -> None:
    """A style stem already opening with the trigger is not doubled."""
    pipe = sd35.load_pipeline()
    sd35.generate(pipe, "a fox", SEED, _small((256, 256)))
    prompt = str(_last_call(pipe)["prompt"])
    check(condition=prompt.startswith("pixel art, "), message=prompt)
    check_equal(actual=prompt.count("pixel art"), expected=1, message=prompt)


def test_the_trigger_is_added_when_the_stem_lacks_it() -> None:
    """A stem without the trigger gets it prepended."""
    pipe = sd35.load_pipeline()
    target = replace(_small((64, 64)), style="crisp")
    sd35.generate(pipe, "a fox", SEED, target)
    check_equal(
        actual=_last_call(pipe)["prompt"],
        expected="pixel art, crisp, a fox",
        message="the trigger is prepended",
    )


def test_generate_requests_the_long_channel_and_full_strength() -> None:
    """The T5 length and the LoRA scale reach the pipeline call."""
    pipe = sd35.load_pipeline()
    image = sd35.generate(pipe, "a fox", SEED, _small((256, 256)))
    call = _last_call(pipe)
    check_equal(
        actual=call["max_sequence_length"],
        expected=sd35.T5_MAX_LEN,
        message="the T5 window is raised past the 256 default",
    )
    check_equal(
        actual=call["joint_attention_kwargs"],
        expected={"scale": sd35.LORA_SCALE},
        message="the LoRA runs at its card's recommended strength",
    )
    check_equal(actual=image.size, expected=(32, 32), message="one render")
