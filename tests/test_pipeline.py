"""Tests for the torch-facing pipeline module.

Every call here runs against the stubs in ``tests/stubs``: the point is to
exercise this module's own branching -- the offline probe, the profiled and
unprofiled load paths, the UNet-only LoRA application -- not to test diffusers.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Any, cast

from artgate_gen import pipeline
from artgate_gen.profiling import RunProfile
from artgate_gen.targets import TARGETS
from tests.assertions import check, check_equal, check_length

RENDER_EDGE = 64
# Small on purpose: the real backdrop target renders at 2304x512, and no test
# allocates a megapixel to prove that width and height are threaded separately.
NON_SQUARE = (128, 32)
SEED = 7


def test_quiet_hub_registers_a_filter() -> None:
    """The deprecation filter is installed without raising."""
    pipeline.quiet_hub()


def test_local_only_is_true_when_every_repo_is_cached() -> None:
    """A complete cache means loads can skip the Hub."""
    check(condition=pipeline.local_only(), message="a full cache is offline-capable")


def test_local_only_is_false_when_a_repo_is_missing() -> None:
    """A partial cache forces the network path."""
    hub = sys.modules["huggingface_hub"]
    hub.missing.add(pipeline.LORA)
    check(
        condition=not pipeline.local_only(),
        message="a missing repo defeats the offline path",
    )


def test_local_only_probes_every_repo() -> None:
    """All three repositories are probed, not just the first."""
    hub = sys.modules["huggingface_hub"]
    pipeline.local_only()
    check_equal(
        actual=set(hub.calls),
        expected={pipeline.MODEL, pipeline.VAE, pipeline.LORA},
        message="every repo is probed",
    )


def test_load_pipeline_moves_to_cuda_and_applies_the_lora() -> None:
    """The loader lands on the GPU with the LoRA merged and bars silenced.

    The stub records these three facts; the Protocol deliberately does not
    declare them, so the concrete type is used for the assertions.
    """
    pipe = cast("Any", pipeline.load_pipeline())
    check_equal(actual=pipe.device, expected="cuda", message="moved to the GPU")
    check(condition=pipe.lora_loaded, message="the LoRA was applied")
    check(condition=pipe.progress_disabled, message="the inference bar is silenced")


def test_load_pipeline_records_phases_when_profiled() -> None:
    """Passing a profile times each load step separately."""
    with RunProfile(interval=0.01) as profile:
        pipeline.load_pipeline(profile)
    names = [phase.name for phase in profile.phases]
    check(condition="load:vae" in names, message="the VAE load is timed")
    check(condition="load:pipeline" in names, message="the pipeline load is timed")
    check(condition="load:lora" in names, message="the LoRA merge is timed")
    check(condition="load:to_cuda" in names, message="the transfer is timed")


def test_load_pipeline_without_a_profile_times_nothing() -> None:
    """The null span keeps the unprofiled path free of bookkeeping."""
    with RunProfile(interval=0.01) as profile:
        pipeline.load_pipeline()
    check_length(sized=profile.phases, expected=0, message="nothing was timed")


def test_generate_returns_one_image() -> None:
    """Rendering yields a single image from the pipeline result."""
    pipe = pipeline.load_pipeline()
    # The target is explicit so this assertion stays about the stub's render
    # size rather than about whatever the 64px icon target happens to render at.
    target = replace(TARGETS[64, 64], render=(RENDER_EDGE, RENDER_EDGE))
    image = pipeline.generate(pipe, "a brass lantern", SEED, target)
    check_equal(
        actual=image.size,
        expected=(RENDER_EDGE, RENDER_EDGE),
        message="the rendered image size",
    )


def test_generate_honours_a_non_square_target() -> None:
    """A non-square target renders at its own width and height."""
    pipe = pipeline.load_pipeline()
    target = replace(TARGETS[256, 64], render=NON_SQUARE)
    image = pipeline.generate(pipe, "a desert highway", SEED, target)
    check_equal(
        actual=image.size,
        expected=NON_SQUARE,
        message="the non-square rendered image size",
    )
