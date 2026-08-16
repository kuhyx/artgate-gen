"""Tests for the structural types describing the diffusion stack.

These Protocols exist so our own call sites stay type-checked while the ML
libraries themselves are opaque. The test that matters is that the stub
pipeline actually satisfies them: if the Protocol and the stub drift apart,
the type checking it buys is fiction.
"""

from __future__ import annotations

from typing import Any, cast

import diffusers

from artgate_gen._types import InferenceResult, Pipeline
from tests.assertions import check, check_equal

SEED = 3
RENDER_EDGE = 64


def test_stub_pipeline_satisfies_the_protocol() -> None:
    """The stub is structurally a Pipeline, so the annotation is honest."""
    pipe = cast("Any", diffusers.DiffusionPipeline())
    check(condition=isinstance(pipe, Pipeline), message="stub matches the protocol")


def test_protocol_methods_are_callable_through_the_annotation() -> None:
    """Every declared member is reachable on a conforming object."""
    pipe = cast("Any", diffusers.DiffusionPipeline())
    check_equal(actual=pipe.to("cuda").device, expected="cuda", message="to()")
    pipe.set_progress_bar_config(disable=True)
    check(condition=pipe.progress_disabled, message="set_progress_bar_config()")
    check(condition=pipe.unet is not None, message="unet is exposed")
    check(condition=pipe.tokenizer is not None, message="tokenizer is exposed")

    state, alphas, metadata = type(pipe).lora_state_dict("repo", weight_name="w")
    check_equal(actual=state, expected={}, message="lora_state_dict state")
    check_equal(actual=alphas, expected={}, message="lora_state_dict alphas")
    check_equal(actual=metadata, expected={}, message="lora_state_dict metadata")

    pipe.load_lora_into_unet(state, unet=pipe.unet)
    check(condition=pipe.lora_loaded, message="load_lora_into_unet()")


def test_inference_result_exposes_images() -> None:
    """Calling a pipeline yields something with an images sequence."""
    pipe = cast("Any", diffusers.DiffusionPipeline())
    result = pipe(prompt="a lantern")
    check(
        condition=isinstance(result, InferenceResult),
        message="the result matches the protocol",
    )
    check_equal(
        actual=result.images[0].size,
        expected=(RENDER_EDGE, RENDER_EDGE),
        message="the result carries a rendered image",
    )
